import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import silhouette_score
from torchdiffeq import odeint
import sentencepiece as spm


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class MLPPotential(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.real_dim = dim * 2
        self.net = nn.Sequential(
            nn.Linear(self.real_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer('lambda_prior', torch.tensor(0.01))

    def forward(self, z):
        x = torch.view_as_real(z).reshape(-1, self.real_dim)
        phi = self.net(x).squeeze(-1)
        prior = self.lambda_prior * torch.sum(x ** 2, dim=-1)
        return phi + prior


class ContextualKahlerODE(nn.Module):
    def __init__(self, potential_net, dim, num_heads=8):
        super().__init__()
        self.potential_net = potential_net
        self.dim = dim
        self.attention = nn.MultiheadAttention(embed_dim=dim * 2, num_heads=num_heads, batch_first=True)

    def forward(self, t, z):
        seq_len = z.shape[1]
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.requires_grad_(True)
            z_real = torch.view_as_real(z).reshape(z.shape[0], seq_len, -1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(z.device)
            context_real, _ = self.attention(z_real, z_real, z_real, is_causal=True, attn_mask=causal_mask)
            context_complex = torch.complex(context_real[..., :self.dim], context_real[..., self.dim:])
            u = context_complex + z
            K = self.potential_net(u).sum()
            grad_u = torch.autograd.grad(K, u, create_graph=True)[0]
        update = -(1.0 + 1.0j) * grad_u
        MAX_FIELD = 10.0
        norm = torch.abs(update).norm(dim=-1, keepdim=True)
        scale = torch.clamp(MAX_FIELD / (norm + 1e-6), max=1.0)
        update = update * scale
        return update


class KahlerTransformerODE(nn.Module):
    def __init__(self, vocab_size, dim=16, num_blocks=3):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.embed_real = nn.Embedding(vocab_size, dim)
        self.embed_imag = nn.Embedding(vocab_size, dim)
        self.potentials = nn.ModuleList([MLPPotential(dim, hidden_dim=dim * 4) for _ in range(num_blocks)])
        self.ode_funcs = nn.ModuleList([ContextualKahlerODE(self.potentials[i], dim, num_heads=8) for i in range(num_blocks)])
        self.project = nn.Linear(dim * 2, vocab_size)

    def embed(self, x):
        real_part = self.embed_real(x)
        imag_part = self.embed_imag(x)
        return torch.complex(real_part, imag_part)

    def forward(self, x, t_span):
        z = self.embed(x)
        for ode_func in self.ode_funcs:
            z = odeint(ode_func, z, t_span, method='euler', options={'step_size': 1.25})[-1]
        zt_real = torch.view_as_real(z).reshape(z.shape[0], z.shape[1], -1)
        return self.project(zt_real), z


class KahlerTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.n_vocab = self.sp.get_piece_size()
        self.eos_id = self.sp.eos_id()
        self.bos_id = self.sp.bos_id()
        self.pad_id = self.sp.pad_id()

    def encode_ordinary(self, text):
        return self.sp.encode(text)

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return self.sp.decode(ids)

    @property
    def eot_token(self):
        return self.eos_id


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return z / (z.norm(dim=1, keepdim=True) + 1e-12)


class MixedEmbeddingDataset(Dataset):
    def __init__(self, template_embs, template_labels, natural_embs):
        self.template_embs = template_embs.astype('float32')
        self.template_labels = template_labels.astype('int64')
        self.natural_embs = natural_embs.astype('float32')

    def __len__(self):
        return len(self.template_embs) + len(self.natural_embs)

    def __getitem__(self, idx):
        if idx < len(self.template_embs):
            return self.template_embs[idx], int(self.template_labels[idx]), 1
        else:
            natural_idx = idx - len(self.template_embs)
            return self.natural_embs[natural_idx], -1, 0


def build_template_sentences(max_count=200):
    subjects = ['Mary', 'John', 'The boy', 'The girl', 'A man', 'The driver', 'The soldier', 'The cat', 'The teacher', 'The child']
    verbs = ['opens', 'closes', 'kicks', 'picks up', 'throws', 'runs to', 'rushes into', 'walks to', 'pushes', 'pulls']
    objects = ['the door', 'the box', 'the old sword', 'the window', 'the gate', 'the bag', 'the book', 'the table', 'the chair', 'the car']
    adverbs = ['quickly', 'silently', 'carefully', 'slowly', 'at night', 'in the morning', 'at dawn', 'without warning', 'hastily', 'calmly']
    action_sentences = []
    for s in subjects:
        for v in verbs[:5]:
            for o in objects[:3]:
                for a in adverbs[:2]:
                    action_sentences.append(f"{s} {v} {o} {a}")
                    if len(action_sentences) >= max_count:
                        break
                if len(action_sentences) >= max_count:
                    break
            if len(action_sentences) >= max_count:
                break
        if len(action_sentences) >= max_count:
            break
    nouns = ['The sky', 'The old house', 'The road', 'The garden', 'The room', 'The city', 'The forest', 'The river', 'The mountain', 'The building']
    adjs = ['dark and cold', 'silent and empty', 'long and winding', 'beautiful and green', 'old and creaky', 'humid and warm', 'quiet and still', 'desolate and bare', 'bright and clean', 'foggy and grey']
    preps = ['on the hill', 'near the sea', 'by the river', 'in the valley', 'under the stars', 'beside the road', 'at the edge of town', 'behind the fence', 'below the cliff', 'across the street']
    descriptive_sentences = []
    for n in nouns:
        for adj in adjs[:5]:
            for p in preps[:2]:
                descriptive_sentences.append(f"{n} was {adj} {p}")
                if len(descriptive_sentences) >= max_count:
                    break
            if len(descriptive_sentences) >= max_count:
                break
        if len(descriptive_sentences) >= max_count:
            break
    return action_sentences[:max_count], descriptive_sentences[:max_count]


def load_natural_sentences(json_path, max_sentences=300):
    with open(json_path, 'r') as f:
        data = json.load(f)
    sentences = []
    for item in data:
        if not isinstance(item, dict):
            continue
        for msg in item.get('messages', []):
            role = msg.get('role')
            if role not in ('user', 'assistant'):
                continue
            text = msg.get('content', '').strip()
            if len(text.split()) < 3:
                continue
            sentences.append(text)
            if len(sentences) >= max_sentences:
                return sentences
    return sentences


def embed_sentences(model, tokenizer, sentences, ODE_t, device):
    embs = []
    with torch.no_grad():
        for text in sentences:
            toks = tokenizer.encode_ordinary(text)
            if len(toks) == 0:
                toks = [tokenizer.bos_id]
            input_ids = torch.tensor([toks], dtype=torch.long, device=device)
            z = model.embed(input_ids)
            for ode_func in model.ode_funcs:
                z = odeint(ode_func, z, ODE_t, method='euler', options={'step_size': 1.25})[-1]
            z_mean = torch.view_as_real(z[0]).reshape(z.shape[1], -1).mean(dim=0)
            embs.append(z_mean.cpu().numpy())
    return np.stack(embs, axis=0)


def contrastive_loss(z, labels=None, temperature=0.07):
    z = z / (z.norm(dim=1, keepdim=True) + 1e-12)
    sim = torch.matmul(z, z.t()) / temperature
    logits_max, _ = torch.max(sim, dim=1, keepdim=True)
    logits = sim - logits_max.detach()
    if labels is not None:
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(z.device)
        self_mask = torch.eye(mask.shape[0], device=z.device)
        mask = mask - self_mask
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        valid = (mask.sum(1) > 0).float()
        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)
        loss = -(mean_log_prob_pos * valid).sum() / valid.sum()
        return loss
    else:
        batch_size = z.shape[0]
        labels = torch.arange(batch_size, device=z.device)
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        self_mask = torch.eye(batch_size, device=z.device)
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob.mean()
        return loss


def augment_embeddings(X, noise_scale=0.1):
    noise = torch.randn_like(X) * noise_scale
    return X + noise


def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--natural-json', default='/home/fausap/Documents/LLM/FINET/darkconvos.json')
    parser.add_argument('--natural-max', type=int, default=300)
    parser.add_argument('--template-count', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--proj-dim', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--output-dir', default='kahler_fineweb')
    parser.add_argument('--init-proj', default='kahler_fineweb/projection_head_best.pt')
    args = parser.parse_args(argv)

    base = Path(__file__).resolve().parent
    output_dir = base / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'mixed_finetune_report.json'
    proj_out_path = output_dir / 'projection_head_mixed.pt'
    emb_out_path = output_dir / 'silhouette_embeddings_proj_mixed.npy'

    config_path = output_dir / 'config.json'
    ckpt_path = output_dir / 'kahler_weights.pt'
    tokenizer_path = base / 'fineweb_tokenizer.model'

    assert config_path.exists(), f'Missing config: {config_path}'
    assert ckpt_path.exists(), f'Missing checkpoint: {ckpt_path}'
    assert tokenizer_path.exists(), f'Missing tokenizer: {tokenizer_path}'
    assert Path(args.natural_json).exists(), f'Missing natural JSON: {args.natural_json}'

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    vocab_size = cfg.get('vocab_size', 16384)
    dim = cfg.get('dim', 384)
    num_blocks = cfg.get('num_blocks', 3)
    ODE_DEPTH = 2.5
    ODE_t = torch.tensor([0.0, ODE_DEPTH], device=get_device())

    device = get_device()
    print('Using device:', device)

    model = KahlerTransformerODE(vocab_size=vocab_size, dim=dim, num_blocks=num_blocks).to(device)
    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)
    model.eval()

    tokenizer = KahlerTokenizer(str(tokenizer_path))
    action_sentences, descriptive_sentences = build_template_sentences(max_count=args.template_count)
    template_sentences = action_sentences + descriptive_sentences
    template_labels = np.array([1] * len(action_sentences) + [0] * len(descriptive_sentences), dtype=int)

    print('Embedding template sentences...')
    template_embs = embed_sentences(model, tokenizer, template_sentences, ODE_t, device)
    natural_sentences = load_natural_sentences(args.natural_json, max_sentences=args.natural_max)
    print(f'Loaded {len(natural_sentences)} natural sentences')
    natural_embs = embed_sentences(model, tokenizer, natural_sentences, ODE_t, device)

    if os.path.exists(args.init_proj):
        init_proj = ProjectionHead(template_embs.shape[1], hidden=256, out_dim=args.proj_dim).to(device)
        init_proj.load_state_dict(torch.load(args.init_proj, map_location=device))
        print(f'Loaded init projection head from {args.init_proj}')
    else:
        init_proj = None

    dataset = MixedEmbeddingDataset(template_embs, template_labels, natural_embs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model_proj = ProjectionHead(template_embs.shape[1], hidden=256, out_dim=args.proj_dim).to(device)
    if init_proj is not None:
        model_proj.load_state_dict(init_proj.state_dict())
    optimizer = torch.optim.AdamW(model_proj.parameters(), lr=args.lr, weight_decay=1e-5)

    best_template_sil = -1e9
    best_epoch = -1
    report = {'epochs': []}

    for epoch in range(1, args.epochs + 1):
        model_proj.train()
        total_loss = 0.0
        steps = 0
        for xb, yb, is_template in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            is_template = is_template.to(device)
            z = model_proj(xb)
            sup_mask = (is_template == 1)
            loss = torch.tensor(0.0, device=device)
            if sup_mask.any():
                sup_z = z[sup_mask]
                sup_labels = yb[sup_mask]
                loss = loss + contrastive_loss(sup_z, labels=sup_labels)
            if (~sup_mask).any():
                nat_z = z[~sup_mask]
                z_a = nat_z
                z_b = augment_embeddings(nat_z, noise_scale=0.08)
                loss = loss + args.alpha * contrastive_loss(torch.cat([z_a, z_b], dim=0), labels=None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().numpy())
            steps += 1
        avg_loss = total_loss / max(1, steps)

        model_proj.eval()
        with torch.no_grad():
            template_proj = model_proj(torch.tensor(template_embs.astype('float32'), device=device)).cpu().numpy()
            natural_proj = model_proj(torch.tensor(natural_embs.astype('float32'), device=device)).cpu().numpy()
        try:
            template_sil = float(silhouette_score(template_proj, template_labels, metric='cosine'))
        except Exception:
            template_sil = None
        raw_sim_action = np.dot(normalize_rows(natural_proj), normalize_rows(template_proj[template_labels == 1]).mean(axis=0))
        raw_sim_desc = np.dot(normalize_rows(natural_proj), normalize_rows(template_proj[template_labels == 0]).mean(axis=0))
        natural_action_ratio = float((raw_sim_action >= raw_sim_desc).mean())

        report['epochs'].append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'template_cosine_silhouette': template_sil,
            'natural_action_ratio': natural_action_ratio,
        })

        if template_sil is not None and template_sil > best_template_sil:
            best_template_sil = template_sil
            best_epoch = epoch
            torch.save(model_proj.state_dict(), proj_out_path)
            np.save(emb_out_path, natural_proj)

        print(f'Epoch {epoch:02d}: loss={avg_loss:.4f} template_sil={template_sil} natural_action_ratio={natural_action_ratio:.4f}')

    report['best'] = {'best_epoch': best_epoch, 'best_template_cosine_silhouette': best_template_sil}
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print('Finished. Best epoch:', best_epoch)
    print('Saved mixed projection head to', proj_out_path)
    print('Saved mixed natural embeddings to', emb_out_path)
    print('Saved report to', report_path)


if __name__ == '__main__':
    main()
