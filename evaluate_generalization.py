import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torchdiffeq import odeint
import sentencepiece as spm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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


def load_json_sentences(path, max_sentences=300):
    with open(path, 'r') as f:
        data = json.load(f)
    sentences = []
    roles = []
    for item in data:
        if 'messages' not in item:
            continue
        for msg in item['messages']:
            if msg.get('role') not in ('user', 'assistant'):
                continue
            text = msg.get('content', '').strip()
            if not text:
                continue
            if len(text.split()) < 3:
                continue
            sentences.append(text)
            roles.append(msg.get('role'))
            if len(sentences) >= max_sentences:
                return sentences, roles
    return sentences, roles


def build_template_sentences(max_count=200):
    subjects = ['Mary', 'John', 'The boy', 'The girl', 'A man', 'The driver', 'The soldier', 'The cat', 'The teacher', 'The child']
    verbs_action = ['opens', 'closes', 'kicks', 'picks up', 'throws', 'runs to', 'rushes into', 'walks to', 'pushes', 'pulls']
    objects = ['the door', 'the box', 'the old sword', 'the window', 'the gate', 'the bag', 'the book', 'the table', 'the chair', 'the car']
    adverbs = ['quickly', 'silently', 'carefully', 'slowly', 'at night', 'in the morning', 'at dawn', 'without warning', 'hastily', 'calmly']

    action_sentences = []
    for s in subjects:
        for v in verbs_action[:5]:
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


def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)


def main():
    base = Path(__file__).resolve().parent
    data_dir = base / 'kahler_fineweb'
    report_path = data_dir / 'generalization_report.json'
    plot_path = data_dir / 'generalization_plot.png'
    pdf_path = data_dir / 'generalization_report.pdf'
    raw_sentences_path = Path('/home/fausap/Documents/LLM/FINET/darkconvos.json')
    proj_state_path = data_dir / 'projection_head_best.pt'
    config_path = data_dir / 'config.json'
    ckpt_path = data_dir / 'kahler_weights.pt'
    tokenizer_path = base / 'fineweb_tokenizer.model'

    assert raw_sentences_path.exists(), f'Missing dataset: {raw_sentences_path}'
    assert config_path.exists(), f'Missing config: {config_path}'
    assert ckpt_path.exists(), f'Missing checkpoint: {ckpt_path}'
    assert tokenizer_path.exists(), f'Missing tokenizer: {tokenizer_path}'
    assert proj_state_path.exists(), f'Missing projection head: {proj_state_path}'

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
    action_sentences, descriptive_sentences = build_template_sentences(max_count=200)
    template_sentences = action_sentences + descriptive_sentences
    template_labels = np.array([1] * len(action_sentences) + [0] * len(descriptive_sentences), dtype=int)

    print('Embedding template sentences...')
    template_embs = embed_sentences(model, tokenizer, template_sentences, ODE_t, device)
    template_proj = None
    if proj_state_path.exists():
        proj_head = ProjectionHead(template_embs.shape[1]).to(device)
        proj_head.load_state_dict(torch.load(proj_state_path, map_location=device))
        proj_head.eval()
        with torch.no_grad():
            template_proj = proj_head(torch.tensor(template_embs.astype('float32'), device=device)).cpu().numpy()

    print('Loading natural sentences...')
    sentences, roles = load_json_sentences(raw_sentences_path, max_sentences=300)
    role_labels = np.array([0 if r == 'user' else 1 for r in roles], dtype=int)
    print(f'Loaded {len(sentences)} sentences from dataset')

    print('Embedding natural sentences...')
    natural_embs = embed_sentences(model, tokenizer, sentences, ODE_t, device)
    natural_proj = None
    if proj_state_path.exists():
        with torch.no_grad():
            natural_proj = proj_head(torch.tensor(natural_embs.astype('float32'), device=device)).cpu().numpy()

    metrics = {}
    if len(set(role_labels)) > 1:
        try:
            metrics['raw_role_silhouette'] = float(silhouette_score(natural_embs, role_labels, metric='cosine'))
        except Exception as e:
            metrics['raw_role_silhouette'] = None
            print('raw role silhouette error', e)
        if natural_proj is not None:
            try:
                metrics['proj_role_silhouette'] = float(silhouette_score(natural_proj, role_labels, metric='cosine'))
            except Exception as e:
                metrics['proj_role_silhouette'] = None
                print('proj role silhouette error', e)

    if template_proj is not None:
        raw_action_centroid = template_embs[template_labels == 1].mean(axis=0)
        raw_desc_centroid = template_embs[template_labels == 0].mean(axis=0)
        proj_action_centroid = template_proj[template_labels == 1].mean(axis=0)
        proj_desc_centroid = template_proj[template_labels == 0].mean(axis=0)

        raw_norm = normalize_rows(natural_embs)
        raw_action_norm = raw_action_centroid / (np.linalg.norm(raw_action_centroid) + 1e-12)
        raw_desc_norm = raw_desc_centroid / (np.linalg.norm(raw_desc_centroid) + 1e-12)
        proj_norm = normalize_rows(natural_proj)
        proj_action_norm = proj_action_centroid / (np.linalg.norm(proj_action_centroid) + 1e-12)
        proj_desc_norm = proj_desc_centroid / (np.linalg.norm(proj_desc_centroid) + 1e-12)

        raw_sim_action = (raw_norm * raw_action_norm).sum(axis=1)
        raw_sim_desc = (raw_norm * raw_desc_norm).sum(axis=1)
        proj_sim_action = (proj_norm * proj_action_norm).sum(axis=1)
        proj_sim_desc = (proj_norm * proj_desc_norm).sum(axis=1)

        metrics.update({
            'raw_mean_cosine_to_action': float(raw_sim_action.mean()),
            'raw_mean_cosine_to_descriptive': float(raw_sim_desc.mean()),
            'proj_mean_cosine_to_action': float(proj_sim_action.mean()),
            'proj_mean_cosine_to_descriptive': float(proj_sim_desc.mean()),
        })

        raw_nearest = (raw_sim_action >= raw_sim_desc).astype(int)
        proj_nearest = (proj_sim_action >= proj_sim_desc).astype(int)
        metrics['raw_nearest_action_ratio'] = float(raw_nearest.mean())
        metrics['proj_nearest_action_ratio'] = float(proj_nearest.mean())
        metrics['raw_nearest_agreement_with_role'] = None
        metrics['proj_nearest_agreement_with_role'] = None
        if len(set(role_labels)) > 1:
            metrics['raw_nearest_agreement_with_role'] = float((raw_nearest == role_labels).mean())
            metrics['proj_nearest_agreement_with_role'] = float((proj_nearest == role_labels).mean())

        example_indices_action = np.argsort(proj_sim_action)[-10:][::-1]
        example_indices_desc = np.argsort(proj_sim_desc)[-10:][::-1]
    else:
        example_indices_action = []
        example_indices_desc = []

    metrics['template_action_count'] = int((template_labels == 1).sum())
    metrics['template_descriptive_count'] = int((template_labels == 0).sum())
    metrics['natural_sentence_count'] = len(sentences)

    print('Saving report and plots...')

    summary = {
        'metrics': metrics,
        'natural_sentences_count': len(sentences),
        'natural_role_counts': {
            'user': int((role_labels == 0).sum()),
            'assistant': int((role_labels == 1).sum()),
        },
        'top_natural_action_like_sentences': [sentences[i] for i in example_indices_action[:10]],
        'top_natural_descriptive_like_sentences': [sentences[i] for i in example_indices_desc[:10]],
    }

    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    for ax, X_data, title in [(axs[0], natural_embs, 'Raw natural embeddings'), (axs[1], natural_proj, 'Projected natural embeddings')]:
        if X_data is None:
            ax.text(0.5, 0.5, 'Missing data', ha='center', va='center')
            ax.set_axis_off()
            continue
        pca = PCA(n_components=2, random_state=42)
        Y = pca.fit_transform(X_data)
        labels_plot = proj_nearest if X_data is natural_proj else raw_nearest
        colors = ['tab:blue' if l == 0 else 'tab:orange' for l in labels_plot]
        markers = ['o' if r == 0 else 's' for r in role_labels]
        for i, (x, y, c, m) in enumerate(zip(Y[:, 0], Y[:, 1], colors, markers)):
            ax.scatter(x, y, c=c, marker=m, s=20, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        action_patch = plt.Line2D([0], [0], marker='o', color='w', label='action-like', markerfacecolor='tab:orange', markersize=8)
        desc_patch = plt.Line2D([0], [0], marker='o', color='w', label='descriptive-like', markerfacecolor='tab:blue', markersize=8)
        user_patch = plt.Line2D([0], [0], marker='o', color='k', label='user', markerfacecolor='none', markersize=8)
        assistant_patch = plt.Line2D([0], [0], marker='s', color='k', label='assistant', markerfacecolor='none', markersize=8)
        ax.legend(handles=[action_patch, desc_patch, user_patch, assistant_patch], loc='best', fontsize='small')

    fig.suptitle('Generalization evaluation of natural FINET sentences')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        fig_text = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        text = [f'Generalization evaluation report', '', f'Natural sentence count: {len(sentences)}', '']
        for k, v in metrics.items():
            text.append(f'{k}: {v}')
        text.append('')
        text.append('Top action-like natural sentences:')
        text.extend([f'- {s}' for s in summary['top_natural_action_like_sentences']])
        text.append('')
        text.append('Top descriptive-like natural sentences:')
        text.extend([f'- {s}' for s in summary['top_natural_descriptive_like_sentences']])
        plt.text(0.01, 0.99, '\n'.join(text), va='top', family='monospace', fontsize=8)
        pdf.savefig(fig_text)
        plt.close(fig_text)

    print('Report saved to', report_path)
    print('Plot saved to', plot_path)
    print('PDF report saved to', pdf_path)


if __name__ == '__main__':
    main()
