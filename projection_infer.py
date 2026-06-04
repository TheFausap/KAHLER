import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import sentencepiece as spm


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


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
            grad_u = torch.autograd.grad(K, u, create_graph=self.training)[0]
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


def build_model(config_path, ckpt_path, tokenizer_path, device=None):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    vocab_size = cfg.get('vocab_size', 16384)
    dim = cfg.get('dim', 384)
    num_blocks = cfg.get('num_blocks', 3)
    device = device or get_device()
    model = KahlerTransformerODE(vocab_size=vocab_size, dim=dim, num_blocks=num_blocks).to(device)
    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)
    model.eval()
    tokenizer = KahlerTokenizer(tokenizer_path)
    return model, tokenizer


def load_projection_head(path, input_dim, device=None):
    device = device or get_device()
    state = torch.load(path, map_location=device)
    if 'net.0.weight' in state or 'net.2.weight' in state:
        proj = ProjectionHead(input_dim).to(device)
        proj.load_state_dict(state)
        proj.eval()
        return proj
    raise ValueError(f'Unable to load projection head from {path}')


def project_embeddings(proj_head, X, device=None):
    device = device or get_device()
    X_t = torch.tensor(X.astype('float32'), device=device)
    with torch.no_grad():
        Z = proj_head(X_t).cpu().numpy()
    return Z


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Project embeddings using a trained projection head.')
    parser.add_argument('--projection-head', default='kahler_fineweb/projection_head_mixed.pt')
    parser.add_argument('--input-emb', type=str, help='Path to a NumPy .npy file of raw embeddings')
    parser.add_argument('--output-emb', type=str, default='kahler_fineweb/projected_embeddings.npy')
    parser.add_argument('--sentence', type=str, help='Single sentence to encode and project')
    parser.add_argument('--sentence-file', type=str, help='Text file with one sentence per line to encode and project')
    parser.add_argument('--config', type=str, default='kahler_fineweb/config.json')
    parser.add_argument('--checkpoint', type=str, default='kahler_fineweb/kahler_weights.pt')
    parser.add_argument('--tokenizer', type=str, default='fineweb_tokenizer.model')
    parser.add_argument('--output-json', type=str, help='Optional JSON summary output')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = get_device()

    if args.input_emb is None and args.sentence is None and args.sentence_file is None:
        raise ValueError('Specify --input-emb, --sentence, or --sentence-file')

    if not os.path.exists(args.projection_head):
        raise FileNotFoundError(f'Projection head not found: {args.projection_head}')

    X = None
    if args.input_emb:
        X = np.load(args.input_emb)
        output_path = Path(args.output_emb)
    else:
        assert args.sentence or args.sentence_file
        model, tokenizer = build_model(args.config, args.checkpoint, args.tokenizer, device=device)
        ODE_t = torch.tensor([0.0, 2.5], device=device)
        sentences = []
        if args.sentence:
            sentences.append(args.sentence)
        if args.sentence_file:
            with open(args.sentence_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sentences.append(line)
        X = embed_sentences(model, tokenizer, sentences, ODE_t, device)
        base = Path(args.sentence_file or 'sentence')
        output_path = Path(args.output_emb) if args.output_emb else Path(f'projected_{base.stem}.npy')

    proj_head = load_projection_head(args.projection_head, input_dim=X.shape[1], device=device)
    projected = project_embeddings(proj_head, X, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, projected)
    print(f'Saved projected embeddings: {output_path}')

    summary = {
        'projection_head': args.projection_head,
        'input_shape': list(X.shape),
        'projected_shape': list(projected.shape),
        'output_path': str(output_path),
    }
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f'Saved summary: {args.output_json}')


if __name__ == '__main__':
    main()
