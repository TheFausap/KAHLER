"""
Run a larger silhouette evaluation for the Kahler ODE model.

Usage (in your local dev environment with PyTorch installed):

python3 KAHLER/eval_silhouette.py

Install dependencies (recommended in a venv):
pip install torch torchvision torchaudio  # or follow official install for CUDA
pip install torchdiffeq scikit-learn sentencepiece numpy

This script loads KAHLER/kahler_fineweb/kahler_weights.pt and KAHLER/fineweb_tokenizer.model
and saves embeddings to KAHLER/kahler_fineweb/silhouette_embeddings.npy

"""
import json
import os
import torch
import numpy as np
from sklearn.metrics import silhouette_score
from torchdiffeq import odeint
import sentencepiece as spm
import torch.nn as nn

# Device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

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
        self.potentials = nn.ModuleList([MLPPotential(dim, hidden_dim=dim*4) for _ in range(num_blocks)])
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


def main():
    config_path = os.path.join('kahler_fineweb', 'config.json')
    ckpt_path = os.path.join('kahler_fineweb', 'kahler_weights.pt')
    tokenizer_path = os.path.join('./', 'fineweb_tokenizer.model')

    assert os.path.exists(config_path), f'config not found: {config_path}'
    assert os.path.exists(ckpt_path), f'checkpoint not found: {ckpt_path}'
    assert os.path.exists(tokenizer_path), f'tokenizer not found: {tokenizer_path}'

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    vocab_size = cfg.get('vocab_size', 16384)
    dim = cfg.get('dim', 384)
    num_blocks = cfg.get('num_blocks', 3)
    ODE_DEPTH = 2.5

    print('Using device:', device)
    model = KahlerTransformerODE(vocab_size=vocab_size, dim=dim, num_blocks=num_blocks).to(device)
    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)

    model.eval()
    enc = KahlerTokenizer(tokenizer_path)

    # Build templated sentences
    subjects = ['Mary','John','The boy','The girl','A man','The driver','The soldier','The cat','The teacher','The child']
    verbs_action = ['opens','closes','kicks','picks up','throws','runs to','rushes into','walks to','pushes','pulls']
    objects = ['the door','the box','the old sword','the window','the gate','the bag','the book','the table','the chair','the car']
    adverbs = ['quickly','silently','carefully','slowly','at night','in the morning','at dawn','without warning','hastily','calmly']

    action_sentences = []
    for s in subjects:
        for v in verbs_action[:5]:
            for o in objects[:3]:
                for a in adverbs[:2]:
                    action_sentences.append(f"{s} {v} {o} {a}")
                    if len(action_sentences) >= 200:
                        break
                if len(action_sentences) >= 200:
                    break
            if len(action_sentences) >= 200:
                break
        if len(action_sentences) >= 200:
            break

    nouns = ['The sky','The old house','The road','The garden','The room','The city','The forest','The river','The mountain','The building']
    adjs = ['dark and cold','silent and empty','long and winding','beautiful and green','old and creaky','humid and warm','quiet and still','desolate and bare','bright and clean','foggy and grey']
    preps = ['on the hill','near the sea','by the river','in the valley','under the stars','beside the road','at the edge of town','behind the fence','below the cliff','across the street']

    descriptive_sentences = []
    for n in nouns:
        for adj in adjs[:5]:
            for p in preps[:2]:
                descriptive_sentences.append(f"{n} was {adj} {p}")
                if len(descriptive_sentences) >= 200:
                    break
            if len(descriptive_sentences) >= 200:
                break
        if len(descriptive_sentences) >= 200:
            break

    action_sentences = action_sentences[:200]
    descriptive_sentences = descriptive_sentences[:200]
    sentences = action_sentences + descriptive_sentences
    labels = np.array([1]*len(action_sentences) + [0]*len(descriptive_sentences))

    embs = []
    ODE_t = torch.tensor([0.0, ODE_DEPTH], device=device)

    with torch.no_grad():
        for s in sentences:
            toks = enc.encode_ordinary(s)
            if len(toks) == 0:
                toks = [enc.bos_id]
            input_ids = torch.tensor([toks], dtype=torch.long, device=device)
            z = model.embed(input_ids)
            for ode_func in model.ode_funcs:
                z = odeint(ode_func, z, ODE_t, method='euler', options={'step_size': 1.25})[-1]
            z_mean = torch.view_as_real(z[0]).reshape(z.shape[1], -1).mean(dim=0)
            embs.append(z_mean.cpu().numpy())

    X = np.stack(embs, axis=0)
    sil = silhouette_score(X, labels)
    print('Silhouette score (action=1 vs descriptive=0):', sil)

    out_dir = 'kahler_fineweb'
    np.save(os.path.join(out_dir, 'silhouette_embeddings.npy'), X)
    np.save(os.path.join(out_dir, 'silhouette_labels.npy'), labels)
    print('Saved embeddings to', out_dir)

if __name__ == '__main__':
    main()
