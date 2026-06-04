#!/usr/bin/env python3
"""
Standalone generation script for the Kähler ODE Transformer.
Loads a saved checkpoint and generates text interactively.

Usage:
    python generate.py                                  # interactive mode
    python generate.py --prompt "The old man walked"    # single prompt
    python generate.py --checkpoint path/to/weights.pt  # custom checkpoint
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import sentencepiece as spm

# ──────────────────────────────────────────────────────────
# 1. MODEL ARCHITECTURE (must match training code exactly)
# ──────────────────────────────────────────────────────────

class MLPPotential(nn.Module):
    """
    4-layer MLP scalar potential phi(z) : C^d -> R.
    Non-convex, so the gradient flow can have multiple attractors.
    Fixed quadratic stabilizer keeps phi bounded below at infinity.
    """
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
            K = self.potential_net(context_complex + z).sum()
            grad_z = torch.autograd.grad(K, z, create_graph=True)[0]

        update = -(1.0 + 1.0j) * grad_z

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

# ──────────────────────────────────────────────────────────
# 2. GENERATION
# ──────────────────────────────────────────────────────────

def generate(model, enc, prompt, seq_len=128, ode_depth=2.0,
             max_new_tokens=100, temperature=0.8, top_k=50,
             repetition_penalty=1.2, device='cpu'):
    """Generate text using a sliding window of up to seq_len tokens.

    No padding is applied — the model receives only real tokens, matching
    the dense contiguous windows it saw during training.  Once the generated
    sequence exceeds seq_len the oldest tokens are dropped (sliding window).

    Args:
        model: loaded KahlerTransformerODE
        enc: tiktoken encoding
        prompt: input text string
        seq_len: context window cap (default 128, must match training)
        ode_depth: ODE integration depth (default 2.0, must match training)
        max_new_tokens: how many tokens to generate
        temperature: sampling temperature (lower = more deterministic)
        top_k: keep only top-k logits before sampling (0 = disabled)
        repetition_penalty: penalise tokens already in the context (1.0 = off)
        device: torch device
    """
    model.eval()
    tokens = enc.encode_ordinary(prompt)
    generated = list(tokens)
    t_span = torch.tensor([0.0, ode_depth], device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Sliding window: use last seq_len tokens (no padding)
            context = generated[-seq_len:]
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            logits, _ = model(input_ids, t_span)
            next_logits = logits[0, -1, :] / temperature

            # Repetition penalty: lower score for tokens already generated
            if repetition_penalty != 1.0:
                for tok_id in set(context):
                    if next_logits[tok_id] > 0:
                        next_logits[tok_id] /= repetition_penalty
                    else:
                        next_logits[tok_id] *= repetition_penalty

            # Suppress EOT so generation doesn't stop prematurely
            next_logits[enc.eot_token] = float('-inf')

            # Top-k filtering
            if top_k > 0:
                top_vals, top_idx = torch.topk(next_logits, min(top_k, next_logits.size(0)))
                mask = torch.full_like(next_logits, float('-inf'))
                mask.scatter_(0, top_idx, top_vals)
                next_logits = mask

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

    # Decode only the generated part (not the prompt)
    output_text = enc.decode(generated[len(tokens):])
    return output_text

# ──────────────────────────────────────────────────────────
# 3. LOADING & MAIN
# ──────────────────────────────────────────────────────────

class KahlerTokenizer:
    """Wrapper around a trained SentencePiece BPE model."""
    def __init__(self, model_path="kahler_tokenizer.model"):
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


def load_model(checkpoint_path, config_path=None, device='cpu'):
    """Load model from checkpoint + config."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")

    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        dim = config.get("dim", 192)
        vocab_size = config.get("vocab_size", 4096)
        num_blocks = config.get("num_blocks", 1)  # Default 1 for old configs
        tokenizer_path = config.get("tokenizer", "kahler_tokenizer.model")
        print(f"[+] Config loaded: dim={dim}, num_blocks={num_blocks}, vocab_size={vocab_size}")
        print(f"    Architecture: {config.get('architecture', 'unknown')}")
        print(f"    Dataset: {config.get('dataset', 'unknown')}")
        print(f"    Tokenizer: {tokenizer_path}")
        print(f"    Step reached: {config.get('step_reached', '?')}")
    else:
        print("[!] No config.json found, using defaults (dim=384, vocab_size=16384, num_blocks=3)")
        dim = 384
        vocab_size = 16384
        num_blocks = 3
        tokenizer_path = "fineweb_tokenizer.model"

    enc = KahlerTokenizer(tokenizer_path)
    model = KahlerTransformerODE(vocab_size=vocab_size, dim=dim, num_blocks=num_blocks).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Model loaded: {total_params / 1e6:.2f}M parameters")
    return model, enc


def main():
    parser = argparse.ArgumentParser(description="Generate text with Kähler ODE Transformer")
    parser.add_argument("--checkpoint", type=str, default="kahler_fineweb/kahler_weights.pt",
                        help="Path to model weights (.pt file)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json (auto-detected if next to checkpoint)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (if omitted, enters interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.1=deterministic, 1.0=creative)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k filtering (0=disabled)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Context window size (must match training)")
    parser.add_argument("--ode-depth", type=float, default=2.0,
                        help="ODE integration depth (must match training)")
    args = parser.parse_args()

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[+] Using device: {device}")

    # Load model
    model, enc = load_model(args.checkpoint, args.config, device=device)

    # Single prompt mode
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        output = generate(model, enc, args.prompt,
                          seq_len=args.seq_len, ode_depth=args.ode_depth,
                          max_new_tokens=args.max_tokens,
                          temperature=args.temperature, top_k=args.top_k,
                          device=device)
        print(f"Output: {output}")
        return

    # Interactive mode
    print("\n" + "="*60)
    print("  Kähler ODE Transformer — Interactive Generation")
    print("  Type a prompt and press Enter. Type 'quit' to exit.")
    print("  Commands: /temp 0.5  /topk 30  /tokens 200")
    print("="*60 + "\n")

    temp = args.temperature
    top_k = args.top_k
    max_tokens = args.max_tokens

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() == 'quit':
            print("Bye!")
            break

        # Handle setting commands
        if prompt.startswith("/temp "):
            temp = float(prompt.split()[1])
            print(f"  Temperature set to {temp}")
            continue
        if prompt.startswith("/topk "):
            top_k = int(prompt.split()[1])
            print(f"  Top-k set to {top_k}")
            continue
        if prompt.startswith("/tokens "):
            max_tokens = int(prompt.split()[1])
            print(f"  Max tokens set to {max_tokens}")
            continue

        output = generate(model, enc, prompt,
                          seq_len=args.seq_len, ode_depth=args.ode_depth,
                          max_new_tokens=max_tokens,
                          temperature=temp, top_k=top_k,
                          device=device)
        print(f"\n{prompt}{output}\n")


if __name__ == "__main__":
    main()
