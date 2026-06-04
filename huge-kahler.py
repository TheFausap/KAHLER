import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
from torchdiffeq import odeint
from datasets import load_dataset
import sentencepiece as spm
import os
import matplotlib
matplotlib.use('Agg') # Forces headless mode (no bouncing Mac icons)
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import csv
import argparse
import json
import time

os.environ["HF_HOME"] = "/home/fausap/Documents/LLM/CACHE/huggingface"
import math

# --- 0. DEVICE SELECTION ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"[+] Using device: {device}")

# --- 1. MLP POTENTIAL (4-layer, non-convex) ---
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

# --- 2. CONTEXTUAL KÄHLER ODE ---
class ContextualKahlerODE(nn.Module):
    def __init__(self, potential_net, dim, num_heads=8):
        super().__init__()
        self.potential_net = potential_net
        self.dim = dim
        self.attention = nn.MultiheadAttention(embed_dim=dim * 2, num_heads=num_heads, batch_first=True)

    def forward(self, t, z):
        seq_len = z.shape[1]
        with torch.enable_grad():
            # Don't detach — backward graph must reach attention & potential params.
            if not z.requires_grad:
                z = z.requires_grad_(True)
            z_real = torch.view_as_real(z).reshape(z.shape[0], seq_len, -1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(z.device)
            context_real, _ = self.attention(z_real, z_real, z_real, is_causal=True, attn_mask=causal_mask)
            context_complex = torch.complex(context_real[..., :self.dim], context_real[..., self.dim:])
            u = context_complex + z
            K = self.potential_net(u).sum()
            # Take the gradient w.r.t 'u' (the local state), NOT 'z'.
            # Taking the gradient w.r.t 'z' backpropagates through the causal attention,
            # which allows future tokens to flow backwards and leak into the current token's vector field.
            grad_u = torch.autograd.grad(K, u, create_graph=True)[0]

        update = -(1.0 + 1.0j) * grad_u

        # Bound the vector field magnitude per position. Without this, a steep
        # region in the potential produces arbitrarily large dz/dt, and no
        # scalar speed penalty can recover once the ODE enters that regime
        # (backprop-through-chaos is unreliable). With this cap, the integral
        # |z_final - z0| is bounded by MAX_FIELD * ODE_DEPTH no matter what
        # shape the potential takes — runaway becomes impossible by construction.
        # Typical stable training had per-position field norm ~2-3, so 10.0
        # leaves plenty of room for legitimate dynamics.
        MAX_FIELD = 10.0
        norm = torch.abs(update).norm(dim=-1, keepdim=True)  # per-position L2
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
        # t_span is divided such that each block takes exactly 2 steps of size 1.25.
        # This keeps the total evaluations at 6, fitting in 6GB VRAM.
        for ode_func in self.ode_funcs:
            z = odeint(ode_func, z, t_span, method='euler', options={'step_size': 1.25})[-1]
        zt_real = torch.view_as_real(z).reshape(z.shape[0], z.shape[1], -1)
        return self.project(zt_real), z

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


def load_projection_head(path, input_dim, device=None):
    device = device if device is not None else globals()['device']
    state = torch.load(path, map_location=device)
    proj = ProjectionHead(input_dim).to(device)
    proj.load_state_dict(state)
    proj.eval()
    return proj


def project_embeddings(embs, proj_head):
    if proj_head is None:
        return embs
    with torch.no_grad():
        X = torch.tensor(embs.astype('float32'), device=device)
        return proj_head(X).cpu().numpy()


def kahler_regularizer(z_start, potentials):
    total_laplacian = 0.0
    total_volume = 0.0
    for p in potentials:
        z = z_start.clone().detach().requires_grad_(True)
        K = p(z)
        grad = torch.autograd.grad(K.sum(), z, create_graph=True)[0]
        total_laplacian += torch.norm(grad, p=2, dim=-1).mean()
        total_volume += torch.log(torch.norm(grad, p=2, dim=-1) + 1e-6).std()
    return total_laplacian + total_volume


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[+] Total Trainable Parameters: {total_params / 1e6:.2f} Million\n")

# --- 3. CUSTOM SENTENCEPIECE TOKENIZER ---
class KahlerTokenizer:
    """Wrapper around a trained SentencePiece BPE model."""
    def __init__(self, model_path="kahler_tokenizer.model"):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.n_vocab = self.sp.get_piece_size()
        self.eos_id = self.sp.eos_id()   # </s> = 2
        self.bos_id = self.sp.bos_id()   # <s> = 1
        self.pad_id = self.sp.pad_id()   # <pad> = 3
    def encode_ordinary(self, text):
        return self.sp.encode(text)
    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return self.sp.decode(ids)
    @property
    def eot_token(self):
        return self.eos_id


def build_tokenizer(model_path="fineweb_tokenizer.model"):
    print("Loading custom Kähler tokenizer...")
    tokenizer = KahlerTokenizer(model_path)
    print(f"Vocab Size: {tokenizer.n_vocab}")
    return tokenizer

class FineWebStreamingDataset(IterableDataset):
    """Streams FineWeb-Edu continuously without blowing up RAM."""
    def __init__(self, seq_len=128):
        super().__init__()
        self.seq_len = seq_len
        print("Initializing FineWeb-Edu 10BT streaming dataset...")

    def __iter__(self):
        dataset = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
        buffer = []
        for item in dataset:
            text = item['text']
            tokens = enc.encode_ordinary(text)
            tokens.append(enc.eot_token)
            buffer.extend(tokens)
            
            # Yield chunks of length seq_len
            while len(buffer) > self.seq_len:
                x = torch.tensor(buffer[:self.seq_len], dtype=torch.long)
                y = torch.tensor(buffer[1:self.seq_len+1], dtype=torch.long)
                yield x, y
                buffer = buffer[self.seq_len:]

# Probe sentences for sentence-level flow analysis
# Action sentences (agent doing something) vs Descriptive (states/settings)
action_sentences = [
    "Mary opens the door at night",
    "The boy ran through the dark forest",
    "He picked up the old sword carefully",
]
descriptive_sentences = [
    "The sky was dark and cold",
    "The old house stood on the hill",
    "It was a long and silent road",
]
# Primary sentence for per-token flow visualization
viz_sentence = "Mary opens the door at night"

def compute_sentence_silhouette(model, proj_head=None):
    """Silhouette on sentence-level embeddings: action vs descriptive.
    Each sentence is mean-pooled over token positions after ODE flow."""
    model.eval()
    all_z = []
    all_labels = []
    with torch.no_grad():
        t_span = torch.tensor([0.0, ODE_DEPTH], device=device)
        for label, group in [(1, action_sentences), (0, descriptive_sentences)]:
            for sentence in group:
                tokens = enc.encode_ordinary(sentence)
                input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                z = model.embed(input_ids)
                for ode_func in model.ode_funcs:
                    z = odeint(ode_func, z, t_span, method='euler', options={'step_size': 1.25})[-1]
                # Mean-pool over token positions → one vector per sentence
                z_mean = torch.view_as_real(z[0]).reshape(z.shape[1], -1).mean(dim=0).cpu().numpy()
                all_z.append(z_mean)
                all_labels.append(label)
    z_array = np.array(all_z)
    if proj_head is not None:
        z_array = project_embeddings(z_array, proj_head)
    labels = np.array(all_labels)
    if not np.isfinite(z_array).all() or len(set(labels)) < 2:
        return float('nan')
    return silhouette_score(z_array, labels)

# --- 3.5 Graphics
def visualize_sentence_flow(model, enc, step, save_dir="kahler_gutenberg"):
    """Visualize per-token ODE trajectories for a probe sentence.
    Content words (nouns/verbs) in red, function words (det/prep) in blue."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    tokens = enc.encode_ordinary(viz_sentence)
    words = [enc.decode([t]) for t in tokens]
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # Heuristic content/function split: short common words are function words
    function_words = {'the', 'a', 'an', 'at', 'on', 'in', 'to', 'of', 'is', 'was', 'it'}
    
    # We must strip any leading/trailing spaces or SentencePiece underscores 
    # to reliably check if a word is a function word.
    colors = ['blue' if w.replace(' ', '').strip().lower() in function_words else 'red' for w in words]

    t_steps = torch.linspace(0.0, ODE_DEPTH, steps=30, device=device)

    with torch.no_grad():
        z0 = model.embed(input_ids)
    
    all_trajs = []
    z_curr = z0
    with torch.set_grad_enabled(True):
        for ode_func in model.ode_funcs:
            # Shape: [time_steps, 1, seq_len, dim]
            traj = odeint(ode_func, z_curr, t_steps, method='euler').detach()
            all_trajs.append(traj)
            z_curr = traj[-1]
            
    # Concatenate all block trajectories sequentially
    zt_traj = torch.cat(all_trajs, dim=0)

    # Reshape: [time_steps, seq_len, real_dim]
    traj_real = torch.view_as_real(zt_traj[:, 0]).reshape(
        zt_traj.shape[0], len(tokens), -1
    ).cpu().numpy()

    final_real = traj_real[-1]
    n_components = min(2, final_real.shape[0] - 1, final_real.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(final_real)

    traj_2d = pca.transform(
        traj_real.reshape(-1, traj_real.shape[-1])
    ).reshape(traj_real.shape[0], len(tokens), n_components)

    pc1 = traj_2d[:, :, 0]
    pc2 = traj_2d[:, :, 1] if n_components >= 2 else np.zeros_like(pc1)
    ev = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(14, 12))
    plt.title(f'Sentence Flow: "{viz_sentence}" (Step {step})', fontsize=16)

    for i in range(len(tokens)):
        plt.plot(pc1[:, i], pc2[:, i], color=colors[i], alpha=0.4, linewidth=2.0)
        plt.scatter(pc1[0, i], pc2[0, i], color=colors[i], alpha=0.5, s=30, marker='x')
        plt.scatter(pc1[-1, i], pc2[-1, i], color=colors[i], alpha=1.0, s=80, edgecolors='k')
        plt.annotate(words[i].strip(),
                     (pc1[-1, i], pc2[-1, i]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=12, fontweight='bold', color=colors[i])

    plt.xlabel(f"PC1 ({ev[0]:.1f}% var)", fontsize=14)
    plt.ylabel(f"PC2 ({ev[1]:.1f}% var)" if len(ev) >= 2 else "PC2", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    red_marker = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='Content (noun/verb)')
    blue_marker = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10, label='Function (det/prep)')
    plt.legend(handles=[red_marker, blue_marker], loc='best', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"sentence_flow_step_{step}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] Visualization saved to: {save_path}")
    plt.close()

def generate_text(model, enc, prompt, max_new_tokens=30, temperature=0.8):
    """Generate text using a sliding window of up to SEQ_LEN tokens.
    No padding — the model receives only real tokens."""
    model.eval()
    print(f"\n--- 📝 GENERATION TEST ---")
    tokens = enc.encode_ordinary(prompt)
    generated = list(tokens)
    current_text = enc.decode(generated)
    print(f"Prompt: {current_text}", end='', flush=True)
    printed_len = len(current_text)
    t_span = torch.tensor([0.0, ODE_DEPTH], device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[-SEQ_LEN:]
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            logits, _ = model(input_ids, t_span)
            next_token_logits = logits[0, -1, :] / temperature
            # Repetition penalty
            for tok_id in set(context):
                if next_token_logits[tok_id] > 0:
                    next_token_logits[tok_id] /= 1.2
                else:
                    next_token_logits[tok_id] *= 1.2
            # Suppress EOT
            next_token_logits[enc.eot_token] = float('-inf')
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            
            # Decode full sequence to properly handle BPE spaces and merges
            new_text = enc.decode(generated)
            word = new_text[printed_len:]
            print(word, end='', flush=True)
            printed_len = len(new_text)
    print("\n\n--------------------------\n")
    model.train()

# --- 4. THE TRAINING LOOP ---
def run_gutenberg_experiment(model, num_epochs=10, max_steps=50000, save_every=2000, save_dir="kahler_gutenberg", proj_head=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file = os.path.join(save_dir, "training_log.csv")
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Step', 'Epoch', 'Total_Tokens', 'Task_Loss', 'Geo_Loss', 'Collapse_Penalty', 'Perplexity', 'Silhouette', 'Mean_Speed'])

    # Cosine LR with warmup to break loss plateaus
    LR_MAX = 1e-4
    LR_MIN = 1e-5
    WARMUP_STEPS = 500
    # LambdaLR multiplies base_lr by lr_lambda — so base_lr must be LR_MAX
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_MAX, weight_decay=1e-4)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS           # 0 → 1.0
        progress = (step - WARMUP_STEPS) / max(1, max_steps - WARMUP_STEPS)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale from 1.0 (= LR_MAX) down to LR_MIN/LR_MAX
        return LR_MIN / LR_MAX + (1.0 - LR_MIN / LR_MAX) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss()
    t_span = torch.tensor([0.0, ODE_DEPTH], device=device)

    dataset = FineWebStreamingDataset(seq_len=SEQ_LEN)
    # IterableDataset does not support shuffle or drop_last
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    tokens_per_step = BATCH_SIZE * SEQ_LEN

    print(f"\nStarting FineWeb Training...")
    print(f"Epochs: {num_epochs} | Max steps: {max_steps:,}")
    print(f"LR schedule: warmup {WARMUP_STEPS} steps → {LR_MAX} → cosine decay → {LR_MIN}")
    print(f"Logging metrics to: {log_file}")
    print(f"{'Step':<8} | {'Epoch':<5} | {'Tokens':<9} |{'Loss':<8} | {'Perplexity':<10} | {'Silhouette':<10} | {'Speed':<8}")

    model.train()
    global_step = 0
    bad_batches = 0

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"  EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*60}")

        for x_batch, y_batch in dataloader:
            if global_step >= max_steps:
                break

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()

            logits, z_final = model(x_batch, t_span)
            z0 = model.embed(x_batch)

            task_loss = criterion(logits.reshape(-1, model.embed_real.num_embeddings), y_batch.reshape(-1))

            speed_diff = torch.abs(z_final - z0).mean()
            with torch.no_grad():
                mean_speed = speed_diff.item() / 2.0

            geo_loss = kahler_regularizer(z0, model.potentials)

            SPEED_MIN = 0.2
            SPEED_MAX = 0.8
            collapse_penalty = (
                F.relu(SPEED_MIN - speed_diff) ** 2
                + F.relu(speed_diff - SPEED_MAX)
            )

            total_loss = task_loss + 0.01 * geo_loss + 1.0 * collapse_penalty

            if not torch.isfinite(total_loss):
                optimizer.zero_grad(set_to_none=True)
                bad_batches += 1
                if bad_batches % 10 == 0:
                    print(f"[warn] {bad_batches} bad batches skipped so far")
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if global_step % 50 == 0:
                with torch.no_grad():
                    total_tokens = global_step * tokens_per_step
                    if total_tokens >= 1_000_000:
                        token_str = f"{total_tokens/1_000_000:.2f}M"
                    else:
                        token_str = f"{total_tokens/1000:.1f}K"

                    ppl = torch.exp(task_loss).item()
                    sil = compute_sentence_silhouette(model, proj_head)
                    print(f"{global_step:<8} | {epoch+1:<5} | {token_str:<9} | {task_loss.item():<8.4f} | {ppl:<10.2f} | {sil:<10.4f} | {mean_speed:<8.4f}")

                    with open(log_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([global_step, epoch+1, total_tokens, task_loss.item(), geo_loss.item(), collapse_penalty.item(), ppl, sil, mean_speed])

            if global_step > 0 and global_step % save_every == 0:
                print(f"\n--- Checkpoint (Step {global_step}, Epoch {epoch+1}) ---")
                weights_path = os.path.join(save_dir, "kahler_weights.pt")
                torch.save(model.state_dict(), weights_path)
                config = {
                    "vocab_size": model.embed_real.num_embeddings,
                    "dim": model.dim,
                    "num_blocks": model.num_blocks,
                    "architecture": "Contextual_MLP_Potential_ODE_v5_Stacked",
                    "dataset": "HuggingFaceFW/fineweb-edu (sample-10BT)",
                    "tokenizer": "fineweb_tokenizer.model",
                    "epoch": epoch + 1,
                    "step_reached": global_step
                }
                with open(os.path.join(save_dir, "config.json"), "w") as f:
                    json.dump(config, f, indent=4)
                print(f"[+] Model weights and config saved.")
                visualize_sentence_flow(model, enc, step=global_step, save_dir=save_dir)
                print("--------------------------------------\n")

            if global_step > 0 and global_step % 5000 == 0:
                generate_text(model, enc, "The old man walked slowly to the", max_new_tokens=40, temperature=0.4)
                generate_text(model, enc, "She looked at him and said", max_new_tokens=40, temperature=0.4)

            global_step += 1

        if global_step >= max_steps:
            break

    print(f"\nTraining complete! Epochs: {epoch+1}, Steps: {global_step}, Bad batches: {bad_batches}")

def parse_args():
    parser = argparse.ArgumentParser(description='Kähler experiment runner with optional projection head.')
    parser.add_argument('--projection-head', type=str, default=None, help='Optional projection head checkpoint to apply during silhouette evaluation')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--max-steps', type=int, default=100000, help='Maximum number of training steps')
    parser.add_argument('--save-every', type=int, default=5000, help='Save checkpoint every N steps')
    parser.add_argument('--save-dir', type=str, default='kahler_fineweb', help='Directory to save checkpoints and logs')
    return parser.parse_args()


def main():
    args = parse_args()
    DIM = 384
    ODE_DEPTH = 2.5
    SEQ_LEN = 128
    BATCH_SIZE = 12

    global enc, vocab_size
    enc = build_tokenizer("fineweb_tokenizer.model")
    vocab_size = enc.n_vocab

    model = KahlerTransformerODE(vocab_size=vocab_size, dim=DIM).to(device)
    print_model_parameters(model)

    proj_head = None
    if args.projection_head is not None:
        proj_head = load_projection_head(args.projection_head, input_dim=DIM * 2, device=device)
        print(f"[+] Loaded projection head: {args.projection_head}")

    run_gutenberg_experiment(
        model,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        save_every=args.save_every,
        save_dir=args.save_dir,
        proj_head=proj_head,
    )


if __name__ == '__main__':
    main()
