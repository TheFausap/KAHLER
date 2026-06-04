import json
from pathlib import Path
import sys

def main():
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import silhouette_score

    base = Path(__file__).resolve().parent
    data_dir = base / "kahler_fineweb"
    emb_path = data_dir / "silhouette_embeddings.npy"
    lbl_path = data_dir / "silhouette_labels.npy"
    out_state = data_dir / "projection_head_best.pt"
    out_emb = data_dir / "silhouette_embeddings_proj.npy"
    report_path = data_dir / "finetune_projection_report.json"

    if not emb_path.exists() or not lbl_path.exists():
        print("Missing embeddings/labels. Run eval_silhouette first.")
        sys.exit(1)

    X = np.load(emb_path)
    y = np.load(lbl_path)
    input_dim = X.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class EmbDataset(Dataset):
        def __init__(self, X, y):
            self.X = X.astype('float32')
            self.y = y.astype('int64')
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return self.X[idx], int(self.y[idx])

    ds = EmbDataset(X, y)
    loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=False)

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
            z = z / (z.norm(dim=1, keepdim=True) + 1e-12)
            return z

    def sup_contrastive_loss(z, labels, temperature=0.07):
        # z: (B, D) normalized
        device = z.device
        sim = torch.matmul(z, z.t()) / temperature
        # mask out self
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - logits_max.detach()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)
        # remove self-positives
        self_mask = torch.eye(mask.shape[0], device=device)
        mask = mask - self_mask
        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
        # only keep positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = - mean_log_prob_pos
        # for samples with no positives (mask.sum==0) we exclude
        valid = (mask.sum(1) > 0).float()
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        loss = (loss * valid).sum() / valid.sum()
        return loss

    model = ProjectionHead(input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_sil = -1e9
    best_epoch = -1
    report = {"epochs": []}

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        steps = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            zb = model(xb)
            loss = sup_contrastive_loss(zb, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu().numpy())
            steps += 1
        avg_loss = total_loss / max(1, steps)

        # evaluate silhouette on full dataset using cosine
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X.astype('float32')).to(device)
            Z = model(X_t).cpu().numpy()
        try:
            sil = float(silhouette_score(Z, y, metric='cosine'))
        except Exception:
            sil = None

        report['epochs'].append({'epoch': epoch, 'avg_loss': avg_loss, 'silhouette_cosine': sil})
        if sil is not None and sil > best_sil:
            best_sil = sil
            best_epoch = epoch
            torch.save(model.state_dict(), out_state)
            np.save(out_emb, Z)

        print(f"Epoch {epoch:02d}: loss={avg_loss:.4f} silhouette_cosine={sil}")

    report['best'] = {'best_epoch': best_epoch, 'best_silhouette_cosine': best_sil}
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Training finished. Best epoch: {best_epoch}, best_silhouette_cosine={best_sil}")
    print(f"Saved best state to {out_state} and embeddings to {out_emb}")


if __name__ == '__main__':
    main()
