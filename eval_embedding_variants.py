import json
from pathlib import Path
import sys

def main():
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    base = Path(__file__).resolve().parent
    data_dir = base / "kahler_fineweb"
    emb_path = data_dir / "silhouette_embeddings.npy"
    lbl_path = data_dir / "silhouette_labels.npy"
    out_report = data_dir / "embedding_variants_report.json"

    if not emb_path.exists() or not lbl_path.exists():
        print("Missing embeddings/labels. Run eval_silhouette first.")
        sys.exit(1)

    X = np.load(emb_path)
    y = np.load(lbl_path)

    results = {}

    # Raw (Euclidean)
    try:
        results['raw_euclidean'] = float(silhouette_score(X, y))
    except Exception as e:
        results['raw_euclidean'] = None

    # L2 normalize (cosine via euclidean on normalized vectors)
    try:
        X_l2 = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        results['l2_cosine'] = float(silhouette_score(X_l2, y, metric='euclidean'))
    except Exception:
        results['l2_cosine'] = None

    # Z-score per dimension
    try:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-12
        X_z = (X - mu) / sd
        results['zscore_euclidean'] = float(silhouette_score(X_z, y))
    except Exception:
        results['zscore_euclidean'] = None

    # PCA whitening (retain components that explain 99% variance)
    try:
        pca = PCA(n_components=min(X.shape[1], X.shape[0]), random_state=42)
        Xp = pca.fit_transform(X)
        # whiten
        comps = pca.explained_variance_ + 1e-12
        Xp_whiten = Xp / (np.sqrt(comps)[None, :])
        results['pca_whiten_euclidean'] = float(silhouette_score(Xp_whiten, y))
    except Exception:
        results['pca_whiten_euclidean'] = None

    # Cosine directly
    try:
        results['raw_cosine'] = float(silhouette_score(X, y, metric='cosine'))
    except Exception:
        results['raw_cosine'] = None

    with open(out_report, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Wrote variant report to: {out_report}")
    print(results)

if __name__ == '__main__':
    main()
