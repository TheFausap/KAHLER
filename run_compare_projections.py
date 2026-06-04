import json
from pathlib import Path
import sys

def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path(__file__).resolve().parent
    data_dir = base / "kahler_fineweb"
    emb_path = data_dir / "silhouette_embeddings.npy"
    lbl_path = data_dir / "silhouette_labels.npy"
    out_img = data_dir / "embedding_compare.png"
    report_path = data_dir / "silhouette_report.json"

    if not emb_path.exists() or not lbl_path.exists():
        print(f"Missing embeddings or labels at {data_dir}. Expected: {emb_path.name}, {lbl_path.name}")
        sys.exit(1)

    X = np.load(emb_path)
    y = np.load(lbl_path)

    results = {}

    # PCA
    try:
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        results['pca_2d_silhouette'] = float(silhouette_score(X_pca, y))
        results['pca_explained_variance_percent'] = (pca.explained_variance_ratio_ * 100).tolist()
    except Exception as e:
        print('PCA or silhouette unavailable:', e)
        X_pca = None

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X)
        try:
            from sklearn.metrics import silhouette_score
            results['umap_2d_silhouette'] = float(silhouette_score(X_umap, y))
        except Exception:
            results['umap_2d_silhouette'] = None
    except Exception as e:
        print('UMAP unavailable:', e)
        X_umap = None

    # t-SNE (try args compatible with different sklearn versions)
    try:
        from sklearn.manifold import TSNE
        X_tsne = None
        # Try simple constructor first (widely compatible)
        try:
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            X_tsne = tsne.fit_transform(X)
        except Exception:
            # Try 'max_iter' (newer sklearn) then 'n_iter' (older fallback)
            tried = False
            try:
                tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
                X_tsne = tsne.fit_transform(X)
                tried = True
            except Exception:
                pass
            if not tried:
                try:
                    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
                    X_tsne = tsne.fit_transform(X)
                except Exception as e:
                    print('t-SNE failed with multiple argument sets:', e)
                    X_tsne = None

        if X_tsne is not None:
            try:
                from sklearn.metrics import silhouette_score
                results['tsne_2d_silhouette'] = float(silhouette_score(X_tsne, y))
            except Exception:
                results['tsne_2d_silhouette'] = None
        else:
            X_tsne = None
    except Exception as e:
        print('t-SNE unavailable:', e)
        X_tsne = None

    # Plot side-by-side
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    plots = [('PCA', X_pca, 'pca_2d_silhouette'), ('UMAP', X_umap, 'umap_2d_silhouette'), ('t-SNE', X_tsne, 'tsne_2d_silhouette')]
    cmap = {0: '#1f77b4', 1: '#d62728'}
    for ax, (name, arr, key) in zip(axs, plots):
        if arr is None:
            ax.text(0.5, 0.5, f'{name} not available', ha='center', va='center')
            ax.set_xticks([]); ax.set_yticks([])
            continue
        for lab in sorted(set(y.tolist())):
            sel = y == lab
            ax.scatter(arr[sel, 0], arr[sel, 1], s=10, alpha=0.8, color=cmap.get(lab, None), label=str(lab))
        # centroids
        cents = [arr[y == lab].mean(axis=0) for lab in sorted(set(y.tolist()))]
        cents = np.stack(cents)
        ax.scatter(cents[:, 0], cents[:, 1], c='k', marker='x', s=80)
        sil = results.get(key)
        title = name + (f' — silhouette: {sil:.3f}' if sil is not None else '')
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    axs[0].legend(title='label')
    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    print(f'Saved comparison image to: {out_img}')

    # update report
    if report_path.exists():
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
        except Exception:
            report = {}
    else:
        report = {}

    report.update(results)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Updated report at: {report_path}')


if __name__ == '__main__':
    main()
