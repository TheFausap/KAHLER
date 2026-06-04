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
    out_img = data_dir / "embedding_umap.png"
    out_npy = data_dir / "silhouette_embeddings_umap.npy"
    report_path = data_dir / "silhouette_report.json"

    if not emb_path.exists() or not lbl_path.exists():
        print(f"Missing embeddings or labels at {data_dir}. Expected: {emb_path.name}, {lbl_path.name}")
        sys.exit(1)

    embeddings = np.load(emb_path)
    labels = np.load(lbl_path)

    try:
        import umap
    except Exception as e:
        print("umap-learn is not installed. Install with: pip install umap-learn")
        raise

    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        silhouette_score = None

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    emb_umap = reducer.fit_transform(embeddings)
    np.save(out_npy, emb_umap)

    plt.figure(figsize=(8, 6))
    for lab in sorted(set(labels.tolist())):
        sel = labels == lab
        plt.scatter(emb_umap[sel, 0], emb_umap[sel, 1], s=12, alpha=0.8, label=str(lab))
    # centroids
    import numpy.linalg as npla
    cents = []
    for lab in sorted(set(labels.tolist())):
        sel = labels == lab
        cents.append(emb_umap[sel].mean(axis=0))
    cents = np.stack(cents)
    plt.scatter(cents[:, 0], cents[:, 1], c='k', marker='x', s=60)
    plt.legend(title='label')
    plt.title('UMAP projection')
    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    print(f"Saved UMAP image to: {out_img}")

    umap_sil = None
    if silhouette_score is not None:
        try:
            umap_sil = float(silhouette_score(emb_umap, labels))
            print(f"UMAP 2D silhouette: {umap_sil}")
        except Exception as e:
            print("Could not compute silhouette on UMAP 2D:", e)

    # update report JSON if present
    if report_path.exists():
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
        except Exception:
            report = {}
    else:
        report = {}

    report['umap_2d_silhouette'] = umap_sil
    report['umap_n_neighbors'] = 15
    report['umap_min_dist'] = 0.1

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Updated report at: {report_path}")


if __name__ == '__main__':
    main()
