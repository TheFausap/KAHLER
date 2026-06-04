"""
Load saved embeddings and labels from `KAHLER/kahler_fineweb`,
produce a 2D PCA scatter PNG and a JSON report with per-class stats.

Run locally:

pip install numpy scikit-learn matplotlib
python3 KAHLER/generate_embedding_report.py

"""
import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out_dir = os.path.join('./', 'kahler_fineweb')
    emb_path = os.path.join(out_dir, 'silhouette_embeddings.npy')
    lab_path = os.path.join(out_dir, 'silhouette_labels.npy')
    if not os.path.exists(emb_path) or not os.path.exists(lab_path):
        raise FileNotFoundError('Expected embeddings and labels in KAHLER/kahler_fineweb')

    X = np.load(emb_path)
    labels = np.load(lab_path)

    # PCA -> 2D
    n_comp = min(2, X.shape[1])
    pca = PCA(n_components=n_comp)
    X2 = pca.fit_transform(X)

    # Plot
    plt.figure(figsize=(8,6))
    cmap = {1: 'red', 0: 'blue'}
    for lbl in np.unique(labels):
        idx = labels == lbl
        plt.scatter(X2[idx,0], X2[idx,1], c=cmap[int(lbl)], label=f'class_{int(lbl)}', alpha=0.6, s=20)

    # centroids
    for lbl in [1,0]:
        idx = labels == lbl
        cent = X2[idx].mean(axis=0)
        plt.scatter(cent[0], cent[1], c='k', marker='x', s=100)
        plt.annotate(f'centroid_{int(lbl)}', (cent[0], cent[1]))

    plt.legend()
    plt.title('PCA of Kahler sentence embeddings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    png_path = os.path.join(out_dir, 'embedding_pca.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    # Metrics
    sil_samples = silhouette_samples(X, labels)
    overall_sil = float(silhouette_score(X, labels))
    mean_sil_per = {int(lbl): float(sil_samples[labels==lbl].mean()) for lbl in np.unique(labels)}
    count_per = {int(lbl): int((labels==lbl).sum()) for lbl in np.unique(labels)}

    cent_full = [X[labels==lbl].mean(axis=0) for lbl in [1,0]]
    centroid_dist = float(np.linalg.norm(cent_full[0]-cent_full[1]))

    intra_var = {int(lbl): float(((X[labels==lbl]-cent_full[i])**2).sum(axis=1).mean()) for i,lbl in enumerate([1,0])}

    report = {
        'overall_silhouette': overall_sil,
        'mean_sil_per_class': mean_sil_per,
        'counts_per_class': count_per,
        'centroid_distance_full_space': centroid_dist,
        'intra_class_variance': intra_var,
        'pca_explained_variance_percent': list((pca.explained_variance_ratio_*100).tolist())
    }

    report_path = os.path.join(out_dir, 'silhouette_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print('Saved PCA plot to', png_path)
    print('Saved JSON report to', report_path)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
