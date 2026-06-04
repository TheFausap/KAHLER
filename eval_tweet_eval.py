"""Evaluate Kähler sentence embeddings on Hugging Face tweet_eval.

This script uses natural English tweets from cardiffnlp/tweet_eval and measures
embedding separability across classification labels.

Usage:
  conda run -n KAHLER python KAHLER/eval_tweet_eval.py --task sentiment --limit 500

Dependencies:
  pip install datasets scikit-learn torch torchdiffeq sentencepiece numpy
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from projection_infer import build_model, embed_sentences, load_projection_head, get_device


ROOT_DIR = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate Kähler embeddings on tweet_eval.')
    parser.add_argument('--task', type=str, default='sentiment', help='tweet_eval task name')
    parser.add_argument('--limit', type=int, default=500, help='Max number of examples per split')
    parser.add_argument('--projection-head', type=str, default=None, help='Optional projection head checkpoint')
    parser.add_argument('--config', type=str, default=str(ROOT_DIR / 'kahler_fineweb' / 'config.json'), help='Kähler config path')
    parser.add_argument('--checkpoint', type=str, default=str(ROOT_DIR / 'kahler_fineweb' / 'kahler_weights.pt'), help='Kähler checkpoint path')
    parser.add_argument('--tokenizer', type=str, default=str(ROOT_DIR / 'fineweb_tokenizer.model'), help='SentencePiece tokenizer path')
    parser.add_argument('--output', type=str, default=str(ROOT_DIR / 'kahler_fineweb' / 'tweet_eval_report.json'), help='Output JSON report')
    parser.add_argument('--sample-seed', type=int, default=42, help='Random seed for subsampling')
    return parser.parse_args(argv)


def sample_split(dataset, limit, seed):
    if limit is None or len(dataset) <= limit:
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def compute_metrics(embs, labels, do_silhouette=True):
    out = {}
    if do_silhouette and len(np.unique(labels)) > 1:
        out['silhouette'] = float(silhouette_score(embs, labels))
    else:
        out['silhouette'] = None

    if embs.shape[0] >= 10:
        emb_norm = normalize(embs, axis=1)
        sims = emb_norm @ emb_norm.T
        same = sims[labels[:, None] == labels[None, :]]
        diff = sims[labels[:, None] != labels[None, :]]
        out['mean_cosine_within'] = float(np.mean(same)) if same.size else None
        out['mean_cosine_between'] = float(np.mean(diff)) if diff.size else None
        out['cosine_gap'] = None if same.size == 0 or diff.size == 0 else float(np.mean(same) - np.mean(diff))
    else:
        out['mean_cosine_within'] = None
        out['mean_cosine_between'] = None
        out['cosine_gap'] = None
    return out


def main(argv=None):
    args = parse_args(argv)
    device = get_device()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading tweet_eval task={args.task}')
    dataset = load_dataset('cardiffnlp/tweet_eval', args.task)
    if args.limit is not None:
        for split in dataset:
            dataset[split] = sample_split(dataset[split], args.limit, args.sample_seed)

    print('Building Kähler model...')
    model, tokenizer = build_model(args.config, args.checkpoint, args.tokenizer, device=device)
    ODE_t = torch.tensor([0.0, 2.5], device=device)

    report = {
        'task': args.task,
        'limit': args.limit,
        'splits': {},
        'projection_head': args.projection_head,
    }

    proj_head = None

    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset:
            continue
        split = dataset[split_name]
        texts = [item['text'] for item in split]
        labels = np.array([int(item['label']) for item in split], dtype=int)
        print(f'Embedding {split_name} split: n={len(texts)} classes={sorted(set(labels))}')

        embs = embed_sentences(model, tokenizer, texts, ODE_t, device)
        split_metrics = compute_metrics(embs, labels)

        embs_proj = None
        if args.projection_head is not None:
            if proj_head is None:
                proj_head = load_projection_head(args.projection_head, input_dim=embs.shape[1], device=device)
            with torch.no_grad():
                embs_proj = proj_head(torch.tensor(embs.astype('float32'), device=device)).cpu().numpy()
            split_metrics['projected'] = compute_metrics(embs_proj, labels)

        report['splits'][split_name] = split_metrics

        if split_name == 'train' and len(labels) > 0:
            if 'validation' in dataset and len(dataset['validation']) > 0:
                val_texts = [item['text'] for item in dataset['validation']]
                val_labels = np.array([int(item['label']) for item in dataset['validation']], dtype=int)
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(embs, labels)
                val_preds = knn.predict(embed_sentences(model, tokenizer, val_texts, ODE_t, device))
                split_metrics['train_to_validation_knn'] = float(accuracy_score(val_labels, val_preds))
                if proj_head is not None and embs_proj is not None:
                    knn_proj = KNeighborsClassifier(n_neighbors=5)
                    knn_proj.fit(embs_proj, labels)
                    with torch.no_grad():
                        val_embs_proj = proj_head(torch.tensor(embed_sentences(model, tokenizer, val_texts, ODE_t, device).astype('float32'), device=device)).cpu().numpy()
                    val_preds_proj = knn_proj.predict(val_embs_proj)
                    split_metrics['train_to_validation_knn_projected'] = float(accuracy_score(val_labels, val_preds_proj))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'Saved tweet_eval report to {output_path}')


if __name__ == '__main__':
    main()
