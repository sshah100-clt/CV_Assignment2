"""Evaluate one model, or the average of several, on the test set.

    python evaluate.py --checkpoint runs/s1_simclr_ft/f{0,1,2,3,4}_s0/last.pt

Prints the accuracy with a 95% confidence interval: on all test images, on
the 15 scene classes only (Flower is trivial to recognise) and without the 4
test images that also occur in the training set. Saves eval_test.json,
test_results.csv and a confusion matrix next to the first checkpoint.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from scenecnn.common import get_device, load_config, merge
from scenecnn.data import build_test_loader
from scenecnn.engine import predict
from scenecnn.models import build_model


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    # Settings added to base.yaml after a model was trained take their default value.
    cfg = merge(load_config('configs/base.yaml'), ckpt['config'])
    model = build_model(cfg, len(ckpt['class_names']))
    model.load_state_dict(ckpt['model'])
    return model.to(device), cfg, ckpt['class_names']


def accuracy(pred, labels, keep):
    """Accuracy on the kept images with a 95% normal-approximation interval."""
    n = int(keep.sum())
    acc = float((pred[keep] == labels[keep]).mean())
    margin = 1.96 * math.sqrt(acc * (1 - acc) / n)
    return {'accuracy': acc, 'n': n, 'ci95': [acc - margin, acc + margin]}


def plot_confusion(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=60, ha='right')
    ax.set_yticks(range(len(class_names)), class_names)
    for i, j in zip(*np.nonzero(cm)):
        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7)
    ax.set(xlabel='predicted', ylabel='true')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', nargs='+', required=True, help='several models are averaged')
    p.add_argument('--split', default='test', help='a labelled folder under data/')
    args = p.parse_args()

    device = get_device()
    probs = 0
    for path in args.checkpoint:
        model, cfg, class_names = load_checkpoint(path, device)
        loader, _, paths = build_test_loader(cfg, args.split)
        model_probs, labels = predict(model, loader, device)
        probs = probs + model_probs / len(args.checkpoint)

    pred, labels = probs.argmax(1).numpy(), labels.numpy()
    in_train = json.loads(Path('splits/duplicates.json').read_text())['test_in_train']
    results = {
        'checkpoints': args.checkpoint,
        'all': accuracy(pred, labels, np.ones(len(labels), bool)),
        'scenes_only': accuracy(pred, labels, labels != class_names.index('Flower')),
        'without_train_duplicates': accuracy(pred, labels, np.array([image not in in_train for image in paths])),
    }
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    np.add.at(cm, (labels, pred), 1)
    results['per_class_accuracy'] = dict(zip(class_names, (cm.diagonal() / cm.sum(1)).tolist()))
    results['confusion_matrix'] = cm.tolist()

    print(f'{args.split}: {len(labels)} images, {len(args.checkpoint)} model(s)')
    for key in ('all', 'scenes_only', 'without_train_duplicates'):
        r = results[key]
        print(f'  {key:<25} {r["accuracy"]:.4f}  (n={r["n"]}, 95% CI {r["ci95"][0]:.3f}-{r["ci95"][1]:.3f})')
    for c, a in sorted(results['per_class_accuracy'].items(), key=lambda kv: kv[1]):
        print(f'    {c:<13} {a:.3f}')

    out = Path(args.checkpoint[0]).parent
    (out / f'eval_{args.split}.json').write_text(json.dumps(results, indent=2))
    plot_confusion(cm, class_names, out / f'confusion_{args.split}.png')
    with open(out / f'{args.split}_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subset', 'accuracy', 'images', 'ci95_low', 'ci95_high'])
        for key in ('all', 'scenes_only', 'without_train_duplicates'):
            r = results[key]
            writer.writerow([key, round(r['accuracy'], 4), r['n'], round(r['ci95'][0], 4), round(r['ci95'][1], 4)])
        for name, acc in results['per_class_accuracy'].items():
            writer.writerow([f'class: {name}', round(acc, 4), int(cm[class_names.index(name)].sum()), '', ''])
    print(f'saved results in {out}')


if __name__ == '__main__':
    main()
