"""Analysis of out-of-fold predictions on the 2,400 training images.

With 5-fold cross-validation, every training image is predicted once per seed
by the model that did not train on it. These predictions give per-class
results on 2,400 images instead of 480, without using the test set.

    python scripts/oof.py --experiment t4_r18_places_ft       # per-class accuracy and confusions
    python scripts/oof.py --compare a6_res128 a12_longer      # is B better than A?

--experiment writes runs/<experiment>/per_class.csv; --compare adds one row
per seed to runs/comparisons.csv.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np


def load_oof(experiment):
    """For each seed with all 5 folds done: labels and probabilities, sorted by image path."""
    by_seed = {}
    for f in Path('runs', experiment).glob('f*_s*/val_predictions.npz'):
        by_seed.setdefault(f.parent.name.split('_s')[1], []).append(np.load(f))
    oof = {}
    for seed, parts in sorted(by_seed.items()):
        if len(parts) == 5:
            order = np.argsort(np.concatenate([p['paths'] for p in parts]))
            labels = np.concatenate([p['labels'] for p in parts])[order]
            probs = np.concatenate([p['probs'] for p in parts])[order]
            oof[seed] = (labels, probs)
    return oof, list(parts[0]['class_names'])


def mcnemar_p(only_a, only_b):
    """Exact two-sided McNemar test: are the images that only one model gets
    right split more unevenly than a fair coin would allow?"""
    n = only_a + only_b
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(a, b):
    """Paired comparison of two experiments on the same images, one row per seed."""
    oof_a, _ = load_oof(a)
    oof_b, _ = load_oof(b)
    path = Path('runs/comparisons.csv')
    write_header = not path.exists()
    diffs = []
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['a', 'b', 'seed', 'acc_a', 'acc_b', 'only_a_right', 'only_b_right', 'mcnemar_p'])
        for seed in sorted(set(oof_a) & set(oof_b)):
            right_a = oof_a[seed][1].argmax(1) == oof_a[seed][0]
            right_b = oof_b[seed][1].argmax(1) == oof_b[seed][0]
            only_a, only_b = int((right_a & ~right_b).sum()), int((right_b & ~right_a).sum())
            p = mcnemar_p(only_a, only_b)
            diffs.append(right_b.mean() - right_a.mean())
            writer.writerow([a, b, seed, round(right_a.mean(), 4), round(right_b.mean(), 4), only_a, only_b, f'{p:.3g}'])
            print(f'seed {seed}: A {right_a.mean():.4f}, B {right_b.mean():.4f} | only A right: {only_a}, '
                  f'only B right: {only_b}, McNemar p = {p:.3g}')
    print(f'B - A: {100 * np.mean(diffs):+.2f} points on average over {len(diffs)} seeds')


def analyse(experiment):
    oof, class_names = load_oof(experiment)
    for seed, (labels, probs) in oof.items():
        print(f'seed {seed}: accuracy {(probs.argmax(1) == labels).mean():.4f}')
    pred = np.mean([probs for _, probs in oof.values()], axis=0).argmax(1)
    print(f'average of the {len(oof)} seeds: accuracy {(pred == labels).mean():.4f}')

    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (labels, pred), 1)
    print('\nper-class accuracy (worst first):')
    for i in np.argsort(cm.diagonal() / cm.sum(1)):
        print(f'  {class_names[i]:<13} {cm[i, i] / cm[i].sum():.3f}')
    errors = cm - np.diag(cm.diagonal())
    with open(Path('runs', experiment, 'per_class.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'oof_accuracy', 'most_often_predicted_as', 'times'])
        for i, name in enumerate(class_names):
            j = errors[i].argmax()
            confused = class_names[j] if errors[i, j] > 0 else ''
            writer.writerow([name, round(cm[i, i] / cm[i].sum(), 4), confused, errors[i, j]])
    print('\nmost frequent confusions (true -> predicted):')
    for k in np.argsort(-errors, axis=None)[:10]:
        i, j = divmod(k, n)
        if errors[i, j] > 0:
            print(f'  {class_names[i]:<13} -> {class_names[j]:<13} {errors[i, j]}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiment', help='one experiment to analyse')
    p.add_argument('--compare', nargs=2, metavar=('A', 'B'))
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        analyse(args.experiment)


if __name__ == '__main__':
    main()
