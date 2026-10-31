"""Cross-validation results per experiment, best first.

    python scripts/summarize.py
    python scripts/summarize.py --filter s    # only experiments whose name starts with 's'

Classifiers (train.py). CV acc: accuracy of the final-epoch predictions on all
2,400 training images, each predicted by the model whose training folds did
not contain it; mean and standard deviation over seeds. Train acc: final
training accuracy (a large gap to CV acc means overfitting).

Pretraining runs (pretrain.py). kNN acc: accuracy of a 20-nearest-neighbour
classifier on the frozen encoder features of the validation folds, pooled
over the 5 folds, before (random encoder) and after pretraining.

Writes runs/summary.md and runs/summary.csv (classifiers) and
runs/summary_pretrain.md (pretraining runs).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def cv_accuracy(runs):
    """Accuracy over all folds of one seed."""
    correct = total = 0
    for run in runs:
        pred = np.load(run['dir'] / 'val_predictions.npz')
        correct += (pred['probs'].argmax(1) == pred['labels']).sum()
        total += len(pred['labels'])
    return correct / total


def complete_seeds(runs):
    """Seeds with all 5 folds done, and their runs."""
    by_seed = defaultdict(list)
    for r in runs:
        by_seed[r['seed']].append(r)
    return {s: rs for s, rs in sorted(by_seed.items()) if len(rs) == 5}


def mean_std(values):
    return 100 * np.mean(values), 100 * (np.std(values, ddof=1) if len(values) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--filter', default='')
    args = p.parse_args()

    experiments = defaultdict(list)
    for path in Path('runs').glob(f'{args.filter}*/f*_s*/metrics.json'):
        run = json.loads(path.read_text())
        run['dir'] = path.parent
        experiments[run['experiment']].append(run)

    rows, pre_rows = [], []
    for name, runs in experiments.items():
        seeds = complete_seeds(runs)
        if not seeds:
            continue
        done = [r for rs in seeds.values() for r in rs]
        if runs[0]['kind'] == 'pretrain':
            # Folds have 480 images each (+-1), so the mean over folds is the pooled accuracy.
            rand = mean_std([np.mean([r['random_knn_acc'] for r in rs]) for rs in seeds.values()])
            knn = mean_std([np.mean([r['final_knn_acc'] for r in rs]) for rs in seeds.values()])
            pretext = 100 * np.mean([r['final_pretext_acc'] for r in done])
            hours = np.mean([r['train_time_s'] for r in done]) / 3600
            pre_rows.append((knn[0], [name, runs[0]['method'], len(seeds), f'{rand[0]:.2f} ± {rand[1]:.2f}',
                                      f'{knn[0]:.2f} ± {knn[1]:.2f}', f'{pretext:.1f}', f'{hours:.2f}']))
        else:
            acc = mean_std([cv_accuracy(rs) for rs in seeds.values()])
            train = 100 * np.mean([r['final_train_acc'] for r in done])
            labelled = int(np.mean([r['labelled_images'] for r in done]))
            rows.append((acc[0], [name, len(seeds), round(acc[0], 2), round(acc[1], 2), round(train, 1),
                                  labelled, round(runs[0]['params'] / 1e6, 2)]))
    rows.sort(key=lambda r: -r[0])
    pre_rows.sort(key=lambda r: -r[0])

    lines = ['| Experiment | Seeds | CV acc (%) | Train acc (%) | Labelled images | Params (M) |',
             '|---|---:|---:|---:|---:|---:|']
    lines += [f'| {r[0]} | {r[1]} | {r[2]:.2f} ± {r[3]:.2f} | {r[4]:.1f} | {r[5]} | {r[6]:.2f} |' for _, r in rows]
    print('\n'.join(lines))
    Path('runs/summary.md').write_text('\n'.join(lines) + '\n')
    with open('runs/summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'seeds', 'cv_acc', 'cv_acc_std', 'train_acc', 'labelled_images', 'params_millions'])
        writer.writerows(r for _, r in rows)

    if pre_rows:
        pre = ['| Pretraining | Method | Seeds | kNN acc, random encoder (%) | kNN acc, pretrained (%) '
               '| Pretext acc (%) | Hours per run |', '|---|---|---:|---:|---:|---:|---:|']
        pre += ['| ' + ' | '.join(str(v) for v in r) + ' |' for _, r in pre_rows]
        print('\n' + '\n'.join(pre))
        Path('runs/summary_pretrain.md').write_text('\n'.join(pre) + '\n')


if __name__ == '__main__':
    main()
