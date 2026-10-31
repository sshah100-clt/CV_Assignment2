"""Figure 1 of the report, from the saved runs.

    python scripts/make_figures.py

(a) kNN accuracy on the held-out fold during SSL pretraining, mean over all
pretraining runs of each method. (b) The same encoders frozen (linear probe)
and fine-tuned: CV accuracy from runs/summary.csv.
"""
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times'], 'font.size': 9,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False})


def knn_curve(experiment):
    curves = [json.load(open(p)) for p in glob.glob(f'runs/{experiment}/f*_s*/history.json')]
    return np.array(curves[0]['knn_epoch']), 100 * np.mean([c['knn_acc'] for c in curves], axis=0)


def cv_acc():
    with open('runs/summary.csv') as f:
        return {r['experiment']: float(r['cv_acc']) for r in csv.DictReader(f)}


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={'width_ratios': [1.35, 1]})
    for exp, color, label in [('p1_simclr', BLUE, 'SimCLR'), ('p1a_no_jitter', AQUA, 'SimCLR, no jitter'),
                              ('p2_rotation', ORANGE, 'Rotation')]:
        x, y = knn_curve(exp)
        ax1.plot(x, y, color=color, lw=2, marker='o', ms=3.5)
        ax1.annotate(f'{label} {y[-1]:.1f}', (x[-1], y[-1]), xytext=(4, 0), textcoords='offset points',
                     va='center', color=INK, fontsize=8.5)
    ax1.set(xlabel='pretraining epoch', ylabel='kNN accuracy (%)', xlim=(0, 640), ylim=(30, 80),
            title='(a) Representation quality during pretraining')
    ax1.set_xticks([0, 100, 200, 300, 400, 500])
    ax1.grid(axis='y', color='#e4e3df', lw=0.6)

    acc = cv_acc()
    # The fine-tuned values are within 2 points, so their labels are spread out (label_y).
    for (probe, ft), color, label, label_y in [(('s0_random_probe', 'b0_supervised'), GRAY, 'Random init', 81.5),
                                               (('s1_simclr_probe', 's1_simclr_ft'), BLUE, 'SimCLR', 86.5),
                                               (('s3_rotation_probe', 's3_rotation_ft'), ORANGE, 'Rotation', 91.5)]:
        y = [acc[probe], acc[ft]]
        ax2.plot([0, 1], y, color=color, lw=2, marker='o', ms=4)
        ax2.annotate(f'{y[0]:.1f}', (0, y[0]), xytext=(-5, 0), textcoords='offset points', ha='right', va='center', fontsize=8.5)
        ax2.annotate(f'{label} {y[1]:.1f}', (1, y[1]), xytext=(1.12, label_y), va='center', fontsize=8.5,
                     arrowprops={'arrowstyle': '-', 'color': color, 'lw': 0.8})
    ax2.set(xlim=(-0.35, 1.9), ylim=(40, 94), ylabel='CV accuracy (%)', title='(b) Frozen vs. fine-tuned encoder')
    ax2.set_xticks([0, 1], ['linear probe', 'fine-tuned'])
    ax2.grid(axis='y', color='#e4e3df', lw=0.6)
    fig.tight_layout()
    out = Path('report/figures/fig1_ssl.png')
    fig.savefig(out, dpi=300)
    print('saved', out)


if __name__ == '__main__':
    main()
