"""Test accuracy of every fold model on its own and of the 5-model average,
plus paired McNemar tests between the averages (Table 1 of the report).

    python scripts/test_analysis.py

Reads runs/<experiment>/f<fold>_s0/last.pt; writes nothing.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import load_checkpoint
from scenecnn.data import build_test_loader
from scenecnn.engine import predict
from scripts.oof import mcnemar_p

EXPERIMENTS = ['b0_supervised', 'b1_supervised_650ep', 's1_simclr_ft', 's3_rotation_ft', 'j1_simclr_joint']
PAIRS = [('b0_supervised', 's3_rotation_ft'), ('b0_supervised', 's1_simclr_ft')]


def main():
    device = torch.device('cpu')
    loader, correct = None, {}
    for experiment in EXPERIMENTS:
        probs = []
        for fold in range(5):
            model, cfg, _ = load_checkpoint(f'runs/{experiment}/f{fold}_s0/last.pt', device)
            if loader is None:
                cfg['data']['num_workers'] = 0
                loader, _, _ = build_test_loader(cfg)
            p, labels = predict(model, loader, device)
            probs.append(p.numpy())
        labels = labels.numpy()
        single = [100 * (p.argmax(1) == labels).mean() for p in probs]
        correct[experiment] = np.mean(probs, axis=0).argmax(1) == labels
        print(f'{experiment:22s} 5-model average {100 * correct[experiment].mean():.2f} | '
              f'single models {np.mean(single):.1f} ± {np.std(single):.1f}')
    for a, b in PAIRS:
        only_a = int((correct[a] & ~correct[b]).sum())
        only_b = int((correct[b] & ~correct[a]).sum())
        print(f'{a} vs {b}: only A right {only_a}, only B right {only_b}, McNemar p = {mcnemar_p(only_a, only_b):.2f}')


if __name__ == '__main__':
    main()
