"""Self-supervised pretraining of an encoder on the training folds of one
cross-validation fold. Labels are not used for training; the validation
fold is not used at all except to monitor a kNN classifier.

    python pretrain.py --config configs/pretrain/p1_simclr.yaml --fold 0 --seed 0

Saves to runs/<experiment>/f<fold>_s<seed>/: the config, a log, metrics.json,
history.json, curves and the encoder weights (encoder.pt), which train.py
loads through model.init.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

from scenecnn.common import apply_overrides, environment_info, get_device, load_config, set_random_seed
from scenecnn.data import build_pretrain_loaders
from scenecnn.engine import build_optimizer, build_scheduler, pretrain_one_epoch
from scenecnn.models import build_encoder
from scenecnn.ssl import build_objective, knn_accuracy


def plot_curves(history, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history['ssl_loss'], label='SSL loss')
    ax1.plot(history['ssl_acc'], label='pretext accuracy')
    ax1.set(xlabel='epoch', title='Pretext task')
    ax1.legend()
    ax2.plot(history['knn_epoch'], history['knn_acc'], marker='o')
    ax2.set(xlabel='epoch', ylabel='accuracy', title='kNN accuracy on the validation fold')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--fold', type=int, default=0, help='validation fold, 0-4')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name', help='experiment name (default: config file name)')
    p.add_argument('--out', default='runs')
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='override config values')
    args = p.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    cfg['data']['fold'] = args.fold
    name = args.name or Path(args.config).stem
    out_dir = Path(args.out) / name / f'f{args.fold}_s{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    log_file = open(out_dir / 'pretrain.log', 'w')

    def log(message):
        print(message, flush=True)
        log_file.write(message + '\n')
        log_file.flush()

    set_random_seed(args.seed)
    device = get_device()
    ssl_loader, bank_loader, val_loader, _ = build_pretrain_loaders(cfg, args.seed)
    encoder = build_encoder(cfg).to(device)
    objective = build_objective(cfg, encoder.out_dim).to(device)
    pt = cfg['pretrain']
    optimizer = build_optimizer(list(encoder.parameters()) + list(objective.parameters()), pt['lr'], pt['weight_decay'])
    scheduler = build_scheduler(optimizer, pt['epochs'], pt['warmup_epochs'], len(ssl_loader))
    params = sum(p.numel() for p in encoder.parameters())
    log(f'{name} | {cfg["ssl"]["method"]} | fold {args.fold} | seed {args.seed} | {device} '
        f'| {params:,} encoder parameters | {len(ssl_loader.dataset)} unlabelled images')

    history = {'ssl_loss': [], 'ssl_acc': [], 'knn_epoch': [], 'knn_acc': []}

    def monitor(epoch):
        acc = knn_accuracy(encoder, bank_loader, val_loader, device)
        history['knn_epoch'].append(epoch)
        history['knn_acc'].append(acc)
        return acc

    log(f'epoch   0 | kNN acc {monitor(0):.3f} (random encoder)')
    start = time.time()
    for epoch in range(1, pt['epochs'] + 1):
        loss, acc = pretrain_one_epoch(encoder, objective, ssl_loader, optimizer, scheduler, device)
        history['ssl_loss'].append(loss)
        history['ssl_acc'].append(acc)
        knn = ''
        if epoch % pt['knn_every'] == 0 or epoch == pt['epochs']:
            knn = f' | kNN acc {monitor(epoch):.3f}'
        log(f'epoch {epoch:3d} | ssl loss {loss:.3f} pretext acc {acc:.3f}{knn} | {time.time() - start:.0f}s')

    torch.save({'encoder': encoder.state_dict(), 'fold': args.fold, 'seed': args.seed, 'config': cfg},
               out_dir / 'encoder.pt')
    metrics = {
        'experiment': name,
        'kind': 'pretrain',
        'fold': args.fold,
        'seed': args.seed,
        'method': cfg['ssl']['method'],
        'random_knn_acc': history['knn_acc'][0],
        'final_knn_acc': history['knn_acc'][-1],
        'final_ssl_loss': history['ssl_loss'][-1],
        'final_pretext_acc': history['ssl_acc'][-1],
        'params': params,
        'train_time_s': round(time.time() - start),
        'overrides': args.set,
        'environment': environment_info(),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (out_dir / 'history.json').write_text(json.dumps(history))
    plot_curves(history, out_dir / 'curves.png')
    log(f'final kNN accuracy {metrics["final_knn_acc"]:.4f}')
    log_file.close()


if __name__ == '__main__':
    main()
