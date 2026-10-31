"""Train one classifier on one cross-validation fold: from scratch, from an
SSL-pretrained encoder (fine-tuning or linear probe), or jointly with an SSL
objective.

    python train.py --config configs/supervised/b0_supervised.yaml --fold 0 --seed 0
    python train.py --config configs/ssl/s1_simclr_ft.yaml --fold 0 --seed 0

With model.init set (e.g. runs/p1_simclr/f{fold}_s{seed}/encoder.pt), the
encoder starts from the weights pretrain.py saved for the same fold and seed,
so it never saw the validation fold.

Saves to runs/<experiment>/f<fold>_s<seed>/: the config, a log, metrics.json,
training curves, the final model (last.pt) and the final-epoch probabilities
for every validation image (val_predictions.npz). The test set is not used here.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from scenecnn.common import apply_overrides, environment_info, get_device, load_config, set_random_seed
from scenecnn.data import build_train_val_loaders
from scenecnn.engine import build_optimizer, build_scheduler, classifier_param_groups, train_one_epoch, validate
from scenecnn.models import build_model
from scenecnn.ssl import build_objective


def plot_curves(history, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for split in ('train', 'val'):
        ax1.plot(history[f'{split}_loss'], label=split)
        ax2.plot(history[f'{split}_acc'], label=split)
    ax1.set(xlabel='epoch', ylabel='loss', title='Classification loss')
    ax2.set(xlabel='epoch', ylabel='accuracy', title='Accuracy')
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_encoder(model, cfg, fold, seed, log):
    path = Path(cfg['model']['init'].format(fold=fold, seed=seed))
    if not path.exists():
        raise SystemExit(f'{path} not found: run pretrain.py for fold {fold}, seed {seed} first')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['fold'] != fold:
        raise SystemExit(f'{path} was pretrained for fold {state["fold"]}, not {fold}')
    model.encoder.load_state_dict(state['encoder'])
    log(f'encoder initialised from {path}')


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
    log_file = open(out_dir / 'train.log', 'w')

    def log(message):
        print(message, flush=True)
        log_file.write(message + '\n')
        log_file.flush()

    set_random_seed(args.seed)
    device = get_device()
    train_loader, val_loader, images = build_train_val_loaders(cfg, args.seed)
    model = build_model(cfg, len(images.classes))
    if cfg['model']['init']:
        load_encoder(model, cfg, args.fold, args.seed, log)
    model = model.to(device)

    groups = classifier_param_groups(model, cfg)
    objective = None
    if cfg['ssl']['joint_weight'] > 0:
        objective = build_objective(cfg, model.encoder.out_dim).to(device)
        groups.append({'params': list(objective.parameters()), 'lr': cfg['optim']['lr']})
    optimizer = build_optimizer(groups, cfg['optim']['lr'], cfg['optim']['weight_decay'])
    t = cfg['train']
    scheduler = build_scheduler(optimizer, t['epochs'], t['warmup_epochs'], len(train_loader))
    params = sum(p.numel() for p in model.parameters())
    log(f'{name} | fold {args.fold} | seed {args.seed} | {device} | {params:,} parameters '
        f'| {len(train_loader.dataset)} labelled training images')

    history = {'train_loss': [], 'train_acc': [], 'train_ssl_loss': [], 'val_loss': [], 'val_acc': []}
    start = time.time()
    for epoch in range(1, t['epochs'] + 1):
        train_loss, train_acc, ssl_loss = train_one_epoch(model, train_loader, optimizer, scheduler, cfg, device, objective)
        val_loss, val_acc, val_probs, val_labels = validate(model, val_loader, device)
        for key, value in zip(history, (train_loss, train_acc, ssl_loss, val_loss, val_acc)):
            history[key].append(value)
        ssl = f' ssl {ssl_loss:.3f}' if objective is not None else ''
        log(f'epoch {epoch:3d} | train loss {train_loss:.3f} acc {train_acc:.3f}{ssl} '
            f'| val loss {val_loss:.3f} acc {val_acc:.3f} | {time.time() - start:.0f}s')

    # Experiments are compared by the final epoch: picking the best of many
    # noisy validation scores would overestimate the accuracy.
    torch.save({'model': model.state_dict(), 'config': cfg, 'class_names': images.classes}, out_dir / 'last.pt')
    paths = [images.paths[i] for i in val_loader.dataset.indices]
    np.savez_compressed(out_dir / 'val_predictions.npz', paths=paths, labels=val_labels.numpy(),
                        probs=val_probs.numpy(), class_names=images.classes)
    metrics = {
        'experiment': name,
        'kind': 'classifier',
        'fold': args.fold,
        'seed': args.seed,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'labelled_images': len(train_loader.dataset),
        'params': params,
        'train_time_s': round(time.time() - start),
        'overrides': args.set,
        'environment': environment_info(),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (out_dir / 'history.json').write_text(json.dumps(history))
    plot_curves(history, out_dir / 'curves.png')
    log(f'final validation accuracy {metrics["final_val_acc"]:.4f}')
    log_file.close()


if __name__ == '__main__':
    main()
