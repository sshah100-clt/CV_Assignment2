"""Run train.py (or pretrain.py) for every combination of config, grid value,
fold and seed, spread over the available GPUs.

    python scripts/launch.py --pretrain configs/pretrain/p1_simclr.yaml
    python scripts/launch.py configs/ssl/s1_simclr_probe.yaml configs/ssl/s1_simclr_ft.yaml
    python scripts/launch.py configs/joint/j1_simclr_joint.yaml --grid ssl.joint_weight=0.1,0.5

Each grid point is a separate experiment, named e.g. j1_simclr_joint__joint_weight0.1.
Runs that already have metrics.json are skipped, so an interrupted launch can be repeated.
Exits with an error if any run failed.
"""
import argparse
import itertools
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


def grid_points(grid):
    """All combinations of the grid values, as (overrides, name suffix) pairs."""
    keys = [item.split('=')[0] for item in grid]
    values = [item.split('=')[1].split(',') for item in grid]
    for combo in itertools.product(*values):
        overrides = [f'{k}={v}' for k, v in zip(keys, combo)]
        suffix = '_'.join(f'{k.split(".")[-1]}{v}' for k, v in zip(keys, combo))
        yield overrides, ('__' + suffix if grid else '')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('configs', nargs='+')
    p.add_argument('--pretrain', action='store_true', help='run pretrain.py instead of train.py')
    p.add_argument('--folds', nargs='+', default=['0', '1', '2', '3', '4'])
    p.add_argument('--seeds', nargs='+', default=['0', '1', '2'])
    p.add_argument('--gpus', nargs='+', default=['0', '1', '2', '3'])
    p.add_argument('--jobs-per-gpu', type=int, default=1, help='more than 1 helps for small models')
    p.add_argument('--grid', nargs='*', default=[], metavar='KEY=V1,V2')
    args = p.parse_args()

    jobs = queue.Queue()
    for config, (overrides, suffix), fold, seed in itertools.product(
            args.configs, list(grid_points(args.grid)), args.folds, args.seeds):
        name = Path(config).stem + suffix
        if not Path('runs', name, f'f{fold}_s{seed}', 'metrics.json').exists():
            command = [sys.executable, 'pretrain.py' if args.pretrain else 'train.py', '--config', config, '--name', name,
                       '--fold', fold, '--seed', seed, '--set', *overrides]
            jobs.put((name, fold, seed, command))
    total = jobs.qsize()
    print(f'{total} runs')
    Path('runs/logs').mkdir(parents=True, exist_ok=True)
    finished = itertools.count(1)
    print_lock = threading.Lock()
    failed = []

    def worker(gpu):
        # Each worker runs jobs one after another on its GPU until none are left.
        while True:
            try:
                name, fold, seed, command = jobs.get_nowait()
            except queue.Empty:
                return
            log_path = Path('runs/logs', f'{name}_{fold}_s{seed}.out')
            with open(log_path, 'w') as log:
                code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu})
            status = 'done' if code == 0 else f'FAILED, see {log_path}'
            if code != 0:
                failed.append(log_path)
            with print_lock:
                print(f'[{next(finished)}/{total}] gpu {gpu}: {name} fold {fold} seed {seed} {status}', flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus * args.jobs_per_gpu]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        sys.exit(f'{len(failed)} run(s) failed; fix the cause and run the same command again '
                 '(finished runs are skipped)')


if __name__ == '__main__':
    main()
