"""Config loading and small helpers."""
import copy
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def merge(base, override):
    """Recursively overwrite the values of `base` with those of `override`."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path):
    """Load a YAML config. A config can inherit from another with
    `_base_: parent.yaml` and then only lists the values it changes."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop('_base_', None)
    if parent:
        cfg = merge(load_config(path.parent / parent), cfg)
    return cfg


def apply_overrides(cfg, overrides):
    """Apply command-line overrides such as `optim.lr=0.001`."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, value = item.split('=', 1)
        *parents, leaf = key.split('.')
        node = cfg
        for name in parents:
            node = node[name]
        node[leaf] = yaml.safe_load(value)
    return cfg


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def environment_info():
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(get_device()),
    }
