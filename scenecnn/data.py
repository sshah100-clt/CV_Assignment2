"""Image transforms, cross-validation splits and data loaders."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class CachedImages(Dataset):
    """The images of one class-folder split (e.g. data/train), decoded once,
    converted to gray and kept in memory. Only Flower images are in colour, so
    gray input stops a model from recognising Flower by its colour alone.
    Decoding the JPEGs in every epoch limited Assignment 1 to ~400 images/s;
    SSL pretraining reads every image twice per step for hundreds of epochs."""

    def __init__(self, root):
        folder = datasets.ImageFolder(root)
        self.root = folder.root
        self.classes = folder.classes
        self.samples = folder.samples
        self.images = [Image.open(path).convert('L') for path, _ in self.samples]
        self.labels = [label for _, label in self.samples]
        self.paths = [str(Path(path).relative_to(self.root)) for path, _ in self.samples]


class Subset(Dataset):
    """Some images of a CachedImages dataset with their own transform, so the
    training and validation images of the same folder can be augmented
    differently (the A1 starter's random_split shared one transform)."""

    def __init__(self, images, indices, transform):
        self.images, self.indices, self.transform = images, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return self.transform(self.images.images[j]), self.images.labels[j]


class MultiView:
    """Applies several transforms to the same image and returns all results,
    e.g. the two augmented views SimCLR compares."""

    def __init__(self, *transforms_):
        self.transforms = transforms_

    def __call__(self, image):
        return [t(image) for t in self.transforms]


def to_tensor():
    # Gray [0, 1] -> [-1, 1]
    return [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]


def build_transform(cfg, train):
    """The supervised transform: random resized crop and flip for training,
    a plain resize for evaluation."""
    d = cfg['data']
    size = d['img_size']
    aug = d['augment']
    if train and aug['random_resized_crop']:
        steps = [transforms.RandomResizedCrop(size, scale=tuple(aug['random_resized_crop']))]
    else:
        steps = [transforms.Resize((size, size))]  # ignores the aspect ratio, as in A1
    if train and aug['hflip']:
        steps.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(steps + to_tensor())


def build_ssl_transform(cfg):
    """One random view for self-supervised learning (cfg['ssl']['augment']).
    SimCLR's most important augmentation is colour distortion: without it two
    crops of an image can be matched by their colour histograms alone. Our
    input is gray, so the equivalent shortcut is the intensity histogram,
    which brightness and contrast jitter break instead."""
    a = cfg['ssl']['augment']
    steps = [transforms.RandomResizedCrop(cfg['data']['img_size'], scale=tuple(a['crop_scale']))]
    if a['hflip']:
        steps.append(transforms.RandomHorizontalFlip())
    if a['jitter_p'] > 0:
        jitter = transforms.ColorJitter(brightness=a['brightness'], contrast=a['contrast'])
        steps.append(transforms.RandomApply([jitter], p=a['jitter_p']))
    if a['blur_p'] > 0:
        size = cfg['data']['img_size'] // 10 * 2 + 1  # odd kernel, ~10% of the image as in SimCLR
        steps.append(transforms.RandomApply([transforms.GaussianBlur(size, sigma=(0.1, 2.0))], p=a['blur_p']))
    return transforms.Compose(steps + to_tensor())


def split_indices(cfg, dataset):
    """Training and validation indices for the configured fold. The folds in
    splits/folds.json are made by scripts/make_splits.py."""
    folds = json.loads(Path(cfg['data']['split_file']).read_text())['assignment']
    fold_of_image = [folds[path] for path in dataset.paths]
    train_idx = [i for i, f in enumerate(fold_of_image) if f != cfg['data']['fold']]
    val_idx = [i for i, f in enumerate(fold_of_image) if f == cfg['data']['fold']]
    return train_idx, val_idx


def keep_label_fraction(indices, labels, fraction, seed):
    """A random `fraction` of the images of every class (at least one each)."""
    if fraction >= 1:
        return indices
    rng = np.random.RandomState(seed)
    kept = []
    for c in sorted(set(labels[i] for i in indices)):
        members = [i for i in indices if labels[i] == c]
        kept += rng.choice(members, max(1, round(fraction * len(members))), replace=False).tolist()
    return sorted(kept)


def make_loader(dataset, cfg, batch_size, shuffle=False, seed=0, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=cfg['data']['num_workers'], pin_memory=torch.cuda.is_available(),
                      persistent_workers=cfg['data']['num_workers'] > 0,
                      generator=torch.Generator().manual_seed(seed))


def load_train_images(cfg):
    return CachedImages(Path(cfg['data']['root']) / 'train')


def build_train_val_loaders(cfg, seed, images=None):
    """Loaders for the labelled training folds and the validation fold of
    data/train. With joint training (ssl.joint_weight > 0) every training
    sample is [supervised view, SSL views...]."""
    images = images or load_train_images(cfg)
    train_idx, val_idx = split_indices(cfg, images)
    train_idx = keep_label_fraction(train_idx, images.labels, cfg['data']['label_fraction'], seed)
    transform = build_transform(cfg, train=True)
    if cfg['ssl']['joint_weight'] > 0:
        transform = MultiView(transform, *[build_ssl_transform(cfg)] * num_ssl_views(cfg))
    bs = cfg['train']['batch_size']
    train_loader = make_loader(Subset(images, train_idx, transform), cfg, bs, shuffle=True, seed=seed,
                               drop_last=cfg['ssl']['joint_weight'] > 0)
    val_loader = make_loader(Subset(images, val_idx, build_transform(cfg, train=False)), cfg, bs)
    return train_loader, val_loader, images


def build_pretrain_loaders(cfg, seed):
    """For SSL pretraining: the training folds without labels (views only),
    and the training and validation folds without augmentation for the kNN
    monitor. The validation fold is never used to train the encoder."""
    images = load_train_images(cfg)
    train_idx, val_idx = split_indices(cfg, images)
    p = cfg['pretrain']
    views = Subset(images, train_idx, MultiView(*[build_ssl_transform(cfg)] * num_ssl_views(cfg)))
    ssl_loader = make_loader(views, cfg, p['batch_size'], shuffle=True, seed=seed, drop_last=True)
    plain = build_transform(cfg, train=False)
    bank_loader = make_loader(Subset(images, train_idx, plain), cfg, 256)
    val_loader = make_loader(Subset(images, val_idx, plain), cfg, 256)
    return ssl_loader, bank_loader, val_loader, images


def num_ssl_views(cfg):
    return {'simclr': 2, 'rotation': 1}[cfg['ssl']['method']]


def build_test_loader(cfg, split='test'):
    """Loader for a labelled folder such as data/test, plus its class names and image paths."""
    images = CachedImages(Path(cfg['data']['root']) / split)
    loader = make_loader(Subset(images, range(len(images.images)), build_transform(cfg, train=False)),
                         cfg, cfg['train']['batch_size'])
    return loader, images.classes, images.paths
