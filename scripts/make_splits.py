"""Find duplicate images and assign the training images to cross-validation folds.

    python scripts/make_splits.py

Two images are treated as duplicates when the correlation of their 32x32
grayscale thumbnails exceeds 0.97; every such pair we inspected was an exact copy.
Writes:
  splits/duplicates.json  duplicate groups in train, and test images that also occur in train
  splits/folds.json       the fold of every training image: 5 folds, stratified by class,
                          with all copies of an image in the same fold
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def thumbnails(paths):
    """Each image as a 32x32 gray vector with zero mean and unit length, so a dot
    product between two vectors is their correlation."""
    vectors = []
    for path in paths:
        v = np.asarray(Image.open(path).convert('L').resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
        v -= v.mean()
        vectors.append(v / (np.linalg.norm(v) + 1e-6))
    return np.stack(vectors)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='data')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threshold', type=float, default=0.97)
    args = p.parse_args()

    train_dir = Path(args.data, 'train')
    paths = sorted(train_dir.glob('*/*.jpg'))
    names = [str(path.relative_to(train_dir)) for path in paths]
    classes = [path.parent.name for path in paths]
    vectors = thumbnails(paths)
    similar = vectors @ vectors.T > args.threshold

    # Start with one group per image and merge the groups of every similar pair.
    group = list(range(len(paths)))
    for i, j in zip(*np.nonzero(similar)):
        if group[i] != group[j]:
            old, new = group[j], group[i]
            group = [new if g == old else g for g in group]
    members = {}
    for i, g in enumerate(group):
        members.setdefault(g, []).append(i)
    duplicates = {'train_groups': [[names[i] for i in m] for m in members.values() if len(m) > 1]}
    print(f'train: {len(duplicates["train_groups"])} duplicate groups')

    test_dir = Path(args.data, 'test')
    test_paths = sorted(test_dir.glob('*/*.jpg'))
    matches = thumbnails(test_paths) @ vectors.T > args.threshold
    duplicates['test_in_train'] = {str(path.relative_to(test_dir)): [names[j] for j in np.flatnonzero(row)]
                                   for path, row in zip(test_paths, matches) if row.any()}
    print(f'test: {len(duplicates["test_in_train"])} images also in train')

    # Within each class, shuffle the groups and put each one in the fold that
    # currently has the fewest images of that class.
    rng = np.random.default_rng(args.seed)
    fold_of = {}
    for cls in sorted(set(classes)):
        groups = [m for g, m in members.items() if classes[g] == cls]
        counts = np.zeros(args.folds, dtype=int)
        for k in rng.permutation(len(groups)):
            fold = int(np.argmin(counts))
            for i in groups[k]:
                fold_of[names[i]] = fold
            counts[fold] += len(groups[k])
    print(f'fold sizes: {np.bincount(list(fold_of.values())).tolist()}')

    Path('splits').mkdir(exist_ok=True)
    Path('splits/duplicates.json').write_text(json.dumps(duplicates, indent=1) + '\n')
    Path('splits/folds.json').write_text(json.dumps({'assignment': dict(sorted(fold_of.items()))}, indent=0) + '\n')


if __name__ == '__main__':
    main()
