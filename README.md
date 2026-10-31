# ITCS 6169/8169 Assignment 2: self-supervised learning for 16-class scene recognition

**Result:** 87.3% test accuracy (95% CI 84.0-90.5%; 86.4% on the 15 scene
classes; 87.1% without the 4 test images that also occur in the training set)
and 88.2 ± 0.1% cross-validation accuracy. No externally pretrained weights and
no extra images are used.

**Model:** the from-scratch CNN of Assignment 1, pretrained with rotation
prediction on the training images (no labels), then fine-tuned on the labels.
The prediction is the average of the five fold models
(`configs/pretrain/p2_rotation.yaml`, then `configs/ssl/s3_rotation_ft.yaml`).

**Checkpoints:** [GitHub Release v1.0](https://github.com/sshah100-clt/CV_Assignment2/releases/tag/v1.0)
(five files, `fold0_last.pt` to `fold4_last.pt`, about 5 MB each).

## How to run

### 1. Get the code and install the dependencies

```bash
git clone https://github.com/sshah100-clt/CV_Assignment2.git
cd CV_Assignment2
pip install -r requirements.txt
```

### 2. Add the dataset

Download the dataset from the link in the assignment handout and put the
images in `data/` inside the repository:

```text
data/train/<class>/*.jpg     2,400 images, 16 class folders
data/test/<class>/*.jpg      400 images, 16 class folders
```

Check that the images match the cross-validation folds (prints `OK`):

```bash
python -c "import json,pathlib; f=json.load(open('splits/folds.json'))['assignment']; imgs={str(p.relative_to('data/train')) for p in pathlib.Path('data/train').glob('*/*.jpg')}; print('OK' if imgs==set(f) else 'MISMATCH')"
```

### 3a. Evaluate the released model (a few minutes, GPU or CPU)

Download the five checkpoints from the
[Release](https://github.com/sshah100-clt/CV_Assignment2/releases/tag/v1.0) and place them like this:

```bash
for f in 0 1 2 3 4; do
  mkdir -p runs/s3_rotation_ft/f${f}_s0
  mv fold${f}_last.pt runs/s3_rotation_ft/f${f}_s0/last.pt
done
python evaluate.py --checkpoint runs/s3_rotation_ft/f{0,1,2,3,4}_s0/last.pt
```

This prints the test accuracy (87.3%) and writes
`runs/s3_rotation_ft/f0_s0/test_results.csv` and a confusion matrix.

### 3b. Train only the final model (about 1.5 hours on 4 GPUs)

Rotation pretraining for each fold, then fine-tuning from the encoder of the
same fold. Move the released checkpoints aside first, because finished runs
are skipped.

```bash
python scripts/launch.py --pretrain configs/pretrain/p2_rotation.yaml --seeds 0 --jobs-per-gpu 2
python scripts/launch.py configs/ssl/s3_rotation_ft.yaml --seeds 0 --jobs-per-gpu 3
python evaluate.py --checkpoint runs/s3_rotation_ft/f{0,1,2,3,4}_s0/last.pt
```

### 3c. Reproduce every experiment and table

The repository already contains the reported results in `runs/` (without the
model weights), and finished runs are skipped. Move them aside first, so that
everything is trained again:

```bash
mv runs runs_reported
SEEDS="0 1 2" bash run_all.sh 1    # all pretraining and classifier runs (about 8 hours on 4 GPUs)
bash run_all.sh 2                  # result tables and McNemar tests
python scripts/test_analysis.py    # test accuracy of Table 1 in the report
python scripts/make_figures.py     # Figure 1 of the report
```

The main comparisons in the report use seeds 0, 1 and 2. The ablations
(`p1a_no_jitter`, `s2a_no_jitter_probe`), B1 and the 10%-label runs use seed 0
only; `SEEDS=0 bash run_all.sh 1` (about 3 hours) runs every experiment with
seed 0. Use `GPUS="0 1"`, `JOBS=2` (classifier runs per GPU) and `PRE_JOBS=1`
(pretraining runs per GPU) to fit other machines. If a run fails, its log is
in `runs/logs/`; fix the cause and run the same command again.

| File | Contents |
|---|---|
| `runs/summary.md`, `runs/summary.csv` | cross-validation accuracy of every classifier experiment |
| `runs/summary_pretrain.md` | kNN accuracy before and after each kind of pretraining |
| `runs/comparisons.csv` | paired McNemar tests used in the report |
| `runs/<experiment>/f0_s0/eval_test.json` | test accuracy of a 5-model average |
| `report/figures/fig1_ssl.png` | Figure 1 of the report |

### Running single experiments

```bash
python pretrain.py --config configs/pretrain/p1_simclr.yaml --fold 0 --seed 0   # one SSL encoder
python train.py --config configs/ssl/s1_simclr_ft.yaml --fold 0 --seed 0        # fine-tune it
python scripts/launch.py configs/ssl/s1_simclr_ft.yaml --seeds 0                # 5 folds
python scripts/summarize.py                                  # cross-validation tables
python scripts/oof.py --compare b0_supervised s3_rotation_ft # paired McNemar test
python scripts/oof.py --experiment s3_rotation_ft            # per-class accuracy and confusions
```

`--set key=value` overrides any config value, e.g. `--set pretrain.epochs=20`.
A classifier config with `model.init` loads the encoder of the same fold and
seed and stops with an error if it is missing or belongs to another fold.

## Experiments

| Config | What it tests |
|---|---|
| `supervised/b0_supervised` | supervised baseline: A1's best from-scratch CNN, 150 epochs |
| `supervised/b1_supervised_650ep` | B0 trained as many epochs as pretraining + fine-tuning |
| `pretrain/p1_simclr` | SimCLR pretraining, gray-image augmentations |
| `pretrain/p1a_no_jitter` | SimCLR without brightness/contrast jitter |
| `pretrain/p2_rotation` | rotation-prediction pretraining |
| `ssl/s0_random_probe` | linear probe on a random encoder (lower bound) |
| `ssl/s1_simclr_probe`, `ssl/s1_simclr_ft` | SimCLR encoder: linear probe, fine-tuning |
| `ssl/s2a_no_jitter_probe` | linear probe of the no-jitter encoder |
| `ssl/s3_rotation_probe`, `ssl/s3_rotation_ft` | rotation encoder: linear probe, fine-tuning (best) |
| `joint/j1_simclr_joint`, `joint/j2_rotation_joint` | joint training: cross-entropy + SSL loss from scratch |

The 10%-label experiments are `b0_supervised` and `s1_simclr_ft` with
`--grid data.label_fraction=0.1` (see `run_all.sh`). The configs in
`configs/backbone/` and the `p1b`/`p1c`/`s2b`/`s2c` ablations were prepared
but not run for the report.

## Validation protocol

Every experiment uses 5-fold cross-validation over the 2,400 training images
(`splits/folds.json`, from Assignment 1: stratified by class, duplicate images
in the same fold). Each SSL encoder is pretrained on the four training folds
only, and the classifier of a fold starts from the encoder of that fold, so no
validation image is seen during pretraining. An experiment is scored by the
final-epoch accuracy over all 2,400 images, each predicted by the model that
did not train on it. Differences are tested with paired McNemar tests. The
test set was used once per reported model, after the recipe was chosen. All
images are converted to grayscale, because only the Flower class is in
colour. The unlabelled `data/test2` folder is not used.

## Repository layout

```text
pretrain.py           SSL pretraining of one encoder on one fold (SimCLR or rotation)
train.py              trains one classifier on one fold: from scratch, from an encoder, or jointly
evaluate.py           test accuracy of one model or an average of several
run_all.sh            runs every experiment and writes the result tables
configs/base.yaml     all settings and their defaults
configs/              one file per experiment (see the table above)
scenecnn/             data loading, models, SSL objectives, training loops, config loading
scripts/              CV splits, job launcher, result tables, McNemar tests, test analysis, figure
splits/               the CV folds and the duplicate list
runs/                 logs, metrics, predictions and result tables of every run
report/               the two-page report and its figure
```

## Environment

The reported results come from 4 NVIDIA RTX A5000 GPUs (24 GB each) with
Python 3.12.3, PyTorch 2.12.1 and CUDA 13.0 (recorded in every run's
`metrics.json`). Pretraining uses bfloat16 on CUDA. Seeds are fixed, but GPU
training is not bit-for-bit deterministic, so a rerun gives values within a
few tenths of a point.
