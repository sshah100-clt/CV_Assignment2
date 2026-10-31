#!/usr/bin/env bash
# Runs every experiment, then writes the result tables. Every experiment is
# 5-fold cross-validation with seed 0 (SEEDS="0 1 2" for more seeds); SSL
# encoders are pretrained per fold and seed, and the classifier for fold f,
# seed s starts from the encoder of fold f, seed s.
#
#   bash run_all.sh                    # all stages
#   bash run_all.sh 1                  # one stage (0, 1 or 2)
#   GPUS="0 1" bash run_all.sh 1       # other GPUs
#   PRE_JOBS=1 JOBS=2 bash run_all.sh  # runs per GPU for pretraining / classifiers
#
# Stage 0: one SimCLR pretraining run, to check speed and GPU memory.
# Stage 1: all experiments in the report:
#   - required: supervised baseline, SSL + linear probe / fine-tuning
#     (SimCLR and rotation prediction), joint SSL + classification;
#   - analysis: does SimCLR need brightness/contrast jitter on gray images,
#     is the SSL gain just longer training, does SSL help more with 10% of
#     the labels.
# Stage 2: result tables and paired McNemar tests.
#
# Finished runs are skipped, so after a failure (or an interruption) fix the
# cause and start the script again.
set -euo pipefail
export PYTHONUNBUFFERED=1   # progress appears at once, also through a pipe (tee)
cd "$(dirname "$0")"
# Under Slurm, use exactly the GPUs the job was given.
GPUS=${GPUS:-$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" | tr , ' ')}
SEEDS=${SEEDS:-0}
JOBS=${JOBS:-3}            # classifiers are small
PRE_JOBS=${PRE_JOBS:-2}    # pretraining: ~8 GB and full GPU use per run
STAGE=${1:-all}
pretrain() { python scripts/launch.py --pretrain "$@" --seeds $SEEDS --gpus $GPUS --jobs-per-gpu "$PRE_JOBS"; }
train() { python scripts/launch.py "$@" --seeds $SEEDS --gpus $GPUS --jobs-per-gpu "$JOBS"; }
stage() { [[ $STAGE == all || $STAGE == "$1" ]]; }

if [[ $STAGE == 0 ]]; then
  python pretrain.py --config configs/pretrain/p1_simclr.yaml --fold 0 --seed 0 --name stage0_timing --set pretrain.epochs=20
  exit
fi

if stage 1; then
  pretrain configs/pretrain/p1_simclr.yaml configs/pretrain/p2_rotation.yaml configs/pretrain/p1a_no_jitter.yaml
  # B1 (650 epochs) is the longest, so it starts first.
  train configs/supervised/b1_supervised_650ep.yaml configs/supervised/b0_supervised.yaml \
        configs/ssl/s0_random_probe.yaml configs/ssl/s1_simclr_probe.yaml configs/ssl/s1_simclr_ft.yaml \
        configs/ssl/s2a_no_jitter_probe.yaml configs/ssl/s3_rotation_probe.yaml configs/ssl/s3_rotation_ft.yaml \
        configs/joint/j1_simclr_joint.yaml configs/joint/j2_rotation_joint.yaml
  # Scarce labels: the classifier sees 10% of the labels; SimCLR saw all training-fold images (no labels).
  train configs/supervised/b0_supervised.yaml configs/ssl/s1_simclr_ft.yaml --grid data.label_fraction=0.1
fi

if stage 2; then
  python scripts/summarize.py
  rm -f runs/comparisons.csv
  python scripts/oof.py --compare b0_supervised s1_simclr_ft
  python scripts/oof.py --compare b0_supervised j1_simclr_joint
  python scripts/oof.py --compare b0_supervised s3_rotation_ft
  python scripts/oof.py --compare s1_simclr_ft j1_simclr_joint
  python scripts/oof.py --compare b1_supervised_650ep s1_simclr_ft
  python scripts/oof.py --compare b1_supervised_650ep s3_rotation_ft
  python scripts/oof.py --compare s1_simclr_ft s3_rotation_ft
  python scripts/oof.py --compare b0_supervised__label_fraction0.1 s1_simclr_ft__label_fraction0.1
fi
