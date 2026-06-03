#!/usr/bin/env bash
set -e

CONFIG="configs/cod-sam-vit-b-pvp.yaml"
GPU=0
EXP_PREFIX="exp_vitb"

for SEED in 123
do
    EXP_NAME="${EXP_PREFIX}_seed${SEED}"

    echo "========================================"
    echo "Running seed ${SEED}"
    echo "Experiment name: ${EXP_NAME}"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=${GPU} torchrun --nproc_per_node=1 train.py \
        --config ${CONFIG} \
        --name ${EXP_NAME} \
        --seed ${SEED}

    echo "Seed ${SEED} finished."
done

echo "All experiments finished."