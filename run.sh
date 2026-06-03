#!/usr/bin/env bash set -e

# python test.py --config save/_cod-sam-vit-b-pvp_seed456/config.yaml --model save/_cod-sam-vit-b-pvp_seed456/model_epoch_best.pth
python test.py --config save/exp_vitb_seed42/config.yaml --model save/exp_vitb_seed42/model_epoch_best.pth
# python test.py --config save/exp_vitb_seed123/config.yaml --model save/exp_vitb_seed123/model_epoch_best.pth


# CUDA_VISIBLE_DEVICES=0 python test.py --config configs/cod-sam-vit-b-pvp.yaml --model save/exp_vitb_pvp/model_epoch_best.pth

# python test.py --config configs/cod-sam-vit-b-pvp.yaml --model save/exp_vitb_pvp/model_epoch_best.pth


