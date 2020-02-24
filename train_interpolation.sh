#!/usr/bin/env bash

python main.py \
--exp_name 'interpolation_exp_2' \
--split 'interpolation' \
--device_idx 0 \
--checkpoint '/media/pinscreen/a/face_neo/coma/out/interpolation_exp_2/checkpoints/checkpoint_300.pt' \
--batch_size 1 \
--epochs 0