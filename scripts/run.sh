#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

target_modality='CT'
pretrained_model_dir='./experiments/snapshots/MR2CT/MPSCL_MR2CT'
log_file='log_batch4_advent.txt'

for i in $(seq 500 500 50000)
do
  pretrained_model_pth="${pretrained_model_dir}/model_${i}.pth"
  python test.py --target_modality "${target_modality}" --pretrained_model_pth "${pretrained_model_pth}" >> "${log_file}" 2>&1 &
  wait
done

