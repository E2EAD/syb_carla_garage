#!/bin/bash
# =====================================================================
# Shell script for online alternating Network-DPMM training (2 GPUs).
#
# Usage:
#   bash core_team_code/online_dpmm/my_shell_train_online_dpmm_2gpu.sh
#
# Adjust the variables below to match your environment.
# =====================================================================
set -euo pipefail

export PROJECT_ROOT=/home/spc/syb_carla_garage
export CARLA_ROOT=/home/spc/carla_0_9_15

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/spc/anaconda3/lib

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1

# --- Paths ---
LOGDIR=${PROJECT_ROOT}/log
DATASET_ROOT=/home/spc/b2d_mini_v2
EXP_ID=syb_online_dpmm_2gpu

# Optional: load a pre-trained model to fine-tune (leave empty to train from scratch)
# LOAD_FILE=${PROJECT_ROOT}/log/syb_tfpp_withB2dTrajFit_stg1/model_0030.pth
LOAD_FILE=""

# --- Build the torchrun command ---
CMD="CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=1 --rdzv_id=251122 --rdzv_backend=c10d \
  core_team_code/online_dpmm/my_train_ability_wTFFdeQtd_online.py \
  --id ${EXP_ID} \
  --use_disk_cache 0 --crop_image 1 --seed 0 --epochs 31 --batch_size 8 --lr 3e-4 --setting all \
  --root_dir ${DATASET_ROOT} \
  --logdir ${LOGDIR} \
  --use_controller_input_prediction 1 --use_wp_gru 0 --continue_epoch 0 --cpu_cores 16 \
  --freeze_backbone 1 --use_depth 0 --use_semantic 0 --detect_boxes 1 --use_bev_semantic 1 \
  --image_architecture regnety_032 --lidar_architecture regnety_032 \
  --sync_batch_norm 1 \
  --online_dpmm 1 \
  --use_traj_front_door_encoder 1 \
  --use_prior_fuseFeat 1 \
  --traj_dpmm_buffer_size 4096 \
  --fusefeat_dpmm_buffer_size 2048 \
  --dpmm_update_start_step 200 \
  --dpmm_update_freq_steps 0 \
  --traj_dpmm_replay_ratio 0.5 \
  --fusefeat_dpmm_replay_ratio 0.5"

# Optional: add --load_file if specified
if [ -n "${LOAD_FILE}" ]; then
    CMD="${CMD} --load_file ${LOAD_FILE} --continue_epoch 1"
fi

echo "Running command:"
echo "${CMD}"
eval "${CMD}"