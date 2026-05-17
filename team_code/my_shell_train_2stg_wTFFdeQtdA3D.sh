#!/bin/bash
set -euo pipefail

export PROJECT_ROOT=/home/spc/syb_carla_garage
export CARLA_ROOT=/home/spc/carla_0_9_15

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/spc/anaconda3/lib

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

# Student init checkpoint (task N model init)
STUDENT_INIT_CKPT=${PROJECT_ROOT}/log/syb_QtdA3D_v2_0-eb_2stg/model_0030.pth
# Frozen teacher directory (task N-1 best model dir with config.json + model_*.pth)
TEACHER_DIR=${PROJECT_ROOT}/log/syb_QtdA3D_v2_0-eb_2stg

torchrun --nnodes=1 --nproc_per_node=1 --max_restarts=1 --rdzv_id=251122 --rdzv_backend=c10d \
  team_code/my_train_ability_QtdHgs.py \
  --id demo_QtdHgs_1-ts_2stg \
  --use_disk_cache 0 --crop_image 1 --seed 0 --epochs 31 --batch_size 2 --lr 3e-4 --setting all \
  --root_dir /home/spc/b2d_mini_v2 \
  --logdir ${PROJECT_ROOT}/log \
  --use_controller_input_prediction 1 --use_wp_gru 0 --continue_epoch 0 --cpu_cores 32 \
  --freeze_backbone 1 --use_depth 0 --use_semantic 0 --detect_boxes 1 --use_bev_semantic 1 \
  --image_architecture regnety_032 --lidar_architecture regnety_032 \
  --load_file ${STUDENT_INIT_CKPT} #--use_a3d 1 --a3d_ref_file ${TEACHER_DIR}
