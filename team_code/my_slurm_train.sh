#!/bin/bash
#SBATCH --job-name=djy_e2e
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:l40:4  # num_gpu
#SBATCH --mail-type=end
#SBATCH --mail-user=2011239@tongji.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# IMPORTANT: Start this script from within team_code folder, otherwise it will not work
## --load_file /share/home/u19666033/djy/carla_garage/results/BEVDrive_0/model_0030.pth

# print info about current job
scontrol show job $SLURM_JOB_ID

pwd
export CARLA_ROOT=/share/home/u19666033/djy/carla2
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/anaconda3/lib
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate garage_2
export OMP_NUM_THREADS=32  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=0 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    train.py --id c_GW_O_M_TS_EB --use_disk_cache 0 --crop_image 1 --seed 0 --epochs 31 --batch_size 16 --lr 3e-4 --setting all \
    --root_dir /share/home/u19666033/djy/carla_dataset \
    --logdir /share/home/u19666033/djy/carla_garage/results \
    --use_controller_input_prediction 1 --use_wp_gru 0 --continue_epoch 0 --cpu_cores 32 --use_depth 0 --use_semantic 0 --detect_boxes 1 --use_bev_semantic 1 \
    --image_architecture regnety_032 --lidar_architecture regnety_032 --load_file /share/home/u19666033/djy/carla_garage/results/c_GW_O_M_TS/model_0030.pthnull
