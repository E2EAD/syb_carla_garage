export CARLA_ROOT=/share/home/u19666033/djy/carla2
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/anaconda3/lib
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate syb_garage_2
export OMP_NUM_THREADS=32  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.

# Student init checkpoint (task N model init)
export PROJECT_ROOT=/share/home/u19666033/syb/carla_garage
STUDENT_INIT_CKPT=${PROJECT_ROOT}/log/syb_QtdA3D_v2_0-eb_2stg/model_0030.pth
# Frozen teacher directory (task N-1 best model dir with config.json + model_*.pth)
TEACHER_DIR=${PROJECT_ROOT}/log/syb_QtdA3D_v2_0-eb_2stg

torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=0 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    team_code/my_train_qtd_a3d_oracle.py --id syb_Qtd+LwF_1-ts_2stg --use_disk_cache 0 --crop_image 1 --seed 0 --epochs 31 --batch_size 32 --lr 3e-4 --setting all \
    --root_dir /share/home/u19666033/syb/pdm_dataset \
    --logdir /share/home/u19666033/syb/carla_garage/log \
    --use_controller_input_prediction 1 --use_wp_gru 0 --continue_epoch 0 --cpu_cores 32 --freeze_backbone 1 --use_depth 0 --use_semantic 0 --detect_boxes 1 --use_bev_semantic 1 \
    --image_architecture regnety_032 --lidar_architecture regnety_032 --load_file ${STUDENT_INIT_CKPT} 
