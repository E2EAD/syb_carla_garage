export CARLA_ROOT=/share/home/u19666033/djy/carla2
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/anaconda3/lib
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate syb_garage_2
export OMP_NUM_THREADS=32  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=0 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    team_code/my_train_tfpp_ability.py --id tfpp_IBS_2-m_2stg --use_disk_cache 0 --crop_image 1 --seed 0 --epochs 31 --batch_size 16 --lr 3e-4 --setting all \
    --root_dir /share/home/u19666033/syb/pdm_dataset \
    --logdir /share/home/u19666033/syb/carla_garage/log \
    --use_controller_input_prediction 1 --use_wp_gru 0 --continue_epoch 0 --cpu_cores 32 --use_depth 0 --use_semantic 0 --detect_boxes 1 --use_bev_semantic 1 \
    --image_architecture regnety_032 --lidar_architecture regnety_032 --load_file /share/home/u19666033/syb/carla_garage/log/tfpp_IBS_1-ts_2stg/model_0030.pth 
