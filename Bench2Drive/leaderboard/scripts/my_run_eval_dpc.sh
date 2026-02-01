#!/bin/bash
#SBATCH --job-name=b2d_009
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=/mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results/logs/b2d_009_%a_%A.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/lustre/work/geiger/bjaeger25/garage_2_cleanup/results/logs/b2d_009_%a_%A.err   # File to which STDERR will be written
#SBATCH --partition=2080-galvani

export CARLA_ROOT=/home/dpc/djy/carla2
export WORK_DIR=/home/dpc/syb/carla_garage/Bench2Drive
export PROJECT_ROOT=/home/dpc/syb/carla_garage

export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:${PROJECT_ROOT}/team_code
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# ====== 可视化设置 ======
export DEBUG_CHALLENGE=1  # 启用调试和可视化
# export UNCERTAINTY_WEIGHT=1
# export TUNED_AIM_DISTANCE=0
# export DIRECT=1
# export SLOWER=0  # 保持原速

BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
# BASE_ROUTES=${WORK_DIR}/leaderboard/data/bench2drive220
BASE_ROUTES=${WORK_DIR}/leaderboard/data/bench2drive_dev10
TEAM_AGENT=${PROJECT_ROOT}/team_code/my_TFFdeQtd_agent.py
# TEAM_CONFIG=${PROJECT_ROOT}/log/
TEAM_CONFIG=/media/dpc/syb_disk_2/garage2_weights/syb_noMoe_wTFdeQtdv15_2stg
# BASE_CHECKPOINT_ENDPOINT=my_eval_bench2drive220
BASE_CHECKPOINT_ENDPOINT=my_eval_dev10
PLANNER_TYPE=my_traj
ALGO=my_tfpp
# SAVE_PATH=${WORK_DIR}/visualizaton_result/TFFdeQtd_v15_eb/eval_bench2drive220_${ALGO}_${PLANNER_TYPE}
SAVE_PATH=/media/dpc/syb_disk_2/visualizaton_result/noMoe_wTFdeQtdv15_2stg/
RESULT_PATH=dev10/noMoe_wTFdeQtdv15_2stg # 'No_Scenario','Give_Way', 'Overtaking', 'Merging', 'Traffic_Sign', 'Emergency_Brake'  

if [ ! -d "${WORK_DIR}/${RESULT_PATH}" ]; then
    mkdir ${WORK_DIR}/${RESULT_PATH}
    echo -e "\033[32m Directory ${RESULT_PATH} created. \033[0m"
else
    echo -e "\033[32m Directory ${RESULT_PATH} already exists. \033[0m"
fi

if [ ! -d "${SAVE_PATH}" ]; then
    mkdir -p ${SAVE_PATH}
    echo -e "\033[32m Directory ${SAVE_PATH} created. \033[0m"
else
    echo -e "\033[32m Directory ${SAVE_PATH} already exists. \033[0m"
fi

echo -e "**************\033[36m Running on single GPU - No splitting needed \033[0m **************"

# 直接使用原始XML文件，不需要分割
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${WORK_DIR}/${RESULT_PATH}/${BASE_CHECKPOINT_ENDPOINT}.json"
mkdir -p "${WORK_DIR}/${RESULT_PATH}"
GPU_RANK=0

echo -e "\033[32m ALGO: $ALGO \033[0m"
echo -e "\033[32m PLANNER_TYPE: $PLANNER_TYPE \033[0m"
echo -e "\033[32m PORT: $BASE_PORT \033[0m"
echo -e "\033[32m TM_PORT: $BASE_TM_PORT \033[0m"
echo -e "\033[32m ROUTES: $ROUTES \033[0m"
echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
echo -e "\033[32m GPU_RANK: $GPU_RANK \033[0m"
echo -e "\033[32m bash ${WORK_DIR}/leaderboard/scripts/run_evaluation.sh $BASE_PORT $BASE_TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \033[0m"
echo -e "***********************************************************************************"

# 直接运行评估脚本
bash -e ${WORK_DIR}/leaderboard/scripts/run_evaluation.sh $BASE_PORT $BASE_TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK

