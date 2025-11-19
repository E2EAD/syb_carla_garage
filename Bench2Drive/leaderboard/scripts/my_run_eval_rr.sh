#!/bin/bash

# 包装脚本：在脚本退出时自动重启

MAX_RESTARTS=15  # 最大重启次数，防止无限循环
RESTART_DELAY=5  # 重启前等待的秒数

export CARLA_ROOT=/share/home/u19666033/djy/carla2
export WORK_DIR=/share/home/u19666033/syb/carla_garage/Bench2Drive
export PROJECT_ROOT=/share/home/u19666033/syb/carla_garage

export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:${PROJECT_ROOT}/team_code
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

restart_count=0

while [ $restart_count -lt $MAX_RESTARTS ]; do
    echo "=========================================="
    echo "启动仿真器运行脚本 (尝试 $((restart_count + 1))/$MAX_RESTARTS)"
    echo "开始时间: $(date)"
    echo "=========================================="
    
    # 运行清理脚本
    echo "清理现有进程..."
    ./Bench2Drive/tools/clean_carla.sh
    
    # 运行你的主脚本
    bash ./Bench2Drive/leaderboard/scripts/my_run_eval_2gpu.sh
    
    exit_code=$?
    echo "=========================================="
    echo "脚本退出，代码: $exit_code"
    echo "退出时间: $(date)"
    echo "=========================================="
    
    # 如果脚本正常退出（不是由于崩溃），则不再重启
    if [ $exit_code -eq 0 ]; then
        echo "脚本正常退出，不再重启"
        break
    fi
    
    restart_count=$((restart_count + 1))
    
    if [ $restart_count -lt $MAX_RESTARTS ]; then
        echo "等待 ${RESTART_DELAY} 秒后重启..."
        sleep $RESTART_DELAY
    else
        echo "达到最大重启次数 ($MAX_RESTARTS)，停止重启"
    fi
donenull
