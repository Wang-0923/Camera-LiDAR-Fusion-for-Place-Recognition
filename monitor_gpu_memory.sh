#!/bin/bash

# 设置初始的显存使用量
prev_mem_gpu_0=0
prev_mem_gpu_1=0

# 输出初始状态
echo "Monitoring GPU memory usage..."

# 监控两个 GPU 的显存变化
while true; do
    # 获取当前时间戳
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # 获取 GPU 0 和 GPU 1 的显存使用情况（单位：MiB）
    current_mem_gpu_0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    current_mem_gpu_1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)

    # 计算显存的变化量
    mem_diff_gpu_0=$((current_mem_gpu_0 - prev_mem_gpu_0))
    mem_diff_gpu_1=$((current_mem_gpu_1 - prev_mem_gpu_1))

    # 输出当前时间，显存变化量以及当前显存使用情况
    echo "$timestamp - GPU 0 Memory Change: +$mem_diff_gpu_0 MiB, GPU 0 Usage: $current_mem_gpu_0 MiB"
    echo "$timestamp - GPU 1 Memory Change: +$mem_diff_gpu_1 MiB, GPU 1 Usage: $current_mem_gpu_1 MiB"

    # 更新上次的显存使用量
    prev_mem_gpu_0=$current_mem_gpu_0
    prev_mem_gpu_1=$current_mem_gpu_1

    # 暂停 5 秒钟后继续
    sleep 1
done

