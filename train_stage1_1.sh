#!/bin/bash

# 设置基本参数
CONFIG_PATH="configs/MSDFont/MSDFont_Train_Stage1_trans_model_predx0_miniUnet.yaml"
GPUS="0,"
MIN_FREE_SPACE_GB=30
CHECK_INTERVAL_SECONDS=600
LOGS_DIR="/home/zihun/workspace/MSD-Font/StableDiffusion/logs/stage1_1"


# 检查磁盘空间函数
check_disk_space() {
    local free_space=$(df -BG --output=avail $LOGS_DIR | tail -n 1 | tr -d 'G')
    echo "当前可用磁盘空间: ${free_space}GB"
    
    if [ $free_space -lt $MIN_FREE_SPACE_GB ]; then
        echo "磁盘空间不足! 当前可用: ${free_space}GB, 最小要求: ${MIN_FREE_SPACE_GB}GB"
        return 1
    fi
    
    return 0
}

# 清理旧的检查点文件
clean_old_checkpoints() {
    echo "开始清理旧的检查点文件..."
    
    # 保留最新的last.ckpt和最新的3个epoch检查点
    for ckpt_dir in $(find $LOGS_DIR -type d -name "checkpoints"); do
        # 保留last.ckpt
        if [ -f "$ckpt_dir/last.ckpt" ]; then
            touch "$ckpt_dir/last.ckpt"  # 更新时间戳
        fi
        
        # 获取最新的3个epoch检查点
        find "$ckpt_dir" -name "epoch=*.ckpt" -type f -printf "%T@ %p\n" | sort -nr | head -n 3 | cut -d' ' -f2- | while read file; do
            touch "$file"  # 更新时间戳
        done
        
        # 删除其他所有检查点
        find "$ckpt_dir" -name "*.ckpt" -type f -not -newermt "1 minute ago" -delete
    done
    
    echo "清理完成"
}

# 主循环
main() {
    # 首先检查磁盘空间
    if ! check_disk_space; then
        echo "尝试清理空间..."
        clean_old_checkpoints
        
        # 再次检查空间
        if ! check_disk_space; then
            echo "清理后磁盘空间仍然不足! 请手动清理空间后重试。"
            exit 1
        fi
    fi
    
    # 启动训练
    echo "启动训练..."
    conda activate MSDFont
    
    # 在后台运行训练
    python main.py --base $CONFIG_PATH -t --gpus $GPUS &
    TRAIN_PID=$!
    
    # 监控磁盘空间
    while kill -0 $TRAIN_PID 2>/dev/null; do
        sleep $CHECK_INTERVAL_SECONDS
        
        if ! check_disk_space; then
            echo "磁盘空间不足，尝试清理..."
            clean_old_checkpoints
            
            if ! check_disk_space; then
                echo "清理后磁盘空间仍然不足! 停止训练..."
                kill $TRAIN_PID
                exit 1
            fi
        fi
    done
    
    # 等待训练进程结束
    wait $TRAIN_PID
    EXIT_CODE=$?
    
    echo "训练结束，退出代码: $EXIT_CODE"
    return $EXIT_CODE
}

# 执行主函数
main
