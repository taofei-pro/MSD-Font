#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import time
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("disk_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DiskMonitor")

class DiskSpaceMonitor:
    """监控磁盘空间，当空间不足时清理旧的检查点文件"""
    
    def __init__(self, logs_dir, min_free_space_gb=50, check_interval_seconds=300):
        """
        初始化磁盘空间监控器
        
        参数:
            logs_dir: 日志目录路径
            min_free_space_gb: 最小可用空间（GB）
            check_interval_seconds: 检查间隔（秒）
        """
        self.logs_dir = Path(logs_dir)
        self.min_free_space_gb = min_free_space_gb
        self.check_interval_seconds = check_interval_seconds
        self.last_check_time = 0
        
    def get_free_space_gb(self, path):
        """获取指定路径所在分区的可用空间（GB）"""
        stat = shutil.disk_usage(path)
        return stat.free / (1024 * 1024 * 1024)  # 转换为GB
    
    def clean_old_checkpoints(self):
        """清理旧的检查点文件，保留最新的last.ckpt和最新的epoch检查点"""
        logger.info("开始清理旧的检查点文件...")
        
        # 查找所有检查点目录
        checkpoint_dirs = []
        for log_dir in self.logs_dir.glob("*"):
            if log_dir.is_dir():
                ckpt_dir = log_dir / "checkpoints"
                if ckpt_dir.exists():
                    checkpoint_dirs.append(ckpt_dir)
        
        # 对每个检查点目录进行清理
        for ckpt_dir in checkpoint_dirs:
            # 保留last.ckpt
            last_ckpt = ckpt_dir / "last.ckpt"
            
            # 获取所有epoch检查点并按修改时间排序
            epoch_ckpts = list(ckpt_dir.glob("epoch=*.ckpt"))
            epoch_ckpts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # 保留最新的3个epoch检查点，删除其余的
            to_keep = epoch_ckpts[:3] if epoch_ckpts else []
            if last_ckpt.exists():
                to_keep.append(last_ckpt)
            
            # 删除其他检查点
            for ckpt_file in ckpt_dir.glob("*.ckpt"):
                if ckpt_file not in to_keep:
                    try:
                        ckpt_file.unlink()
                        logger.info(f"已删除检查点文件: {ckpt_file}")
                    except Exception as e:
                        logger.error(f"删除文件 {ckpt_file} 时出错: {e}")
        
        logger.info("清理完成")
    
    def check_and_clean(self):
        """检查磁盘空间，如果不足则清理"""
        current_time = time.time()
        
        # 检查是否到了检查时间
        if current_time - self.last_check_time < self.check_interval_seconds:
            return True  # 默认返回True
        
        self.last_check_time = current_time
        free_space_gb = self.get_free_space_gb(self.logs_dir)
        
        logger.info(f"当前可用磁盘空间: {free_space_gb:.2f} GB")
        
        if free_space_gb < self.min_free_space_gb:
            logger.warning(f"磁盘空间不足! 当前可用: {free_space_gb:.2f} GB, 最小要求: {self.min_free_space_gb} GB")
            self.clean_old_checkpoints()
            
            # 再次检查空间
            new_free_space_gb = self.get_free_space_gb(self.logs_dir)
            logger.info(f"清理后可用磁盘空间: {new_free_space_gb:.2f} GB")
            
            if new_free_space_gb < self.min_free_space_gb:
                logger.error("清理后磁盘空间仍然不足!")
                return False
        
        return True

# 使用示例
if __name__ == "__main__":
    monitor = DiskSpaceMonitor("/home/zihun/workspace/MSD-Font/logs", min_free_space_gb=50)
    monitor.check_and_clean()
