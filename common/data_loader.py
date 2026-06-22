"""
数据加载模块 - 从 /tmp/IDE_logs/ 加载IDE日志数据
"""
import pandas as pd
import numpy as np


def load_ide_logs(logs_path='/tmp/IDE_logs/IDE_logs.csv',
                  passed_path='/tmp/IDE_logs/passed.csv'):
    """加载IDE日志和标签数据

    Returns:
        ide_logs: DataFrame (student, part, exercise, eventType, timestamp, timeToDeadline)
        passed: DataFrame (student, passed)
    """
    print("Loading IDE logs...")
    ide_logs = pd.read_csv(logs_path)
    passed = pd.read_csv(passed_path)

    ide_logs['timestamp'] = pd.to_datetime(ide_logs['timestamp'])

    print(f"  IDE logs shape: {ide_logs.shape}")
    print(f"  Passed shape: {passed.shape}")
    print(f"  Event types: {ide_logs['eventType'].unique().tolist()}")

    return ide_logs, passed


def get_device():
    """获取计算设备 (CPU/GPU 自动检测)"""
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    """设置随机种子"""
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
