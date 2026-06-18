"""
数据预处理模块
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_data(ide_logs_path, passed_path):
    """加载IDE日志和标签数据"""
    print("Loading data...")
    ide_logs = pd.read_csv(ide_logs_path)
    passed = pd.read_csv(passed_path)
    
    print(f"IDE logs shape: {ide_logs.shape}")
    print(f"Passed shape: {passed.shape}")
    print(f"Event types: {ide_logs['eventType'].unique()}")
    
    return ide_logs, passed

def get_event_types(ide_logs):
    """获取所有事件类型"""
    return ide_logs['eventType'].unique().tolist()

def aggregate_by_student(ide_logs, passed):
    """按学生聚合事件数据"""
    print("Aggregating by student...")
    
    # 确保时间戳是datetime类型
    ide_logs['timestamp'] = pd.to_datetime(ide_logs['timestamp'])
    
    # 按学生聚合
    student_events = {}
    for student_id, group in ide_logs.groupby('student'):
        student_events[student_id] = group.sort_values('timestamp')
    
    print(f"Number of students: {len(student_events)}")
    return student_events

if __name__ == "__main__":
    ide_logs, passed = load_data(
        '/tmp/IDE_logs/IDE_logs.csv',
        '/tmp/IDE_logs/passed.csv'
    )
