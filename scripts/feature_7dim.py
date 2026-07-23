"""
7维原始特征构建模块

从预处理序列文件 /tmp/IDE_logs/out/X_seq.npy 加载原始11维数据，
取前7维（核心事件类型）的均值聚合作为7维原始特征。

与46维增强特征形成对比，验证特征工程的效果。

7维特征 (每学生144个exercise的均值):
    text_insert, text_remove, text_paste, focus_gained, focus_lost, run, submit
"""
import numpy as np


EVENT_TYPES_7 = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit'
]


def build_feature_matrix_7dim():
    """从X_seq.npy加载7维原始特征（每学生144个exercise的均值聚合）

    X_seq: (n_students=473, n_exercises=144, n_features=11)
    前7维为核心事件原始计数的z-score标准化值，
    取每个学生在144个exercise上的均值作为7维特征。

    Returns:
        X: np.array (n_students, 7)
        y: np.array (n_students,)
        student_ids: list (0..472)
    """
    X_seq = np.load('/tmp/IDE_logs/out/X_seq.npy')   # (473, 144, 11)
    y = np.load('/tmp/IDE_logs/out/y.npy')            # (473,)

    # 前7维: 核心事件类型的z-score值, 在exercise维度求均值
    X = X_seq[:, :, :7].mean(axis=1)   # (473, 7)

    print(f"  7-dim Feature matrix: {X.shape}  (from X_seq mean across 144 exercises)")
    print(f"  Passed: {int(y.sum())}, Failed: {int((y==0).sum())}")
    print(f"  Feature names: {EVENT_TYPES_7}")

    return X, y, None
