"""
Step 4: 原型发现 - K-Means 聚类

对 Step 3 提取的多尺度表示进行 K-Means 聚类,
发现学生行为模式的原型 (prototypes)。

聚类后, 每个原型代表一种典型的学习行为模式:
  - 高频编辑 + 频繁运行: 积极调试型
  - 低频事件 + 接近截止日期: 临时抱佛脚型
  - 大量文本插入: 直接复制粘贴型
  - 少量事件 + 早提交: 快速完成型

聚类结果用于:
  1. 分析不同学生群体的行为特征
  2. 为后续微调提供原型特征
  3. 可解释性分析
"""

import os
import sys
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# sys.path 设置
# ============================================================
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def run_kmeans(representations, labels, n_clusters=4):
    """
    Step 4: K-Means 聚类发现行为原型

    Args:
        representations: np.array (n_samples, d_model) 多尺度特征表示
        labels: np.array (n_samples,) risk标签 (0=passed, 1=at-risk)
        n_clusters: 聚类数量

    Returns:
        tuple: (kmeans_model, cluster_assignments)
            - kmeans_model: 训练好的 KMeans 模型
            - cluster_assignments: np.array (n_samples,) 聚类分配
    """
    print(f"\n[Step 4] 原型发现 - K-Means 聚类")
    print(f"  样本数: {len(representations)}")
    print(f"  聚类数: {n_clusters}")
    print(f"  特征维度: {representations.shape[1]}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_assignments = kmeans.fit_predict(representations)

    # 打印聚类统计信息
    print(f"\n  聚类统计:")
    print(f"  {'聚类':>6s} | {'样本数':>6s} | {'通过率':>6s} | {'挂科率':>6s} | {'风险比':>6s}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    for c in range(n_clusters):
        mask = (cluster_assignments == c)
        n_in_cluster = mask.sum()
        if n_in_cluster == 0:
            print(f"  {c:>6d} | {0:>6d} | {'N/A':>6s} | {'N/A':>6s} | {'N/A':>6s}")
            continue

        passed = (labels[mask] == 0).sum()
        failed = (labels[mask] == 1).sum()
        pass_rate = passed / n_in_cluster * 100
        fail_rate = failed / n_in_cluster * 100
        risk_ratio = failed / n_in_cluster

        print(f"  {c:>6d} | {n_in_cluster:>6d} | {pass_rate:>5.1f}% | {fail_rate:>5.1f}% | {risk_ratio:>6.3f}")

    # 原型中心间距离
    print(f"\n  原型中心间距离矩阵:")
    centers = kmeans.cluster_centers_
    dists = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            dists[i, j] = np.linalg.norm(centers[i] - centers[j])

    header = "        " + "  ".join([f"  C{j}  " for j in range(n_clusters)])
    print(header)
    for i in range(n_clusters):
        row = f"  C{i}  | " + "  ".join([f"{dists[i, j]:.3f}" for j in range(n_clusters)])
        print(row)

    print(f"\n  聚类完成, 惯性 (inertia): {kmeans.inertia_:.4f}")

    return kmeans, cluster_assignments


if __name__ == '__main__':
    from models.mamba.steps.step1_preprocessing import preprocess
    from models.mamba.steps.step3_multiscale import extract_representations
    from models.mamba.model import create_model
    from common.data_loader import get_device, set_seed

    set_seed(42)
    device = get_device()

    # Step 1: 加载数据
    samples, student_ids, labels = preprocess()

    # Step 3: 提取表示
    model = create_model(device)
    reprs, lbls = extract_representations(model, samples, device)

    # Step 4: K-Means 聚类
    kmeans, assignments = run_kmeans(reprs, lbls, n_clusters=4)

    print(f"\n聚类测试完成")
