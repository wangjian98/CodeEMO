"""
数据增强工具 - 适用于 46 维静态特征
增强方法:
  1. Gaussian Noise   - 加性高斯噪声
  2. Feature Dropout  - 随机置零部分特征
  3. SMOTE-like       - 同类样本间线性插值
  4. Mixup            - 跨类样本线性插值 (软标签)
"""
import numpy as np


def gaussian_noise(X, y, sigma=0.1, seed=None):
    """加性高斯噪声 - 标签保持不变"""
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.randn(*X.shape) * sigma
    return X + noise, y.copy()


def feature_dropout(X, y, p=0.1, seed=None):
    """随机将 p 比例的特征置零 - 标签保持不变"""
    if seed is not None:
        np.random.seed(seed)
    mask = np.random.binomial(1, 1 - p, X.shape).astype(np.float32)
    return X * mask, y.copy()


def smote_like(X, y, n_samples=None, k=5, seed=None):
    """SMOTE 风格: 同类样本间线性插值,生成新样本"""
    if seed is not None:
        np.random.seed(seed)
    if n_samples is None:
        n_samples = len(X)
    classes = np.unique(y)
    new_X, new_y = [], []
    for c in classes:
        X_c = X[y == c]
        if len(X_c) < 2:
            continue
        idxs = np.random.randint(0, len(X_c), size=n_samples)
        for i in idxs:
            j = np.random.randint(0, len(X_c))
            lam = np.random.uniform(0.3, 0.7)
            new_X.append(X_c[i] * lam + X_c[j] * (1 - lam))
            new_y.append(c)
    return np.array(new_X), np.array(new_y)


def mixup(X, y, n_samples=None, alpha=0.4, seed=None):
    """Mixup: 两样本线性插值,标签为软标签"""
    if seed is not None:
        np.random.seed(seed)
    if n_samples is None:
        n_samples = len(X)
    new_X, new_y = [], []
    for _ in range(n_samples):
        i, j = np.random.randint(0, len(X), size=2)
        lam = np.random.beta(alpha, alpha)
        new_X.append(X[i] * lam + X[j] * (1 - lam))
        new_y.append(y[i] * lam + y[j] * (1 - lam))
    return np.array(new_X), np.array(new_y)


def combined_augment(X, y, factor=3, sigma=0.08, drop_p=0.1, seed=42):
    """组合增强: 原数据 + 噪声 + 特征丢弃 + SMOTE,数据量扩大 factor 倍

    factor=3 -> 原数据占 1/3, 增强占 2/3
    """
    np.random.seed(seed)
    n = len(X)
    aug_X = [X]
    aug_y = [y.astype(np.float32)]

    # 1. 高斯噪声 (30%)
    n_noise = int(n * (factor - 1) * 0.4)
    Xn, yn = gaussian_noise(X, y, sigma=sigma, seed=seed + 1)
    aug_X.append(Xn[:n_noise])
    aug_y.append(yn[:n_noise].astype(np.float32))

    # 2. 特征丢弃 (30%)
    n_drop = int(n * (factor - 1) * 0.3)
    Xd, yd = feature_dropout(X, y, p=drop_p, seed=seed + 2)
    aug_X.append(Xd[:n_drop])
    aug_y.append(yd[:n_drop].astype(np.float32))

    # 3. SMOTE (40%)
    n_smote = int(n * (factor - 1) * 0.3)
    Xs, ys = smote_like(X, y, n_samples=n_smote, seed=seed + 3)
    if len(Xs) > 0:
        aug_X.append(Xs)
        aug_y.append(ys.astype(np.float32))

    X_out = np.vstack(aug_X)
    y_out = np.concatenate(aug_y)
    return X_out, y_out