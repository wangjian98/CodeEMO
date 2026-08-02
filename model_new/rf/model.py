"""随机森林模型 - 传统机器学习基线"""
from sklearn.ensemble import RandomForestClassifier


def create_model(n_estimators=100, max_depth=10, random_state=42):
    """创建随机森林模型"""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
