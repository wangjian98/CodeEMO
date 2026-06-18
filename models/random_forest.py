"""
随机森林模型
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def create_random_forest(n_estimators=100, max_depth=10, random_state=42):
    """创建随机森林模型"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    return model

if __name__ == "__main__":
    model = create_random_forest()
    print("Random Forest model created")
