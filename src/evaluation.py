# -*- coding: utf-8 -*-
"""評価ユーティリティ: CV 計測 / 比較表 / 特徴量重要度の可視化."""
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold

RANDOM_STATE = 42


def make_cv(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def cv_accuracy(model, X, y, cv=None):
    """5-fold CV の平均 Accuracy。"""
    cv = cv or make_cv()
    return cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()


def evaluate_models(models, X, y, cv=None):
    """{name: model} を一括評価して比較 DataFrame を返す。"""
    cv = cv or make_cv()
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        rows.append({"モデル": name, "CV平均Accuracy": scores.mean(), "標準偏差": scores.std()})
    return pd.DataFrame(rows).sort_values("CV平均Accuracy", ascending=False).reset_index(drop=True)


def plot_feature_importance(model, feature_names, out_path, title="Feature Importance"):
    """学習済みツリーモデルの特徴量重要度を横棒グラフで保存。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, len(imp) * 0.35)))
    ax.barh(imp.index, imp.values, color="#3b7dd8")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return imp.sort_values(ascending=False)


def make_submission(model, X_train, y, X_test, passenger_ids, out_path):
    """学習→予測→提出 CSV 生成 (gender_submission.csv と同形式)。"""
    model.fit(X_train, y)
    pred = model.predict(X_test).astype(int)
    sub = pd.DataFrame({"PassengerId": passenger_ids, "Survived": pred})
    sub.to_csv(out_path, index=False)
    return sub
