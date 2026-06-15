# -*- coding: utf-8 -*-
"""モデル定義・チューニング・学習.

- build_baseline_models : Phase 1 の 3 モデル (LR / DT / RF)
- build_tuned_models    : Phase 2 で GridSearch から得た最適パラメータを固定したモデル
- grid_search_*         : チューニングを再現したい場合のグリッド探索
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42


def build_baseline_models():
    """Phase 1: ロジスティック回帰 / 決定木 / ランダムフォレスト。"""
    return {
        "ロジスティック回帰": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "決定木": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "ランダムフォレスト": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }


def build_tuned_models():
    """Phase 2 の GridSearch で確定した最適構成 (CV 再現に使用)。"""
    return {
        "RF (tuned)": RandomForestClassifier(
            n_estimators=300, max_depth=None, max_features=0.5,
            min_samples_leaf=4, random_state=RANDOM_STATE),
        "XGBoost (tuned)": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, subsample=1.0,
            eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0),
        "LightGBM (tuned)": LGBMClassifier(
            n_estimators=400, num_leaves=15, learning_rate=0.03, subsample=0.8,
            random_state=RANDOM_STATE, verbose=-1),
        "SVM (tuned)": make_pipeline(
            StandardScaler(), SVC(C=1, gamma="scale", probability=True, random_state=RANDOM_STATE)),
    }


def build_ensembles(tuned):
    """チューニング済みモデルから構成するアンサンブル群。"""
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    et = ExtraTreesClassifier(n_estimators=400, random_state=RANDOM_STATE)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                    learning_rate=0.05, random_state=RANDOM_STATE)
    rf, xgb, lgb, svm = (tuned["RF (tuned)"], tuned["XGBoost (tuned)"],
                         tuned["LightGBM (tuned)"], tuned["SVM (tuned)"])
    return {
        "Voting(RF,XGB,LGB,SVM,LR)": VotingClassifier(
            [("rf", rf), ("xgb", xgb), ("lgb", lgb), ("svm", svm), ("lr", lr)],
            voting="soft", weights=[2, 2, 2, 1, 1]),
        "Voting(+ET,+GB)": VotingClassifier(
            [("rf", rf), ("xgb", xgb), ("lgb", lgb), ("et", et), ("gb", gb), ("svm", svm)],
            voting="soft", weights=[2, 2, 2, 1, 2, 1]),
        "Voting(LGB重視)": VotingClassifier(
            [("lgb", lgb), ("xgb", xgb), ("rf", rf)], voting="soft", weights=[3, 2, 1]),
    }


# --- グリッド探索 (チューニングを最初から再現したい場合) -----------------
def grid_search_rf(X, y, cv):
    g = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [300, 500], "max_depth": [5, 7, 9, None],
         "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5]},
        cv=cv, scoring="accuracy", n_jobs=-1)
    g.fit(X, y)
    return g


def grid_search_xgb(X, y, cv):
    g = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0),
        {"n_estimators": [200, 400], "max_depth": [3, 4, 5],
         "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.8, 1.0]},
        cv=cv, scoring="accuracy", n_jobs=-1)
    g.fit(X, y)
    return g
