# -*- coding: utf-8 -*-
"""特徴量エンジニアリング.

2 つの設計思想を関数として提供する:
  - build_features_v1 : Phase 1 (人間がドメイン知識で設計した最小構成)
  - build_features_v2 : Phase 2 (相互作用項・敬称・Deck 等を加えた拡張構成)

補填(欠損処理)の基準値は必ず train から計算し test に適用する
(data leakage を避けるため)。
"""
import pandas as pd

TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Mme": "Mrs", "Ms": "Miss", "Mlle": "Miss",
}


def _extract_title(name_series):
    raw = name_series.str.extract(r",\s*([^\.]+)\.")[0]
    return raw.map(lambda t: TITLE_MAP.get(t, "Rare"))


# ---------------------------------------------------------------------------
# Phase 1: ベースライン特徴量 (指示通りの整形)
# ---------------------------------------------------------------------------
def build_features_v1(df, base):
    """Pclass 別 Age 中央値補填 / cabin_missing / Embarked 最頻値 / family_size。
    不要列 (PassengerId 以外: Ticket, Name, Fare, Cabin) を削除。
    """
    df = df.copy()

    age_median = base.groupby("Pclass")["Age"].median()
    df["Age"] = df.apply(
        lambda r: age_median[r["Pclass"]] if pd.isnull(r["Age"]) else r["Age"], axis=1)

    df["cabin_missing"] = df["Cabin"].isnull().astype(int)
    df = df.drop(columns=["Cabin"])

    df["Embarked"] = df["Embarked"].fillna(base["Embarked"].mode()[0])
    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    df = df.drop(columns=["Ticket", "Name", "Fare"])
    return df


V1_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked",
               "cabin_missing", "family_size"]


# ---------------------------------------------------------------------------
# Phase 2: 拡張特徴量 (Fare 復活 + Title + Deck + 相互作用項 ...)
# ---------------------------------------------------------------------------
def build_features_v2(df, base):
    """Phase 1 に経済力・社会的地位・相互作用項を追加した拡張構成。"""
    df = df.copy()
    df["Title"] = _extract_title(df["Name"])

    # Age: Title x Pclass 中央値で補填 (Pclass 単独より精緻)
    grp = base.copy()
    grp["Title"] = _extract_title(grp["Name"])
    age_med = grp.groupby(["Title", "Pclass"])["Age"].median()
    global_med = grp["Age"].median()
    df["Age"] = df.apply(
        lambda r: r["Age"] if pd.notnull(r["Age"])
        else age_med.get((r["Title"], r["Pclass"]), global_med), axis=1)

    df["cabin_missing"] = df["Cabin"].isnull().astype(int)
    df["Deck"] = df["Cabin"].astype(str).str[0].where(df["Cabin"].notnull(), "U")
    df = df.drop(columns=["Cabin"])

    df["Embarked"] = df["Embarked"].fillna(base["Embarked"].mode()[0])
    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["family_size"] == 1).astype(int)

    fare_med = base.groupby("Pclass")["Fare"].median()
    df["Fare"] = df.apply(
        lambda r: fare_med[r["Pclass"]] if pd.isnull(r["Fare"]) else r["Fare"], axis=1)
    df["Fare_per_person"] = df["Fare"] / df["family_size"]

    tc = base["Ticket"].value_counts()
    df["TicketGroup"] = df["Ticket"].map(tc).fillna(1)
    df["FareBin"] = pd.qcut(df["Fare"], 5, labels=False, duplicates="drop")
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100], labels=False)

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df["Title"] = df["Title"].map({"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4})
    deck_map = {d: i for i, d in enumerate(["U", "A", "B", "C", "D", "E", "F", "G", "T"])}
    df["Deck"] = df["Deck"].map(deck_map).fillna(0).astype(int)

    # 相互作用項: 「上流階級の女性/子供」を明示的に表現
    df["Sex_x_Pclass"] = df["Sex"] * df["Pclass"]
    df["Age_x_Pclass"] = df["Age"] * df["Pclass"]
    df["Title_x_Pclass"] = df["Title"] * df["Pclass"]
    df["FamilySurvival"] = df["family_size"] * df["Sex"]

    df = df.drop(columns=["Ticket", "Name"])
    return df


# v2 のうち「コア」となる手設計特徴 (Phase1 結果分析で採用したもの)
V2_CORE_FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title",
    "cabin_missing", "family_size", "IsAlone",
    "Sex_x_Pclass", "Age_x_Pclass", "Fare_per_person",
]


def feature_columns(df):
    """学習に使う特徴量列 (目的変数と ID を除く)。"""
    return [c for c in df.columns if c not in ("Survived", "PassengerId")]
