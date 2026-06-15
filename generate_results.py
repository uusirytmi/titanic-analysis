# -*- coding: utf-8 -*-
"""src モジュールを用いて results/ 配下の成果物を生成する。
  - results/accuracy_comparison.csv : Phase1/Phase2 精度比較表
  - results/feature_importance.png  : RF(v2) 特徴量重要度
  - results/training_log.txt        : 全プロセスのテキストログ
同時に data/submission.csv を最終モデルで再生成する。
"""
import warnings; warnings.filterwarnings("ignore")
import io, sys, os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from feature_engineering import (build_features_v1, build_features_v2,
                                  V1_FEATURES, feature_columns)
from model_training import build_baseline_models, build_tuned_models, build_ensembles
from evaluation import (make_cv, evaluate_models, cv_accuracy,
                        plot_feature_importance, make_submission)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")

buf = io.StringIO()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line); buf.write(line + "\n")

train_raw = pd.read_csv(os.path.join(DATA, "train.csv"))
test_raw = pd.read_csv(os.path.join(DATA, "test.csv"))
base = train_raw
y = train_raw["Survived"]
# 全フェーズで同一の CV (層化5-fold, シャッフル) に統一し再現性を担保する
cv = make_cv(5)

log("=" * 70)
log("Titanic Survival Prediction — results 生成ログ")
log("=" * 70)
log(f"train={train_raw.shape}, test={test_raw.shape}, 全体生存率={y.mean():.3f}")

# ---- Phase 1 ----
train_v1 = build_features_v1(train_raw, base)
X1 = train_v1[V1_FEATURES]
res_v1 = evaluate_models(build_baseline_models(), X1, y, cv)
log("\n[Phase 1] ベースライン 3 モデル (5-fold CV)")
log(res_v1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---- Phase 2 (tuned + ensemble) ----
train_v2 = build_features_v2(train_raw, base)
test_v2 = build_features_v2(test_raw, base)
feats2 = feature_columns(train_v2)
X2 = train_v2[feats2]

# Phase 1 の到達点: v2 手設計特徴 × デフォルトモデル (公称 0.8204 の出所)
res_v2_default = evaluate_models(build_baseline_models(), X2, y, cv)
log("\n[Phase 1.5] v2手設計特徴 × デフォルト3モデル (5-fold CV)")
log(res_v2_default.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

tuned = build_tuned_models()
res_tuned = evaluate_models(tuned, X2, y, cv)
log("\n[Phase 2-a] チューニング済みモデル (5-fold CV)")
log(res_tuned.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

res_ens = evaluate_models(build_ensembles(tuned), X2, y, cv)
log("\n[Phase 2-b] アンサンブル (5-fold CV)")
log(res_ens.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---- 精度比較表 (統合) ----
def tag(df, phase):
    d = df.rename(columns={"モデル": "model", "CV平均Accuracy": "cv_accuracy", "標準偏差": "std"}).copy()
    d["phase"] = phase
    return d[["phase", "model", "cv_accuracy", "std"]] if "std" in d else d

comp = pd.concat([
    tag(res_v1, "Phase1 (v1 baseline)"),
    tag(res_v2_default, "Phase1 (v2 human FE)"),
    tag(res_tuned, "Phase2 (tuned)"),
    tag(res_ens.assign(**{"標準偏差": float("nan")}), "Phase2 (ensemble)"),
], ignore_index=True).sort_values("cv_accuracy", ascending=False).reset_index(drop=True)
comp.to_csv(os.path.join(RESULTS, "accuracy_comparison.csv"), index=False)
log("\n[精度比較表] results/accuracy_comparison.csv に保存")
log(comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

best_row = comp.iloc[0]
log(f"\n>>> 最高精度: {best_row['model']} ({best_row['phase']}) = {best_row['cv_accuracy']:.4f}")

# ---- 特徴量重要度プロット (RF tuned, v2) ----
rf = tuned["RF (tuned)"].fit(X2, y)
imp = plot_feature_importance(rf, feats2, os.path.join(RESULTS, "feature_importance.png"),
                              title="Feature Importance (RandomForest tuned, v2)")
log("\n[特徴量重要度] results/feature_importance.png に保存")
log(imp.to_string(float_format=lambda x: f"{x:.4f}"))

# ---- 最終 submission 再生成 (RF tuned) ----
make_submission(tuned["RF (tuned)"], X2, y, test_v2[feats2],
                test_raw["PassengerId"], os.path.join(DATA, "submission.csv"))
log("\n[提出] data/submission.csv を RF(tuned) で生成")

# ---- 生存要因サマリ ----
log("\n[生存要因] 属性別 生存率")
log("  性別: " + train_raw.groupby("Sex")["Survived"].mean().round(3).to_dict().__str__())
log("  クラス: " + train_raw.groupby("Pclass")["Survived"].mean().round(3).to_dict().__str__())

with open(os.path.join(RESULTS, "training_log.txt"), "w") as f:
    f.write(buf.getvalue())
log("\nresults/training_log.txt に全ログを保存。完了。")
