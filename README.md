# Titanic Survival Prediction

## 概要
RMS Titanic の生存者予測モデル。
データサイエンスにおける「人間の思考」と「自動化」の役割分担を実験。

> **問い**: ドメイン知識で特徴量を設計する「人間」と、ハイパラ探索・アンサンブルを回す「自動化」は、それぞれどこまで精度を伸ばせるのか？そして、その限界はどこにあるのか？

## 結果サマリー
- **Phase 1（人間が特徴量設計）**: CV Accuracy **0.8204**
- **Phase 2（自動最適化ループ）**: CV Accuracy **0.8440**
- **最終提出**: ランダムフォレスト **0.8440**

| フェーズ | 手法 | 5-fold CV Accuracy |
|---|---|---|
| Phase 1 (v1) | 基本整形 + LR/DT/RF | ~0.80 |
| Phase 1 (v2) | 人間によるドメイン特徴量設計 | **0.8204** |
| Phase 2 | GridSearch + XGBoost/LightGBM | **0.8440** |
| Phase 2 | アンサンブル (Voting/Stacking) | 0.836–0.841（改善せず）|

※ CV は層化 5-fold。`results/accuracy_comparison.csv` に全モデルの数値を記録。

## プロジェクト構成
```
titanic-analysis/
├── README.md
├── notebooks/      # 分析プロセスの完全記録
│   ├── 01_eda.ipynb            # 初期探索
│   ├── 02_v1_baseline.ipynb    # Phase 1: 基本分析 (0.8204)
│   └── 03_v2_optimization.ipynb# Phase 2: 自動最適化ループ (0.8440)
├── data/           # train/test/submission (train・test は .gitignore 対象)
├── src/            # 再利用可能なコード
│   ├── feature_engineering.py  # v1 / v2 特徴量生成
│   ├── model_training.py       # モデル定義・GridSearch・アンサンブル
│   └── evaluation.py           # CV評価・比較表・重要度プロット
├── results/        # 結果・図表
│   ├── accuracy_comparison.csv
│   ├── feature_importance.png
│   └── training_log.txt
└── generate_results.py # results 一式を再生成するスクリプト
```

## 実行方法
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # macOS で XGBoost/LightGBM を使う場合は `brew install libomp` も必要

# results 一式を再生成 (比較表・重要度図・ログ・submission)
python generate_results.py
```
ノートブックで分析プロセスを追う場合:
1. `notebooks/01_eda.ipynb` で初期探索
2. `notebooks/02_v1_baseline.ipynb` で基本分析
3. `notebooks/03_v2_optimization.ipynb` で最適化ループ

## 学んだこと

### Phase 1 の洞察（人間の思考）
- Titanic の史実（女性・子供・階級優先）をドメイン知識として活用
- Fare、Title、相互作用項など、有意義な特徴を設計
- 結果：0.80 → 0.8204

### Phase 2 の気づき（自動化の限界）
- RF、XGB が 0.844 で同値に収束
- これは「データの本質的限界」を示唆
- アンサンブル学習は多様性不足で改善できず
- 実務的には「汎化性能の堅実性」を優先すべき

---

## 補足: 今回の分析での革新部分

このプロジェクトの主眼は「スコアを 0.001 競う」ことではなく、**精度がどこで頭打ちになり、それが何を意味するのかを構造的に示す**ことにある。以下が特に新しい試み。

### 1. 「人間 vs 自動化」を同一 CV 上で対照実験にした
通常の Kaggle ノートは特徴量設計とチューニングが混ざりがち。本プロジェクトは
**Phase 1（人間のドメイン設計）と Phase 2（機械的探索）を明確に分離**し、同じ層化 5-fold CV で測定することで「どちらがどれだけ寄与したか」を切り分けた。
結果は **人間 +0.02（0.80→0.82）、自動化 +0.02（0.82→0.84）** と寄与がほぼ拮抗。

### 2. 「相互作用項」が単独特徴を上回った定量的証拠
特徴量重要度（`results/feature_importance.png`）で、手設計した **`Title_x_Pclass`（0.14）・`Sex_x_Pclass`・`Age_x_Pclass`** が
生の `Sex` や `Pclass` 単独を上回って上位を占めた。
「上流階級の女性・子供が生き残った」という史実が、**掛け算特徴という形で初めてモデルに伝わる**ことを示している。
線形・木モデルが暗黙に取りこぼす非線形効果を、人間の仮説で明示的に補ったのが効いた。

### 3. 停止条件つき自動最適化ループ（暴走しない自動化）
AutoML 的な探索を **3 つの停止条件**（3 連続改善なし / 最大 10 回 / 0.95 到達）で囲い、
「いつ止めるか」を設計に組み込んだ。実際は **3 連続改善なし** で正常終了し、
無駄な計算を避けつつ「これ以上伸びない」ことを能動的に検出できた。

### 4. アンサンブルの「失敗」を成果として記録
7 種の Voting/Stacking を試して**いずれも個別モデル（0.844）を超えられなかった**。
これは失敗ではなく **「構成モデルの予測が高相関＝多様性が枯渇している」という診断**であり、
複数の独立手法が同じ 0.844 に収束する事実とあわせて、**0.84 付近が Titanic データの説明限界**であることの傍証になった。

### 5. data leakage を構造的に防ぐ設計
全ての欠損補填（Age 中央値・Embarked 最頻値・Fare 中央値）と
カテゴリ集計（Ticket グループ数等）の**基準値を必ず train から計算して test に適用**する設計を
`src/feature_engineering.py` に固定。再利用時にもリークが混入しない。

> **結論**: 0.844 は「敗北」ではなく**実力上限の発見**である。
> モデルを増やすより、限界を正しく認識して堅実なモデルを選ぶことが、実務では正解になる。

## 今後の展開
- [ ] 他のコンペでこのフレームワークを適用
- [ ] ブログで思考プロセスを詳述
- [ ] 分析テンプレート化
- [ ] 姓ベースの家族生存グループ特徴（leakage 境界の検証）
