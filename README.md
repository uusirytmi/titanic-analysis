# Titanic Survival Prediction

Titanic 教材を使い、モデルと特徴量を比較する実験リポジトリです。
**前回の比較では、固定候補の外側 CV 最高値は 0.8351、内側で候補を選ぶ手順の外側評価は 0.8205 でした。Kaggle 採点は未確認です。**

## 追加比較: ensemble（2026-10-12）

この枝は前回 commit `1d0eef8` から分岐した3仮説のうち `ensemble` を実行したものです。
[共通計画](experiments/NEXT_PLAN.md)、[3枝の比較と解釈](results/next_round/REPORT.md)、
[比較CSV](results/next_round/comparison.csv)、[この枝の実測結果](results/next_round/ensemble/summary.csv)を参照してください。
通常の層化CVでは共通HGBが平均0.8369で追加候補は上回らず、別目的のグループCVではRF家族が0.8332（同条件HGB比+0.0135）でした。
過去OOFを踏まえた探索であり、独立データでの確証ではありません。

再現（データと依存関係の準備は下記参照）:

```bash
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.next_round ensemble --out results/next_round/new_run
python -m experiments.audit_next results/next_round/ensemble
```

以下は共通出発点の前回実験の記録です。

## 2026-10-12: fold 内前処理による比較

[実験計画](experiments/PLAN.md)を実行前に固定しました。891 行のラベル付きデータに層化外側 5-fold、各学習部分に内側 3-fold を用い、15 候補を同じ分割で比較しました。探索は内側の平均 Accuracy、評価は未使用の外側 fold で行います。乱数 seed は 20261012（内側は 20261013 + fold 番号）。

欠損補完、運賃の四分位境界、カテゴリ符号化、標準化を sklearn Pipeline 内で学習します。検証行や test の分布で基準値を作り直しません。氏名から敬称のみを抽出し、氏名・チケット ID・目的変数の集計は使いません。特徴量は累積追加、モデル設定は固定し、結果を見た追加探索はしていません。

数値は外側 5-fold の平均 Accuracy。RF は120本、HGB は100反復です。

| 特徴量群（累積） | Logistic regression | Random forest | Histogram gradient boosting |
|---|---:|---:|---:|
| 基本 | 0.8047 | 0.8171 | 0.8283 |
| + 敬称 | 0.8250 | 0.8317 | 0.8261 |
| + 家族人数・単独 | 0.8249 | 0.8339 | 0.8305 |
| + 対数運賃・家族当たり運賃・運賃区分 | 0.8182 | 0.8283 | 0.8340 |
| + 性別×階級・敬称×階級・年齢×階級・子供 | 0.8283 | 0.8261 | **0.8351** |

- **内側選択手順の外側評価: 0.8205 ± 0.0303**（fold間標本標準偏差）、平均 log loss 0.4173。
- 固定候補の探索的最高値: HGB＋全特徴量 **0.8351 ± 0.0380**、平均 log loss 0.4281。
- 多数派予測: 0.6162。HGBの基本特徴量から全特徴量への差は +0.0068、RFの基本から家族特徴量までの差は +0.0168。特徴量の追加で悪化する例もありました。
- 計305 fit（内側225、固定候補の外側75、dummy 5）、約40.5秒、単一スレッド、失敗0件。時間は実行環境依存です。

最高値 0.8351 は外側の結果を見た順位であり、それを採用した後の不偏な最終性能ではありません。候補選択を含めた評価は 0.8205 です。foldごとの選択は RF全特徴量 / RF全特徴量 / LR全特徴量 / HGB敬称 / RF運賃となりました。小標本かつfoldは相関するため、標準偏差を信頼区間と解釈せず、有意な改善とは主張しません。同一家族が異なるfoldに入る可能性があり、未知の家族への一般化は未検証です。次の試行には新たな評価設計が必要です。

## 保存した証拠

- [summary.csv](results/fold_safe/run1/summary.csv): 全候補の平均・標準偏差・log loss
- [fold_scores.csv](results/fold_safe/run1/fold_scores.csv): fold別成績、選択候補、時間
- [inner_scores.csv](results/fold_safe/run1/inner_scores.csv): 探索用225件の成績・時間
- [oof.csv](results/fold_safe/run1/oof.csv): 行位置とOOF確率（乗客の氏名等は含めない）
- [splits.json](results/fold_safe/run1/splits.json): 外側・内側の固定行位置
- [manifest.json](results/fold_safe/run1/manifest.json): データ・コード・計画ハッシュ、バージョン、全設定、総時間
- [failures.json](results/fold_safe/run1/failures.json): 失敗記録（今回は空）
- [実行ログ](results/fold_safe/run1.log)、[実行時コード](results/fold_safe/run1/runner.py)

`fold_scores.seconds` は固定候補では内側3fit＋外側fit＋test変換の経過秒数、nested_selectionではfold全体の経過秒数です。dummyの時間は未計測（0）。独立レビュー後、実験本体の予測ロジックは変えず、全候補失敗時のログ保存・失敗段階と時間・コード自動保存を補強しました。run1のハッシュは保存した実行時コードに対応します。

## 再現・検査

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r experiments/requirements.txt
# Kaggle Titanic の train.csv / test.csv を data/ に配置
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.fold_safe --out results/fold_safe/new_run
# 保存済みrun1と同じデータで整合性を検査
python -m experiments.audit_results
```

今回のローカル入力は[公開ミラー](https://github.com/agconti/kaggle-titanic/tree/master/data)から取得し、SHA256をmanifestに記録しました。元CSVはgitignore対象のままです。新しい実行は既存出力先を上書きしません。テストはfold外の極端値・未知カテゴリが学習済み統計に混入しないこと、同じ行の変換が同時入力の分布や順序に依存しないこと、cloneが各foldで再学習すること、全候補失敗時の記録を確認します。監査は全分割の非重複・包含、17方式×891行のOOF、選択規則、成績の再計算とハッシュを検査します。

前回run1では test.csv の変換・予測が有限値になることだけを確認し、別seed反復・家族単位分割は未実施でした。今回の追加比較では3seed反復と家族・同行者のグループ分割を実施しました。提出ファイルの新規作成・Kaggle提出・採点、外部検証は今回も行っていません。

## 過去の実験（評価条件が異なる参考記録）

`notebooks/`、`src/`、`generate_results.py`、旧 `results/accuracy_comparison.csv` は過去の教材として残しています。旧スクリプトは全trainで前処理してからCVを行い、運賃のqcut境界もtrain/testで不一致になりえます。設定探索と評価の再利用もあるため、新規実験には上記のfold内Pipelineを使ってください。

保存CSVのRFは基本特徴量 **0.799071**、特徴量追加 **0.819277**、調整後 **0.843977**、XGBoostは **0.843971** でした。以前のREADMEの0.8204という説明は特徴量追加RFの保存値と一致していませんでした。0.844はKaggleスコアではなく旧CV値です。今回との直接の改善比較や「データの実力上限」の証明には使えません。`should_stop` は旧実行経路で未使用であり、自動停止が実行されたという以前の記述も取り下げます。旧 `data/submission.csv` は保存済み予測ファイルで、採点済みである証拠はありません。
