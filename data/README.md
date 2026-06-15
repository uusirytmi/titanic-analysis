# data/

Kaggle [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data) のデータを配置する。

`train.csv` と `test.csv` は Kaggle の規約に従い本リポジトリには含めていない（`.gitignore` 対象）。
各自で以下からダウンロードして本ディレクトリに置くこと:

```bash
kaggle competitions download -c titanic
unzip titanic.zip -d data/
```

- `train.csv` … 学習データ（891 行、Survived ラベル付き）
- `test.csv`  … 予測対象（418 行）
- `submission.csv` … 本プロジェクトの最終予測（RandomForest tuned, CV 0.8440）※リポジトリに含む
- `gender_submission.csv` … Kaggle 提供のサンプル提出形式
