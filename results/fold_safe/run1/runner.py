"""Bounded, nested Titanic benchmark; run python -m experiments.fold_safe."""
import argparse
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

SEED = 20261012
GROUPS = ['base', 'title', 'family', 'fare', 'interactions']
BASE = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


class Features(TransformerMixin, BaseEstimator):
    def __init__(self, group='base'):
        self.group = group

    def fit(self, X, y=None):
        if self.group not in GROUPS:
            raise ValueError(self.group)
        self.age_median_ = X.Age.median()
        self.fare_median_ = X.Fare.median()
        self.fare_edges_ = np.unique(np.quantile(X.Fare.fillna(self.fare_median_), [.25, .5, .75]))
        return self

    def transform(self, X):
        check_is_fitted(self, 'fare_edges_')
        z = X[BASE].copy()
        # These statistics come exclusively from fit's training rows.
        z['Age'] = z.Age.fillna(self.age_median_)
        z['Fare'] = z.Fare.fillna(self.fare_median_)
        level = GROUPS.index(self.group)
        if level >= 1:
            title = X.Name.str.extract(r',\s*([^.]*)\.', expand=False).fillna('Unknown')
            z['Title'] = title.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        if level >= 2:
            z['FamilySize'] = X.SibSp + X.Parch + 1
            z['IsAlone'] = (z.FamilySize == 1).astype(float)
        if level >= 3:
            z['LogFare'] = np.log1p(z.Fare.clip(lower=0))
            z['FarePerPerson'] = z.Fare / z.FamilySize
            z['FareBin'] = np.searchsorted(self.fare_edges_, z.Fare, side='right').astype(str)
        if level >= 4:
            z['SexClass'] = z.Sex.astype(str) + '_' + z.Pclass.astype(str)
            z['TitleClass'] = z.Title + '_' + z.Pclass.astype(str)
            z['AgeClass'] = z.Age * z.Pclass
            z['IsChild'] = (z.Age < 16).astype(float)
        return z


def pipeline(group, model):
    prep = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), make_column_selector(dtype_include=np.number)),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
         make_column_selector(dtype_exclude=np.number)),
    ])
    return Pipeline([('features', Features(group)), ('preprocess', prep), ('model', model)])


def candidates():
    models = {
        'lr': LogisticRegression(C=1, max_iter=2000, random_state=SEED),
        'rf': RandomForestClassifier(n_estimators=120, min_samples_leaf=3,
                                     max_features='sqrt', n_jobs=1, random_state=SEED),
        'hgb': HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                             min_samples_leaf=20, learning_rate=.08,
                                             l2_regularization=1, early_stopping=False, random_state=SEED),
    }
    return {f'{name}_{group}': pipeline(group, clone(model))
            for group in GROUPS for name, model in models.items()}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(data, out):
    out.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(data / 'train.csv')
    test = pd.read_csv(data / 'test.csv')
    assert len(train) == 891 and len(test) == 418
    assert train.PassengerId.is_unique and test.PassengerId.is_unique
    assert set(train.Survived) == {0, 1}
    assert not set(train.PassengerId) & set(test.PassengerId)
    X, y = train.drop(columns='Survived'), train.Survived
    specs = candidates()
    outer = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    splits = []
    for fold, (tr, va) in enumerate(outer):
        inner = StratifiedKFold(3, shuffle=True, random_state=SEED + 1 + fold)
        splits.append({'fold': fold, 'train': tr.tolist(), 'validation': va.tolist(),
                       'inner': [{'train': tr[a].tolist(), 'validation': tr[b].tolist()}
                                 for a, b in inner.split(X.iloc[tr], y.iloc[tr])]})
    (out / 'splits.json').write_text(json.dumps(splits))
    manifest = {'seed': SEED, 'python': platform.python_version(),
                'versions': {m.__name__: m.__version__ for m in [np, pd, scipy, sklearn]},
                'data_sha256': {p.name: digest(p) for p in [data/'train.csv', data/'test.csv']},
                'source': 'https://github.com/agconti/kaggle-titanic/tree/master/data',
                'code_sha256': digest(__file__), 'plan_sha256': digest(Path(__file__).with_name('PLAN.md')),
                'parameters': {k: {p: str(v) for p, v in s.get_params().items()} for k, s in specs.items()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows, inner_rows, oof, failures = [], [], [], []
    start = time.perf_counter()
    for split in splits:
        fold, tr, va = split['fold'], split['train'], split['validation']
        fold_start = time.perf_counter()
        valid = {}
        predictions = {}
        for name, spec in specs.items():
            t = time.perf_counter()
            try:
                scores = []
                for j, ins in enumerate(split['inner']):
                    ti = time.perf_counter()
                    fitted = clone(spec).fit(X.iloc[ins['train']], y.iloc[ins['train']])
                    score = accuracy_score(y.iloc[ins['validation']], fitted.predict_proba(X.iloc[ins['validation']])[:, 1] >= .5)
                    scores.append(score)
                    inner_rows.append(dict(fold=fold, inner_fold=j, candidate=name, accuracy=score,
                                           seconds=time.perf_counter()-ti))
                valid[name] = np.mean(scores)
                fitted = clone(spec).fit(X.iloc[tr], y.iloc[tr])
                prob = fitted.predict_proba(X.iloc[va])[:, 1]
                # Transform test with the SAME fitted preprocessing, never fit on it.
                assert np.isfinite(fitted.predict_proba(test)).all()
                predictions[name] = prob
                rows.append(dict(fold=fold, candidate=name, accuracy=accuracy_score(y.iloc[va], prob >= .5),
                                 log_loss=log_loss(y.iloc[va], prob), seconds=time.perf_counter()-t))
                oof.extend(dict(row=i, fold=fold, candidate=name, probability=float(p)) for i, p in zip(va, prob))
            except Exception:
                valid.pop(name, None)
                failures.append(dict(fold=fold, candidate=name, traceback=traceback.format_exc()))
            print(f'fold={fold} {name} done', flush=True)
        if not valid:
            raise RuntimeError('All candidates failed')
        selected = max(valid, key=valid.get)
        prob = predictions[selected]
        rows.append(dict(fold=fold, candidate='nested_selection', selected=selected,
                         accuracy=accuracy_score(y.iloc[va], prob >= .5), log_loss=log_loss(y.iloc[va], prob),
                         seconds=time.perf_counter()-fold_start))
        oof.extend(dict(row=i, fold=fold, candidate='nested_selection', probability=float(p)) for i,p in zip(va,prob))
        dummy = DummyClassifier(strategy='prior').fit(np.zeros((len(tr), 1)), y.iloc[tr])
        prob = dummy.predict_proba(np.zeros((len(va), 1)))[:, 1]
        rows.append(dict(fold=fold, candidate='dummy', accuracy=accuracy_score(y.iloc[va], prob >= .5),
                         log_loss=log_loss(y.iloc[va], prob), seconds=0))
        oof.extend(dict(row=i, fold=fold, candidate='dummy', probability=float(p)) for i,p in zip(va,prob))
        pd.DataFrame(rows).to_csv(out/'fold_scores.csv', index=False)
        pd.DataFrame(inner_rows).to_csv(out/'inner_scores.csv', index=False)
        pd.DataFrame(oof).to_csv(out/'oof.csv', index=False)
        (out/'failures.json').write_text(json.dumps(failures, indent=2))
    scores = pd.DataFrame(rows)
    summary = scores.groupby('candidate').agg(mean_accuracy=('accuracy','mean'), std_accuracy=('accuracy','std'),
                                             mean_log_loss=('log_loss','mean'), folds=('fold','count'))
    summary.sort_values('mean_accuracy', ascending=False).to_csv(out/'summary.csv')
    manifest.update(total_seconds=time.perf_counter()-start, failures=len(failures), complete=not failures)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(), flush=True)
    if failures:
        raise RuntimeError('Incomplete comparison: see failures.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('results/fold_safe/run1'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data, args.out)
