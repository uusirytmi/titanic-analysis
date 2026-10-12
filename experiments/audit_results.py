"""Audit saved splits, nested choices, hashes and metrics against local labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from experiments.fold_safe import digest


def audit(data=Path('data'), out=Path('results/fold_safe/run1')):
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['complete'] and manifest['failures'] == 0
    assert json.loads((out/'failures.json').read_text()) == []
    assert digest(out/'runner.py') == manifest['code_sha256']
    assert digest(Path('experiments/PLAN.md')) == manifest['plan_sha256']
    for name, sha in manifest['data_sha256'].items():
        assert digest(data/name) == sha
    splits = json.loads((out/'splits.json').read_text())
    folds = pd.read_csv(out/'fold_scores.csv')
    inner = pd.read_csv(out/'inner_scores.csv')
    oof = pd.read_csv(out/'oof.csv')
    y = pd.read_csv(data/'train.csv').Survived
    assert len(inner) == 225 and len(folds) == 85 and len(oof) == 17 * 891
    names = list(manifest['parameters'])
    all_valid = []
    for split in splits:
        fold, tr, va = split['fold'], set(split['train']), set(split['validation'])
        assert tr.isdisjoint(va) and tr | va == set(range(891))
        all_valid.extend(va)
        inner_valid = []
        for ins in split['inner']:
            a, b = set(ins['train']), set(ins['validation'])
            assert a.isdisjoint(b) and a | b == tr
            inner_valid.extend(b)
        assert sorted(inner_valid) == sorted(tr)
        scores = inner[inner.fold == fold].groupby('candidate').accuracy.mean()
        selected = max(names, key=lambda name: scores[name])
        row = folds[(folds.fold == fold) & (folds.candidate == 'nested_selection')].iloc[0]
        assert row.selected == selected
        nested = oof[(oof.fold == fold) & (oof.candidate == 'nested_selection')].sort_values('row')
        fixed = oof[(oof.fold == fold) & (oof.candidate == selected)].sort_values('row')
        np.testing.assert_array_equal(nested.probability, fixed.probability)
        for name in oof.candidate.unique():
            p = oof[(oof.fold == fold) & (oof.candidate == name)]
            assert set(p.row) == va
            row = folds[(folds.fold == fold) & (folds.candidate == name)].iloc[0]
            np.testing.assert_allclose(row.accuracy, accuracy_score(y.iloc[p.row], p.probability >= .5))
            np.testing.assert_allclose(row.log_loss, log_loss(y.iloc[p.row], p.probability))
    assert sorted(all_valid) == list(range(891))
    for _, p in oof.groupby('candidate'):
        assert sorted(p.row) == list(range(891))
        assert p.probability.between(0, 1).all()
    summary = pd.read_csv(out/'summary.csv').set_index('candidate')
    for name, f in folds.groupby('candidate'):
        np.testing.assert_allclose(summary.loc[name, 'mean_accuracy'], f.accuracy.mean())
        np.testing.assert_allclose(summary.loc[name, 'std_accuracy'], f.accuracy.std())
    print('PASS: hashes, 5 outer/15 inner splits, 225 inner scores, 85 fold scores,')
    print('17 x 891 OOF records, nested choices/probabilities and all saved metrics.')


if __name__ == '__main__':
    audit()
