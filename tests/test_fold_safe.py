import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from experiments.fold_safe import Features, pipeline


def fixture():
    return pd.DataFrame({'Pclass': [1,2,3,1,2,3], 'Sex': ['male','female']*3,
                         'Age': [20.,30.,np.nan,40.,50.,60.], 'SibSp': [0,1,0,0,1,0],
                         'Parch': [0]*6, 'Fare': [5.,10.,20.,30.,40.,np.nan],
                         'Embarked': ['S','C',np.nan,'S','C','S'],
                         'Name': ['A, Mr. B','C, Mrs. D']*3})


class FoldSafety(unittest.TestCase):
    def test_statistics_and_categories_exclude_validation(self):
        train = fixture()
        held = train.iloc[[2]].copy()
        held['Fare'] = 1000000.
        held['Age'] = np.nan
        held['Embarked'] = 'HELD_OUT_ONLY'
        p = pipeline('interactions', LogisticRegression()).fit(train, [0,1,0,1,0,1])
        f = p.named_steps['features']
        self.assertEqual(f.age_median_, 40.)
        np.testing.assert_allclose(f.fare_edges_, np.quantile(train.Fare.fillna(20), [.25,.5,.75]))
        before = f.fare_edges_.copy()
        np.testing.assert_allclose(f.transform(held).Age, [40.])
        self.assertTrue(np.isfinite(p.predict_proba(held)).all())
        enc = p.named_steps['preprocess'].named_transformers_['categorical'].named_steps['encode']
        self.assertFalse(any('HELD_OUT_ONLY' in c for c in enc.categories_))
        np.testing.assert_array_equal(before, f.fare_edges_)
        imp = p.named_steps['preprocess'].named_transformers_['categorical'].named_steps['impute']
        self.assertIn('S', imp.statistics_)

    def test_transform_is_independent_of_batch_distribution_and_order(self):
        train = fixture()
        f = Features('interactions').fit(train)
        one = train.iloc[[1]].copy()
        shifted = train.copy()
        shifted['Fare'] = 1e9
        shifted['Age'] = 1e9
        batch = pd.concat([one, shifted], ignore_index=True)
        pd.testing.assert_frame_equal(f.transform(one).reset_index(drop=True), f.transform(batch).iloc[[0]])
        pd.testing.assert_frame_equal(f.transform(train).sort_index(), f.transform(train.iloc[::-1]).sort_index())
        missing = one.copy()
        missing[['Fare','Age']] = np.nan
        self.assertEqual(f.transform(missing).Fare.iloc[0], 20.)
        self.assertEqual(f.transform(missing).Age.iloc[0], 40.)

    def test_clone_refits_on_training_fold_and_ignores_target(self):
        X = fixture()
        p = pipeline('fare', LogisticRegression())
        a = clone(p).fit(X.iloc[:4], [0,1,0,1])
        b = clone(p).fit(X.iloc[2:], [0,1,0,1])
        self.assertEqual(a.named_steps['features'].age_median_, 30.)
        self.assertEqual(b.named_steps['features'].age_median_, 50.)
        pd.testing.assert_frame_equal(Features('fare').fit(X, [0]*6).transform(X),
                                      Features('fare').fit(X, [1]*6).transform(X))
        self.assertFalse(hasattr(p.named_steps['features'], 'fare_edges_'))


class FailureRecording(unittest.TestCase):
    def test_all_failed_candidates_are_persisted(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from experiments.fold_safe import run
        train = pd.concat([fixture()] * 149, ignore_index=True).iloc[:891].copy()
        train['PassengerId'] = np.arange(891)
        train['Survived'] = np.arange(891) % 2
        test = train.iloc[:418].drop(columns='Survived').copy()
        test['PassengerId'] = np.arange(891, 1309)
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            train.to_csv(data/'train.csv', index=False)
            test.to_csv(data/'test.csv', index=False)
            with patch('experiments.fold_safe.candidates', return_value={
                    'deliberate_failure': pipeline('base', LogisticRegression(C=-1))}):
                with self.assertRaisesRegex(RuntimeError, 'All candidates failed'):
                    run(data, data/'out')
            failures = json.loads((data/'out/failures.json').read_text())
            self.assertEqual(failures[0]['candidate'], 'deliberate_failure')
            self.assertIn('Traceback', failures[0]['traceback'])
            self.assertEqual(failures[0]['stage'], 'inner')
            self.assertGreater(failures[0]['seconds'], 0)
            self.assertFalse(json.loads((data/'out/manifest.json').read_text())['complete'])


if __name__ == '__main__':
    unittest.main()
