import unittest
import pandas as pd
from experiments.ten_candidates import exceeds,primary,candidates,SEEDS


class StopRules(unittest.TestCase):
    def test_threshold_is_strict_and_unrounded(self):
        self.assertFalse(exceeds(.844))
        self.assertFalse(exceeds(.843999999))
        self.assertTrue(exceeds(.844000001))
    def test_all_fifteen_folds_required_and_seeds_equal_weighted(self):
        frame=pd.DataFrame([dict(seed=s,fold=f,accuracy=a) for s,a in zip(SEEDS,[.8,.84,.88]) for f in range(5)])
        self.assertAlmostEqual(primary(frame),.84)
        with self.assertRaises(AssertionError):primary(frame.iloc[:-1])
        with self.assertRaises(AssertionError):primary(pd.concat([frame.iloc[:-1],frame.iloc[[0]]]))
    def test_ten_fixed_candidates_no_validation_tuning(self):
        specs=candidates()
        self.assertEqual(len(specs),10)
        self.assertEqual(len({name for name,_ in specs}),10)
        self.assertEqual(specs[-1][1].weights,[2,2,1])


class FailureRecording(unittest.TestCase):
    def test_summary_failure_is_persisted_once(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from experiments.ten_candidates import guarded_run
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            (out/'status.json').write_text(json.dumps(dict(state='running',active_trial=1,technical_failures=0)))
            with patch('experiments.ten_candidates.OUT',out), patch('experiments.ten_candidates.run',side_effect=ValueError('synthetic summary failure')):
                with self.assertRaises(ValueError):guarded_run(out)
                status=json.loads((out/'status.json').read_text())
                self.assertEqual(status['state'],'technical_failure')
                self.assertEqual(status['technical_failures'],1)
                self.assertIn('synthetic summary failure',(out/'execution_failure.json').read_text())
                with self.assertRaises(ValueError):guarded_run(out)
                self.assertEqual(json.loads((out/'status.json').read_text())['technical_failures'],1)


    def test_completed_result_is_not_overwritten_on_accidental_rerun(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from experiments.ten_candidates import guarded_run
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);status=dict(state='exhausted_without_target',completed_trials=10,technical_failures=0)
            (out/'status.json').write_text(json.dumps(status))
            with patch('experiments.ten_candidates.OUT',out),patch('experiments.ten_candidates.run',side_effect=AssertionError('already complete')):
                with self.assertRaises(AssertionError):guarded_run(out)
            self.assertEqual(json.loads((out/'status.json').read_text()),status)
            self.assertFalse((out/'execution_failure.json').exists())


if __name__=='__main__':unittest.main()
