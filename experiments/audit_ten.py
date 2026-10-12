"""Audit the executed snapshot and all saved OOF against local labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from experiments.fold_safe import digest


def audit(data=Path('data')):
    out=Path('results/ten_candidates')
    plan=json.loads((out/'plan.json').read_text());status=json.loads((out/'status.json').read_text())
    assert digest(out/'runner.py')==plan['code_sha256']
    assert digest(Path('experiments/fold_safe.py'))==plan['preprocessing_sha256']
    assert digest(data/'train.csv')==plan['data_sha256']
    assert digest(out/'splits.json')==plan['splits_sha256']
    assert digest(out/'splits.json')==digest(Path('results/next_round/seeds/splits.json'))
    assert plan['seeds']==[20261012,20261029,20261113] and plan['threshold']==.844
    assert len(plan['candidate_order'])==10 and plan['maximum_trials']==10
    y=pd.read_csv(data/'train.csv').Survived
    splits=[s for s in json.loads((out/'splits.json').read_text()) if s['protocol']=='stratified']
    summary=pd.read_csv(out/'summary.csv');n=len(summary)
    assert list(summary.trial)==list(range(1,n+1)) and 1<=n<=10
    assert status['completed_trials']==n and status['technical_failures']==0
    for row in summary.itertuples():
        planned=plan['candidate_order'][row.trial-1]
        assert row.candidate==planned['name']
        directory=out/f'{row.trial:02d}_{row.candidate}'
        oof=pd.read_csv(directory/'oof.csv');scores=pd.read_csv(directory/'fold_scores.csv')
        assert len(oof)==3*891 and len(scores)==15
        assert not oof.duplicated(['seed','row']).any()
        assert oof.probability.between(0,1).all()
        recomputed=[]
        for s in splits:
            tr,va=set(s['train']),set(s['validation'])
            assert tr.isdisjoint(va) and tr|va==set(range(891))
            p=oof[(oof.seed==s['seed'])&(oof.fold==s['fold'])]
            assert set(p.row)==va
            score=accuracy_score(y.iloc[p.row],p.probability>=.5)
            ll=log_loss(y.iloc[p.row],p.probability)
            saved=scores[(scores.seed==s['seed'])&(scores.fold==s['fold'])].iloc[0]
            np.testing.assert_allclose([saved.accuracy,saved.log_loss],[score,ll],atol=1e-14)
            recomputed.append(dict(seed=s['seed'],fold=s['fold'],accuracy=score))
        recalculated=pd.DataFrame(recomputed)
        means=recalculated.groupby('seed').accuracy.mean()
        primary=float(means.mean())
        np.testing.assert_allclose([row.primary_accuracy,row.fold_sd,row.seed_sd,row.seed_min,row.seed_max],
            [primary,recalculated.accuracy.std(),means.std(),means.min(),means.max()],atol=1e-14)
        assert bool(row.exceeded)==(primary>.844)
        if row.trial<n:assert not primary>.844
        saved_seed=pd.read_csv(directory/'seed_scores.csv').set_index('seed')
        np.testing.assert_allclose(saved_seed.accuracy,means)
        baseline=pd.read_csv('results/next_round/seeds/fold_scores.csv').query("candidate=='hgb_reference'")
        pairs=recalculated.merge(baseline[['seed','fold','accuracy']],on=['seed','fold'],suffixes=('','_baseline'))
        d=pairs.accuracy-pairs.accuracy_baseline
        np.testing.assert_allclose(row.delta_baseline,d.mean(),atol=1e-14)
        assert row.worse_folds==int((d<0).sum())
    assert (n==10 and not summary.exceeded.any()) or bool(summary.iloc[-1].exceeded)
    assert status['state'] in ['awaiting_review','reviewed_target_reached','exhausted_without_target']
    print(f'PASS: {n} consecutive candidates, {15*n} folds, {2673*n} OOF records; hashes, all metrics and strict stopping rule verified.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data'));args=p.parse_args();audit(args.data)
