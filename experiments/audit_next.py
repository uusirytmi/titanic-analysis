"""Recompute metrics and selection from saved OOF; check split/data provenance."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from experiments.fold_safe import digest
from experiments.next_round import SEEDS, group_ids, select_weights, WEIGHTS, specs_for


def audit(out, data):
    m=json.loads((out/'manifest.json').read_text())
    assert m['complete'] and m['failures']==0
    assert json.loads((out/'failures.json').read_text())==[]
    assert m['code_sha256']==digest(out/'runner.py')==digest(Path('experiments/next_round.py'))
    assert m['plan_sha256']==digest(Path('experiments/NEXT_PLAN.md'))
    assert m['base_code_sha256']==digest(Path('experiments/fold_safe.py'))
    assert m['splits_sha256']==digest(out/'splits.json')
    assert m['data_sha256']==digest(data/'train.csv')
    X=pd.read_csv(data/'train.csv'); y=X.Survived
    groups=np.array(json.loads((out/'groups.json').read_text()))
    np.testing.assert_array_equal(groups,group_ids(X.drop(columns='Survived')))
    splits=json.loads((out/'splits.json').read_text())
    scores=pd.read_csv(out/'fold_scores.csv')
    oof=pd.read_csv(out/'oof.csv')
    protocols=['stratified','group'] if m['branch']=='groups' else ['stratified']
    expected=list(specs_for(m['branch']))+(['equal_average','inner_weighted'] if m['branch']=='ensemble' else [])
    assert set(scores.candidate)==set(expected)
    assert len(scores)==len(protocols)*15*len(expected)
    assert len(oof)==len(protocols)*3*891*len(expected)
    assert m['fit_count']=={'seeds':60,'groups':120,'ensemble':180}[m['branch']]
    assert not oof.duplicated(['protocol','seed','candidate','row']).any()
    for protocol in protocols:
        for seed in SEEDS:
            partition=[]
            for s in [s for s in splits if s['protocol']==protocol and s['seed']==seed]:
                tr,va=set(s['train']),set(s['validation']);partition.extend(va)
                assert tr.isdisjoint(va) and tr|va==set(range(891))
                if protocol=='group':
                    assert set(groups[list(tr)]).isdisjoint(groups[list(va)])
                ins_va=[]
                for ins in s['inner']:
                    a,b=set(ins['train']),set(ins['validation'])
                    assert a.isdisjoint(b) and a|b==tr
                    ins_va.extend(b)
                assert sorted(ins_va)==sorted(tr)
                for name in expected:
                    p=oof[(oof.protocol==protocol)&(oof.seed==seed)&(oof.fold==s['fold'])&(oof.candidate==name)]
                    assert set(p.row)==va and p.probability.between(0,1).all()
                    r=scores[(scores.protocol==protocol)&(scores.seed==seed)&(scores.fold==s['fold'])&(scores.candidate==name)].iloc[0]
                    np.testing.assert_allclose(r.accuracy,accuracy_score(y.iloc[p.row],p.probability>=.5))
                    np.testing.assert_allclose(r.log_loss,log_loss(y.iloc[p.row],p.probability))
                    np.testing.assert_allclose(r.delta_accuracy,r.accuracy-r.baseline_accuracy)
            assert sorted(partition)==list(range(891))
    if m['branch']=='ensemble':
        inner=pd.read_csv(out/'inner_oof.csv')
        choices=json.loads((out/'selections.json').read_text())
        assert len(choices)==15
        assert not inner.duplicated(['seed','fold','candidate','row']).any()
        for s in [s for s in splits if s['protocol']=='stratified']:
            names=['hgb_reference','rf_family','lr_interactions']
            part=inner[(inner.seed==s['seed'])&(inner.fold==s['fold'])]
            for j,ins in enumerate(s['inner']):
                for name in names:
                    assert set(part[(part.inner_fold==j)&(part.candidate==name)].row)==set(ins['validation'])
            matrix=part.pivot(index='row',columns='candidate',values='probability')[names]
            assert set(matrix.index)==set(s['train'])
            infold=part.drop_duplicates('row').set_index('row').loc[matrix.index,'inner_fold'].to_numpy()
            choice,losses=select_weights(matrix.to_numpy(),y.iloc[matrix.index].to_numpy(),infold)
            saved=next(c for c in choices if c['seed']==s['seed'] and c['fold']==s['fold'])
            assert saved['choice']==choice and saved['weights']==WEIGHTS[choice]
            np.testing.assert_allclose(saved['inner_log_losses'],losses)
            outer=oof[(oof.seed==s['seed'])&(oof.fold==s['fold'])].pivot(index='row',columns='candidate',values='probability')
            np.testing.assert_allclose(outer.inner_weighted,outer[names].to_numpy()@WEIGHTS[choice])
            np.testing.assert_allclose(outer.equal_average,outer[names].to_numpy()@WEIGHTS[0])
    seed_scores=pd.read_csv(out/'seed_scores.csv')
    for row in seed_scores.itertuples():
        f=scores[(scores.protocol==row.protocol)&(scores.candidate==row.candidate)&(scores.seed==row.seed)]
        np.testing.assert_allclose([row.accuracy,row.delta,row.log_loss],[f.accuracy.mean(),f.delta_accuracy.mean(),f.log_loss.mean()])
    summary=pd.read_csv(out/'summary.csv')
    for row in summary.itertuples():
        f=scores[(scores.protocol==row.protocol)&(scores.candidate==row.candidate)]
        s=seed_scores[(seed_scores.protocol==row.protocol)&(seed_scores.candidate==row.candidate)]
        np.testing.assert_allclose([row.accuracy,row.fold_sd,row.delta,row.delta_fold_sd,row.seed_sd,row.delta_seed_sd,row.log_loss],
            [f.accuracy.mean(),f.accuracy.std(),f.delta_accuracy.mean(),f.delta_accuracy.std(),s.accuracy.std(),s.delta.std(),f.log_loss.mean()],atol=1e-14)
        assert row.worse_folds==(f.delta_accuracy<0).sum()
    print(f"PASS {m['branch']}: {len(scores)} scores, {len(oof)} OOF rows, disjoint splits, group isolation, hashes, metrics, selection")


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);parser.add_argument('--data',type=Path,default=Path('data'))
    args=parser.parse_args();audit(args.out,args.data)
