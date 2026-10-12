"""At most ten prespecified candidates, sequentially; no outer-label tuning."""
import argparse
import json
import platform
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
from threadpoolctl import threadpool_limits
from experiments.fold_safe import pipeline, digest

SEED=20261012
SEEDS=[20261012,20261029,20261113]
THRESHOLD=.844
OUT=Path('results/ten_candidates')
SPLITS=Path('results/next_round/seeds/splits.json')


def candidates():
    models=[
      ('gb_depth3',GradientBoostingClassifier(n_estimators=200,learning_rate=.04,max_depth=3,min_samples_leaf=5,random_state=SEED)),
      ('gb_depth2',GradientBoostingClassifier(n_estimators=300,learning_rate=.04,max_depth=2,min_samples_leaf=5,random_state=SEED)),
      ('gb_leaf10',GradientBoostingClassifier(n_estimators=300,learning_rate=.03,max_depth=3,min_samples_leaf=10,random_state=SEED)),
      ('rf_depth8',RandomForestClassifier(n_estimators=500,max_depth=8,min_samples_leaf=2,max_features=.8,n_jobs=1,random_state=SEED)),
      ('rf_leaf2',RandomForestClassifier(n_estimators=500,min_samples_leaf=2,max_features=.8,n_jobs=1,random_state=SEED)),
      ('extra_trees',ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,max_features=1.,n_jobs=1,random_state=SEED)),
      ('svc_c3',SVC(C=3,gamma=.03,probability=True,random_state=SEED)),
      ('svc_c10',SVC(C=10,gamma=.01,probability=True,random_state=SEED)),
      ('hgb_small_leaves',HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=15,l2_regularization=3,early_stopping=False,random_state=SEED)),
    ]
    specs=[(name,pipeline('interactions',model)) for name,model in models]
    specs.append(('fixed_vote',VotingClassifier(estimators=[('gb',clone(specs[0][1])),('rf',clone(specs[3][1])),('svc',clone(specs[6][1]))],voting='soft',weights=[2,2,1],n_jobs=1)))
    return specs


def exceeds(value):
    return float(value)>THRESHOLD


def primary(scores):
    assert len(scores)==15 and set(scores.seed)==set(SEEDS)
    assert not scores.duplicated(['seed','fold']).any()
    for seed in SEEDS:
        assert set(scores[scores.seed==seed].fold)==set(range(5))
    return float(scores.groupby('seed').accuracy.mean().mean())


def prepare(data):
    OUT.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,OUT/'runner.py')
    shutil.copyfile(SPLITS,OUT/'splits.json')
    planned=[dict(trial=i,name=name,parameters={k:str(v) for k,v in spec.get_params().items()})
             for i,(name,spec) in enumerate(candidates(),1)]
    plan=dict(definition='One candidate on all 5 folds x 3 fixed seeds; no cherry-picking.',
        parent='120bf665b47d273ffe5c46f8ff926f4298a1c7b4',seeds=SEEDS,threshold=THRESHOLD,
        decision='strict unrounded mean of 3 seed means > threshold; accuracy at probability >= .5',
        candidate_order=planned,maximum_trials=10,maximum_pipeline_fits=180,
        tuning='None. All settings and ensemble weights fixed before execution.',
        svc='probability=True uses training-only internal probability estimation; no outer validation passed.',
        interpretation='Exploratory reuse of previously inspected data/splits, not independent confirmation. Historical .844 is a different protocol.',
        stop='First complete candidate strictly above threshold: stop execution for independent review. Otherwise stop after trial 10. Technical errors stop for diagnosis; record before any rerun.',
        group_cv='Not used for target decision; no group CV in this round.',
        data_sha256=digest(data/'train.csv'),splits_sha256=digest(SPLITS),code_sha256=digest(__file__),
        preprocessing_sha256=digest(Path('experiments/fold_safe.py')),
        versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    (OUT/'status.json').write_text(json.dumps(dict(state='planned',completed_trials=0,technical_failures=0)))


def run(data):
    plan=json.loads((OUT/'plan.json').read_text())
    assert plan['code_sha256']==digest(__file__)
    assert plan['data_sha256']==digest(data/'train.csv')
    assert plan['splits_sha256']==digest(OUT/'splits.json')
    status=json.loads((OUT/'status.json').read_text())
    assert status['state']=='planned', 'Use a new documented attempt after failures; never silently overwrite.'
    train=pd.read_csv(data/'train.csv');X=train.drop(columns='Survived');y=train.Survived
    assert len(train)==891 and train.PassengerId.is_unique
    splits=[s for s in json.loads((OUT/'splits.json').read_text()) if s['protocol']=='stratified']
    baseline=pd.read_csv('results/next_round/seeds/fold_scores.csv').query("candidate == 'hgb_reference'")
    summary=[]
    for trial,(name,spec) in enumerate(candidates(),1):
        directory=OUT/f'{trial:02d}_{name}';directory.mkdir()
        rows=[];oof=[];start=time.perf_counter()
        status.update(state='running',active_trial=trial)
        (OUT/'status.json').write_text(json.dumps(status,indent=2))
        for split in splits:
            t=time.perf_counter();context={k:split[k] for k in ['seed','fold']}
            try:
                model=clone(spec).fit(X.iloc[split['train']],y.iloc[split['train']])
                p=model.predict_proba(X.iloc[split['validation']])[:,1]
                assert np.isfinite(p).all()
                rows.append(dict(**context,accuracy=accuracy_score(y.iloc[split['validation']],p>=.5),
                                 log_loss=log_loss(y.iloc[split['validation']],p),seconds=time.perf_counter()-t))
                oof.extend(dict(**context,row=i,probability=float(prob)) for i,prob in zip(split['validation'],p))
            except Exception:
                failure=dict(trial=trial,candidate=name,**context,seconds=time.perf_counter()-t,traceback=traceback.format_exc())
                (directory/'failure.json').write_text(json.dumps(failure,indent=2))
                status.update(state='technical_failure',technical_failures=status['technical_failures']+1)
                (OUT/'status.json').write_text(json.dumps(status,indent=2))
                raise
            finally:
                pd.DataFrame(rows).to_csv(directory/'fold_scores.csv',index=False)
                pd.DataFrame(oof).to_csv(directory/'oof.csv',index=False)
        scores=pd.DataFrame(rows)
        seed_scores=scores.groupby('seed').agg(accuracy=('accuracy','mean'),log_loss=('log_loss','mean'))
        seed_scores.to_csv(directory/'seed_scores.csv')
        value=primary(scores)
        paired=scores.merge(baseline[['seed','fold','accuracy']],on=['seed','fold'],suffixes=('','_baseline'),validate='one_to_one')
        paired['delta']=paired.accuracy-paired.accuracy_baseline
        paired.to_csv(directory/'paired_scores.csv',index=False)
        result=dict(trial=trial,candidate=name,primary_accuracy=value,fold_sd=float(scores.accuracy.std()),
                    seed_sd=float(seed_scores.accuracy.std()),seed_min=float(seed_scores.accuracy.min()),
                    seed_max=float(seed_scores.accuracy.max()),delta_baseline=float(paired.delta.mean()),
                    worse_folds=int((paired.delta<0).sum()),log_loss=float(scores.log_loss.mean()),
                    seconds=time.perf_counter()-start,exceeded=exceeds(value),outer_fits=15,
                    component_fits=45 if trial==10 else 15)
        (directory/'result.json').write_text(json.dumps(result,indent=2))
        summary.append(result);pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
        status.update(completed_trials=trial,state='awaiting_review' if exceeds(value) else 'running')
        (OUT/'status.json').write_text(json.dumps(status,indent=2))
        print(json.dumps(result),flush=True)
        if exceeds(value):
            return
    status.update(state='exhausted_without_target')
    (OUT/'status.json').write_text(json.dumps(status,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--data',type=Path,default=Path('data'));args=parser.parse_args()
    with threadpool_limits(limits=1):
        (prepare if args.action=='prepare' else run)(args.data)
