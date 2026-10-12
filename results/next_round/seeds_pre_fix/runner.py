"""Three bounded hypotheses sharing a frozen protocol. See NEXT_PLAN.md."""
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
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, log_loss
from threadpoolctl import threadpool_limits
from experiments.fold_safe import Features, candidates, digest

SEEDS = [20261012, 20261029, 20261113]
PARENT = '1d0eef8cbc074bcb6332d8c8f957ccf0fa2ed2a3'
WEIGHTS = [[1/3]*3, [.6,.2,.2], [.2,.6,.2], [.2,.2,.6]]


def keys(X):
    ticket = X.Ticket.fillna('').str.upper().str.replace(r'\s+', '', regex=True)
    size = X.SibSp + X.Parch + 1
    family = X.Name.str.split(',').str[0].str.strip().str.upper() + '_' + size.astype(str)
    return ticket.where(ticket != '', None), family.where(size > 1, None)


def group_ids(X):
    """Connected components use covariates only; no y accepted."""
    parent = list(range(len(X)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for series in keys(X):
        seen = {}
        for i, value in enumerate(series):
            if value is None:
                continue
            if value in seen:
                parent[root(i)] = root(seen[value])
            else:
                seen[value] = i
    return np.array([root(i) for i in range(len(X))])


class Companions(Features):
    def fit(self, X, y=None):
        super().fit(X, y)
        self.training_ids_ = set(X.PassengerId)
        self.counts_ = [s.value_counts().to_dict() for s in keys(X)]
        return self

    def transform(self, X):
        z = super().transform(X)
        z['Deck'] = X.Cabin.fillna('Unknown').str[0]
        z['TicketPrefix'] = X.Ticket.fillna('').str.upper().str.replace(r'[^A-Z]', '', regex=True).replace('', 'NONE')
        size = X.SibSp + X.Parch + 1
        z['FamilyBand'] = pd.cut(size, [0,1,4,np.inf], labels=['alone','small','large']).astype(str)
        for label, s, counts in zip(['TicketCompanions','FamilyCompanions'], keys(X), self.counts_):
            own = (X.PassengerId.isin(self.training_ids_) & s.notna()).astype(int)
            z[label] = (s.map(counts).fillna(0) - own).clip(lower=0)
        return z


def specs_for(branch):
    old = candidates()
    specs = {'hgb_reference': old['hgb_interactions'], 'rf_family': old['rf_family']}
    if branch == 'seeds':
        specs['hgb_regularized'] = clone(old['hgb_interactions']).set_params(
            model__max_leaf_nodes=7, model__min_samples_leaf=30, model__l2_regularization=5)
        specs['lr_regularized'] = clone(old['lr_interactions']).set_params(model__C=.1)
    elif branch == 'groups':
        for name, base in [('hgb_companions','hgb_interactions'),('rf_companions','rf_family')]:
            specs[name] = clone(old[base]).set_params(features=Companions(old[base].named_steps['features'].group))
    elif branch == 'ensemble':
        specs['lr_interactions'] = old['lr_interactions']
    else:
        raise ValueError(branch)
    return specs


def make_splits(X, y):
    groups = group_ids(X)
    result = []
    for protocol in ['stratified','group']:
        for seed in SEEDS:
            cv = (StratifiedKFold(5, shuffle=True, random_state=seed) if protocol == 'stratified'
                  else StratifiedGroupKFold(5, shuffle=True, random_state=seed))
            iterator = cv.split(X, y) if protocol == 'stratified' else cv.split(X, y, groups)
            for fold, (tr, va) in enumerate(iterator):
                inner = StratifiedKFold(3, shuffle=True, random_state=seed+100+fold)
                result.append(dict(protocol=protocol, seed=seed, fold=fold, train=tr.tolist(), validation=va.tolist(),
                                   inner=[dict(train=tr[a].tolist(), validation=tr[b].tolist())
                                          for a,b in inner.split(X.iloc[tr], y.iloc[tr])]))
    return result, groups


def select_weights(inner_predictions, y, inner_fold):
    """Only outer-training OOF labels are provided here."""
    losses = [float(np.mean([log_loss(y[inner_fold == f], (inner_predictions @ w)[inner_fold == f])
                            for f in np.unique(inner_fold)])) for w in WEIGHTS]
    return int(np.argmin(losses)), losses


def summarize(scores, out):
    baseline = scores[scores.candidate == 'hgb_reference'][['protocol','seed','fold','accuracy']].rename(columns={'accuracy':'baseline_accuracy'})
    paired = scores.merge(baseline, on=['protocol','seed','fold'], validate='many_to_one')
    paired['delta_accuracy'] = paired.accuracy - paired.baseline_accuracy
    paired.to_csv(out/'fold_scores.csv', index=False)
    seeds = paired.groupby(['protocol','candidate','seed']).agg(accuracy=('accuracy','mean'),
        delta=('delta_accuracy','mean'), log_loss=('log_loss','mean'), seconds=('seconds','sum')).reset_index()
    seeds.to_csv(out/'seed_scores.csv', index=False)
    summary = paired.groupby(['protocol','candidate']).agg(accuracy=('accuracy','mean'),
        fold_sd=('accuracy','std'), delta=('delta_accuracy','mean'), delta_fold_sd=('delta_accuracy','std'),
        worse_folds=('delta_accuracy',lambda s: int((s < 0).sum())), log_loss=('log_loss','mean'),
        seconds=('seconds','sum')).join(seeds.groupby(['protocol','candidate']).agg(seed_sd=('accuracy','std'),
        delta_seed_sd=('delta','std')))
    summary.to_csv(out/'summary.csv')


def run(branch, data, out):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/'runner.py')
    train = pd.read_csv(data/'train.csv')
    assert len(train) == 891 and train.PassengerId.is_unique and set(train.Survived) == {0,1}
    X, y = train.drop(columns='Survived'), train.Survived
    splits, groups = make_splits(X,y)
    (out/'splits.json').write_text(json.dumps(splits))
    (out/'groups.json').write_text(json.dumps(groups.tolist()))
    specs = specs_for(branch)
    manifest = dict(branch=branch, parent=PARENT, seeds=SEEDS, complete=False, python=platform.python_version(),
        versions={'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
        data_sha256=digest(data/'train.csv'), code_sha256=digest(__file__),
        base_code_sha256=digest(Path(__file__).with_name('fold_safe.py')),
        plan_sha256=digest(Path(__file__).with_name('NEXT_PLAN.md')),
        splits_sha256=digest(out/'splits.json'), weights=WEIGHTS,
        parameters={k:{p:str(v) for p,v in s.get_params().items()} for k,s in specs.items()})
    rows, oof, inner_oof, selections, failures = [], [], [], [], []
    start = time.perf_counter()
    fit_count = 0
    def flush():
        pd.DataFrame(rows).to_csv(out/'fold_scores.csv',index=False)
        pd.DataFrame(oof).to_csv(out/'oof.csv',index=False)
        pd.DataFrame(inner_oof).to_csv(out/'inner_oof.csv',index=False)
        (out/'selections.json').write_text(json.dumps(selections,indent=2))
        (out/'failures.json').write_text(json.dumps(failures,indent=2))
        manifest.update(seconds=time.perf_counter()-start, fit_count=fit_count, failures=len(failures))
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    flush()
    for split in splits:
        if branch != 'groups' and split['protocol'] != 'stratified':
            continue
        context = {k:split[k] for k in ['protocol','seed','fold']}
        tr, va = split['train'], split['validation']
        preds = {}
        elapsed = {}
        fold_start = time.perf_counter()
        stage = 'outer'
        name = ''
        try:
            for name, spec in specs.items():
                t = time.perf_counter()
                fitted = clone(spec).fit(X.iloc[tr],y.iloc[tr]); fit_count += 1
                preds[name] = fitted.predict_proba(X.iloc[va])[:,1]
                elapsed[name] = time.perf_counter()-t
            if branch == 'ensemble':
                names = ['hgb_reference','rf_family','lr_interactions']
                matrix = np.column_stack([preds[n] for n in names])
                preds['equal_average'] = matrix @ WEIGHTS[0]
                elapsed['equal_average'] = sum(elapsed.values())
                stage = 'inner'
                ins_pred = np.full((len(X),3),np.nan)
                ins_fold = np.full(len(X),-1)
                for j, ins in enumerate(split['inner']):
                    for c, name in enumerate(names):
                        fitted = clone(specs[name]).fit(X.iloc[ins['train']],y.iloc[ins['train']]); fit_count += 1
                        ps = fitted.predict_proba(X.iloc[ins['validation']])[:,1]
                        ins_pred[ins['validation'],c] = ps
                        inner_oof.extend(dict(**context,inner_fold=j,row=i,candidate=name,probability=float(p))
                                         for i,p in zip(ins['validation'],ps))
                    ins_fold[ins['validation']] = j
                choice, losses = select_weights(ins_pred[tr],y.iloc[tr].to_numpy(),ins_fold[tr])
                selections.append(dict(**context,choice=choice,weights=WEIGHTS[choice],inner_log_losses=losses))
                preds['inner_weighted'] = matrix @ WEIGHTS[choice]
                elapsed['inner_weighted'] = time.perf_counter()-fold_start
            for name, p in preds.items():
                assert np.isfinite(p).all()
                rows.append(dict(**context,candidate=name,accuracy=accuracy_score(y.iloc[va],p>=.5),
                                 log_loss=log_loss(y.iloc[va],p),seconds=elapsed[name]))
                oof.extend(dict(**context,candidate=name,row=i,probability=float(prob)) for i,prob in zip(va,p))
        except Exception:
            failures.append(dict(**context,candidate=name,stage=stage,seconds=time.perf_counter()-fold_start,
                                 traceback=traceback.format_exc()))
            flush()
            raise
        flush()
        print(branch,context,'completed',flush=True)
    try:
        summarize(pd.DataFrame(rows),out)
    except Exception:
        failures.append(dict(stage='summary', seconds=time.perf_counter()-start,
                             traceback=traceback.format_exc()))
        flush()
        raise
    manifest.update(complete=True, seconds=time.perf_counter()-start)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.read_csv(out/'summary.csv').to_string(index=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('branch',choices=['seeds','groups','ensemble'])
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.branch,args.data,args.out)
