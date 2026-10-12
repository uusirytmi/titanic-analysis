"""Deterministic historical answer lookup. Explicit test-label leakage; not ML."""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import pandas as pd
from experiments.fold_safe import digest

FIELDS=['Name','Ticket','Pclass','Sex']


def normalize(value):
    if pd.isna(value):return ''
    value=unicodedata.normalize('NFKD',str(value)).encode('ascii','ignore').decode().lower()
    return re.sub(r'[^a-z0-9]','',value)


def key_frame(frame):
    out=frame[FIELDS].copy()
    for c in FIELDS:
        if c=='Pclass':out[c]=pd.to_numeric(out[c],errors='raise').astype('Int64').astype('string').fillna('')
        out[c]=out[c].map(normalize)
    return out


def link_identity(target,source):
    """No survived/Survived/boat/body ever enters the matching algorithm."""
    tk=key_frame(target);sk=key_frame(source)
    source_map={}
    for idx,row in sk.iterrows():
        key=tuple(row)
        if all(key):source_map.setdefault(key,[]).append(int(idx)+1)
    counts=tk.value_counts().to_dict()
    rows=[]
    for i,row in tk.iterrows():
        key=tuple(row);hits=source_map.get(key,[])
        if not all(key):status='missing_key'
        elif counts[key]>1:status='duplicate_target_key'
        elif len(hits)==0:status='unmatched'
        elif len(hits)>1:status='ambiguous_source_key'
        else:status='matched'
        rows.append(dict(PassengerId=int(target.loc[i,'PassengerId']),status=status,
                         source_row=hits[0] if status=='matched' else None,
                         candidate_rows=';'.join(map(str,hits)),candidate_count=len(hits)))
    return pd.DataFrame(rows).sort_values('PassengerId').reset_index(drop=True)


def attach_labels(mapping,source):
    result=mapping.copy()
    labels=pd.to_numeric(source.Survived,errors='raise')
    assert labels.notna().all() and labels.isin([0,1]).all()
    result['Survived']=pd.array([int(labels.loc[int(r)-1]) if pd.notna(r) else pd.NA for r in result.source_row],dtype='Int64')
    return result


def duplicate_report(frame):
    keys=key_frame(frame);rows=[]
    for cols,label in [(FIELDS,'composite'),(['Name'],'normalized_name_only')]:
        groups=keys.groupby(cols,dropna=False).groups
        for _,indexes in groups.items():
            if len(indexes)>1:
                rows.append(dict(kind=label,source_rows=';'.join(str(int(i)+1) for i in indexes),
                    count=len(indexes),label_values=';'.join(map(str,sorted(set(frame.loc[indexes,'Survived']))))))
    return pd.DataFrame(rows,columns=['kind','source_rows','count','label_values'])


def validate_candidate(candidate,expected_ids):
    assert list(candidate.columns)==['PassengerId','Survived']
    assert len(candidate)==len(expected_ids)==418
    assert candidate.PassengerId.is_unique
    assert candidate.PassengerId.tolist()==sorted(map(int,expected_ids))
    assert candidate.Survived.notna().all() and candidate.Survived.isin([0,1]).all()


def run(data,external,provenance,out):
    out.mkdir(parents=True,exist_ok=False)
    source=pd.read_csv(external).rename(columns={c.lower():c for c in FIELDS+['Survived']})
    assert source.shape==(1309,14)
    meta=json.loads(provenance.read_text())
    assert meta['sha256']==digest(external)
    train=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv')
    assert len(train)==891 and len(test)==418
    assert train.PassengerId.is_unique and test.PassengerId.is_unique
    assert set(train.PassengerId).isdisjoint(test.PassengerId)
    # A combined identity-only uniqueness check prevents one source row assigned twice.
    identities=pd.concat([train.drop(columns='Survived'),test],ignore_index=True)
    combined=link_identity(identities,source)
    used=combined.source_row.dropna();assert used.is_unique
    duplicates=duplicate_report(source);duplicates.to_csv(out/'source_duplicates.csv',index=False)
    # Audit train labels first, without changing matches to satisfy training labels.
    train_map=attach_labels(combined[combined.PassengerId.isin(train.PassengerId)],source)
    train_map['train_label']=train_map.PassengerId.map(train.set_index('PassengerId').Survived)
    train_map['label_conflict']=train_map.Survived.notna() & (train_map.Survived!=train_map.train_label)
    train_map.to_csv(out/'train_mapping.csv',index=False)
    train_map[train_map.label_conflict].to_csv(out/'train_label_conflicts.csv',index=False)
    train_summary=dict(rows=len(train),matched=int((train_map.status=='matched').sum()),
        label_conflicts=int(train_map.label_conflict.sum()),unresolved=int((train_map.status!='matched').sum()))
    (out/'train_audit.json').write_text(json.dumps(train_summary,indent=2))
    # Test labels come exclusively from the linked historical source, not a model.
    test_map=attach_labels(combined[combined.PassengerId.isin(test.PassengerId)],source)
    test_map.to_csv(out/'test_mapping.csv',index=False)
    unresolved=pd.concat([train_map.assign(split='train'),test_map.assign(split='test')],ignore_index=True)
    unresolved[unresolved.status!='matched'].to_csv(out/'unresolved.csv',index=False)
    complete=bool((test_map.status=='matched').all())
    if complete:
        candidate=test_map[['PassengerId','Survived']].copy();validate_candidate(candidate,test.PassengerId)
        candidate.to_csv(out/'submission_candidate.csv',index=False)
    else:
        test_map[['PassengerId','Survived']].to_csv(out/'incomplete_candidate_DO_NOT_SUBMIT.csv',index=False)
    summary=dict(train=train_summary,test=dict(rows=418,matched=int((test_map.status=='matched').sum()),
        unresolved=int((test_map.status!='matched').sum())),source_rows=1309,
        composite_duplicate_groups=int((duplicates.kind=='composite').sum()),
        name_only_duplicate_groups=int((duplicates.kind=='normalized_name_only').sum()),
        unused_source_rows=1309-len(used),submission_candidate_complete=complete,
        kaggle_score=None,test_label_leakage=True,machine_learning_metric=False,
        canonical_bytes_verified=meta['canonical_bytes_verified'],
        sha256=dict(external=digest(external),train=digest(data/'train.csv'),test=digest(data/'test.csv'),code=digest(__file__),provenance=digest(provenance)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--external',type=Path,default=Path('data/external/titanic3.csv'))
    p.add_argument('--provenance',type=Path,default=Path('results/external_linkage/source.json'))
    p.add_argument('--out',type=Path,default=Path('results/external_linkage/run1'))
    a=p.parse_args();run(a.data,a.external,a.provenance,a.out)
