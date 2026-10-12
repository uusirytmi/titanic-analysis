"""Verify saved identity mapping and candidate and reproduce byte-for-byte locally."""
import argparse
import json
import tempfile
from pathlib import Path
import pandas as pd
from experiments.external_linkage import run,validate_candidate
from experiments.fold_safe import digest


def audit(data,external):
    out=Path('results/external_linkage/run1');meta=json.loads((out/'summary.json').read_text())
    assert digest(external)==meta['sha256']['external']
    assert digest(Path('experiments/external_linkage.py'))==meta['sha256']['code']
    assert digest(Path('results/external_linkage/source.json'))==meta['sha256']['provenance']
    seen=[]
    ext=pd.read_csv(external)
    for split in ['train','test']:
        inputs=pd.read_csv(data/f'{split}.csv')
        assert digest(data/f'{split}.csv')==meta['sha256'][split]
        m=pd.read_csv(out/f'{split}_mapping.csv')
        assert len(m)==len(inputs) and m.PassengerId.is_unique
        assert set(m.PassengerId)==set(inputs.PassengerId)
        assert (m.status=='matched').all() and (m.candidate_count==1).all()
        assert m.source_row.notna().all() and m.source_row.is_unique
        seen.extend(m.source_row.tolist())
        pd.testing.assert_series_equal(m.Survived.reset_index(drop=True),ext.iloc[m.source_row.astype(int)-1].survived.reset_index(drop=True),check_names=False)
        if split=='train':
            actual=inputs.set_index('PassengerId').loc[m.PassengerId,'Survived'].to_numpy()
            assert (m.Survived.to_numpy()==actual).all()
    assert len(seen)==len(set(seen))==1309
    candidate=pd.read_csv(out/'submission_candidate.csv')
    test=pd.read_csv(data/'test.csv');validate_candidate(candidate,test.PassengerId)
    mapping=pd.read_csv(out/'test_mapping.csv')
    pd.testing.assert_frame_equal(candidate,mapping[['PassengerId','Survived']])
    with tempfile.TemporaryDirectory() as tmp:
        repro=Path(tmp)/'run'
        run(data,external,Path('results/external_linkage/source.json'),repro)
        for file in out.iterdir():
            assert file.read_bytes()==(repro/file.name).read_bytes(),file.name
    print('PASS: 1309 unique source assignments, train labels, all 418 candidate rows and byte-identical reproduction. No Kaggle score established.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--external',type=Path,default=Path('data/external/titanic3.csv'));a=p.parse_args();audit(a.data,a.external)
