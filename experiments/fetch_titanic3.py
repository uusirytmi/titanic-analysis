"""Fetch only the public Vanderbilt CSV, with provenance; never Kaggle private inputs."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import pandas as pd
URL='https://hbiostat.org/data/repo/titanic3.csv'
DOCS=['https://hbiostat.org/data/','https://hbiostat.org/data/repo/titanic.html',
      'https://search.r-project.org/CRAN/refmans/PASWR/html/titanic3.html']

MIRROR='https://raw.githubusercontent.com/jbryer/CompStats/0a544f2e4f8fe8b75c2bb9c09dfa6f7d3bdd5e0c/Data/titanic3.csv'

def fetch(out,provenance,public_mirror=False):
    if out.exists() or provenance.exists():raise FileExistsError('Preserve existing evidence; choose fresh paths')
    out.parent.mkdir(parents=True,exist_ok=True)
    actual_url=MIRROR if public_mirror else URL
    with urlopen(actual_url,timeout=60) as response:
        data=response.read()
        meta=dict(canonical_url=URL,url=actual_url,canonical_bytes_verified=not public_mirror,final_url=response.url,content_type=response.headers.get('Content-Type'),
                  last_modified=response.headers.get('Last-Modified'),retrieved_utc=datetime.now(timezone.utc).isoformat(),
                  sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    out.write_bytes(data)
    frame=pd.read_csv(out)
    assert frame.shape==(1309,14)
    assert set(frame.survived)=={0,1}
    meta.update(rows=len(frame),columns=frame.columns.tolist(),missing_survived=int(frame.survived.isna().sum()),documents=[])
    for url in DOCS:
        try:
            with urlopen(url,timeout=60) as response:
                raw=response.read()
                meta['documents'].append(dict(url=url,retrieved_utc=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(raw).hexdigest()))
        except Exception as e:
            meta['documents'].append(dict(url=url,download_error=str(e),review='No review inferred by this downloader; document retrieval failed and no raw-byte hash is available.'))
    meta.update(permission='Distribution page grants use with attribution and requests citing original sources. No specific standardized license or explicit redistribution clause identified; raw CSV is not committed.',
                attribution='Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.',
                warning='Known survival labels, including test people: answer lookup and test-label leakage, not prediction/generalization.')
    provenance.parent.mkdir(parents=True,exist_ok=True)
    provenance.write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:meta[k] for k in ['rows','columns','sha256','retrieved_utc']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('data/external/titanic3.csv'))
    p.add_argument('--provenance',type=Path,default=Path('results/external_linkage/source.json'))
    p.add_argument('--public-mirror',action='store_true',help='Explicit pinned public mirror fallback; does not claim canonical byte identity')
    a=p.parse_args();fetch(a.out,a.provenance,a.public_mirror)
