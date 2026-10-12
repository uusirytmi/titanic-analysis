# Historical answer lookup: complete linkage, Kaggle unscored

This separate branch uses historical survival records to look up known answers.
**It deliberately introduces test-label leakage. It is not machine learning,
CV improvement, or evidence of generalization.** The completed ten-candidate
prediction experiment remains closed and its results are unchanged.

| Audit | Result |
|---|---:|
| Actual external input | 1,309 rows × 14 columns |
| Train unique identity matches | 891 / 891 (100%) |
| Train survival-label contradictions | 0 |
| Test unique identity matches | 418 / 418 (100% coverage) |
| Missing / ambiguous / unresolved matches | 0 |
| Composite-key duplicate groups | 0 |
| Name-only duplicate groups | 2 |
| External rows reused / unused | 0 / 0 |

Normalize Name, Ticket, Pclass and Sex; match their **full composite key** only
when unique in both datasets. There is no name-only fallback, fuzzy matching,
label-assisted identity choice, manual correction or filling of unresolved rows.
The two name-only duplicate groups are explicitly listed by source row in
`run1/source_duplicates.csv`; the other fields disambiguate them. The identity
matcher does not access survived, boat or body. After matching, survived is
copied from the source; train labels are used only for the separate contradiction
audit. Boat/body are never used as predictive features or matching fields.

`run1/test_mapping.csv` contains 418 PassengerIds, source row numbers, match
status and historical labels. `run1/submission_candidate.csv` contains exactly
PassengerId,Survived in ascending ID order 892–1309, all 418 IDs unique and all
labels binary. It is a **submission-format candidate**, not a submitted file.
Train mapping, empty conflict/unresolved tables, duplicate diagnostics and hashes
are retained. New generated tables contain no passenger names, tickets, addresses
or the raw source table.

**Kaggle score: unknown. No 1.00000 / 100% test accuracy claim is made.** The 100%
figures above are identity coverage and train label agreement, not hidden-test
scoring. Notebook-private inputs were not fetched. The answer-lookup method's
competition-rule compatibility remains unconfirmed.

## Actual source and limitation

Requested canonical https://hbiostat.org/data/repo/titanic3.csv could not be
retrieved because the execution proxy returned 403, including an approved
escalated read. We used the publicly accessible jbryer/CompStats copy referenced
by CRAN outliertree, pinned to commit 0a544f2e4f8fe8b75c2bb9c09dfa6f7d3bdd5e0c.
**Its identity with current canonical CSV bytes is unverified.** This limitation
is recorded in source.json and summary.json, not silently treated as a canonical
successful download.

Retrieved 2026-10-12T07:36:09.712476+00:00, 116,752 bytes, SHA256:
`db6df9666818c69a753cd85d743e01502c8518b00579b6aada0d4fc5a66ccb9d`.
All 14 columns and the date/URLs are recorded in source.json. See SOURCES.md for
historical provenance, variable definitions, browser review, use permission,
attribution and redistribution limitations. Raw external data is ignored by git;
only the retrieval code, provenance and requested generated audit/candidate
outputs are committed.

## Checks and reproduction

19 tests pass (6 new linkage tests plus 13 previous tests). New tests cover exact
normalization, missing keys, no name-only fallback, duplicate source/target keys,
no resolution even when duplicate labels agree, target/source label independence,
boat/body independence, input-order stability and malformed submission rejection.
A full audit verifies the 1,309 one-to-one assignments, labels and every candidate
row; a second run produces byte-identical files. Independent review reimplemented
the four-field matching in the standard library and confirmed every saved match,
label, hash and output constraint. See tests.log, audit.log and REVIEW.md.

```bash
# Existing experiments/requirements.txt is sufficient.
# Fresh checkout: put train.csv / test.csv into data/ (remain untracked).
# Reproduce the actual pinned-mirror bytes, preserving original provenance:
python -m experiments.fetch_titanic3 --public-mirror --provenance results/external_linkage/refetch.json
python -m experiments.external_linkage --out results/external_linkage/reproduced
python -m unittest discover -s tests -v
python -m experiments.audit_linkage
# To attempt canonical retrieval separately, use fresh paths:
# python -m experiments.fetch_titanic3 --out data/external/canonical.csv --provenance results/external_linkage/canonical_source.json
```

The checked-in source.json preserves the original acquisition record. To fetch
again without replacing it, pass `--provenance` with a fresh path and then pass
that same path to the linker. To reproduce the original audit, use the pinned
mirror bytes whose SHA256 matches the saved source.json; canonical bytes might
be a different version. Downloader/linker refuse to overwrite their outputs.
No master merge, Kaggle upload, charge or authentication change was performed.
