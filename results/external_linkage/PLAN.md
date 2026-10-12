# Historical record linkage, separate from completed prediction experiments

Match only normalized Name + Ticket + Pclass + Sex, with all four nonmissing.
Normalize Unicode decomposition to ASCII, case folding and removal of punctuation
and whitespace. Require a unique composite key in both combined train/test and
external records. No name-only matches, fuzzy matching, manual corrections or
fallback labels. Repeated or ambiguous keys remain unresolved even if labels
agree. Source rows are one-based data row numbers in the hashed CSV.

Identity matching accepts covariates only and never uses either dataset's
survival labels, boat, body or other post-outcome information. After identity
matching, audit train against historical survived. Preserve every contradiction;
never overwrite source labels to make train agree. Then produce the 418-row test
mapping and an explicitly labelled submission-format candidate only if complete.

Initial identity-only inspection found 891/891 train and 418/418 test exact
composite-key matches, with no composite duplicates. No survival comparison had
been used to design or adjust the matching rule. Source normalizations and all
rows are deterministic; no model or CV is trained. This is answer lookup with
explicit test-label leakage, not improved predictive performance.

The requested hbiostat.org endpoint and previous Vanderbilt endpoint failed with
proxy CONNECT 403, including an approved escalated read. A public mirror referenced
by the CRAN outliertree vignette was used instead, pinned to jbryer/CompStats commit
0a544f2e4f8fe8b75c2bb9c09dfa6f7d3bdd5e0c. The current canonical CSV bytes could
not be verified. Record the actual URL/date/SHA and this limitation prominently.
Do not commit raw source or Kaggle input CSVs. No Kaggle private inputs or submission.
