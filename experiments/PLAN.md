# Frozen first comparison (2026-10-12, before execution)

- Input: 891 labelled Kaggle Titanic rows, 418 unlabelled rows. Local only;
  public mirror https://github.com/agconti/kaggle-titanic/tree/master/data.
  SHA256 recorded in manifest; no raw passenger records are added to git.
- Primary metric accuracy at probability >= 0.5. Secondary log loss.
- Outer StratifiedKFold(5, shuffle=True, random_state=20261012).
  Inner StratifiedKFold(3, shuffle=True, random_state=20261013+outer_fold).
- Same outer and inner splits for every candidate. Inner mean accuracy selects
  one candidate per outer fold (ties: listed order). Outer labels never select
  that fold's model. Report this nested selection procedure separately from
  exploratory outer rankings of fixed candidates. Do not claim the maximum
  outer score is an unbiased estimate of a subsequently selected winner.
- Fifteen fixed candidates: 3 models x 5 cumulative feature groups. Models:
  logistic regression C=1; random forest 120 trees, min_samples_leaf=3,
  max_features=sqrt; histogram gradient boosting 100 iterations,
  max_leaf_nodes=15, min_samples_leaf=20, learning_rate=.08, l2=1,
  early_stopping=False. No adaptive search or reruns after seeing results.
- Groups: base; +title; +family; +fare; +interactions. Base includes class,
  sex, age, siblings/spouses, parents/children, fare, embarkation. Family adds
  family size/alone; fare adds log fare, fare per family member and quartile;
  interactions add sex/class, title/class, age/class, child indicator.
- All learned medians, modes, encodings, scales and fare boundaries are fitted
  inside each training fold, through sklearn Pipeline. No target aggregates,
  surnames, ticket identifiers/counts or external survival information.
- Total bound: 225 inner + 75 outer fits plus 5 dummy fits. Single CPU thread.
  Failures recorded with traceback; any failed candidate cannot be selected;
  incomplete results must not be interpreted as a full comparison.
- Save OOF probabilities (row position only), split indices, fold metrics,
  inner scores, model parameters, timings, versions, data/code hashes and logs.
  Transform local test data as a smoke check only; no Kaggle submission.
- Small correlated folds do not support a strong significance claim. Random
  passenger splits may share families; performance on unseen families is not
  established. Future changes require fresh evaluation; Kaggle score unknown.
- Historical ~0.844 used different, globally preprocessed/selection-reused CV;
  do not directly compare as improvement or infer a performance ceiling.
