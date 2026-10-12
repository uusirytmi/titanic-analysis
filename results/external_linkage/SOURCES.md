# Public sources and access record (reviewed 2026-10-12 UTC)

- Distribution and permission: https://hbiostat.org/data/ . The page grants
  dataset use and asks for the original source citation and attribution below.
  No standardized license or explicit raw-redistribution clause was identified.
  Accordingly neither the external raw CSV nor Kaggle input CSVs are committed.
- Canonical requested CSV: https://hbiostat.org/data/repo/titanic3.csv . The
  workspace proxy returned CONNECT 403. An approved escalated public read also
  returned CONNECT 403; no policy approval was rejected. HTTP variant and the
  older https://biostat.app.vumc.org/wiki/pub/Main/DataSets/titanic3.csv endpoint
  also failed. The browser could read documentation but not import CSV bytes
  (unsupported text/csv). No network controls were disabled or credentials changed.
- Public mirror used: https://github.com/jbryer/CompStats/blob/0a544f2e4f8fe8b75c2bb9c09dfa6f7d3bdd5e0c/Data/titanic3.csv .
  The CRAN outliertree vignette explicitly names this repository's titanic3.csv:
  https://cran.r-project.org/web/packages/outliertree/vignettes/Explainable_Outlier_Detection_in_Titanic_dataset.html .
  This is a public copy, not a verified byte-identical retrieval of the current
  canonical endpoint. Actual download URL/date/hash are in source.json.
- Historical provenance: https://hbiostat.org/data/repo/titanic.html attributes
  titanic3 to Thomas Cason's updates using Encyclopedia Titanica and describes
  the historical snapshot as 2 August 1999. Earlier sources include Eaton & Haas
  (1994), Titanic: Triumph and Tragedy, Patrick Stephens Ltd.
- Variable documentation: https://search.r-project.org/CRAN/refmans/PASWR/html/titanic3.html .
  It documents 1309 observations, 14 variables, survived as 0/1, and boat/body
  as lifeboat/body-identification fields. Harrell (2001), Regression Modeling
  Strategies, Springer, is its listed reference.

Required attribution:
“Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt University
Department of Biostatistics.”

The three documentation pages were read using the browser tool in this session.
The downloader's raw-document requests failed through the workspace proxy, so
source.json correctly has no raw-document checksum. Its review note describes
this session only; the current downloader no longer auto-asserts browser review
on future failures. This file is the actual manual review record.

Reference notebook supplied in the request:
https://www.kaggle.com/code/tarunpaparaju/titanic-competition-how-top-lb-got-their-score
The request's prior investigation reported Best Score 1.00000 V2 and PRIVATE
input data. Our browser returned no readable notebook content, and we did not
fetch restricted input. Therefore this is an independently implemented public
record-lookup route, not a verified line-for-line reproduction of that notebook.

Rules: https://www.kaggle.com/competitions/titanic/rules . The request reported
“Rules acceptance is not required for this competition.” Our browser returned
no readable rules text. Neither that message nor public data availability
establishes permission for answer lookup. Competition-rule compatibility is
unconfirmed, and no submission was performed.
