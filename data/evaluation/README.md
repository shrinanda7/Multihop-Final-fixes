# Evaluation artifacts

`historical_unvalidated_benchmark.json` is retained only for provenance. It contains pre-fix scores and graph-preservation values that were not reliably measured; **do not cite or use it as evidence** in a paper.

A fresh run of the benchmark endpoint writes `latest_benchmark.json`. Before reporting results, verify that each configuration has successful samples, valid judge scores, and independently computed graph metrics. Failed query attempts and missing judge scores are now represented explicitly rather than silently omitted or assigned fabricated values.
