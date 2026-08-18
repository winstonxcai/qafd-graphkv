# QAFD + GraphKV 250-question optimization log

## Objective and fixed evaluation contract

- Dataset: the first 250 entries in `results_hotpotqa.json`, unchanged between attempts.
- Retrieval source: QAFD HotpotQA output generated with `gpt-4o-mini` and `nvidia/nv-embed-v2`.
- Default passage pool: the same retrieved top-20 passages per question; each attempt records any smaller `k` or selection rule it uses.
- Sequential baseline: EM `0.732` (183/250), F1 `0.109042`, average request latency `2.061916 s`.
- Required margin: at least `+0.05` EM over sequential.
- Operational success threshold: at least 196/250 EM hits, i.e. EM `0.784`.
- Eligibility: the winning strategy must use both QAFD-derived graph information and GraphKV inference/cache machinery.
- Machine-readable ledger: `artifacts/results/qafd_graphkv_250_results.csv`.

The benchmark's existing `EM` field is retained for comparability. It scores a
prediction as correct when a normalized gold answer is contained in the
normalized generated response; it is not the official strict HotpotQA EM
implementation.

## Imported reference

### Attempt 0 — sequential baseline

- Completed: 2026-08-18 23:01:11 +08:00
- Strategy: concatenate the same QAFD top-15 passages and run ordinary sequential generation.
- Result: EM `0.732` (183/250), F1 `0.109042`, average latency `2.061916 s`.
- Status: fixed comparison baseline; not eligible as a QAFD+GraphKV solution.

## QAFD + GraphKV attempts

### Attempt 1 — QAFD h<=0 ordering + GraphKV

- Completed: 2026-08-18 23:03:52 +08:00
- Strategy: order the same QAFD top-15 passages by connected components in the `h<=0` passage graph, then use GraphKV's official `gapemp` generation path.
- Result: EM `0.692` (173/250), F1 `0.092156`, average latency `2.706527 s`.
- Delta from sequential: `-0.040` EM.
- Outcome: did not beat the baseline.

### Attempt 2 — QAFD h<=1 ordering + GraphKV

- Completed: 2026-08-18 23:03:26 +08:00
- Strategy: order the same QAFD top-15 passages by connected components in the `h<=1` passage graph, then use GraphKV's official `gapemp` generation path.
- Result: EM `0.684` (171/250), F1 `0.092888`, average latency `2.597660 s`.
- Delta from sequential: `-0.048` EM.
- Outcome: the positive signal from the earlier 50-question run did not persist.

### Attempt 3 — QAFD h<=2 ordering + GraphKV

- Completed: 2026-08-18 23:03:53 +08:00
- Strategy: order the same QAFD top-15 passages by connected components in the `h<=2` passage graph, then use GraphKV's official `gapemp` generation path.
- Result: EM `0.684` (171/250), F1 `0.091143`, average latency `2.711113 s`.
- Delta from sequential: `-0.048` EM.
- Outcome: extra graph connectivity did not improve generation.

### Attempt 4 — recursive QAFD h<=1, T=2

- Completed: 2026-08-18 23:05:04 +08:00
- Strategy: reorder QAFD top-15 passages with the `h<=1` graph, propagate neighboring passage KV caches for two rounds, merge the resulting caches, and generate one answer.
- Result: EM `0.192` (48/250), F1 `0.056391`, average latency `2.993303 s`.
- Delta from sequential: `-0.540` EM.
- Outcome: operationally stable but semantically incorrect; do not scale this cache formulation without fixing its positional semantics.

