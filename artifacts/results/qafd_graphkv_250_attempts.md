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


### Attempt 5 — qafd_h0_k3

- Completed: 2026-08-19T09:31:52+08:00
- Strategy: QAFD h<=0 ordering of the top-3 retrieved passages followed by GraphKV gapemp
- Result: EM `0.676000` (169/250), F1 `0.086491`, average latency `2.284109 s`.
- Delta from sequential: `-0.056000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 6 — qafd_h0_k5

- Completed: 2026-08-19T09:31:52+08:00
- Strategy: QAFD h<=0 ordering of the top-5 retrieved passages followed by GraphKV gapemp
- Result: EM `0.680000` (170/250), F1 `0.090020`, average latency `2.297495 s`.
- Delta from sequential: `-0.052000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 7 — qafd_h0_k8

- Completed: 2026-08-19T09:31:52+08:00
- Strategy: QAFD h<=0 ordering of the top-8 retrieved passages followed by GraphKV gapemp
- Result: EM `0.712000` (178/250), F1 `0.093001`, average latency `2.321299 s`.
- Delta from sequential: `-0.020000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 8 — qafd_h0_k10

- Completed: 2026-08-19T09:31:52+08:00
- Strategy: QAFD h<=0 ordering of the top-10 retrieved passages followed by GraphKV gapemp
- Result: EM `0.692000` (173/250), F1 `0.088409`, average latency `2.451696 s`.
- Delta from sequential: `-0.040000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 9 — qafd_h1_k3

- Completed: 2026-08-19T09:31:52+08:00
- Strategy: QAFD h<=1 ordering of the top-3 retrieved passages followed by GraphKV gapemp
- Result: EM `0.676000` (169/250), F1 `0.086840`, average latency `2.303784 s`.
- Delta from sequential: `-0.056000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 10 — qafd_h1_k5

- Completed: 2026-08-19T09:31:52+08:00
- Strategy: QAFD h<=1 ordering of the top-5 retrieved passages followed by GraphKV gapemp
- Result: EM `0.684000` (171/250), F1 `0.089143`, average latency `2.327345 s`.
- Delta from sequential: `-0.048000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 11 — qafd_h1_k8

- Completed: 2026-08-19T09:31:52+08:00
- Strategy: QAFD h<=1 ordering of the top-8 retrieved passages followed by GraphKV gapemp
- Result: EM `0.712000` (178/250), F1 `0.093407`, average latency `2.322360 s`.
- Delta from sequential: `-0.020000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 12 — qafd_h1_k10

- Completed: 2026-08-19T09:31:52+08:00
- Strategy: QAFD h<=1 ordering of the top-10 retrieved passages followed by GraphKV gapemp
- Result: EM `0.688000` (172/250), F1 `0.089292`, average latency `2.375846 s`.
- Delta from sequential: `-0.044000` EM.
- Outcome: target not met; continue experimentation.
