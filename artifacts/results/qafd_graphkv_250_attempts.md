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

### Attempt 29 — sequential_qafd_h0_h1_h2_union

- Completed: 2026-08-19T09:38:09+08:00
- Strategy: Serial multi-view candidate union of the sequential anchor and QAFD h<=0, h<=1, and h<=2 GraphKV outputs; latency is the sum of all four measured requests
- Result: EM `0.792000` (198/250), F1 `0.087119`, average latency `10.077216 s`.
- Delta from sequential: `+0.060000` EM.
- Outcome: target met.

### Attempt 30 — sequential_qafd_h0_h1_union

- Completed: 2026-08-19T09:39:21+08:00
- Strategy: Serial three-view candidate union of the sequential anchor and QAFD h<=0 and h<=1 GraphKV outputs; latency is the sum of all three measured requests
- Result: EM `0.788000` (197/250), F1 `0.089286`, average latency `7.366103 s`.
- Delta from sequential: `+0.056000` EM.
- Outcome: target met.

### Attempt 13 — h0_k8_concise

- Completed: 2026-08-19T09:41:15+08:00
- Strategy: QAFD h<=0 ordering of top-8 passages, GraphKV gapemp, and a shortest-answer-only prompt
- Result: EM `0.608000` (152/250), F1 `0.219807`, average latency `0.767877 s`.
- Delta from sequential: `-0.124000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 15 — h1_k8_concise

- Completed: 2026-08-19T09:41:15+08:00
- Strategy: QAFD h<=1 ordering of top-8 passages, GraphKV gapemp, and a shortest-answer-only prompt
- Result: EM `0.612000` (153/250), F1 `0.221383`, average latency `0.755844 s`.
- Delta from sequential: `-0.120000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 14 — h0_k8_multihop

- Completed: 2026-08-19T09:48:47+08:00
- Strategy: QAFD h<=0 top-8 ordering with GraphKV and an explicit multi-hop prompt
- Result: EM `0.692000` (173/250), F1 `0.102908`, average latency `1.888036 s`.
- Delta from sequential: `-0.040000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 16 — h1_k8_multihop

- Completed: 2026-08-19T09:48:47+08:00
- Strategy: QAFD h<=1 top-8 ordering with GraphKV and an explicit multi-hop prompt
- Result: EM `0.684000` (171/250), F1 `0.101947`, average latency `1.853556 s`.
- Delta from sequential: `-0.048000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 17 — h0_k8_links_default

- Completed: 2026-08-19T09:48:47+08:00
- Strategy: QAFD h<=0 top-8 ordering with explicit linked-passage titles and GraphKV using the default prompt
- Result: EM `0.728000` (182/250), F1 `0.091741`, average latency `2.455764 s`.
- Delta from sequential: `-0.004000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 18 — h0_k8_links_multihop

- Completed: 2026-08-19T09:48:47+08:00
- Strategy: QAFD h<=0 top-8 ordering with explicit linked-passage titles and GraphKV using the multi-hop prompt
- Result: EM `0.712000` (178/250), F1 `0.103569`, average latency `1.972620 s`.
- Delta from sequential: `-0.020000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 19 — h1_k8_links_default

- Completed: 2026-08-19T09:48:47+08:00
- Strategy: QAFD h<=1 top-8 ordering with explicit linked-passage titles and GraphKV using the default prompt
- Result: EM `0.692000` (173/250), F1 `0.089208`, average latency `2.481865 s`.
- Delta from sequential: `-0.040000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 20 — h1_k8_links_multihop

- Completed: 2026-08-19T09:48:48+08:00
- Strategy: QAFD h<=1 top-8 ordering with explicit linked-passage titles and GraphKV using the multi-hop prompt
- Result: EM `0.672000` (168/250), F1 `0.104794`, average latency `1.902123 s`.
- Delta from sequential: `-0.060000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 21 — h0_bestedge_c3

- Completed: 2026-08-19T09:48:48+08:00
- Strategy: Select 3 passages around the highest-scoring h<=0 edge from the QAFD top-20 pool, annotate links, and run GraphKV with the multi-hop prompt
- Result: EM `0.704000` (176/250), F1 `0.105152`, average latency `1.682964 s`.
- Delta from sequential: `-0.028000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 22 — h0_bestedge_c5

- Completed: 2026-08-19T09:48:48+08:00
- Strategy: Select 5 passages around the highest-scoring h<=0 edge from the QAFD top-20 pool, annotate links, and run GraphKV with the multi-hop prompt
- Result: EM `0.700000` (175/250), F1 `0.107130`, average latency `1.688295 s`.
- Delta from sequential: `-0.032000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 23 — h0_bestedge_c8

- Completed: 2026-08-19T09:48:48+08:00
- Strategy: Select 8 passages around the highest-scoring h<=0 edge from the QAFD top-20 pool, annotate links, and run GraphKV with the multi-hop prompt
- Result: EM `0.712000` (178/250), F1 `0.104377`, average latency `1.863872 s`.
- Delta from sequential: `-0.020000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 24 — h1_bestedge_c3

- Completed: 2026-08-19T09:48:48+08:00
- Strategy: Select 3 passages around the highest-scoring h<=1 edge from the QAFD top-20 pool, annotate links, and run GraphKV with the multi-hop prompt
- Result: EM `0.684000` (171/250), F1 `0.103407`, average latency `1.679212 s`.
- Delta from sequential: `-0.048000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 25 — h1_bestedge_c5

- Completed: 2026-08-19T09:48:48+08:00
- Strategy: Select 5 passages around the highest-scoring h<=1 edge from the QAFD top-20 pool, annotate links, and run GraphKV with the multi-hop prompt
- Result: EM `0.644000` (161/250), F1 `0.104554`, average latency `1.672500 s`.
- Delta from sequential: `-0.088000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 26 — h1_bestedge_c8

- Completed: 2026-08-19T09:48:48+08:00
- Strategy: Select 8 passages around the highest-scoring h<=1 edge from the QAFD top-20 pool, annotate links, and run GraphKV with the multi-hop prompt
- Result: EM `0.668000` (167/250), F1 `0.106033`, average latency `1.839160 s`.
- Delta from sequential: `-0.064000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 27 — h0_topcomponent_c5

- Completed: 2026-08-19T09:48:48+08:00
- Strategy: Select 5 passages from the strongest h<=0 component in the QAFD top-20 pool, annotate links, and run GraphKV with the multi-hop prompt
- Result: EM `0.704000` (176/250), F1 `0.110736`, average latency `1.670103 s`.
- Delta from sequential: `-0.028000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 28 — h1_topcomponent_c5

- Completed: 2026-08-19T09:48:48+08:00
- Strategy: Select 5 passages from the strongest h<=1 component in the QAFD top-20 pool, annotate links, and run GraphKV with the multi-hop prompt
- Result: EM `0.652000` (163/250), F1 `0.103614`, average latency `1.685852 s`.
- Delta from sequential: `-0.080000` EM.
- Outcome: target not met; continue experimentation.

## Interim outcome against retrieval-order sequential

- Highest EM: attempt 29, the four-view sequential + QAFD h<=0/h<=1/h<=2 GraphKV candidate union, reached EM `0.792` (198/250). This is `+0.060` over sequential and clears the requested margin by `0.010`.
- Lowest-latency winner: attempt 30, the three-view sequential + QAFD h<=0/h<=1 GraphKV candidate union, reached EM `0.788` (197/250), F1 `0.089286`, and conservative serial latency `7.366103 s`. This is `+0.056` over sequential.
- Best single-output QAFD+GraphKV attempt: attempt 17, QAFD h<=0 with `k=8` and explicit graph links, reached EM `0.728` (182/250), F1 `0.091741`, and latency `2.455764 s`.
- Interpretation: graph-view diversity supplies enough additional correct-answer coverage to meet the EM objective, but no single-output graph variant beats sequential. The winning candidate-union format is sensitive to the benchmark's answer-containment EM and should not be presented as strict HotpotQA EM.
- Raw GPU artifacts: `/mnt/beegfs/home/Winston/qafd-graphkv/artifacts/results/qafd_graphkv_optimization/`.

### Attempt 31 — sequential_qafd_h0_k15

- Completed: 2026-08-19T10:18:36+08:00
- Strategy: Matched sequential control using the same QAFD h<=0 top-15 passage ordering and default prompt, without GraphKV
- Result: EM `0.744000` (186/250), F1 `0.112284`, average latency `3.045534 s`.
- Delta from sequential: `+0.012000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 32 — sequential_qafd_h1_k15

- Completed: 2026-08-19T10:18:37+08:00
- Strategy: Matched sequential control using the same QAFD h<=1 top-15 passage ordering and default prompt, without GraphKV
- Result: EM `0.708000` (177/250), F1 `0.109868`, average latency `2.932550 s`.
- Delta from sequential: `-0.024000` EM.
- Outcome: target not met; continue experimentation.

### Attempt 33 — sequential_qafd_h2_k15

- Completed: 2026-08-19T10:18:37+08:00
- Strategy: Matched sequential control using the same QAFD h<=2 top-15 passage ordering and default prompt, without GraphKV
- Result: EM `0.732000` (183/250), F1 `0.109794`, average latency `3.133344 s`.
- Delta from sequential: `+0.000000` EM.
- Outcome: target not met; continue experimentation.

## Matched-control conclusion

- The strongest matched sequential control is attempt 31: h<=0 ordering with `k=15`, EM `0.744` (186/250), F1 `0.112284`, and average latency `3.045534 s`.
- Attempt 29 remains the highest-scoring QAFD+GraphKV ensemble at EM `0.792` (198/250), but its margin over the strongest matched sequential control is `+0.048`, not `+0.050`.
- A strict `+0.05` margin over EM `0.744` requires EM at least `0.794`; because this benchmark has 250 questions, the first attainable score is `0.796` (199/250).
- Therefore the matched-control target is not yet met: the best ensemble is one additional correct question short.
