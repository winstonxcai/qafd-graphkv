# Single-pipeline QAFD + GraphKV F1 experiments

## Fixed evaluation contract

- Questions: the same 250 HotpotQA questions, preserved in QID order `0..249`.
- Success criterion: `QAFD+GraphKV F1 - matched Sequential F1 >= 0.05`.
- Every attempt produces one final answer per question from one unified pipeline.
- No ensembles, oracle routing, output unions, majority voting, or ground-truth-dependent behavior.
- Every QAFD+GraphKV run has a Sequential control with the same model, passages, ordering/serialization inputs, prompt, decoding, precision, and evaluation code. Only the graph/cache mechanism under test may differ.
- EM, F1, average request latency, deltas, and latency ratio are recorded.
- Raw predictions remain on the GPU server under `artifacts/results/qafd_graphkv_f1_search/`.

The historical EM-oriented ensemble ledger is separate and is not valid evidence
for this objective.


## Attempt 1 — h0_k8_concise_gapemp

**Timestamp:** 2026-08-19T10:37:39+08:00  
**Hypothesis:** A concise answer contract plus QAFD's lower-noise h<=0 ordering will let GraphKV integrate the relevant top-8 passages more precisely than full sequential attention.  
**Changes:** Restarted the F1 search with one answer per question, concise prompting, k=8, h<=0 ordering, and freshly reran both methods.  
**Matched Sequential configuration:** Tulu3-Block-FT, identical QAFD top-8 h<=0 order, concise prompt, greedy decoding, bfloat16 A800 inference, and identical scoring.  
**Important hyperparameters:** k=8; h=0; prompt=concise; GraphKV=gapemp; limit=250; temperature=1; greedy decoding.  

### Results

| Method | EM | F1 | Avg Latency |
|---|---:|---:|---:|
| Sequential | 0.644000 | 0.527573 | 0.437426 s |
| QAFD + GraphKV | 0.608000 | 0.219807 | 0.818715 s |

**Delta EM:** -0.036000  
**Delta F1:** -0.307766  
**Latency ratio:** 1.871666x  
**Outcome:** target not met.  

### Interpretation

The hypothesis failed. Sequential F1 was 0.527573 while GraphKV F1 was 0.219807. GraphKV responses averaged 11.2 words versus 8.1 for Sequential, indicating that the standard all-context cache transformation damages concise extraction and adds answer noise.

### Next Experiment

Replace global gapemp with a graph-specific QAFD center/neighbor cache path. Use the highest-scoring h<=0 edge from the top-20 pool, the same selected passages for Sequential, and the same concise 256-token decoding budget.
