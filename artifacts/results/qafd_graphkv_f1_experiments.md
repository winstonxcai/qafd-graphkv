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

## Attempt 2 — graph_specific_bestedge_h0_n4

**Timestamp:** 2026-08-19T10:46:18+08:00  
**Hypothesis:** Replacing global gapemp with QAFD-selected center/neighbor caches will remove unrelated context interactions and close or reverse the F1 gap to matched Sequential.  
**Changes:** Introduced a graph-specific server and paired benchmark. Both methods receive the same center, neighbors, serialization, concise prompt, 256-token greedy decoding, and scoring; only GraphKV cache construction differs.  
**Matched Sequential configuration:** Tulu3-Block-FT, identical QAFD best-edge h<=0 context selected from top-20, at most four neighbors, concise prompt, 256-token greedy decoding, bfloat16 A800 inference.  
**Important hyperparameters:** pool_k=20; h=0; center_rule=best_edge; max_neighbors=4; prompt=concise; max_new_tokens=256; limit=250.  

### Results

| Method | EM | F1 | Avg Latency |
|---|---:|---:|---:|
| Sequential | 0.612000 | 0.499500 | 0.420993 s |
| QAFD + GraphKV | 0.616000 | 0.466921 | 0.665596 s |

**Delta EM:** +0.004000  
**Delta F1:** -0.032579  
**Latency ratio:** 1.581015x  
**Outcome:** target not met.  

### Interpretation

The graph-specific path dramatically reduced the Attempt 1 deficit, but still trailed Sequential by 0.032579 F1. GraphKV gained one EM hit yet averaged 12.2 words versus 10.2 for Sequential. The architecture is promising, but four neighbors may omit useful second-hop evidence.

### Next Experiment

Increase the same best-edge h<=0 neighborhood cap from four to eight. This is motivated by 156 of 250 questions saturating the four-neighbor cap; all other settings remain fixed and both methods are rerun.

## Attempt 3 — graph_specific_bestedge_h0_n8

**Timestamp:** 2026-08-19T10:53:21+08:00  
**Hypothesis:** The four-neighbor cap excluded useful second-hop evidence on 156 of 250 questions, so admitting up to eight graph-linked neighbors should improve GraphKV more than matched serialization.  
**Changes:** Increased only the maximum direct QAFD-graph neighbors from four to eight relative to Attempt 2; preserved all 250 examples and the isolated-node fallback.  
**Matched Sequential configuration:** Fresh Sequential run with the identical QAFD h<=0 selected center, neighbor passages and order, Tulu3-Block-FT model, concise prompt, greedy decoding, 256-token cap, BF16 FlashAttention2, and A800 hardware.  
**Important hyperparameters:** pool_k=20; h=0; center_rule=best_edge; max_neighbors=8; prompt_style=concise; limit=250; greedy; max_new_tokens=256.  

### Results

| Method | EM | F1 | Avg Latency |
|---|---:|---:|---:|
| Sequential | 0.640000 | 0.508352 | 0.475665 s |
| QAFD + GraphKV | 0.612000 | 0.460841 | 0.793014 s |

**Delta EM:** -0.028000  
**Delta F1:** -0.047511  
**Latency ratio:** 1.667167x  
**Outcome:** target not met.  

### Interpretation

Increasing graph fan-in degraded both methods and widened the deficit: QAFD+GraphKV fell from -0.032579 F1 at four neighbors to -0.047511 F1 at eight. The overlapping parallel neighbor windows add interference, so broader fan-in is not the path forward.

### Next Experiment

Change the graph direction rather than fan-in: choose the likely bridge-target endpoint as center using query-to-passage lexical overlap, with the question-facing endpoint as its neighbor, and rerun a matched 250-question pair.

## Attempt 4 — graph_specific_bridge_target_h0_n4

**Timestamp:** 2026-08-19T11:01:36+08:00  
**Hypothesis:** Making the likely second-hop answer passage the GraphKV center should let it integrate the question-facing bridge passage and improve over the identical flat Sequential order.  
**Changes:** Added answer-independent content-word query overlap to orient the strongest QAFD edge. The lower-overlap endpoint becomes center, with retrieval score and index as deterministic tie-breakers.  
**Matched Sequential configuration:** Fresh Sequential run with the identical bridge-target center, neighbor passages and order, Tulu3-Block-FT model, concise prompt, greedy 256-token cap, BF16 FlashAttention2, and A800 hardware.  
**Important hyperparameters:** pool_k=20; h=0; center_rule=bridge_target; max_neighbors=4; prompt_style=concise; limit=250; greedy; max_new_tokens=256.  

### Results

| Method | EM | F1 | Avg Latency |
|---|---:|---:|---:|
| Sequential | 0.588000 | 0.478807 | 0.449320 s |
| QAFD + GraphKV | 0.592000 | 0.434540 | 0.738978 s |

**Delta EM:** +0.004000  
**Delta F1:** -0.044267  
**Latency ratio:** 1.644661x  
**Outcome:** target not met.  

### Interpretation

The query-directed orientation changed most stars but reduced both methods and left QAFD+GraphKV 0.044267 F1 behind. Lower lexical overlap is not a reliable enough proxy for the answer-bearing endpoint, so a single directed star remains brittle.

### Next Experiment

Represent multiple QAFD components inside one GraphKV inference: select the best directed star from each of the two highest-scoring connected components and merge their caches once, while matched Sequential receives the same grouped passages.
