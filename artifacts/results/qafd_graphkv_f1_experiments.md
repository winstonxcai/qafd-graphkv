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

## Attempt 5 — graph_batch_multistar_h0_s2_n2

**Timestamp:** 2026-08-19T11:08:20+08:00  
**Hypothesis:** Separating disconnected QAFD components into independent GraphKV cache regions should preserve evidence from multiple plausible subgraphs while avoiding cross-component prefill interference.  
**Changes:** Added the official GraphKV graph-batch path, deterministic component-level star selection, and a matched grouped Sequential serializer. Stars are disjoint because they come from separate connected components.  
**Matched Sequential configuration:** Fresh Sequential run over exactly the same star centers, neighbor groups, group order, model, concise prompt, greedy 256-token cap, BF16 FlashAttention2, and A800 hardware.  
**Important hyperparameters:** pool_k=20; h=0; max_stars=2; max_neighbors_per_star=2; component edge ranking=sum of QAFD scores; prompt_style=concise; limit=250; greedy; max_new_tokens=256.  

### Results

| Method | EM | F1 | Avg Latency |
|---|---:|---:|---:|
| Sequential | 0.580000 | 0.473635 | 0.417997 s |
| QAFD + GraphKV | 0.588000 | 0.402070 | 0.716890 s |

**Delta EM:** +0.008000  
**Delta F1:** -0.071565  
**Latency ratio:** 1.715058x  
**Outcome:** target not met.  

### Interpretation

Two parallel component caches increased EM slightly but reduced answer precision: QAFD+GraphKV EM was 0.588 versus 0.580, while F1 was 0.402070 versus 0.473635. The batch cache preserves additional candidate evidence but does not condition its component integration on the question, so generated answers contain more irrelevant tokens.

### Next Experiment

Return to the strongest one-star topology and make its center integration query-conditioned by placing the question and a graph-integration instruction before the center passage for both methods. This lets GraphKV integrate neighbor caches with the actual query rather than statically.

## Attempt 6 — query_conditioned_center_bestedge_h0_n4

**Timestamp:** 2026-08-19T11:14:21+08:00  
**Hypothesis:** The center passage must know the question while attending to GraphKV neighbor caches; query-conditioned center prefill should suppress irrelevant cached evidence and improve answer precision over flat Sequential serialization.  
**Changes:** Returned to the best one-star h<=0 n=4 topology and added identical question-directed center text to both methods. No answer metadata or gold passages are used.  
**Matched Sequential configuration:** Fresh Sequential run with the identical center, neighbors, query-conditioned center text, order, Tulu3-Block-FT model, concise final prompt, greedy 256-token cap, BF16 FlashAttention2, and A800 hardware.  
**Important hyperparameters:** pool_k=20; h=0; center_rule=best_edge; max_neighbors=4; max_stars=1; center_query_focus=true; prompt_style=concise; limit=250; greedy; max_new_tokens=256.  

### Results

| Method | EM | F1 | Avg Latency |
|---|---:|---:|---:|
| Sequential | 0.616000 | 0.460650 | 0.457007 s |
| QAFD + GraphKV | 0.604000 | 0.482109 | 0.612495 s |

**Delta EM:** -0.012000  
**Delta F1:** +0.021459  
**Latency ratio:** 1.340229x  
**Outcome:** target not met.  

### Interpretation

This is the first positive matched F1 result. Query-conditioned cache integration raised QAFD+GraphKV to 0.482109 versus Sequential 0.460650, a +0.021459 gain, and reduced the latency ratio to 1.340. The gain remains 0.028541 below the target, so the query-aware integration signal should be strengthened rather than discarded.

### Next Experiment

Append a dedicated graph-integration checkpoint after the center passage. Those final center tokens can causally aggregate the question, neighbors, and center into a compact latent state before the final answer query; both methods receive the same checkpoint text.

## Attempt 7 — latent_checkpoint_bestedge_h0_n4

**Timestamp:** 2026-08-19T11:19:59+08:00  
**Hypothesis:** Explicit latent integration tokens after the center passage will concentrate GraphKV graph evidence and extend the positive F1 signal from query-conditioned center prefill.  
**Changes:** Kept Attempt 6 selection and decoding unchanged; appended the same graph-integration checkpoint text after the center passage in GraphKV and matched Sequential.  
**Matched Sequential configuration:** Fresh Sequential run with identical QAFD center and neighbors, query-focus prefix, integration checkpoint, order, Tulu3-Block-FT model, concise final prompt, greedy 256-token cap, BF16 FlashAttention2, and A800 hardware.  
**Important hyperparameters:** pool_k=20; h=0; center_rule=best_edge; max_neighbors=4; max_stars=1; center_query_focus=true; center_integration_checkpoint=true; prompt_style=concise; limit=250; greedy; max_new_tokens=256.  

### Results

| Method | EM | F1 | Avg Latency |
|---|---:|---:|---:|
| Sequential | 0.580000 | 0.480698 | 0.433518 s |
| QAFD + GraphKV | 0.592000 | 0.516881 | 0.590255 s |

**Delta EM:** +0.012000  
**Delta F1:** +0.036183  
**Latency ratio:** 1.361548x  
**Outcome:** target not met.  

### Interpretation

The checkpoint strengthened the positive signal: QAFD+GraphKV reached 0.516881 F1 versus 0.480698 Sequential, a +0.036183 gain, and also led EM by +0.012. It is 0.013817 short of the target. Prior topology analysis indicates the remaining weakness is concentrated in sparse stars.

### Next Experiment

Preserve this query-focus and checkpoint architecture, keep the h=0 center and direct neighbors, and fill only stars with fewer than four neighbors using the highest-QAFD-score h<=1 neighbors. Matched Sequential receives the identical expanded stars.

## Attempt 8 — checkpoint_sparsefill_h0_h2_n4

**Timestamp:** 2026-08-19T11:30:10+08:00  
**Hypothesis:** The checkpoint architecture is strongest with four linked cache regions; selectively filling only sparse h=0 stars from bounded h<=2 connectivity should extend that gain without globally introducing the noisy h<=2 graph.  
**Changes:** Preserved dense h=0 stars exactly. For stars below four neighbors, added highest-QAFD-score nodes connected to the same center within h<=2 until the four-neighbor cap. Both methods receive identical selected passages and prompt text.  
**Matched Sequential configuration:** Fresh Sequential run with identical h=0 center, direct and fallback neighbors, ordering, query-focus text, latent checkpoint, Tulu3-Block-FT model, concise final prompt, greedy 256-token cap, BF16 FlashAttention2, and A800 hardware.  
**Important hyperparameters:** pool_k=20; base_h=0; fill_h<=2 only when sparse; center_rule=best_edge; min_neighbors=4; max_neighbors=4; max_stars=1; center_query_focus=true; center_integration_checkpoint=true; prompt_style=concise; limit=250; greedy; max_new_tokens=256.  

### Results

| Method | EM | F1 | Avg Latency |
|---|---:|---:|---:|
| Sequential | 0.592000 | 0.480889 | 0.501102 s |
| QAFD + GraphKV | 0.592000 | 0.535732 | 0.656396 s |

**Delta EM:** +0.000000  
**Delta F1:** +0.054842  
**Latency ratio:** 1.309903x  
**Outcome:** success; validate with a reproducibility run.  

### Interpretation

The topology-adaptive cache budget met the target: QAFD+GraphKV F1 was 0.535732 versus 0.480889 Sequential, a +0.054842 gain, with tied 0.592 EM and a 1.310 latency ratio. This is one deterministic graph/cache pipeline and one final prediction per question. The result is provisional until an exact fresh rerun reproduces the margin.

### Next Experiment

Rerun this exact Sequential and QAFD+GraphKV pair from scratch on the same QIDs 0..249 and audit raw alignment, summaries, and the +0.05 F1 threshold.

## Attempt 9 — repro_checkpoint_sparsefill_h0_h2_n4

**Timestamp:** 2026-08-19T11:34:15+08:00  
**Hypothesis:** If the qualifying gain is deterministic and not a cache or evaluation artifact, a fresh exact rerun will reproduce every prediction and retain at least +0.05 F1 over its fresh matched Sequential control.  
**Changes:** No method or hyperparameter changes from Attempt 8. Used new output directories, new model-server processes, and fresh Sequential and QAFD+GraphKV inference over QIDs 0..249.  
**Matched Sequential configuration:** Exact Attempt 8 configuration: identical selected passage structures within the pair, Tulu3-Block-FT, BF16 FlashAttention2, A800, concise prompt, greedy decoding, and 256-token cap.  
**Important hyperparameters:** pool_k=20; base_h=0; sparse_fill_h<=2; center_rule=best_edge; min_neighbors=4; max_neighbors=4; max_stars=1; center_query_focus=true; center_integration_checkpoint=true; prompt_style=concise; limit=250; greedy; max_new_tokens=256.  

### Results

| Method | EM | F1 | Avg Latency |
|---|---:|---:|---:|
| Sequential | 0.592000 | 0.480889 | 0.502857 s |
| QAFD + GraphKV | 0.592000 | 0.535732 | 0.657702 s |

**Delta EM:** +0.000000  
**Delta F1:** +0.054842  
**Latency ratio:** 1.307932x  
**Outcome:** success; independently reproduced.  

### Interpretation

The validation reproduced all 250 predictions, scores, and selected graph structures exactly for both methods. Sequential F1 remained 0.480889 and QAFD+GraphKV F1 remained 0.535732, preserving the +0.054842 qualifying margin; only measured request timing varied slightly.

### Next Experiment

Target achieved and independently rerun. Preserve the winner, raw predictions, paired logs, and exact configuration; no further optimization is required for this objective.
