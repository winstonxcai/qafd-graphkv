# QAFD–GraphKV: Results and Integrity Report

## Executive summary

This project evaluated QAFD retrieval, GraphKV cache manipulation, recursive
cache propagation, and a new joint-prefill contextual sparse-attention (CSA)
prototype on HotpotQA with Tulu3-Block-FT.

The main conclusion is negative but well-controlled: no single QAFD+GraphKV
cache pipeline has beaten matched sequential full-context inference on the
strict 250-question comparison. Recursive cache propagation substantially
degrades accuracy. The CSA prototype preserves accuracy within the development
margin, but its Python/gathered-SDPA implementation is slower than dense
attention and its selected configuration gives QAFD no influence over routing
(`beta=0`).

There are two useful positive findings, both conditional:

1. A query-conditioned, checkpointed sparse-fill cache pipeline reproduced a
   `+0.054842` F1 gain over its exact restricted four-passage Sequential control
   on 250 questions. It tied that control on accuracy and did not beat full
   top-20 Sequential.
2. The CSA grid's best raw accuracy was `0.685`, compared with `0.680` for
   dense SDPA. The paired interval included zero, and the configuration was
   slower, so this is a development signal rather than a demonstrated win.

The compact machine-readable results are in the linked artifacts below. Raw
prediction JSONL files for GPU experiments remain on the A800 server unless
otherwise noted.

## Evaluation contract

The project used several contracts over time; they must not be merged into one
number.

### Current primary metric

The primary metric for the matched Test 4 and CSA evaluations is GraphKV-style
normalized token-intersection accuracy: lowercase prediction and reference,
remove articles and punctuation, normalize whitespace, and count a question as
correct when the normalized strings share at least one token. Token F1 is
diagnostic only.

### Historical metric

Earlier ledgers labeled a non-strict answer-containment score as `EM`. Those
values are retained for historical traceability but are not strict HotpotQA EM
and are not directly comparable with the current accuracy tables.

### Fairness rules

Matched comparisons keep the question IDs, retrieval outputs, model, prompt,
decoding, precision, and token cap fixed. When a method uses a smaller or
structured context, its Sequential control receives the same selected context.
Such a control is informative, but it is not a replacement for the full
top-20 Sequential baseline.

## Implementation and validation

The repository was initialized with QAFD-RAG and GraphKV as untouched
submodules. Project-owned logic lives under `src/`; no upstream source files
were modified.

### Test 0 — untouched GraphKV

Ten HotpotQA requests were run on one A800 through the official GraphKV server.

| Mode | Examples | Wall time |
|---|---:|---:|
| Vanilla | 10 | 26.13 s |
| GraphKV `gapemp` | 10 | 34.27 s |

The official client did not expose TTFT, and no concurrent VRAM sampler was
running. Those two measurements were therefore not captured and are not
retroactively inferred.

### Test 1 — recursive propagation sanity check

The synthetic three-passage RED/BLUE chain passed:

```text
round-1 P0 distance: 0.0
round-2 P0 distance: 5.9296875
```

This verifies that information cannot reach `P0` after one propagation round,
but can reach it after two rounds.

### Test 2 — QAFD retrieval

The initial 20-question retrieval-only run completed with the prebuilt QAFD KG:

| Metric | Value |
|---|---:|
| Retrieval time | 25.3 s |
| QAFD time | 15.3 s |
| Recall@1 | 0.400 |
| Recall@2 | 0.750 |
| Recall@5 | 0.925 |
| Recall@10 | 0.975 |
| Recall@20 | 1.000 |

The optional OpenAI reranker returned HTTP 401 because no API key was
configured. QAFD fell back to embedding-ranked facts; this validates the local
retrieval path, not an OpenAI-reranker comparison.

For CSA, a fresh 450-question QAFD slice covering QIDs 500–949 was generated
with exactly five passages and entity-flow traces. Recall@5 was `0.8889`.
The artifact is pinned by SHA-256
`1d2d58ed7b85b04cc7f9ec8c085d917106116770e82b0ef0dce9ce7f57849901`.

### Test 3 — topology analysis

On 250 questions with `k=15`, the bounded QAFD passage graph showed the expected
sparsity/connectivity tradeoff:

| Entity-hop bound | Avg edge density | Avg components | Avg diameter | Fully connected |
|---|---:|---:|---:|---:|
| `h <= 0` | 0.245 | 5.50 | 3.11 | 6.4% |
| `h <= 1` | 0.522 | 1.78 | 2.96 | 58.0% |
| `h <= 2` | 0.851 | 1.07 | 1.98 | 94.0% |

`h <= 1` was selected as the structural compromise. `h <= 0` was too often
disconnected; `h <= 2` was close to a clique. The full per-question topology
outputs are in `artifacts/results/test3_topology_h0_250/`,
`test3_topology_h1_250/`, and `test3_topology_h2_250/`.

## Test 4 — matched QAFD/GraphKV comparison

The main strict-accuracy comparison used 250 questions, `k=20`, a concise
matched prompt, Tulu3-Block-FT, greedy BF16 decoding, and a 256-token cap.

| Method | Accuracy | F1 diagnostic | Avg latency |
|---|---:|---:|---:|
| Sequential, full top-20 | **0.836** | 0.095505 | 1.980 s |
| Block-RAG, full top-20 | 0.692 | 0.076250 | 2.576 s |
| Original GraphKV, full top-20 | 0.772 | 0.085841 | 3.262 s |
| QAFD ordering, `h <= 0` | 0.784 | 0.082398 | 3.119 s |
| QAFD ordering, `h <= 1` | **0.804** | 0.087408 | 3.222 s |
| QAFD ordering, `h <= 2` | 0.776 | 0.087003 | 3.090 s |
| Recursive QAFD-GraphKV, `T=2` | 0.384 | 0.048243 | 3.763 s |
| Adaptive recursive QAFD-GraphKV | 0.404 | 0.050646 | 4.252 s |

Sequential full top-20 is the strongest baseline. QAFD `h <= 1` improves over
the other QAFD orderings but remains `0.032` below Sequential. Both recursive
variants fail substantially, showing that the initial cache-propagation
formulation is not semantically reliable even though the synthetic propagation
test passes.

The machine-readable comparison is
[test4_all_methods_comparison.csv](artifacts/results/test4_all_methods_comparison.csv),
with the corresponding explanation in
[test4_all_methods_comparison.md](artifacts/results/test4_all_methods_comparison.md).

The earlier 10-question sparse-recursive smoke test was directionally
misleading: its strongest `h=0, B=2, T=1, MSF` setting scored `0.80` versus
`0.70` for Sequential, but the matched 250-question run scored `0.668` versus
`0.748`. The larger run is the result used for conclusions.

## Cache-manipulation experiments

The corrected 250-question cache-manipulation run used `h=0`, `B=2`, one
round, MSF restoration, and matched controls.

| Method | Accuracy | F1 | Avg latency |
|---|---:|---:|---:|
| Sequential retrieval-order reference | 0.748 | 0.576956 | 0.484 s |
| QAFD `h=0` vanilla full attention | **0.756** | **0.590612** | 0.463 s |
| Canonical raw caches | 0.624 | 0.504169 | 0.982 s |
| Neighbor replacement | 0.680 | 0.545197 | 1.659 s |
| Query residual, alpha=0.25 | 0.632 | 0.491878 | 1.708 s |
| Query residual, alpha=0.5 | 0.680 | 0.551082 | 1.696 s |

The main failure is independently encoding and merging passage caches: raw
caches lose `0.132` accuracy against matched QAFD-order vanilla attention.
Neighbor replacement recovers some of that loss but remains below full
attention. This is evidence against treating independently formed KV caches as
equivalent to joint prefill.

The detailed paired intervals are in
[cache_manipulation_250.md](artifacts/results/cache_manipulation_250.md).

## Best restricted single-pipeline result

The strongest internally matched F1 result came from a query-conditioned
QAFD+GraphKV star with a latent integration checkpoint and sparse fill. It was
independently reproduced on the same 250 questions:

| Method | Accuracy | F1 | Avg latency |
|---|---:|---:|---:|
| Restricted Sequential control | 0.592 | 0.480889 | 0.503 s |
| QAFD+GraphKV sparse-fill pipeline | 0.592 | **0.535732** | 0.658 s |
| Difference | 0.000 | **+0.054842** | 1.308× |

This is a valid conditional gain over the exact selected four-passage control.
It is not a win over full top-20 Sequential, and the restricted context makes
it unsuitable as the project's general baseline claim.

The full attempt history and integrity contract are in
[qafd_graphkv_f1_experiments.md](artifacts/results/qafd_graphkv_f1_experiments.md).

### Historical optimization search

The earlier EM-oriented ledger found a candidate-union ensemble at `0.792`
reported EM versus its original `0.732` Sequential reference. That result
combined several separately generated answers and used the non-strict
containment score, so it is not evidence that one QAFD+GraphKV pipeline beat
Sequential. Against the later matched `k=15` Sequential control (`0.744`), the
same ensemble's margin was `+0.048`, below the requested `+0.05` threshold.
The complete attempt log is
[qafd_graphkv_250_attempts.md](artifacts/results/qafd_graphkv_250_attempts.md).

## Untouched holdout: QIDs 250–499

The independent holdout used the same four baseline methods and a locked
sparse-fill pipeline. The source artifact labels the primary metric `EM`; it is
kept separate from the strict-accuracy table above.

| Method | Reported EM | F1 | Avg latency |
|---|---:|---:|---:|
| Sequential, full top-20 | 0.756 | **0.119818** | 2.060 s |
| Block-RAG, full top-20 | 0.680 | 0.098491 | 2.952 s |
| Original GraphKV, full top-20 | 0.720 | 0.110086 | 3.078 s |
| Original QAFD, `h <= 1` ordering | 0.756 | 0.112858 | 3.165 s |
| Locked sparse-fill pipeline | 0.640 | 0.529109 | 0.704 s |

The sparse-fill pipeline did not beat full-top-20 Sequential. Its F1 gain over
its exact selected-star control was `+0.028932`, which is a conditional
within-pipeline result rather than a universal improvement.

See [test4_holdout_250_baselines.md](artifacts/results/test4_holdout_250_baselines.md).

## Joint-Prefill Soft-Graph CSA

CSA was introduced to preserve cross-passage interaction during the original
prefill rather than reconstructing it from independent caches. The project
implementation:

- uses a question-first segmented prompt with five passages;
- projects joint hidden states to pre-RoPE Q/K/V at every layer;
- routes only to earlier passages, preserving decoder causality;
- uses normalized-token mean pooling per head with GQA mapping;
- combines standardized LLM routing scores with a QAFD `h <= 1` prior;
- provides dense-reference and gathered-block SDPA backends; and
- records spans, scores, routing traces, attended-pair counts, latency, VRAM,
  and provenance for every prediction.

The implementation passed 10/10 server-side tests, a three-passage GPU smoke
test, and B=4 first-token equivalence checks against dense controls on ten real
prompts.

### Development grid

The development slice was QIDs 500–699. The grid tested normalized-token mean
pooling across `beta ∈ {0, 0.25, 0.5, 1, 2}` and `B ∈ {1, 2, 3, 4}`, with
B=4 inference deduplicated. Plain mean pooling was tested at the selected
normalized configuration.

| Configuration | Accuracy | F1 | Avg latency | QK ratio |
|---|---:|---:|---:|---:|
| Dense vanilla SDPA | 0.680 | 0.5872 | 0.253 s | 1.000 |
| Best raw CSA: `B=1, beta=0.25` | **0.685** | 0.5806 | 0.384 s | 0.616 |
| Frozen CSA: `B=2, beta=0` | 0.675 | 0.5763 | 0.376 s | 0.795 |
| Plain-mean ablation: `B=2, beta=0` | 0.665 | 0.5703 | 0.382 s | 0.796 |
| Dense vanilla FlashAttention2 | 0.675 | 0.5866 | 0.273 s | 1.000 |

The frozen configuration was selected among configurations within `0.01` of
the best development accuracy, prioritizing latency, then F1, lower `B`, lower
beta, normalized pooling, and strategy name. It is an LLM-only router because
`beta=0`; QAFD still determines retrieval and passage order, but its graph prior
does not affect the selected routes.

The complete grid is in
[csa_dev_grid.csv](artifacts/results/csa_development/csa_dev_grid.csv) and
[csa_dev_grid.md](artifacts/results/csa_development/csa_dev_grid.md). The
frozen provenance record is in
[csa_frozen_configuration.json](artifacts/results/csa_development/csa_frozen_configuration.json).

### CSA interpretation

The frozen CSA result is within the planned accuracy band of dense attention:
the paired accuracy difference is `-0.005`, with 95% bootstrap interval
`[-0.030, +0.020]`. The interval includes zero. However, the gathered Python
prototype is about 49% slower than dense SDPA. Its theoretical QK-pair count is
lower, but per-layer Python routing and multiple SDPA dispatches dominate at
`k=5`; decoding latency is essentially unchanged because generation remains
dense.

The QAFD prior did not provide a demonstrated benefit. Small beta helped one
configuration, but stronger beta values generally reduced accuracy, and the
latency-selected frozen winner used beta zero. A custom block-sparse kernel,
route sharing across layers, or a larger passage/context regime would be
needed before treating CSA as a speedup.

## Integrity and scope boundaries

- No claim combines the historical containment-style `EM` values with current
  strict accuracy.
- Candidate-union ensembles are reported as ensembles, not as evidence that a
  single QAFD+GraphKV pipeline beats Sequential.
- Four-passage sparse-fill results are compared to their exact four-passage
  controls and are not compared directly with full top-20 results.
- QAFD retrieval can fall back from the unavailable OpenAI reranker to local
  embedding-ranked facts; this is recorded wherever it occurred.
- CSA development predictions were restricted to QIDs 500–699. QIDs 700–949
  remained prediction-free for a future frozen evaluation.
- The frozen CSA code commit was `1d1437f805930ebeea4e99403adbee7fe65ae8d6`;
  the compact report artifacts were committed in `1d3a856`.

## Conclusion

QAFD is a useful retrieval and topology instrument, but the current GraphKV
cache transformations do not preserve the quality of ordinary joint attention.
The strongest evidence is the matched 250-question Test 4 table: full-top-20
Sequential reaches `0.836` accuracy, while recursive QAFD-GraphKV falls to
`0.384–0.404`. Query-conditioned checkpointed cache integration recovers a
useful restricted-context F1 signal, but not a general full-context win.

CSA is the cleanest architectural direction because it preserves interaction
during joint prefill. The first implementation is validated and nearly
accuracy-preserving, but not faster, and its selected router does not use the
QAFD prior. The next credible milestone is an optimized sparse-attention
backend evaluated under the same matched contract—not a new claim based on the
existing Python prototype or on a different restricted context.
