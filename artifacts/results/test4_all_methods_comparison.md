# Test 4 — all-method comparison on 250 questions

The canonical table uses the same 250 fixed HotpotQA questions and the same
top-15 QAFD passages for every method. It uses the original Test 4 default
prompt and the GraphKV server configuration used by the existing 250-question
run. Recursive rows use the project recursive endpoint with 128 generated
tokens; the other GraphKV rows use the official server endpoint.

| Method | EM | F1 | Avg latency |
|---|---:|---:|---:|
| Sequential | 0.732 | 0.109042 | 2.062 s |
| Block-RAG | 0.640 | 0.090980 | 5.513 s |
| Original Graph-KV | 0.664 | 0.089655 | 2.877 s |
| QAFD ordering, h≤0 | 0.692 | 0.092156 | 2.707 s |
| QAFD ordering, h≤1 | 0.684 | 0.092888 | 2.598 s |
| QAFD ordering, h≤2 | 0.684 | 0.091143 | 2.711 s |
| Recursive QAFD-GraphKV, T=2 | 0.192 | 0.056391 | 2.993 s |
| Adaptive recursive QAFD-GraphKV, T=min(D,3) | 0.204 | 0.057100 | 4.132 s |

Under this canonical configuration, Sequential is the strongest baseline by
both EM and F1. The recursive implementations are currently substantially
behind the non-recursive methods, so they should not be presented as the
winning result.

## Separately matched winning experiment

The winning unified QAFD+GraphKV pipeline was intentionally tuned under a
different contract: retrieval pool 20, concise answer-only prompting, a
query-conditioned h≤0 center, h≤2 sparse-fill to four passages, and the
latent integration checkpoint. Its matched Sequential comparison is included
below, but these rows must not be ranked as if they were the same
configuration as the canonical table.

| Method | k | EM | F1 | Avg latency |
|---|---:|---:|---:|---:|
| Sequential, winner-matched | 20 | 0.592 | 0.480889 | 0.501 s |
| QAFD+GraphKV sparse-fill winner | 20 | 0.592 | **0.535732** | 0.656 s |

The winning pipeline improves F1 by `+0.054842` over its matched Sequential
baseline on the same 250 questions. The exact reproducible experiment is
recorded in `artifacts/results/qafd_graphkv_f1_experiments.md`.

## Raw artifact locations

Canonical rows:

```text
/mnt/beegfs/home/Winston/qafd-graphkv/artifacts/results/test4_250_k15_parallel_rerun/
```

Adaptive row:

```text
/mnt/beegfs/home/Winston/qafd-graphkv/artifacts/results/test4_250_k15_all_methods/qafd_adaptive_recursive_h1/
```

The machine-readable version of this table is
`artifacts/results/test4_all_methods_comparison.csv`.
