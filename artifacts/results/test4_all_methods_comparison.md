# Test 4 — matched all-method comparison on 250 questions

This table is the corrected comparison. Every method uses the same 250 fixed
HotpotQA questions, the same QAFD retrieval results, the same retrieval pool
(`k=20`), Tulu3-Block-FT, concise answer-only prompt, greedy decoding, BF16
FlashAttention2, and a 256-token generation cap. The methods differ only in
how they consume or order the retrieved passages and caches.

| Method | EM | F1 | Avg latency |
|---|---:|---:|---:|
| Sequential, full top-20 | 0.728 | 0.114041 | 1.878 s |
| Block-RAG, full top-20 | 0.640 | 0.091590 | 2.626 s |
| Original Graph-KV, full top-20 | 0.648 | 0.101775 | 3.145 s |
| QAFD ordering, h≤0 | 0.680 | 0.099437 | 3.152 s |
| QAFD ordering, h≤1 | **0.700** | 0.104190 | 3.113 s |
| QAFD ordering, h≤2 | 0.672 | 0.102493 | 3.056 s |
| Recursive QAFD-GraphKV, h≤1, T=2 | 0.176 | 0.056818 | 3.563 s |
| Adaptive recursive QAFD-GraphKV, h≤1, T=min(D,3) | 0.200 | 0.061052 | 4.592 s |

Under this strictly matched environment, Sequential has the best EM and F1.
QAFD h≤1 is the strongest QAFD ordering variant by both metrics. Both
recursive variants remain substantially behind the non-recursive methods.

## Winning sparse-fill pipeline

The winning QAFD+GraphKV sparse-fill pipeline is included for completeness.
It uses the same model, questions, k=20 retrieval pool, concise prompt, and
256-token cap, but its method-specific operation selects a query-conditioned
h≤0 center, fills sparse stars from h≤2 connectivity to four neighbors, and
adds the graph-integration checkpoint. Its exact within-pipeline matched
control is therefore reported separately rather than conflated with the
full-top-20 Sequential row above.

| Method | Context operation | EM | F1 | Avg latency |
|---|---|---:|---:|---:|
| Sequential, winner-pipeline control | identical selected four-passage stars | 0.592 | 0.480889 | 0.501 s |
| QAFD+GraphKV sparse-fill winner | h≤0 center + h≤2 sparse fill + checkpoint | 0.592 | **0.535732** | 0.656 s |

The winner improves F1 by `+0.054842` over its exact within-pipeline control.
That result is a separate structural comparison because the control receives
the winner’s selected four-passage star rather than the full top-20 context.

## Matched-run artifacts

The fresh per-method JSONL and summaries are on the GPU server at:

```text
/mnt/beegfs/home/Winston/qafd-graphkv/artifacts/results/test4_250_winner_matched_methods/
```

The machine-readable summary is
`artifacts/results/test4_all_methods_comparison.csv`.
