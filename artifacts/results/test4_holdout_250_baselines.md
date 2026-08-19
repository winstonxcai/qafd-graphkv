# Untouched holdout baseline comparison — QIDs 250–499

This is an independent 250-question holdout. Questions 0–249 were excluded
from this run. The four baseline methods use the same fresh QAFD retrieval
file, k=20, Tulu3-Block-FT, concise prompting, greedy BF16 decoding, and a
256-token cap.

| Method | EM | F1 | Avg latency |
|---|---:|---:|---:|
| Sequential, full top-20 | 0.756 | **0.119818** | 2.060 s |
| Block-RAG, full top-20 | 0.680 | 0.098491 | 2.952 s |
| Original Graph-KV, full top-20 | 0.720 | 0.110086 | 3.078 s |
| Original QAFD, h≤1 ordering | 0.756 | 0.112858 | 3.165 s |

The locked winning sparse-fill pipeline was also run on the same holdout:

| Method | Context operation | EM | F1 | Avg latency |
|---|---|---:|---:|---:|
| Winner’s exact Sequential control | selected four-passage stars | 0.648 | 0.500177 | 0.498 s |
| QAFD+GraphKV sparse-fill winner | h≤0 center + h≤2 fill + checkpoint | 0.640 | 0.529109 | 0.704 s |

The winner improves F1 by `+0.028932` over its exact within-pipeline control
on this untouched slice. It does not beat full-top-20 Sequential on this
holdout, so the result should be reported as a conditional pipeline gain,
not as a universal win over all baselines.

## Provenance

- QAFD retrieval source: `/mnt/beegfs/home/Winston/qafd-graphkv/artifacts/results/test4_holdout_500_qafd_retrieval.json`
- Baseline outputs: `/mnt/beegfs/home/Winston/qafd-graphkv/artifacts/results/test4_holdout_250_baselines/`
- Winner outputs: `/mnt/beegfs/home/Winston/qafd_holdout_winner/`
- Verified JSONL QID range: 250–499 for every method.
