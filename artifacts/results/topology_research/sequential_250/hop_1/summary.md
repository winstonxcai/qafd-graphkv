# Faithful Graph-KV MoreHopQA reproduction

Dataset: `/mnt/beegfs/home/Winston/old_data/qafd-graphkv/artifacts/results/morehop_faithful_reproduction/input/with_human_verification_ascend.jsonl`; selected hop: `1`; rows: `99`.
The protocol uses the official processed MoreHopQA release, ten documents, Tulu3-Block-FT, the upstream MoreHopQA prompt, a 256-token greedy cap, and the upstream final `Answer:` scorer.

| Method | Questions | Paper-compatible accuracy | Strict-final accuracy | Explicit-answer rate | Average latency (s) |
|---|---:|---:|---:|---:|---:|
| sequential | 99 | 0.6768 | 0.5960 | 0.8384 | 2.895 |

## Accuracy by hop

| Method | Hop-1 | Hop-2 | Hop-3 | Hop-4 | Hop-5 |
|---|---:|---:|---:|---:|---:|
| sequential | 0.6768 | — | — | — | — |
