# Faithful Graph-KV MoreHopQA reproduction

Dataset: `/mnt/beegfs/home/Winston/old_data/qafd-graphkv/artifacts/results/morehop_faithful_reproduction/input/with_human_verification_ascend.jsonl`; selected hop: `4`; rows: `3`.
The protocol uses the official processed MoreHopQA release, ten documents, Tulu3-Block-FT, the upstream MoreHopQA prompt, a 256-token greedy cap, and the upstream final `Answer:` scorer.

| Method | Questions | Paper-compatible accuracy | Strict-final accuracy | Explicit-answer rate | Average latency (s) |
|---|---:|---:|---:|---:|---:|
| sequential | 3 | 1.0000 | 0.6667 | 0.6667 | 4.691 |

## Accuracy by hop

| Method | Hop-1 | Hop-2 | Hop-3 | Hop-4 | Hop-5 |
|---|---:|---:|---:|---:|---:|
| sequential | — | — | — | 1.0000 | — |
