# Faithful Graph-KV MoreHopQA reproduction

Dataset: `/mnt/beegfs/home/Winston/old_data/qafd-graphkv/artifacts/results/morehop_faithful_reproduction/input/with_human_verification_ascend.jsonl`; selected hop: `3`; rows: `154`.
The protocol uses the official processed MoreHopQA release, ten documents, Tulu3-Block-FT, the upstream MoreHopQA prompt, a 1024-token greedy cap, and the upstream final `Answer:` scorer.

| Method | Questions | Accuracy | Average latency (s) |
|---|---:|---:|---:|
| graphkv_full | 154 | 0.2597 | 4.296 |

## Accuracy by hop

| Method | Hop-1 | Hop-2 | Hop-3 | Hop-4 | Hop-5 |
|---|---:|---:|---:|---:|---:|
| graphkv_full | — | — | 0.2597 | — | — |
