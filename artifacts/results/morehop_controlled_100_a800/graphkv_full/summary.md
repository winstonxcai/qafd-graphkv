# Faithful Graph-KV MoreHopQA reproduction

Dataset: `/mnt/beegfs/home/Winston/old_data/qafd-graphkv/artifacts/results/morehop_faithful_reproduction/input/with_human_verification_ascend.jsonl`; selected hop: `None`; rows: `100`.
The protocol uses the official processed MoreHopQA release, ten documents, Tulu3-Block-FT, the upstream MoreHopQA prompt, a 256-token greedy cap, and the upstream final `Answer:` scorer.

| Method | Questions | Paper-compatible accuracy | Strict-final accuracy | Explicit-answer rate | Average latency (s) |
|---|---:|---:|---:|---:|---:|
| graphkv_full | 100 | 0.4200 | 0.3200 | 0.8100 | 3.984 |

## Accuracy by hop

| Method | Hop-1 | Hop-2 | Hop-3 | Hop-4 | Hop-5 |
|---|---:|---:|---:|---:|---:|
| graphkv_full | 0.7000 | 0.2703 | 0.1429 | 0.0000 | 0.2500 |
