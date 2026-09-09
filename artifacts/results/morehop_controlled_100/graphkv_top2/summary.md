# Faithful Graph-KV MoreHopQA reproduction

Dataset: `/mnt/beegfs/home/Winston/old_data/qafd-graphkv/artifacts/results/morehop_faithful_reproduction/input/with_human_verification_ascend.jsonl`; selected hop: `None`; rows: `100`.
The protocol uses the official processed MoreHopQA release, ten documents, Tulu3-Block-FT, the upstream MoreHopQA prompt, a 256-token greedy cap, and the upstream final `Answer:` scorer.

| Method | Questions | Paper-compatible accuracy | Strict-final accuracy | Explicit-answer rate | Average latency (s) |
|---|---:|---:|---:|---:|---:|
| graphkv_top2 | 100 | 0.4200 | 0.3700 | 0.8500 | 3.962 |

## Accuracy by hop

| Method | Hop-1 | Hop-2 | Hop-3 | Hop-4 | Hop-5 |
|---|---:|---:|---:|---:|---:|
| graphkv_top2 | 0.7000 | 0.2432 | 0.2143 | 1.0000 | 0.1250 |
