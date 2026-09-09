# GraphKV Full engine parity

**Decision: PASS**

The untouched upstream `pcw.gapemp` cache-construction path is compared with the project-owned Full implementation on two deterministic MoreHopQA questions per hop bucket.

| Check | Result | Gate |
|---|---:|---:|
| Identical token IDs | 10/10 | 10/10 |
| Matching cache shapes | 10/10 | 10/10 |
| Matching first token | 10/10 | 10/10 |
| Matching complete greedy output | 10/10 | 10/10 |
| Maximum relative logit RMS | 0 | ≤0.01 |
| Maximum relative cache RMS | 0 | diagnostic |

Divergent QIDs: `[]`.
