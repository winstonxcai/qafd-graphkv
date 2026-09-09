# MoreHopQA topology research log

## 2026-09-10 — Protocol and baseline audit

Objective: identify a reproducible topology/KV-routing improvement over the strongest matched baseline: at least +0.02 accuracy, or matching/beating it with at most 30% of Full passage edges. Report public-evaluator-compatible containment and strict explicit-final-answer containment separately, overall and H1–H5. Significance requires a paired confidence interval excluding zero; exploratory selection alone is not confirmation.

Current source commit: `67647ab`. Main checkout contains unrelated uncommitted work; parallel changes use separate worktrees. No earlier experiment results are overwritten. Existing model snapshot is Tulu3-Block-FT `62eecbdfa4e12821a487e91ed2ef847a85426efe`; GraphKV is `ba01bbcc98fc9a352582dd1e75ab88fa6cac0dfb`.

### Baseline evidence

The 100-question A800 comparison uses identical prompt hashes, ordered documents, and 256-token greedy generation. Top-1: 0.480 public-compatible / 0.390 strict; Sequential: 0.420 / 0.340; Full: 0.420 / 0.320. Its H1–H5 counts are 40/37/14/1/8; H4 cannot support a trend claim. Full engine parity previously passed exactly on ten cases. Full 1118-question historical outputs are being rescored before establishing the overall strongest baseline.

Actual upstream code reencodes all target passages with source caches, including selected source passages themselves, and retains raw plus updated document caches for decoding. Thus with ten passages Full has 100 source-target pairs (90 cross-passage + 10 diagonal); global source Top-1 has 10 (9 cross + 1 diagonal), Top-3 has 30 (27 cross + 3 diagonal). The simplified disjoint bipartite explanation does not describe the implementation exactly.

### First experiment batch

Freeze the existing 100-question calibration manifest. Compare released-last1/last3/Full with BM25 query-selected source k=1 and k=3, query-seeded PPR k=1, random source k=1 seed42, and first-in-released-order source k=1. Preserve document/token order, prompt, cache duplication, self-source convention, model and decoding. Gold answers and annotated hop counts never enter topology selection. These are cheap source-selection tests. PPR is computed over a lexical passage graph; it is not claimed to reproduce QAFD.

Gate new engine against upstream Full and Top-k on A800 before inference. Record actual adjacency, diagonal/cross edges, source token exposure, source choice, latency, configuration/source hashes, complete generated text, both scores. One model per worker; GPU jobs through Slurm only. Compare source selection at matched k before attributing changes to topology. A later degree-preserving randomization and staged propagation comparison remain required.

After the first batch, freeze a candidate and test beyond these 100 questions against matched strongest controls. The remaining MoreHopQA questions have historical baseline predictions, so describe them as a prospective candidate evaluation split, not a never-accessed blind holdout. Keep H1–H5 denominators and paired intervals explicit.

### Parallel workers

Requested model: `gpt-5.6-luna`, reasoning `medium` for all four CLI terminals.

| Role | Branch | Worktree | CLI session |
|---|---|---|---|
| Baseline/control audit | research/baseline-audit | /tmp/qg-baseline-20260910 | 01a0877d-cbea-7370-ac91-ed9923934e2f |
| Query source selection | research/query-topology | /tmp/qg-query-20260910 | 01a0877d-cbea-7ec3-bc0c-efc9bd83d67f |
| PPR/staged paths | research/path-topology | /tmp/qg-path-20260910 | 01a0877d-cbf2-78d0-b51e-5194d1e08f2e |
| General routing engine | research/routing-engine | /tmp/qg-engine-20260910 | 01a0877d-cbea-7b10-9ad9-76b204eda67a |

CLI v0.36.0 rejected Luna before work. Retried with app-bundled v0.153.1 using the same requested model/effort. This is an infrastructure attempt, not a model experiment. SSH/Slurm works; at inspection no a800-debug jobs were running and its four debug GPUs were unallocated. Existing unrelated user job 18492 is out of scope.

### Implementation review and source-selection diagnostics

BM25 and PPR modules implemented by Luna workers; 17 local tests pass across query selection, PPR, staged definitions, reversal, degree swaps, and graph statistics. Lead review tightened PPR convergence/parameter validation and rejected repeated stage ownership. On the frozen 100 questions, BM25 k=1 equals the released last source for 83 questions; PPR k=1 equals it for 10; BM25 and PPR agree for 15. No gold labels were used in these diagnostics.

Raw 256-token full-dataset predictions rescored with the current project scorer: Sequential 505/1118 = 0.4517 public / 415/1118 = 0.3712 strict; Block-RAG 500/1118 = 0.4472 public / 429/1118 = 0.3837 strict; Top-1 499/1118 = 0.4463 public / 428/1118 = 0.3828 strict; Top-3 486/1118 = 0.4347 public / 411/1118 = 0.3676 strict; Full 490/1118 = 0.4383 public / 424/1118 = 0.3792 strict. Source: `artifacts/results/morehop_faithful_reproduction/hop_*/<method>.jsonl`. Historical full-run revision provenance is less complete than the frozen 100-A800 manifest. Do not substitute the 100-question winner for the full-dataset winner.

Lead review rejected the first routing draft before GPU inference: cache indexing confused layers and passages, selected source caches lacked rotary reapplication, and document tokenization lacked upstream trailing newlines/context truncation. Worker is implementing corrections and an executable GPU parity gate. Also returned the baseline audit draft for correction of static unsupported claims; numerical findings must be derived from rows.

### A800 routing screen completed

The executable parity gate passed on A800: the new engine matched upstream/project GraphKV Full, Top-1, and Top-3 generated text and both scorer outputs for all 100 control questions. The routing screen therefore isolates source-topology choices rather than introducing a new cache implementation.

| Method | Public-compatible | Strict-final | Mean latency (s) | Mean block-read edges |
|---|---:|---:|---:|---:|
| released-last1 (exact GraphKV Top-1 control) | 48/100 | 39/100 | 3.688 | 10 |
| query-BM25 k=1 | 47/100 | 39/100 | 3.683 | 10 |
| released-last3 (exact GraphKV Top-3 control) | 43/100 | 35/100 | 3.602 | 30 |
| query-BM25 k=3 | 43/100 | 35/100 | 3.758 | 30 |
| reversed-rank k=1 | 45/100 | 33/100 | 3.629 | 10 |
| random k=1, seed 42 | 42/100 | 36/100 | 3.736 | 10 |
| Full (exact GraphKV Full control) | 42/100 | 32/100 | 3.745 | 100 |
| query-seeded PPR k=1 | 40/100 | 30/100 | 3.645 | 10 |
| sequential | 42/100 | 34/100 | 3.178 | — |

The best public difference against sequential is +0.06 for released-last1 and +0.05 for BM25 k=1. Their 95% paired-bootstrap intervals are [0.00, 0.12] and [-0.01, 0.11], respectively; neither is a statistically significant improvement at this sample size. Released-last1 is not a new method—it is an exact GraphKV Top-1 reproduction. The only non-released candidate that merits confirmation is BM25 k=1, and it must be compared on a predeclared larger split.

The 250-question confirmation manifest is frozen at `manifest_250.json` with seed `20260910:topology-confirmation-250` and hop counts 99/93/35/3/20. A matched sequential baseline and BM25 k=1/k=3, released-last1, PPR k=1, and Full runs are queued or running on A800. The first 100-question screen is not being treated as a final win.

### 250-question confirmation result

The complete matched confirmation falsifies the apparent 100-question improvement. Sequential scored 117/250 = 0.468 public-compatible and 98/250 = 0.392 strict-final. The best tested non-sequential method was query-BM25 k=3 at 113/250 = 0.452 public-compatible and 98/250 = 0.392 strict-final, a public difference of −0.016 with 95% paired-bootstrap interval [−0.080, +0.048]. Other results were released-last1/GraphKV Top-1 110/250 = 0.440, query-BM25 k=1 110/250 = 0.440, PPR k=1 109/250 = 0.436, and Full 102/250 = 0.408. No tested public-accuracy difference excludes zero, and no topology beats sequential on this confirmation.

The corrected sequential run used job 23457 and exactly the frozen 250 IDs. Two earlier sequential attempts were rejected before analysis: job 23392 omitted the QID manifest and evaluated full hop buckets; job 23445 passed the manifest while filtering too early and exited before predictions. Their logs were moved to `sequential_wrong_scope_23392/` and `sequential_failed_manifest_validation_23445/`; they are not part of any score. The evaluator now applies manifest selection before hop filtering, with unit tests passing.

This is a negative result for rank/query/PPR source selection under the current GraphKV cache engine. The next experiment should change the mechanism being tested—e.g. target-specific routing or cache-content manipulation—rather than continue tuning the same global source-set family on this split.

### Target-conditioned routing

Target-conditioned BM25 was then evaluated on the same frozen 250-question split. It selects sources separately for each target using `question + target passage` and does not use answers or hop labels. k=1 scored 120/250 = 0.480 public-compatible and 101/250 = 0.404 strict-final, versus sequential 117/250 = 0.468 and 98/250 = 0.392. The public paired difference was +0.012 with 95% interval [−0.052, +0.076], so this is not significant and does not meet the +0.05 goal. k=3 scored 112/250 = 0.448.

The k=1 graph had 9.06 diagonal edges and only 1.20 cross edges on average, showing that the apparent small gain is mostly a self-routing control rather than substantial cross-passage transfer. A no-self k=1 variant is the next preregistered mechanism check; it will force one non-self source per target while retaining the same lexical, answer-blind scoring.
