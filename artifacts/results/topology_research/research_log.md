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

No new topology accuracy result is available yet.
