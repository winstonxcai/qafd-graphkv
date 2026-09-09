# A800 MoreHopQA Topology Screen

Scope: matched 100-question A800 MoreHopQA control; n=100. Question-order SHA-256: `f1797ca933794df4c1893cce1faccd734f1a69fa3970a72a32657b871f4daec5`.

This is a matched 100-question control. Every topology uses the same prompt, model revision, question IDs, released document order, 256-token cap, greedy decoding, and scorer. Retrieval and serialization are excluded from latency.

## Results

| Method | Public correct | Public accuracy | Strict correct | Strict accuracy | Avg latency (s) | Avg block-read edges | Public Δ vs sequential | 95% paired CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| released_last1 | 48/100 | 0.480 | 39/100 | 0.390 | 3.688 | 10.0 | +0.060 | [+0.000, +0.120] |
| query_bm25_k1 | 47/100 | 0.470 | 39/100 | 0.390 | 3.683 | 10.0 | +0.050 | [-0.010, +0.110] |
| reversed_rank_k1 | 45/100 | 0.450 | 33/100 | 0.330 | 3.629 | 10.0 | +0.030 | [-0.060, +0.120] |
| released_last3 | 43/100 | 0.430 | 35/100 | 0.350 | 3.602 | 30.0 | +0.010 | [-0.060, +0.080] |
| query_bm25_k3 | 43/100 | 0.430 | 35/100 | 0.350 | 3.758 | 30.0 | +0.010 | [-0.070, +0.090] |
| sequential | 42/100 | 0.420 | 34/100 | 0.340 | 3.178 | nan | +0.000 | [+0.000, +0.000] |
| random_k1_seed42 | 42/100 | 0.420 | 36/100 | 0.360 | 3.736 | 10.0 | +0.000 | [-0.080, +0.080] |
| full | 42/100 | 0.420 | 32/100 | 0.320 | 3.745 | 100.0 | +0.000 | [-0.070, +0.070] |
| ppr_k1 | 40/100 | 0.400 | 30/100 | 0.300 | 3.645 | 10.0 | -0.020 | [-0.090, +0.050] |

The public scorer is the paper-compatible whole-response answer-containment rule. Strict-final scoring is shown only as a diagnostic and requires an explicit final-answer marker.

## Hop breakdown

| Method | H1 | H2 | H3 | H4 | H5 |
|---|---:|---:|---:|---:|---:|
| released_last1 | 0.725 | 0.324 | 0.214 | 1.000 | 0.375 |
| query_bm25_k1 | 0.725 | 0.297 | 0.214 | 1.000 | 0.375 |
| reversed_rank_k1 | 0.625 | 0.297 | 0.357 | 0.000 | 0.500 |
| released_last3 | 0.725 | 0.216 | 0.214 | 1.000 | 0.250 |
| query_bm25_k3 | 0.700 | 0.189 | 0.286 | 1.000 | 0.375 |
| sequential | 0.675 | 0.243 | 0.143 | 1.000 | 0.375 |
| random_k1_seed42 | 0.600 | 0.270 | 0.286 | 1.000 | 0.375 |
| full | 0.700 | 0.270 | 0.143 | 0.000 | 0.250 |
| ppr_k1 | 0.625 | 0.216 | 0.214 | 1.000 | 0.375 |

## Integrity checks

The parity controls below compare the new routing engine against the existing project GraphKV outputs on the same 100 questions:

- `released_last1` vs `graphkv_top1`: **PASS**; generated text equal=True, prompt hash equal=True, scorer outputs equal=True.
- `released_last3` vs `graphkv_top3`: **PASS**; generated text equal=True, prompt hash equal=True, scorer outputs equal=True.
- `full` vs `graphkv_full`: **PASS**; generated text equal=True, prompt hash equal=True, scorer outputs equal=True.

On this 100-question confirmation, `released_last1` is the best non-sequential result at 0.480, versus sequential at 0.420 (Δ +0.060); this is exploratory until a predeclared replication confirms it.
The released-last-1 control is an exact reproduction of GraphKV Top-1, not a new topology. Confidence intervals are paired over the identical question IDs; none of the tested public-accuracy differences excludes zero.

## Reproduction

```bash
PYTHONPATH=. python -m src.eval.morehop_topology_results \
  --routing-dir artifacts/results/topology_research/routing_a800 \
  --sequential artifacts/results/morehop_controlled_100_a800/sequential/sequential.jsonl \
  --reference-root artifacts/results/morehop_controlled_100_a800 \
  --csv artifacts/results/topology_research/routing_summary.csv \
  --markdown artifacts/results/topology_research/routing_summary.md
```
