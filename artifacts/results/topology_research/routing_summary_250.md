# A800 MoreHopQA Topology Screen

Scope: matched 250-question A800 MoreHopQA control; n=250. Question-order SHA-256: `2d48f5a6303367f79772f47f5391eddce8ae531aaad9c288558a20dfcb5f8a8d`.

This is a matched 250-question control. Every topology uses the same prompt, model revision, question IDs, released document order, 256-token cap, greedy decoding, and scorer. Retrieval and serialization are excluded from latency.

## Results

| Method | Public correct | Public accuracy | Strict correct | Strict accuracy | Avg latency (s) | Avg block-read edges | Public Δ vs sequential | 95% paired CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target_bm25_k1 | 120/250 | 0.480 | 101/250 | 0.404 | 3.922 | 10.3 | +0.012 | [-0.052, +0.076] |
| sequential | 117/250 | 0.468 | 98/250 | 0.392 | 3.151 | nan | +0.000 | [+0.000, +0.000] |
| query_bm25_k3 | 113/250 | 0.452 | 98/250 | 0.392 | 4.150 | 30.8 | -0.016 | [-0.080, +0.048] |
| target_bm25_k3 | 112/250 | 0.448 | 95/250 | 0.380 | 3.915 | 30.8 | -0.020 | [-0.084, +0.044] |
| released_last1 | 110/250 | 0.440 | 94/250 | 0.376 | 3.818 | 10.3 | -0.028 | [-0.092, +0.032] |
| query_bm25_k1 | 110/250 | 0.440 | 93/250 | 0.372 | 4.127 | 10.3 | -0.028 | [-0.092, +0.036] |
| ppr_k1 | 109/250 | 0.436 | 91/250 | 0.364 | 3.760 | 10.3 | -0.032 | [-0.092, +0.028] |
| target_bm25_k1_noself | 104/250 | 0.416 | 91/250 | 0.364 | 3.929 | 10.3 | -0.052 | [-0.112, +0.008] |
| full | 102/250 | 0.408 | 87/250 | 0.348 | 3.910 | 111.8 | -0.060 | [-0.124, +0.000] |

The public scorer is the paper-compatible whole-response answer-containment rule. Strict-final scoring is shown only as a diagnostic and requires an explicit final-answer marker.

## Hop breakdown

| Method | H1 | H2 | H3 | H4 | H5 |
|---|---:|---:|---:|---:|---:|
| target_bm25_k1 | 0.707 | 0.333 | 0.257 | 0.667 | 0.400 |
| sequential | 0.677 | 0.333 | 0.229 | 1.000 | 0.400 |
| query_bm25_k3 | 0.687 | 0.269 | 0.286 | 0.333 | 0.450 |
| target_bm25_k3 | 0.667 | 0.301 | 0.286 | 0.667 | 0.300 |
| released_last1 | 0.677 | 0.290 | 0.286 | 0.333 | 0.250 |
| query_bm25_k1 | 0.687 | 0.258 | 0.314 | 0.333 | 0.300 |
| ppr_k1 | 0.667 | 0.290 | 0.257 | 0.667 | 0.250 |
| target_bm25_k1_noself | 0.636 | 0.312 | 0.171 | 0.333 | 0.250 |
| full | 0.636 | 0.258 | 0.257 | 0.000 | 0.300 |

## Integrity checks

The routing engine parity gate was passed separately on A800; the available stored 100-question reference uses different IDs, so cross-run equivalence is not recomputed in this scope.


On this 250-question confirmation, `target_bm25_k1` is the best non-sequential result at 0.480, versus sequential at 0.468 (Δ +0.012); this is exploratory until a predeclared replication confirms it.
The released-last-1 control is an exact reproduction of GraphKV Top-1, not a new topology. Confidence intervals are paired over the identical question IDs; none of the tested public-accuracy differences excludes zero.

## Reproduction

```bash
PYTHONPATH=. python -m src.eval.morehop_topology_results \
  --routing-dir artifacts/results/topology_research/routing_250 \
  --sequential artifacts/results/topology_research/sequential_250 \
  --reference-root artifacts/results/morehop_controlled_100_a800 \
  --csv artifacts/results/topology_research/routing_summary_250.csv \
  --markdown artifacts/results/topology_research/routing_summary_250.md
```
