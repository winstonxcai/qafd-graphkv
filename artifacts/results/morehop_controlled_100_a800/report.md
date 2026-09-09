# Matched MoreHopQA topology comparison

Frozen manifest: `artifacts/results/morehop_controlled_100_a800/manifest.json` (`2948f5776290f9587d28fc2a4b6636793bdff897cd5d39681cf88178f01775de`). All methods used the same 100 questions, prompt hashes, document order, model revision, greedy decoding, and 256-token cap.

The paper-compatible and strict-final metrics are separate analyses. They are never combined into one table cell.

All six methods ran with one NVIDIA A800 per worker via Slurm a800-debug job 21061. Parity ran on A800 as job 21060.

| Method | Role | Paper-compatible accuracy | Δ vs Sequential (95% CI) | Strict-final accuracy | Δ vs Sequential (95% CI) | Explicit `Answer:` | Avg latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| sequential | preregistered control | 0.420 | +0.000 [+0.000, +0.000] | 0.340 | +0.000 [+0.000, +0.000] | 0.750 | 3.178 |
| block_rag | preregistered control | 0.390 | -0.030 [-0.110, +0.050] | 0.300 | -0.040 [-0.120, +0.040] | 0.820 | 3.590 |
| graphkv_top1 | preregistered control | 0.480 | +0.060 [+0.000, +0.120] | 0.390 | +0.050 [-0.030, +0.130] | 0.850 | 3.932 |
| graphkv_top2 | exploratory | 0.420 | +0.000 [-0.080, +0.080] | 0.370 | +0.030 [-0.060, +0.120] | 0.850 | 3.901 |
| graphkv_top3 | preregistered control | 0.430 | +0.010 [-0.060, +0.080] | 0.350 | +0.010 [-0.070, +0.090] | 0.800 | 3.863 |
| graphkv_full | preregistered control | 0.420 | +0.000 [-0.070, +0.070] | 0.320 | -0.020 [-0.100, +0.060] | 0.810 | 3.984 |

This 100-question calibration run is an implementation/topology check, not a confirmation result. Confidence intervals are paired question-level bootstrap intervals with 20,000 resamples.

All latency values are directly comparable because every method ran on the same A800 node type under the same Slurm partition.
