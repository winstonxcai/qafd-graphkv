# Matched MoreHopQA topology comparison

Frozen manifest: `artifacts/results/morehop_controlled_100/manifest.json` (`65b2ba28a4aebffbdec9b15aa5c01545f3f016015623119d25a2c0202e1b7268`). All methods used the same 100 questions, prompt hashes, document order, model revision, greedy decoding, and 256-token cap.

The paper-compatible and strict-final metrics are separate analyses. They are never combined into one table cell.

| Method | Paper-compatible accuracy | Δ vs Sequential (95% CI) | Strict-final accuracy | Δ vs Sequential (95% CI) | Explicit `Answer:` | Avg latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| sequential | 0.420 | +0.000 [+0.000, +0.000] | 0.340 | +0.000 [+0.000, +0.000] | 0.750 | 3.213 |
| block_rag | 0.390 | -0.030 [-0.110, +0.050] | 0.300 | -0.040 [-0.120, +0.040] | 0.820 | 3.652 |
| graphkv_top1 | 0.480 | +0.060 [+0.000, +0.120] | 0.390 | +0.050 [-0.030, +0.130] | 0.850 | 3.974 |
| graphkv_top3 | 0.430 | +0.010 [-0.060, +0.080] | 0.350 | +0.010 [-0.070, +0.090] | 0.800 | 3.926 |
| graphkv_full | 0.420 | +0.000 [-0.070, +0.070] | 0.320 | -0.020 [-0.100, +0.060] | 0.810 | 3.961 |

This 100-question calibration run is an implementation/topology check, not a confirmation result. Confidence intervals are paired question-level bootstrap intervals with 20,000 resamples.
