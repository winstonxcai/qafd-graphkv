# Joint-Prefill Soft-Graph CSA Development Results

Generated: 2026-08-25T12:42:54.424793+00:00

Only HotpotQA QIDs 500-699 are included. QIDs 700-949 remain prediction-free.

| Method | Pooling | beta | B | Accuracy | F1 | Avg latency (s) | QK ratio | Delta vs dense | 95% paired CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_vanilla_sdpa | not_applicable | 0 | 4 | 0.6800 | 0.5872 | 0.2530 | 1.0000 | +0.0000 | [+0.0000, +0.0000] |
| dense_vanilla_flash2 | not_applicable | 0 | 4 | 0.6750 | 0.5866 | 0.2728 | 1.0000 | -0.0050 | [-0.0150, +0.0000] |
| csa_norm_b1_beta0 | normalized_token_mean | 0 | 1 | 0.6750 | 0.5732 | 0.3785 | 0.6157 | -0.0050 | [-0.0400, +0.0300] |
| csa_norm_b1_beta0p25 | normalized_token_mean | 0.25 | 1 | 0.6850 | 0.5806 | 0.3841 | 0.6157 | +0.0050 | [-0.0300, +0.0400] |
| csa_norm_b1_beta0p5 | normalized_token_mean | 0.5 | 1 | 0.6750 | 0.5737 | 0.3831 | 0.6163 | -0.0050 | [-0.0400, +0.0300] |
| csa_norm_b1_beta1 | normalized_token_mean | 1 | 1 | 0.6800 | 0.5738 | 0.3872 | 0.6212 | +0.0000 | [-0.0350, +0.0350] |
| csa_norm_b1_beta2 | normalized_token_mean | 2 | 1 | 0.6600 | 0.5655 | 0.3866 | 0.6256 | -0.0200 | [-0.0550, +0.0150] |
| csa_norm_b2_beta0 | normalized_token_mean | 0 | 2 | 0.6750 | 0.5763 | 0.3762 | 0.7955 | -0.0050 | [-0.0300, +0.0200] |
| csa_norm_b2_beta0p25 | normalized_token_mean | 0.25 | 2 | 0.6750 | 0.5747 | 0.3785 | 0.7951 | -0.0050 | [-0.0300, +0.0200] |
| csa_norm_b2_beta0p5 | normalized_token_mean | 0.5 | 2 | 0.6800 | 0.5794 | 0.3830 | 0.7950 | +0.0000 | [-0.0250, +0.0250] |
| csa_norm_b2_beta1 | normalized_token_mean | 1 | 2 | 0.6500 | 0.5629 | 0.3836 | 0.7967 | -0.0300 | [-0.0550, -0.0100] |
| csa_norm_b2_beta2 | normalized_token_mean | 2 | 2 | 0.6550 | 0.5709 | 0.3819 | 0.8011 | -0.0250 | [-0.0500, -0.0050] |
| csa_norm_b3_beta0 | normalized_token_mean | 0 | 3 | 0.6750 | 0.5853 | 0.3901 | 0.9282 | -0.0050 | [-0.0300, +0.0150] |
| csa_norm_b3_beta0p25 | normalized_token_mean | 0.25 | 3 | 0.6750 | 0.5839 | 0.3851 | 0.9275 | -0.0050 | [-0.0300, +0.0150] |
| csa_norm_b3_beta0p5 | normalized_token_mean | 0.5 | 3 | 0.6750 | 0.5839 | 0.3907 | 0.9274 | -0.0050 | [-0.0300, +0.0150] |
| csa_norm_b3_beta1 | normalized_token_mean | 1 | 3 | 0.6700 | 0.5796 | 0.3784 | 0.9284 | -0.0100 | [-0.0300, +0.0100] |
| csa_norm_b3_beta2 | normalized_token_mean | 2 | 3 | 0.6700 | 0.5825 | 0.3803 | 0.9299 | -0.0100 | [-0.0300, +0.0100] |
| csa_norm_b4_beta0 | normalized_token_mean | 0 | 4 | 0.6750 | 0.5856 | 0.3802 | 1.0000 | -0.0050 | [-0.0250, +0.0100] |
| csa_norm_b4_beta0p25 | normalized_token_mean | 0.25 | 4 | 0.6750 | 0.5856 | 0.3802 | 1.0000 | -0.0050 | [-0.0250, +0.0100] |
| csa_norm_b4_beta0p5 | normalized_token_mean | 0.5 | 4 | 0.6750 | 0.5856 | 0.3802 | 1.0000 | -0.0050 | [-0.0250, +0.0100] |
| csa_norm_b4_beta1 | normalized_token_mean | 1 | 4 | 0.6750 | 0.5856 | 0.3802 | 1.0000 | -0.0050 | [-0.0250, +0.0100] |
| csa_norm_b4_beta2 | normalized_token_mean | 2 | 4 | 0.6750 | 0.5856 | 0.3802 | 1.0000 | -0.0050 | [-0.0250, +0.0100] |
| csa_plain_b2_beta0 | plain_mean | 0 | 2 | 0.6650 | 0.5703 | 0.3815 | 0.7958 | -0.0150 | [-0.0400, +0.0050] |

Selected configuration: `csa_norm_b2_beta0` (accuracy=0.6750, average latency=0.3762s).

Selection rule: within 0.01 of maximum development accuracy, then lowest average latency, higher F1, lower B, lower beta, normalized-token pooling, and strategy name.

Prototype criterion: **not yet promising**. It must beat dense accuracy or remain within 0.01 while reducing measured latency.
