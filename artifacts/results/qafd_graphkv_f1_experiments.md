# Single-pipeline QAFD + GraphKV F1 experiments

## Fixed evaluation contract

- Questions: the same 250 HotpotQA questions, preserved in QID order `0..249`.
- Success criterion: `QAFD+GraphKV F1 - matched Sequential F1 >= 0.05`.
- Every attempt produces one final answer per question from one unified pipeline.
- No ensembles, oracle routing, output unions, majority voting, or ground-truth-dependent behavior.
- Every QAFD+GraphKV run has a Sequential control with the same model, passages, ordering/serialization inputs, prompt, decoding, precision, and evaluation code. Only the graph/cache mechanism under test may differ.
- EM, F1, average request latency, deltas, and latency ratio are recorded.
- Raw predictions remain on the GPU server under `artifacts/results/qafd_graphkv_f1_search/`.

The historical EM-oriented ensemble ledger is separate and is not valid evidence
for this objective.

