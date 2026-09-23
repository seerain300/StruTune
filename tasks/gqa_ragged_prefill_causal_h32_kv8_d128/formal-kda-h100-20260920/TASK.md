# Task Contract

- Task: optimize official FlashInfer task `gqa_ragged_prefill_causal_h32_kv8_d128` on NVIDIA H100 (`sm_90`).
- Read `task/definition.json` for the exact signature, reference implementation, dtypes, shapes, and tolerances.
- Submission: `solution/solution.py` exposing the required `run(...)` entry point.
- Primary implementation must use Triton. PyTorch is allowed only for tensor metadata and launch plumbing.
- No Torch computational fallback, CPU/NumPy fallback, CUDA-extension fallback, or alternate implementation fallback.
- Feedback evaluation runs the FULL official workload set (coarse: warmup 2 / 10 iterations) from `task/feedback_workloads.jsonl`, so every shape — including boundary cases — is checked during the search.
- One immutable kernel version over the full feedback workload set counts as one candidate evaluation.
- Candidate budget: 100 evaluations. Token soft limit: 16,000,000; normal completion limit: 18,000,000; absolute limit: 20,000,000.
- Token budget includes uncached input, cache creation input, cache-read input, and output tokens.
- Final evaluation: one full 21-workload evaluation for the best valid candidate, only after operator approval.
- Primary ranking metric: geometric mean speedup; every selected workload must pass correctness.
