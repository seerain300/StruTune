# Task Contract

- Task: optimize official SOL-ExecBench task `L1/002_vae_conv3x3_groupnorm_silu_residual_fused` on NVIDIA A800 (`sm_80`).
- Read `task/definition.json` for the exact signature, reference implementation, dtypes, shapes, and tolerances.
- Submission: `solution/solution.py` exposing the required `run(...)` entry point.
- Primary implementation must use Triton. PyTorch is allowed only for tensor metadata and launch plumbing.
- No Torch computational fallback, CPU/NumPy fallback, CUDA-extension fallback, or alternate implementation fallback.
- Feedback evaluation uses the fixed five official workloads in `task/feedback_workloads.jsonl`.
- One immutable kernel version over all five feedback workloads counts as one candidate evaluation.
- Candidate budget: 100 evaluations. Token budget: managed externally by the operator (timer-windowed continuation phase, since 2026-09-18); the configured numeric limits are non-binding backstops only.
- Token budget includes uncached input, cache creation input, cache-read input, and output tokens.
- Final evaluation: one full 20-workload evaluation for the best valid candidate, only after operator approval.
- Primary ranking metric: geometric mean speedup; every selected workload must pass correctness.
