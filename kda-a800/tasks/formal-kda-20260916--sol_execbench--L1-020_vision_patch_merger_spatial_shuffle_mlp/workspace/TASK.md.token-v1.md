# Task Contract

- Task: optimize official SOL-ExecBench task `L1/020_vision_patch_merger_spatial_shuffle_mlp` on NVIDIA A800 (`sm_80`).
- Read `task/definition.json` for the exact signature, reference implementation, dtypes, shapes, and tolerances.
- Submission: `solution/solution.py` exposing the required `run(...)` entry point.
- Primary implementation must use Triton. PyTorch is allowed only for tensor metadata and launch plumbing.
- No Torch computational fallback, CPU/NumPy fallback, CUDA-extension fallback, or alternate implementation fallback.
- Feedback evaluation uses the fixed five official workloads in `task/feedback_workloads.jsonl`.
- One immutable kernel version over all five feedback workloads counts as one candidate evaluation.
- Candidate budget: 100 evaluations. Token soft limit: 1,000,000; normal completion limit: 1,500,000; absolute limit: 1,650,000.
- Token budget includes uncached input, cache creation input, cache-read input, and output tokens.
- Final evaluation: one full 15-workload evaluation for the best valid candidate, only after operator approval.
- Primary ranking metric: geometric mean speedup; every selected workload must pass correctness.
