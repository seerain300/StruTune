# Decision Notes / Candidate Post-Mortems

## c001 — INVALID (RUNTIME_ERROR on all 5 workloads)

**Result:** `runs/candidates/c001/feedback.json` → `valid:false, passed:0/5`, every workload
`RUNTIME_ERROR`, `max_abs=0.0` (i.e. the solution produced no comparable output → it raised before/at
kernel execution, not a numerical mismatch). Evaluator log carries no Python traceback, and local
`python3` is sandbox-denied, so root cause was inferred by static reasoning, not reproduced.

**Uniform failure across all 5 configs and both GEMM shapes** ⇒ a single defect present on every code
path (not shape/occupancy specific). Ranked suspects in the c001 source:

1. **`tl.math.erf` (highest suspicion).** Used only in the FC1 epilogue (K2), but K2 executes in every
   workload before K3, so an erf compile/lookup failure fails all 5. `tl.math.erf` has moved between
   Triton releases (now often under `libdevice`) and may not resolve in the installed version.
2. **`tl.trans(w)` inside `tl.dot`.** Standard in modern Triton but a possible version-sensitivity.
3. Less likely: K1 constructs (`tl.sum` on 1D, scalar `tl.load` of `src_idx`), or the `min(...)` L2
   swizzle (this matches the official matmul tutorial, so should be fine). Host `_build_src_idx` is
   plain torch/CPU and unlikely to raise.

**Fixes to fold into the next candidate (c002), in priority order — do NOT reuse c001's ID:**
- **Remove the `tl.math.erf` dependency entirely.** Implement exact-erf GELU via the
  Abramowitz–Stegun 7.1.26 polynomial in fp32 (max err ~1.5e-7, well inside atol 0.0014). This is
  version-portable and needs no `tl.math`/`libdevice` import. Keep the reference-matching
  bf16-before-GELU rounding.
- **Eliminate `tl.trans`:** load the weight tile directly as `[BLOCK_K, BLOCK_N]`
  (`w_ptrs = W + offs_k[:,None]*stride_wk + offs_n[None,:]*stride_wn`) and call `tl.dot(a, w)`
  directly. Correctness-first; revisit contiguity for perf later.
- Keep K1 (LN+shuffle), fp32 accumulators, both biases, M-tail masking, host index (option A) as-is.

**Guardrail update:** the static pre-eval checklist must add "no `tl.math.*` / `libdevice` symbols of
uncertain availability; prefer polynomial intrinsics or verified `tl.*` ops." Because the harness gives
no traceback and local execution is blocked, favor the most version-robust constructs and change the
fewest risky primitives per candidate so an invalid result narrows the cause.

**Next action (next turn):** implement **c002** with the two fixes above, run the static checklist,
evaluate once via `./scripts/evaluate_candidate.sh feedback c002`, append its record. c002 becomes the
correctness anchor if it passes all 5.
