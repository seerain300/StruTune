# Draft — L1/053 `gaussian_topk_sparse_activation`

Target: NVIDIA A800 (`sm_80`), Triton primary implementation. Submission is
`solution/solution.py` exposing `run(inputs, target_sparsity)`. Ranking is geomean
speedup over the reference across the 5 fixed feedback workloads; every scored
workload must also pass correctness.

---

## 1. What the operation computes

Signature (from `task/definition.json`):

- `inputs`: `[batch_size, seq_len, intermediate_size]`, **bfloat16**.
- `target_sparsity`: **float32 scalar** in `[0, 1]`.
- `output`: same shape as `inputs`, **bfloat16**.

Reference semantics (exact, from the embedded `reference`):

1. **Early exit:** if `target_sparsity == 0.0`, return `inputs` unchanged (identity,
   same object/dtype, no copy).
2. Upcast: `inputs_f32 = inputs.to(float32)` (bf16→f32 is exact / lossless).
3. Per-row statistics over the **last dim** (`intermediate_size = H`), `keepdim=True`:
   - `inputs_mean = mean(inputs_f32, dim=-1)`  → shape `[B, S, 1]`
   - `inputs_std  = std(inputs_f32, dim=-1, unbiased=False)` → `[B, S, 1]`
     i.e. population std `sqrt( mean((x-mean)^2) )`, divisor = `H` (not `H-1`).
4. `std_multiplier = _ndtri(target_sparsity)` — the inverse standard-normal CDF
   (quantile) of the scalar sparsity, computed in **float32**. This is a **single
   scalar**, independent of the tensor data.
5. `cutoff_threshold = inputs_mean + inputs_std * std_multiplier` → `[B, S, 1]`.
6. `output = relu(inputs_f32 - cutoff_threshold)` (broadcast threshold over H).
7. `output.to(bfloat16)`.

**Structural view.** This is a **row-wise reduction + broadcast elementwise map**:
`M = B·S` independent rows, each of length `H`. Each row needs its own mean and std,
then a per-element `max(0, x - thr_row)`. Exactly the LayerNorm/RMSNorm memory
pattern (reduce over the contiguous last dim, then a fused elementwise pass).

**Key simplification.** `std_multiplier` depends only on the scalar `target_sparsity`.
`_ndtri` is a fixed rational (Abramowitz–Stegun 26.2.23) approximation with 3 regions.
It can be evaluated **once on the host in float32** and passed to the kernel as a
single `float` argument. No per-element quantile work belongs in the kernel.

`_ndtri` sanity (host reasoning, matches the reference formula):
z(0.1) ≈ −1.2816, z(0.2) ≈ −0.8416, z(0.3) ≈ −0.5244. All feedback sparsities are
< 0.5 ⇒ z < 0 ⇒ `thr = mean + std·z` sits **below** the row mean, so the op keeps the
upper `(1−sparsity)` fraction and zeros the lower `sparsity` fraction (consistent with
"sparsity 0.1 = zero ~10%").

---

## 2. Feedback workloads (fixed; do not modify)

| WL | B | S | H (intermediate) | M = B·S | sparsity | in-bytes (bf16) |
|----|---|---|------|---------|----------|-----------------|
| 1  | 16 | 1163 | 8192  | 18608 | 0.1 | ~305 MB |
| 2  | 2  | 293  | 12288 | 586   | 0.3 | ~14 MB  |
| 3  | 1  | 512  | 12288 | 512   | 0.2 | ~13 MB  |
| 4  | 1  | 8192 | 4096  | 8192  | 0.3 | ~67 MB  |
| 5  | 4  | 541  | 8192  | 2164  | 0.3 | ~35 MB  |

Tolerance (all workloads): `max_atol = 1e-5`, `max_rtol = 0.05`. Inputs `random`.

Observations:
- `H ∈ {4096, 8192, 12288}` — **max row length 12288**. `next_pow2(H) ≤ 16384`, so a
  full row can fit in a single Triton block if we want single-read behavior.
- `M` ranges 512 → 18608. Even the smallest (512 rows) gives enough programs to fill
  an A800 (108 SMs) when each program = one row; larger workloads are abundant.
- WL1 dominates total bytes (~305 MB in) and is the main geomean lever.

---

## 3. Constraints & environment

- **Triton-only compute.** PyTorch allowed solely for metadata / launch plumbing and
  host-side scalar (`_ndtri`) evaluation. No Torch/CPU/NumPy/CUDA-extension fallback;
  a failing Triton kernel is invalid and must not be swapped for a fallback.
- Immutable candidates `c001, c002, …`; any meaningful source/config/launch change
  ⇒ new candidate ID. One kernel version over all 5 workloads = one evaluation.
- Budget: 100 evaluations; token soft 1.0M / hard 1.2M. Converge early.
- **No local execution** of CUDA / profiler / nvidia-smi / the raw evaluator / any
  alternate correctness harness. The **only** measurement channel is
  `./scripts/evaluate_candidate.sh feedback cNNN` (trusted launcher → official
  evaluator). Even a host Python sanity run was denied in this sandbox, so all
  pre-evaluation reasoning must be static.
- `final` (12-workload) only with explicit operator approval.

---

## 4. Numerical-risk analysis

Both reference and candidate produce **bf16** output, so the comparison is between two
bf16 tensors under `|a−b| ≤ atol + rtol·|b|` with `atol=1e-5, rtol=5%`.

Risk inventory:

1. **Threshold accuracy near the ReLU boundary (dominant risk).** Elements with
   `x ≈ thr_row` are near zero after ReLU; reference may keep a tiny positive while we
   round to 0 (or vice versa) if our `thr` differs. For such borderline elements the
   allowed tolerance collapses toward `atol=1e-5`. The *fraction* of elements that can
   mismatch scales with the threshold error, so we must reproduce `mean` and `std`
   faithfully. Mitigation: compute stats in **float32** and use a **two-pass variance**
   (mean first, then `sum((x-mean)^2)/H`) rather than the cancellation-prone
   `E[x^2]-E[x]^2` shortcut; this matches torch's stable reduction closely.
2. **Reduction order.** torch uses tree/pairwise reduction; Triton `tl.sum` also uses a
   tree reduction in f32. Residual differences are ~1e-6 relative and vanish after bf16
   rounding. Low risk.
3. **`std_multiplier` mismatch.** Eliminated by replicating the reference `_ndtri`
   formula exactly in float32 on the host (identical constants, same 3-region branch).
   Converting the f32 result through a Python float and back to f32 in the kernel is
   exact. Even a 1-ULP difference in z would perturb `thr` by `std·ε` ≪ tolerance.
4. **Population vs sample std.** Must use divisor `H` (`unbiased=False`). Using `H-1`
   would bias `std` by factor `sqrt(H/(H-1))` ≈ 1 + 1/(2H) — tiny for H≥4096 but
   avoidable; use `H` exactly.
5. **Masked lanes** (when `BLOCK_H > H`): set loaded value `other=0`, force the
   `(x-mean)` term to 0 on masked lanes (via `tl.where`) before squaring so padding
   never pollutes the variance, and store with the row mask so padding is never written.
   `x - thr` on padded lanes can be positive (since `thr<0` here) — harmless because the
   store is masked.
6. **Early-exit path.** For `target_sparsity == 0.0` return `inputs` directly to match
   the reference bit-for-bit (no kernel launch). (No feedback WL uses 0.0, but final /
   robustness may.)

Conclusion: a float32, two-pass, single-row kernel with host-precomputed z should sit
comfortably inside `rtol=5%` for bulk values and satisfy `atol=1e-5` at the boundary.

---

## 5. Performance analysis (roofline)

The op is **memory-bandwidth bound** (a reduction + one elementwise pass; negligible
FLOPs). A800 HBM ≈ 1.6–2.0 TB/s.

**Why the reference is slow (the opportunity):** the Torch reference materializes a
full `float32` copy (`inputs.to(float32)`, 4·MH bytes) and executes several separate
CUDA kernels — upcast, mean, std, subtract, relu, downcast — each streaming the f32
temporary through HBM. Effective traffic is on the order of ~20–30·MH bytes plus
multiple launch/overhead points.

**Fused Triton kernel traffic:**
- Ideal single-read design: read `inputs` bf16 once (2·MH) + write bf16 out once
  (2·MH) = **4·MH bytes**, one launch.
- Two-pass (reload from HBM) design: 3·MH read/write.

So the fused kernel cuts HBM traffic by roughly **5–8×** versus the reference, bounding
the achievable speedup similarly (minus launch/reduction overhead). Target: strong
multi-× geomean, with WL1 (largest) carrying the most weight.

Illustrative lower bound (single-read, ~4·MH at ~1.8 TB/s): WL1 ≈ 305 MB in ⇒ ~610 MB
moved ⇒ ~0.34 ms; the reference should be several× that.

---

## 6. Triton design space

**Parallelization unit.** One program per row (`grid = (M,)`), row = contiguous `H`
bf16 elements at `row*H`. Coalesced, vectorizable (128-bit ⇒ 8 bf16/txn). M is always
large enough to fill the GPU.

**Reduction strategy (mean + var over H):**

- **(A) Single-block, row-resident, two-pass-in-registers.** `BLOCK_H = next_pow2(H)`
  (≤16384). Load the whole row once → compute `mean` → recompute `var` from the *same*
  registers → apply `max(0, x-thr)` → store. **1 read + 1 write** (minimal traffic).
  Risk: register pressure at H=12288 (BLOCK 16384). Mitigate with larger `num_warps`
  (16/32) so per-thread element count and register footprint drop; watch for spills.
- **(B) Blocked two-pass over HBM.** Small `BLOCK` (e.g. 1024/2048), loop over H twice:
  pass 1 accumulates sum & (deferred) variance, pass 2 reloads and applies. **2 reads +
  1 write** (+50% traffic vs A) but low register pressure / high occupancy. Fallback if
  (A) spills badly.
- **(C) Blocked single-pass sum + sumsq (Welford or moments).** 1 read for stats but
  still needs a second read to apply ⇒ same traffic as (B) unless row is cached; and
  `E[x^2]−E[x]^2` risks cancellation (see §4.1). Lower priority.
- **(D) Split-row reduction (multi-program per row + atomics / 2-kernel).** Only useful
  if a single row can't fill the GPU — not our regime (M ≥ 512). Skip.

**Given max H = 12288, design (A) is the primary choice**: it achieves the memory-optimal
1-read/1-write and a numerically clean in-register two-pass. (B) is the fallback for
register pressure.

**Tunables to sweep (later candidates):**
- `num_warps ∈ {4, 8, 16, 32}` (esp. scaling with H to relieve register pressure).
- `num_stages ∈ {1, 2, 3, 4}` (pipeline the load).
- `BLOCK_H` = `next_pow2(H)` per shape (constexpr per workload ⇒ recompiles per H; fine
  — only 3 distinct H values).
- Optional: multiple rows per program for small-H workloads (WL4 H=4096) to amortize.
- `tl.load` with masking `other=0.0`; cast to f32 for reduction; store `.to(bf16)`.
- Consider `tl.sqrt` precision (f32) — adequate.

**Kernel skeleton (conceptual, not final code):**
```
pid  = program_id(0)                 # row in [0, M)
offs = arange(0, BLOCK_H); m = offs < H
x    = load(in_ptr + pid*H + offs, mask=m, other=0.0).to(f32)
mean = sum(x, 0) / H                  # padded lanes are 0 → correct
xc   = where(m, x - mean, 0.0)
var  = sum(xc*xc, 0) / H
thr  = mean + sqrt(var) * Z          # Z = host-precomputed f32 scalar arg
out  = maximum(x - thr, 0.0).to(bf16)
store(out_ptr + pid*H + offs, out, mask=m)
```

Host wrapper: validate/contiguous `inputs`, view as `[M, H]`, allocate bf16 output,
early-return on `target_sparsity==0.0`, compute `Z=_ndtri_f32(target_sparsity)`, launch,
reshape back to `[B,S,H]`.

---

## 7. Validation strategy

Because local execution is disallowed, validation is **static reasoning + the official
evaluator only**:

1. **Static correctness review** before each candidate: dtype path (bf16→f32→bf16),
   divisor `H`, masked-lane handling, `Z` replicated from reference `_ndtri`, early-exit
   parity, shape/stride/contiguity of output.
2. **First candidate = correctness anchor.** Ship the simplest correct design (A) and
   run `./scripts/evaluate_candidate.sh feedback c001`. Require **all 5 pass** before any
   perf tuning. If a workload fails correctness, prioritize the boundary/threshold
   numerics (§4) over speed.
3. **Per-candidate evaluation** captures per-workload pass/fail + speedup; compute the
   geomean over the 5. Record each as one JSON line in `candidates.jsonl` (parent, source
   hash, hypothesis, validation, per-WL result, geomean, decision, cumulative eval count,
   skill usage). Never rewrite prior records.
4. **One change per candidate ID.** Change kernel or launch config ⇒ new ID; never reuse
   an ID for changed source.
5. **Convergence / stop.** Stop when geomean improvement flattens across a few tuning
   candidates, or at budget. Create `SEARCH_COMPLETE` with the reason. `final` only on
   explicit operator approval.

Correctness watch-list to check in evaluator output: WL1 (largest, dominates geomean),
and the H=12288 workloads (WL2/WL3) where design (A)'s register pressure is highest.

---

## 8. Candidate roadmap (tentative; each is its own immutable ID)

- **c001** — Design (A): one row/program, `BLOCK_H=next_pow2(H)`, f32 two-pass stats,
  host `_ndtri` scalar, heuristic `num_warps`. Goal: 5/5 correct + baseline speedup.
- **c002+** — Tune `num_warps`/`num_stages` (relieve register pressure at H=12288;
  pipeline loads).
- **c003+** — If (A) spills at H=12288: design (B) blocked two-pass for large H,
  keep (A) for small H (H-dependent `BLOCK_H`/config heuristic).
- **c004+** — Small-H amortization (rows-per-program) for WL4; vectorization / eviction
  hints; possibly `sum`+`sumsq` single-pass if it stays within tolerance.
- Iterate only while geomean improves; then `SEARCH_COMPLETE`.

---

## 9. Risks & mitigations (summary)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Threshold error near ReLU boundary | correctness (atol 1e-5) | f32 two-pass stats; exact `_ndtri`; divisor H |
| Register spill at H=12288 (design A) | perf regression / occupancy | larger `num_warps`; fallback design (B) |
| Masked-lane pollution of variance | wrong std | `other=0` load, `tl.where` before square, masked store |
| `std_multiplier` mismatch | small threshold bias | replicate reference `_ndtri` in float32 host-side |
| Torch fallback temptation on failure | invalid submission | never fall back; fix Triton numerics/config |
| No local test channel | slow iteration | rigorous static review; spend evals deliberately |

---

## 10. Skill usage plan

- **KernelWiki** — consult for A800/`sm_80` (Ampere) reduction + fused-elementwise
  (LayerNorm-style) Triton idioms, `num_warps`/`num_stages` tuning, vectorized bf16
  load/store, and register-pressure guidance for wide rows. (Note: skill emphasizes
  Blackwell/Hopper; apply only the Ampere-relevant, hardware-agnostic Triton advice.)
- **ncu-report-skill** — targets B200/sm_100 and local profiling, which is disallowed
  here; not used for measurement. May consult only for conceptual bandwidth-bound
  reasoning if needed.

Any skill use will be logged per candidate in `candidates.jsonl`.

---

**Next step (separate turn):** write `docs/plan.md` (executable optimization plan), then
implement `c001`. No code is produced in this draft turn.
