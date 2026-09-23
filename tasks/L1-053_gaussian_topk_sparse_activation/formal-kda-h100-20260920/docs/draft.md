# Draft — L1/053 Gaussian Top-K Sparse Activation (H100, sm_90)

## 0. Purpose of this document

Analyze the operation, constraints, numerical risks, Triton design space, and a
validation strategy **before** writing any code or the executable plan. No
`docs/plan.md` and no `solution/` code are produced this turn.

---

## 1. Operation semantics (exact)

Signature (from `task/definition.json`):

```
run(inputs: bf16[B, S, H], target_sparsity: float32 scalar) -> bf16[B, S, H]
```

`H = intermediate_size` is the last (contiguous) axis. The reference algorithm:

1. **Early exit:** if `target_sparsity == 0.0`, return `inputs` unchanged (same
   bf16 tensor, no copy, no cast).
2. Upcast `inputs` to fp32 → `x` (materializes a full fp32 copy in the reference).
3. Per-row statistics over the **last dim** `H` (`keepdim=True`), shape `[B,S,1]`:
   - `mean = mean(x, dim=-1)`
   - `std  = std(x, dim=-1, unbiased=False)` → population std (divide by `N=H`).
4. `std_multiplier = _ndtri(target_sparsity)` — inverse standard-normal CDF
   (quantile) evaluated at the **scalar** `target_sparsity` via the
   Abramowitz–Stegun 26.2.23 rational approximation embedded in the reference.
   **Because `target_sparsity` is a scalar, `std_multiplier` is a single number
   shared by every row.**
5. `cutoff = mean + std * std_multiplier`  (shape `[B,S,1]`, broadcast).
6. `output = relu(x - cutoff)` in fp32, then cast to bf16 (round-to-nearest-even).

So per element: `out[b,s,h] = max(0, x[b,s,h] - (mean_row + std_row * z))`, where
`z = ndtri(target_sparsity)` is a global scalar and `mean_row`, `std_row` are the
row's fp32 mean/std computed from the fp32-upcast bf16 values.

### Sign/magnitude of `z` for the feedback set
`_ndtri` regions: `p_low = 0.02425`, `p_high = 0.97575`. All feedback sparsities
(0.1, 0.2, 0.3) lie in the **central region**, so the central rational branch is
used. Reference-consistent values:
- `ndtri(0.1) ≈ -1.2816`, `ndtri(0.2) ≈ -0.8416`, `ndtri(0.3) ≈ -0.5244`.

All are negative ⇒ `cutoff = mean − |z|·std < mean`. Higher target sparsity ⇒
larger (less negative) `z` ⇒ higher cutoff ⇒ more elements zeroed. Consistent.

---

## 2. Key observations that shape the design

- **`std_multiplier` can be computed on the host.** It depends only on the scalar
  `target_sparsity`, not on tensor data. I will port the *documented* A&S formula
  (this is the operation's own math, not evaluator code) to host Python, evaluate
  it once per call in fp64, and pass the resulting fp32 scalar into the kernel.
  This removes the entire rational-polynomial + branch logic from the GPU kernel.
- **The reduction axis is the contiguous last dim.** Each row of `H` elements is
  contiguous ⇒ natural coalesced access if consecutive lanes map to consecutive
  `h`. Treat the tensor as `[M, H]` with `M = B*S` independent rows.
- **This is a bandwidth-bound reduction+elementwise op.** Compute is trivial
  (a mean, a variance, a subtract, a relu). Runtime is dominated by HBM traffic.
  The reference incurs *extra* traffic: an fp32 materialization of the whole
  tensor plus multiple separate CUDA kernels (`to`, `mean`, `std`, `sub`, `relu`,
  `to`). A single fused Triton kernel that reads bf16 directly and writes bf16
  once should win primarily by eliminating that traffic and those launches.
- **bf16 I/O, fp32 math.** Load bf16 → convert to fp32, accumulate/threshold in
  fp32, store bf16. This mirrors the reference exactly (which upcasts before
  computing stats and downcasts only at the end).
- **All feedback `H` are multiples of 4096** (4096, 8192, 12288, 16384). A
  power-of-two `BLOCK_H ∈ {256,512,1024,2048,4096}` divides every `H` exactly ⇒
  the inner tile loop needs no `H`-mask on this set (keep a mask anyway for
  generality; the cost is negligible).
- **`target_sparsity == 0.0` never appears** in the 12 feedback workloads (only
  0.1/0.2/0.3), but I will still implement the early-return for parity/safety.

---

## 3. Workload characterization (feedback == full 12-workload set)

`M = B*S` rows, each of length `H`. Input+output are bf16 (2 B/elem).

| # | B | S | H | M=B·S | elems | in+out (bf16) | sparsity |
|---|---|---|---|-------|-------|---------------|----------|
| 1 | 1 | 512 | 12288 | 512 | 6.3M | ~25 MB | 0.2 |
| 2 | 4 | 2048 | 12288 | 8192 | 100.7M | ~403 MB | 0.1 |
| 3 | 32 | 128 | 12288 | 4096 | 50.3M | ~201 MB | 0.3 |
| 4 | 2 | 211 | 8192 | 422 | 3.5M | ~14 MB | 0.2 |
| 5 | 1 | 8192 | 4096 | 8192 | 33.6M | ~134 MB | 0.3 |
| 6 | 1 | 1024 | 16384 | 1024 | 16.8M | ~67 MB | 0.3 |
| 7 | 16 | 1163 | 8192 | 18608 | 152.5M | ~610 MB | 0.1 |
| 8 | 4 | 541 | 8192 | 2164 | 17.7M | ~71 MB | 0.3 |
| 9 | 4 | 449 | 4096 | 1796 | 7.4M | ~29 MB | 0.2 |
| 10 | 64 | 1024 | 8192 | 65536 | 536.9M | ~2.1 GB | 0.3 |
| 11 | 2 | 131 | 4096 | 262 | 1.1M | ~4 MB | 0.1 |
| 12 | 2 | 293 | 12288 | 586 | 7.2M | ~29 MB | 0.3 |

Observations:
- **Large / bandwidth-dominated:** #10 (2.1 GB), #7, #2, #3, #5. These decide the
  geomean's high end; achieving near-peak HBM bandwidth matters most here.
- **Small row counts:** #11 (M=262), #4 (422), #1 (512), #12 (586). With
  one-program-per-row these barely exceed the 132 SMs — occupancy/latency risk.
  Each program still does heavy per-row work (H=4096–12288), so warps have work,
  but a split-row strategy may help.
- **Tiny total volume / launch-overhead bound:** #11 (~4 MB), #4, #9, #12. Here
  the reference's ~6 separate kernel launches dominate; a single fused kernel
  should give a large speedup almost regardless of bandwidth efficiency.
- Every workload has `M ≥ 262 ≥ 132` rows, so a per-row grid always yields ≥ ~2
  waves; parallelism is adequate, the question is per-SM efficiency.

---

## 4. Constraints (from CLAUDE.md / TASK.md)

- **Triton is the primary implementation.** PyTorch only for metadata/launch
  plumbing (shape, reshape, empty output, scalar host math). No Torch/CPU/NumPy/
  CUDA-extension computational fallback — a failing Triton kernel is *invalid*,
  not something to paper over with `torch`.
- **Correctness is judged only by the official evaluator** via
  `./scripts/evaluate_candidate.sh feedback <id>`. I must not run CUDA directly,
  not run `nvidia-smi`, not run the external evaluator by hand, and not build any
  alternate correctness harness. The full 12-workload feedback run = **one**
  candidate evaluation.
- **Immutable candidates:** any meaningful source/config/launch change ⇒ new
  candidate ID; never reuse an ID for changed source; append one JSON record per
  eval to `candidates.jsonl`, never rewrite prior records.
- **Budget:** 100 candidate evaluations; token soft limit 5.0M / normal 6.0M /
  hard 6.5M. Stop at budget or on genuine convergence, then write
  `SEARCH_COMPLETE`. `final` only with explicit operator approval.
- **Profiling** is allowed *only* through the `ncu-report-skill` workflow against
  a profiling harness built inside this workspace, and **never concurrently with
  an evaluation** (a foreign process on the locked GPU ⇒ controller discards the
  measurement, return code 3, wasting one budget slot). One at a time.
- **Isolation:** work only in this workspace; permitted external knowledge =
  `KernelWiki` and `ncu-report-skill` skills only.
- Tolerances (all 12 workloads): `max_atol = 1e-5`, `max_rtol = 0.05`.

---

## 5. Numerical risks & mitigations

1. **Variance formula / catastrophic cancellation.**
   - Naive one-pass: `var = sum(x²)/N − mean²`. One read of the row, cheapest.
     Risks cancellation when `|mean| ≫ std` (subtracting two large near-equal
     fp32 numbers). torch's `std` uses a stable two-pass, so results can diverge.
   - For random inputs centered near 0 (e.g. `randn`), `mean ≈ 0`, no
     cancellation — naive is accurate. For offset/large-magnitude data the naive
     var could lose precision.
   - **Mitigation / decision branch:** start with naive one-pass fp32 (2 total
     reads: stats + apply). If the evaluator flags a correctness failure, switch
     to a stable **two-pass** (mean pass, then `sum((x−mean)²)/N` pass, then apply
     = 3 reads) or **Welford** accumulation. This is a correctness↔bandwidth
     trade recorded explicitly as candidate progression.
2. **`sqrt` of a slightly negative variance.** fp round-off in the naive formula
   can make `var` a tiny negative ⇒ `NaN`. Clamp `var = max(var, 0)` before
   `sqrt`. (The stable two-pass cannot go negative but the clamp is harmless.)
3. **`ndtri` must match the reference.** I re-implement the *identical* A&S
   constants/branches on the host. Evaluating in fp64 then narrowing to fp32
   differs from the reference's fp32 evaluation by ~1e-7 relative; multiplied by
   `std ~ O(1)` this perturbs `cutoff` by ~1e-7 — far below `atol = 1e-5`.
4. **Threshold sensitivity near the relu knee.** For elements just above `cutoff`,
   `x − cutoff` is small, so *absolute* error in `mean/std/z` matters (governed by
   `atol = 1e-5`). fp32 accumulation over H ≤ 16384 bf16 values keeps mean/std
   errors ≪ 1e-5 for well-conditioned (near-zero-mean) data. This reinforces
   preferring fp32 accumulators (never bf16/fp16 accumulation).
5. **bf16 rounding of the output.** Reference `.to(bfloat16)` and Triton
   `.to(tl.bfloat16)` both round-to-nearest-even ⇒ identical last-step rounding;
   `rtol = 0.05` (≈ 12× bf16's ~2⁻⁸ relative granularity) gives ample margin for
   values well above the knee.
6. **Reduction accumulator ordering.** Tiled partial sums summed in fp32 differ
   from torch's reduction tree only in the last bits; irrelevant at these tols.
7. **Empty / constant rows.** If a row is constant, `std = 0`, `cutoff = mean`,
   `relu(x − mean) = 0`. The clamp + fp32 path handle this cleanly (no div/zero;
   there is no division by std anywhere).
8. **Contiguity.** Assume `[B,S,H]` is contiguous (gate_proj output); reshape to
   `[M,H]` for coalesced row access. Guard by passing the row stride to the kernel
   (or `.contiguous()` only if a non-contiguous input is ever observed — avoided
   by default since it adds a full copy of traffic).

---

## 6. Triton design space

### 6.1 Baseline: fused, one program per row, two-pass over H
- **Grid:** `M` programs (one per row). Program `pid` owns row `pid`, base offset
  `pid*H`.
- **Pass 1 (stats):** loop `h` in `BLOCK_H` tiles across `H`; load bf16 → fp32;
  accumulate scalar `acc_sum += tl.sum(tile)` and `acc_sqr += tl.sum(tile*tile)`.
- Compute `mean = acc_sum/H`, `var = max(acc_sqr/H − mean*mean, 0)`,
  `std = sqrt(var)`, `cutoff = mean + std*z` (`z` passed in as fp32 scalar).
- **Pass 2 (apply):** loop tiles again; `out = max(x_f32 − cutoff, 0)`; store bf16.
- **Traffic:** 2× read + 1× write. The row just read in pass 1 is likely still
  resident in L2 (50 MB) when pass 2 re-reads it, so effective *HBM* re-read cost
  is often well below a full 1×. This is the simplest correct design → **c001**.

### 6.2 Whole-row caching (single HBM read)
Hold the entire row in registers with `BLOCK_H = next_pow2(H)` (single masked
tile), compute stats, apply, store — 1× read + 1× write. **Problem:** for
`H = 16384` (and even 8192/12288) a `[BLOCK_H]` register array is enormous ⇒
Triton spills to local (global) memory, re-introducing traffic and killing
occupancy. Viable only for small `H`; not attractive for this set. Consider only
if profiling shows pass-2 re-reads actually hit HBM (not L2).

### 6.3 Two-kernel split (reduction kernel + elementwise kernel)
- Kernel A: per-row `mean`,`std` → write `cutoff[M]` (tiny). Kernel B: fully
  elementwise `out = relu(x − cutoff[row])`. Total 2× read + 1× write, same as
  §6.1, but Kernel B parallelizes perfectly regardless of `M`, and Kernel A can be
  split across many blocks (atomics or a two-stage reduction) for small-`M` cases.
- Cost: extra launch + a global round-trip of the tiny `cutoff` vector. Useful
  fallback for the small-row workloads (#11/#4/#1/#12) if §6.1 underutilizes.

### 6.4 Split-row reduction for small M (atomics / two-stage)
For `M` near the SM count, split each row across `P` programs computing partial
`(sum, sqr)` into a scratch `[M, 2]`, `atomic_add`, then a second stage finalizes
`cutoff` and applies. Increases occupancy for #11/#4/#1/#12. Adds complexity and a
scratch buffer; only pursue if profiling of §6.1 confirms these shapes are
latency/occupancy-limited rather than launch-overhead-limited.

### 6.5 Tuning axes (autotune candidates)
- `BLOCK_H` ∈ {256, 512, 1024, 2048, 4096} (all divide every feedback `H`).
- `num_warps` ∈ {4, 8, 16} — more warps per row → more in-flight loads for the
  large-`H`, bandwidth-bound shapes.
- `num_stages` ∈ {2, 3, 4} — software pipelining of the tile-load loop to overlap
  HBM latency; important for the sequential per-row loop.
- Vectorized/aligned loads: `H` multiples of 4096 and bf16 give naturally aligned
  128-bit accesses; ensure the tile access pattern is contiguous per lane.
- Possibly a heuristic that switches design (§6.1 vs §6.3/§6.4) on `M` vs a
  threshold (e.g. `M < 4 * num_SMs`).
- Hopper-specific options to evaluate via **KernelWiki** (TMA loads, `cp.async`
  pipelining, L2 residency hints) — only if simple tiling leaves bandwidth on the
  table.

### 6.6 Rough performance model (roofline)
H100 SXM HBM3 ≈ 3.35 TB/s. For #10 (~2.1 GB min traffic at 1R+1W): lower bound
≈ 2.1 GB / 3.35 TB/s ≈ 0.63 ms; the §6.1 two-pass (≤ 3× traffic if all re-reads
miss L2) ≈ ≤ 0.95 ms. The reference moves substantially more (fp32
materialization ≈ +2× the tensor written and re-read, plus separate mean/std/relu
passes) and pays ~6 launch latencies, so a **2–4×** geomean speedup looks
plausible, with the largest gains on the tiny launch-overhead-bound shapes.

---

## 7. Validation strategy

Because I may **not** build an alternate correctness harness or run CUDA directly,
correctness is established by (a) careful analytical reasoning and (b) the official
evaluator as the sole ground truth.

1. **Analytical pre-checks (before each eval, to conserve budget):**
   - Confirm kernel math reproduces the reference expression element-wise:
     fp32 upcast, population variance (`/H`), `cutoff = mean + std·z`, relu, bf16
     RNE downcast.
   - Confirm the host `ndtri` port reproduces the A&S constants/branches exactly
     and that 0.1/0.2/0.3 hit the central branch with the expected z-values.
   - Confirm `BLOCK_H | H` for all feedback `H` (no silent mask bugs); confirm the
     `var = max(·,0)` clamp; confirm fp32 accumulators everywhere.
   - Confirm reshape assumes contiguity and the grid covers all `M` rows.
2. **Official evaluation:** run `./scripts/evaluate_candidate.sh feedback c001`
   once the baseline is implemented. The evaluator reports per-workload
   correctness (atol/rtol) and timing; the full 12-workload run is one budget
   unit. Record parent, source hash, hypothesis, per-workload pass/fail + speedup,
   geomean, decision, cumulative eval count, and skill usage into
   `candidates.jsonl` (append-only).
3. **Correctness branch:** if any workload fails the tolerance, do not fall back to
   torch. Diagnose analytically (most likely the variance formula, the sqrt clamp,
   the z-value, or a rounding/stride issue) and spin a **new** candidate with the
   stable two-pass/Welford variance (§5.1) or fixed indexing.
4. **Performance branch (ncu, never during an eval):** if a candidate is correct
   but slow, build an in-workspace profiling harness and use the
   `ncu-report-skill` workflow to measure achieved DRAM throughput, L2 hit rate on
   the pass-2 re-read, occupancy, and warp-stall reasons. Use findings to choose
   among §6.1/§6.2/§6.3/§6.4 and the §6.5 knobs. Ensure no evaluation is running
   before launching `ncu` (and vice versa).
5. **Convergence:** stop when successive candidates no longer improve geomean
   meaningfully (or on budget), then write `SEARCH_COMPLETE` with the reason.
   Never invoke `final` without explicit operator approval.

---

## 8. Open questions / to resolve during search

- **Input distribution** ("random") — determines whether naive one-pass variance
  is safe. Treated as a c001→c002 branch rather than an assumption; validated by
  the evaluator, since inspecting the generator is out of scope.
- **Does the pass-2 re-read hit L2 or HBM** for the large shapes (#10/#7/#2)? This
  decides whether §6.2/§6.3 single-read designs are worth the complexity —
  answerable only via ncu profiling.
- **Small-M occupancy** (#11/#4/#1/#12): launch-overhead-bound (then §6.1 already
  wins) vs bandwidth/occupancy-bound (then §6.3/§6.4). Decide from ncu, not a
  priori.
- **Best `(BLOCK_H, num_warps, num_stages)`** per shape — autotune vs a small
  hand-picked set keyed on `H`/`M`.

## 9. Planned first candidate (for the plan, not implemented this turn)
`c001` = §6.1 fused one-program-per-row, naive one-pass fp32 stats, host-computed
`z`, `var` clamp, bf16 RNE store, a modest `BLOCK_H`/`num_warps`/`num_stages`
starting point, early-return on `target_sparsity == 0.0`. Establish correctness +
a baseline geomean, then iterate per §6/§7.
