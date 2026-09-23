# Draft: L1/020 Vision Patch Merger (Spatial Shuffle + LayerNorm + 2-layer GELU MLP)

Target HW: NVIDIA H100 (`sm_90`). Primary metric: geometric-mean speedup vs the PyTorch
reference over the full 15-workload feedback set, with every selected workload passing
correctness. Implementation must be Triton (PyTorch only for metadata / launch plumbing;
no Torch/CPU/NumPy computational fallback).

---

## 1. Operation analysis

### 1.1 Reference semantics (from `task/definition.json`)

Constants: `hidden_size C = 1536`, `merge_size = 2`, `hidden_size_expanded E = 6144`
(`= 4*C = merge^2 * C`), `out_hidden_size O = 3584`, `eps = 1e-6`.

Inputs:
- `hidden`           `[N, 1536]` bf16      — `N = num_patches`.
- `grid_thw`         `[G, 3]` int64        — `G = num_grids`, rows `[T, H, W]`.
- `ln_weight`        `[1536]` bf16
- `ln_bias`          `[1536]` bf16
- `fc1_weight`       `[6144, 6144]` bf16   — Linear weight, i.e. `y = x @ Wᵀ`.
- `fc1_bias`         `[6144]` bf16
- `fc2_weight`       `[3584, 6144]` bf16
- `fc2_bias`         `[3584]` bf16
- `eps`              fp32 scalar

Output: `[M, 3584]` bf16, `M = num_merged_patches = Σ_g T·(H/2)·(W/2)` (`= N/4` for these
workloads since generator makes grids sum to `N`).

Pipeline (reference `run`):
1. **LayerNorm (pre-shuffle, over the 1536 channel dim)**, computed in fp32:
   `mean, var(unbiased=False)` over last dim; `x̂ = (x-mean)/sqrt(var+eps)`;
   `x̂ = x̂*ln_weight + ln_bias` in fp32; **cast back to bf16**. LN is per input patch (row).
2. **Spatial shuffle / 2×2 merge**, per grid, on the bf16 LN output:
   view `(T, H/2, 2, W/2, 2, C)` → permute `(0,1,3,2,4,5)` → `(T, H/2, 2, W/2, 2, C)` order
   `(T,Hm,Wm,a,b,C)` → reshape `(T·Hm·Wm, 4·C=6144)`. Concatenate across grids → `[M, 6144]` bf16.
3. **FC1**: `linear(x, fc1_weight, fc1_bias)` → `[M,6144]` bf16 (bf16 in, fp32 accum in cuBLAS, bf16 out).
4. **GELU**: `F.gelu` default = **exact erf** GELU `0.5·x·(1+erf(x/√2))`, applied to bf16 → bf16.
5. **FC2**: `linear(x, fc2_weight, fc2_bias)` → `[M,3584]` bf16.

### 1.2 Exact shuffle index math (critical for correctness)

Within one grid `(T,H,W)`, input patches are row-major `l = ((t·H + h)·W + w)`.
Decompose `h = hm·2 + a`, `w = wm·2 + b`, `a,b ∈ {0,1}`, `Hm=H/2`, `Wm=W/2`.
The reshape/permute maps this input patch to:
- output row (within grid) `o = (t·Hm + hm)·Wm + wm`
- sub-block `s = 2·a + b ∈ {0,1,2,3}` occupying columns `[s·C, (s+1)·C)` of the 6144 vector.

So each **input** row (global) maps to exactly one `(output_row, sub-block)` → a **bijection**
between the `Σ T·H·W` covered input rows and the `M·4` output slots. Equivalently, define a
scatter offset `dst[r] = o_global·6144 + s·1536` for global input row `r`; then LN(row r) is
written contiguously to `out_shuffled.flat[dst[r] : dst[r]+1536]`. Grid global offsets:
input offset `Σ_{g'<g} T·H·W`, output-row offset `Σ_{g'<g} T·Hm·Wm`.

Note: if `Σ T·H·W < N` (generator dropped remainder), trailing `hidden` rows are unused — must
mirror this by driving everything from `grid_thw`, not from `N`. `M` must be computed from
`grid_thw` too (`Σ T·(H/2)·(W/2)`).

### 1.3 Cost / roofline (why the win exists)

Two GEMMs dominate FLOPs: FC1 `M·6144·6144`, FC2 `M·3584·6144` (fp32-accum bf16).
Weights are fixed and re-read from HBM: fc1 = 75.5 MB, fc2 = 44 MB.

Weight-dominated arithmetic intensity ≈ `M` FLOP/byte. H100 SXM bf16 roofline knee
≈ 990 TFLOP/s ÷ 3.35 TB/s ≈ ~295 FLOP/byte. Classification per workload (by `M`):

| uuid (short) | num_patches | M | regime |
|---|---|---|---|
| 38e61f30 | 512 | 128 | memory-bound (weights), low-occupancy |
| 133134ee | 576 | 144 | memory-bound |
| 9d5b4822 / 9376d72b | 1024 | 256 | memory-bound |
| 67d44c8f | 1600 | 400 | memory-bound |
| 31eaac46 | 2048 | 512 | transition |
| a2be6188 | 3072 | 768 | transition→compute |
| 3b69084d | 4096 | 1024 | compute-bound |
| 24f153b1 | 6400 | 1600 | compute-bound |
| b52d9dc5 | 7168 | 1792 | compute-bound |
| 42de85c1 | 8192 | 2048 | compute-bound |
| 1f411501 | 12288 | 3072 | compute-bound |
| ffadc1e3 | 16384 | 4096 | compute-bound |
| c60491d2 | 32768 | 8192 | compute-bound |
| ea969b1d | 65536 | 16384 | compute-bound |

geomean weights every workload equally in log space → both regimes matter. Wins come from:
- **Small/transition M (memory-bound):** remove PyTorch's separate LN kernel, the
  permute/reshape/`cat` copies, the GELU round-trip, and the per-grid Python loop with
  `.item()` host syncs. Weights need only be streamed once. Kernel-launch + sync overhead is a
  large fraction of runtime here → biggest relative speedups.
- **Large M (compute-bound):** the two GEMMs are the floor; a well-tiled Triton bf16 GEMM must
  approach cuBLAS, and fusion (GELU in FC1 epilogue, fused LN+shuffle prologue) removes several
  full `[M,6144]` HBM round-trips (~200 MB each at M=16384).

Reference kernel count per call ≈ LN + (per-grid views are free but `cat` copies) + FC1 + GELU +
FC2 + host syncs. Our target: **2–3 Triton launches, zero host syncs on the hot path.**

---

## 2. Constraints & workflow rules

- Triton-only compute; the two Linear layers **must** be Triton `tl.dot` GEMMs (cannot call
  `F.linear`/cuBLAS as the primary path). PyTorch allowed only for shape math, index-tensor
  construction, output allocation.
- Entry point: `solution/solution.py::run(hidden, grid_thw, ln_weight, ln_bias, fc1_weight,
  fc1_bias, fc2_weight, fc2_bias, eps)` returning bf16 `[M,3584]`; signature matches reference.
- Immutable candidates `c001, c002, …`; one full feedback run = one eval; budget 100 evals.
  Token soft/normal/hard = 9M / 10M / 11M.
- Evaluate only via `./scripts/evaluate_candidate.sh feedback cNNN`. Do **not** run CUDA/nvidia-smi
  directly, the external evaluator, or any alternate correctness harness.
- Profiling only via `./scripts/ncu_profile.sh …` (ncu-report-skill workflow), **never**
  concurrently with an evaluation (foreign process on the locked GPU ⇒ return code 3, wasted eval).
- A failed Triton kernel is invalid — no Torch/CPU fallback permitted.
- `final` only with explicit operator approval.

---

## 3. Numerical-risk analysis (tolerances are tight)

Per-workload tolerances: `max_atol ≈ 1.3e-3 … 3.8e-3`, `max_rtol = 0.05`. These roughly match a
bf16 ULP at the output magnitude — the comparison is against the **bf16-rounded reference**, so
we must reproduce the reference’s rounding structure, not merely be "more accurate."

Risks and mitigations:
1. **LayerNorm precision.** Must accumulate mean/variance in **fp32**, use **population variance**
   (`unbiased=False`, divide by `C=1536`), `1/sqrt(var+eps)` with `eps=1e-6`, apply
   `weight/bias` in fp32, then cast to bf16 *before* the shuffle/GEMM (reference feeds bf16 into
   FC1). Deviating (bf16 accum, biased/unbiased mismatch, eps outside sqrt) is the top risk.
2. **GELU must be exact erf**, computed in fp32 (`0.5·x·(1+erf(x·0.70710678))`). The tanh
   approximation deviates by ~1e-3 and could break the tighter-atol workloads → **use
   `tl.math.erf`/libdevice `erf`, verify availability at build time.**
3. **Intermediate bf16 rounding.** Reference rounds FC1 output to bf16 *before* GELU and rounds
   GELU output to bf16 *before* FC2. Safest match: in the FC1 epilogue take fp32 accum → round to
   bf16 (= `hidden_fc1`) → upcast → erf-GELU in fp32 → round to bf16 (= `hidden_gelu`). Keeping
   full fp32 through GELU is *more* accurate but drifts from the reference; start by matching
   rounding, relax only if it demonstrably helps and stays in tol.
4. **GEMM accumulation.** bf16 inputs with **fp32 accumulator** (`tl.dot(..., out_dtype=fp32)` /
   fp32 acc), matching cuBLAS. No tf32 concerns (inputs already bf16).
5. **Weight orientation.** `linear` is `x @ Wᵀ`; `W` is row-major `[out,in]`. In the K-loop load
   the `[BK,BN]` B-tile from `W[n,k]` (stride-N = K). Getting the transpose wrong silently passes
   shapes but fails numerics.
6. **Index correctness / coverage.** `M` and every scatter offset derived from `grid_thw`;
   handle `Σ T·H·W ≤ N`. Off-by-one in `(a,b)`/`(hm,wm)` ordering flips channel blocks — will
   fail. The smallest grids (`H=W=2`, one merged patch) are the boundary check.
7. **ln_weight=1, ln_bias=0 in `get_inputs`.** Do not special-case; apply generally (correct and
   robust) — but this means LN reduces to standardization in these tests, a useful cross-check.
8. **Empty/degenerate grids** unlikely given generator, but guard `M>0`.

---

## 4. Triton design space

### 4.1 Kernel decomposition options

- **A (baseline, 3 kernels):**
  1. `ln_shuffle`: grid = covered input rows; each program LN-normalizes one `[1536]` row and
     **scatters** it to `out_shuffled.flat[dst[r]:+1536]` → materialize `[M,6144]` bf16.
  2. `gemm1_gelu`: `[M,6144] = shuffled @ fc1_wᵀ + fc1_b`, erf-GELU epilogue → bf16.
  3. `gemm2`: `[M,3584] = gelu @ fc2_wᵀ + fc2_b` → bf16.
- **B (2 kernels, fused prologue):** fold LN+shuffle into GEMM1’s A-load. Each A-tile
  `[BM,BK]` gathers the appropriate input rows (via the src-index map) and LN-normalizes on the
  fly, removing the `[M,6144]` intermediate (saves ~2×200 MB HBM at large M). Since a merged row =
  4 contiguous `C=1536` sub-blocks and `BK|1536` divides cleanly, the K-tiling aligns to
  sub-blocks. More complex; strong candidate after A is correct.
- Cannot fuse both GEMMs: FC2 needs the full `E=6144` FC1 row, and a `BM×6144` fp32 tile
  (≥3 MB) exceeds SMEM — keep FC1/FC2 as separate kernels (standard MLP structure).

### 4.2 Index construction (remove host syncs)

- **v0:** read `grid_thw` to host once (`.tolist()`, tiny [G,3], single sync), build the int32
  scatter/gather index with vectorized torch ops on-device per grid (`G ≤ 8`). Simple; the one
  small sync is far cheaper than the reference’s per-grid `.item()` loop + `cat`.
- **v1 (optimization):** zero host syncs — pass `grid_thw` (device) + device cumsum prefixes of
  `T·H·W` and `T·Hm·Wm`; each program locates its grid by a ≤8-iter in-kernel scan and computes
  `(o,s)` arithmetically. Removes all D2H. Evaluate if v0 overhead shows up on small M.

### 4.3 GEMM tiling / scheduling

- bf16 `tl.dot`, fp32 acc, K-loop over `K=6144` with `BK ∈ {32,64,128}`, `num_stages 3–4`,
  `num_warps 4–8`; swizzled/grouped tile raster for L2 reuse of the shared weight.
- **Large M (compute-bound):** classic `BM×BN ∈ {128×256,128×128,256×128}`, `num_stages 3–4`,
  `num_warps 8`; grouped-M raster. Aim to approach cuBLAS; fusion supplies the edge.
- **Small M (memory-bound / low occupancy):** few M-tiles ⇒ under-fill 132 SMs. Options: small
  `BM` (64) with fine `BN` to raise N-tile count; **split-K / atomic or two-stage reduction** over
  `K=6144` to spread the weight read across more CTAs and hide HBM latency; ensure weights stream
  once. Per Blackwell/Hopper roofline guidance (KernelWiki `pattern-memory-bound`,
  `pattern-tail-effect`) the levers are wide vectorized loads, occupancy, and tail mitigation, not
  compute.
- **Autotune vs heuristics:** prefer a *small curated* config set keyed on `M` (heuristics) to
  bound JIT/autotune wall-clock across 15 distinct shapes; autotune only if the config sweep stays
  cheap. Autotune benchmarking lands in warmup (not the timed region), but excessive compile time
  risks eval wall-clock and flakiness — keep the space tight.
- Epilogue fusion (bias + erf-GELU) in FC1 per KernelWiki `technique-epilogue-fusion` /
  `technique-kernel-fusion`: fold bias add + activation into the store, avoiding a `[M,6144]`
  read+write round trip.

### 4.4 Candidate roadmap (sequential, immutable)

- `c001` — Option A, correctness-first: straightforward LN-scatter + two `tl.dot` GEMMs, exact
  erf GELU, reference-matched bf16 rounding, conservative fixed configs. Establish correctness on
  all 15 shapes (incl. boundary M=128 and `H=W=2` grids) and a baseline geomean.
- `c002+` — tune GEMM tiling / stages / warps (M-keyed heuristics); grouped raster.
- next — split-K / low-occupancy handling for small-M workloads.
- next — Option B fused LN+shuffle prologue into GEMM1 (drop the `[M,6144]` intermediate).
- next — v1 sync-free index; micro-tune epilogue.
- Stop when geomean converges or budget/token limits approached; then write `SEARCH_COMPLETE`.

---

## 5. Validation strategy

- **No local CUDA / alternate harness** (forbidden). Correctness is established solely by
  `./scripts/evaluate_candidate.sh feedback cNNN`, which checks all 15 workloads (coarse: warmup 2
  / 10 iters) against the reference within per-workload `atol/rtol` and reports geomean speedup.
  One full run = one candidate eval.
- **Pre-eval static verification** (cheap; before spending an eval): re-derive the shuffle index
  bijection by hand for a tiny grid (`T=1,H=W=2` → 1 merged row, blocks `(a,b)`), confirm LN
  formula (fp32, unbiased=False, eps-in-sqrt), weight-transpose orientation, GELU = erf, and the
  bf16 rounding points. Confirm `tl.math.erf` (or `tl.extra.libdevice.erf`) is importable in the
  build.
- **Correctness-first ordering:** `c001` deliberately simple so a pass validates all index/dtype
  logic across shapes before any perf tuning. A failing workload ⇒ candidate invalid; first
  suspects: GELU approximation, LN variance/eps, weight transpose, or shuffle index ordering.
- **Performance diagnosis:** only after a correct baseline, use `./scripts/ncu_profile.sh` (ncu
  report skill) on representative shapes — one small-M (memory-bound, e.g. M=256) and one large-M
  (compute-bound, e.g. M=16384) — to confirm regime (DRAM-throughput vs tensor-core utilization,
  tail/occupancy) and target the right lever. **Never profile while an eval runs** (return code 3
  wastes an eval); serialize the two.
- **Record-keeping:** append one JSON object per evaluated candidate to `candidates.jsonl`
  (parent, source hash, hypothesis, validation, per-workload result, geomean, decision, cumulative
  eval count, skill usage); never rewrite prior records.

---

## 6. Open questions / to confirm during implementation

1. Installed Triton version and exact `erf` symbol path (affects GELU import).
2. Whether reference-matched intermediate bf16 rounding is required for tol or full-fp32 GELU
   passes with more headroom (test empirically once c001 is correct).
3. Cost of per-call index construction on the smallest workloads (decide v0 vs v1).
4. Achievable Triton-vs-cuBLAS GEMM efficiency at `K=6144` on H100 for the large-M cases — sets
   the ceiling for the compute-bound workloads.
