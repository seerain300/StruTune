# Draft — L2/036 ConvNeXtV2 Layer (NHWC persistence) Backward

Target: NVIDIA H100 (`sm_90`). Entry point: `solution/solution.py::run(...)` with the exact
signature in `task/definition.json`. Primary implementation must be Triton; PyTorch is allowed
only for tensor metadata / launch plumbing (no Torch/CPU/NumPy/CUDA-extension computational
fallback). Ranking metric: geometric-mean speedup vs. the reference `run`, subject to every
selected workload passing correctness.

---

## 1. What the operation computes

This is the **full backward pass** of one ConvNeXtV2 block, stored with NHWC persistence. The
forward block (recorded in `get_inputs`, only for context; we never recompute it) was:

```
residual (NCHW)
  -> x_dwconv   = conv2d(residual, dwconv_weight, pad=3, groups=C)      # depthwise 7x7
  -> x_nhwc     = permute NCHW->NHWC
  -> LayerNorm over C: mean, var, x_normalized, x_ln = x_normalized * layernorm_weight
  -> x_expanded = x_ln @ pwconv1_weight.T                               # (C -> C4) linear
  -> x_gelu     = GELU_tanh(x_expanded)
  -> GRN: global_features = ||x_gelu||_2 over (H,W); gf_mean = mean_C4(global_features)
          norm_features = global_features/(gf_mean+eps)
          x_grn_scaled  = x_gelu * norm_features
          x_grn         = grn_weight * x_grn_scaled + x_gelu
  -> x_projected = x_grn @ pwconv2_weight.T                             # (C4 -> C) linear
  -> permute NHWC->NCHW, drop_path scale, + residual  = output (NCHW)
```

Crucially, **all forward intermediates are provided as inputs** (`x_dwconv, x_nhwc, mean, var,
x_normalized, x_ln, x_expanded, x_gelu, global_features, gf_mean, norm_features, x_grn_scaled,
x_grn`). So the backward is a straight reverse sweep — no forward recomputation is needed. We only
must reproduce the reference's exact arithmetic (including its quirks; see §5).

### Reverse computational graph (must be matched bit-for-tolerance)

Let `M = B*H*W`, `C = 128`, `C4 = 512`. NHWC tensors are logically `(M, C)` or `(M, C4)`.

1. **Drop path + permute**: `keep = 1 - drop_path_prob` (=0.9). `grad_x_nchw = grad_output *
   drop_mask / keep` (drop_mask is per-batch `(B,1,1,1)`). `grad_x_projected = permute(grad_x_nchw)`
   → NHWC `(M, C)`. Separately `grad_residual = grad_output` (un-scaled).
2. **pwconv2 backward** (`pwconv2_weight` is `(C, C4)`):
   - `grad_x_grn = grad_x_projected @ pwconv2_weight`  → `(M, C4)`  *(GEMM, K=C=128)*
   - `grad_pwconv2_weight = grad_x_projected.T @ x_grn`  → `(C, C4)` *(GEMM, K=M)*
   - `grad_pwconv2_bias = sum_M grad_x_projected` → `(C,)`
3. **GRN backward** (`grn_weight`,`grad_grn_bias` are `(1,1,1,C4)`):
   - `grad_x_gelu = grad_x_grn` (residual branch of `x_grn`)
   - `grad_x_grn_scaled = grad_x_grn * grn_weight`
   - `grad_grn_weight = sum_M(grad_x_grn * x_grn_scaled)` → `(1,1,1,C4)`
   - `grad_grn_bias  = sum_M(grad_x_grn)` → `(1,1,1,C4)`
   - `grad_x_gelu += grad_x_grn_scaled * norm_features`  (norm_features `(B,1,1,C4)`)
   - `grad_norm_features = sum_{H,W}(grad_x_grn_scaled * x_gelu)` per batch → `(B,1,1,C4)`
4. **norm_features / gf_mean backward** (see §5 — replicate the reference's *non-summed* form):
   - `grad_global_features = grad_norm_features/(gf_mean+eps)`
   - `grad_gf_mean = -grad_norm_features * global_features/((gf_mean+eps)^2)`   *(kept per-C4!)*
   - `grad_global_features += grad_gf_mean / C4`
5. **global_features (L2 norm) backward**:
   - `grad_x_gelu += x_gelu * grad_global_features / (global_features + eps)`  (broadcast over H,W)
6. **GELU (tanh approx) backward** on `x_expanded`:
   - `inner = a*(x + 0.044715 x^3)`, `a=0.7978845608028654`, `t = tanh(inner)`
   - `cdf = 0.5(1+t)`, `pdf = 0.5(1-t^2)*a*(1 + 3*0.044715 x^2)`
   - `gelu_grad = cdf + x*pdf`; `grad_x_expanded = grad_x_gelu * gelu_grad`  → `(M, C4)`
7. **pwconv1 backward** (`pwconv1_weight` is `(C4, C)`):
   - `grad_x_ln = grad_x_expanded @ pwconv1_weight`  → `(M, C)` *(GEMM, K=C4=512)*
   - `grad_pwconv1_weight = grad_x_expanded.T @ x_ln` → `(C4, C)` *(GEMM, K=M)*
   - `grad_pwconv1_bias = sum_M grad_x_expanded` → `(C4,)`
8. **LayerNorm affine backward** (`layernorm_weight` `(C,)`):
   - `grad_x_normalized = grad_x_ln * layernorm_weight`
   - `grad_layernorm_weight = sum_M(grad_x_ln * x_normalized)` → `(C,)`
   - `grad_layernorm_bias  = sum_M(grad_x_ln)` → `(C,)`
9. **LayerNorm normalization backward** (reduce over C, `N=C=128`, `std=sqrt(var+eps)`):
   - `gxn = grad_x_normalized`; `xc = x_nhwc - mean`
   - `grad_var = -sum_C(gxn*xc)/(2*(var+eps)*std)`
   - `grad_mean = -sum_C(gxn/std) + grad_var*(-2*sum_C(xc)/N)`
   - `grad_x_nhwc = gxn/std + grad_var*(2*xc/N) + grad_mean/N`  → `(M, C)`
10. **permute NHWC->NCHW** → `grad_x_dwconv (B,C,H,W)`.
11. **Depthwise conv backward** (7x7, pad=3, groups=C):
    - `grad_x = conv_transpose2d(grad_x_dwconv, dwconv_weight, pad=3, groups=C) + grad_residual`
      → `(B,C,H,W)`  (input-gradient = full/transposed correlation, per-channel, 49 taps)
    - `grad_dwconv_weight[c,0,i,j] = sum_{b,y,x} residual[b,c,y+i-3,x+j-3] * grad_x_dwconv[b,c,y,x]`
      → `(C,1,7,7)` (zero-padded outside).
    - `grad_dwconv_bias = sum_{B,H,W} grad_x_dwconv` → `(C,)`.

### Output inventory (11 tensors, all fp32)

| output | shape | how |
|---|---|---|
| grad_x | (B,C,H,W) | dwconv input-grad (transposed corr) + grad_output |
| grad_dwconv_weight | (C,1,7,7) | spatial+batch correlation of residual & grad_x_dwconv |
| grad_dwconv_bias | (C,) | sum over B,H,W of grad_x_dwconv |
| grad_layernorm_weight | (C,) | sum_M(grad_x_ln * x_normalized) |
| grad_layernorm_bias | (C,) | sum_M(grad_x_ln) |
| grad_pwconv1_weight | (C4,C) | grad_x_expanded.T @ x_ln |
| grad_pwconv1_bias | (C4,) | sum_M(grad_x_expanded) |
| grad_grn_weight | (1,1,1,C4) | sum_M(grad_x_grn * x_grn_scaled) |
| grad_grn_bias | (1,1,1,C4) | sum_M(grad_x_grn) |
| grad_pwconv2_weight | (C,C4) | grad_x_projected.T @ x_grn |
| grad_pwconv2_bias | (C,) | sum_M(grad_x_projected) |

Return order (tuple): grad_x, grad_dwconv_weight, grad_dwconv_bias, grad_layernorm_weight,
grad_layernorm_bias, grad_pwconv1_weight, grad_pwconv1_bias, grad_grn_weight, grad_grn_bias,
grad_pwconv2_weight, grad_pwconv2_bias.

---

## 2. Shapes, constants, and the 14 feedback workloads

Constants: `C=128`, `C4=512`, `eps=1e-6`, `drop_path_prob=0.1` (keep=0.9), kernel 7x7 pad 3.

| # | B | H | W | M=B·H·W | max_atol |
|---|---|---|---|---|---|
| 1 | 16 | 14 | 14 | 3136 | 0.96 |
| 2 | 8 | 28 | 28 | 6272 | 1.5 |
| 3 | 8 | 14 | 14 | 1568 | 0.72 |
| 4 | 1 | 14 | 14 | 196 | 0.2 |
| 5 | 4 | 56 | 56 | 12544 | 1.3 |
| 6 | 32 | 28 | 28 | 25088 | 2.9 |
| 7 | 1 | 56 | 56 | 3136 | 0.75 |
| 8 | 8 | 56 | 56 | 25088 | 2.8 |
| 9 | 1 | 28 | 28 | 784 | 0.51 |
| 10 | 64 | 14 | 14 | 12544 | 1.3 |
| 11 | 4 | 28 | 28 | 3136 | 1.0 |
| 12 | 32 | 56 | 56 | 100352 | 4.6 |
| 13 | 16 | 56 | 56 | 50176 | 5.2 |
| 14 | 2 | 28 | 28 | 1568 | 0.68 |

All workloads: `max_rtol = 0.001`, `required_match_ratio = 0.98`.

Observations:
- `M` spans 196 → 100352 (a ~500× range). Kernels must handle tiny (B=1,H=14) and huge
  (B=32,H=56) tensors well; a fixed launch config will underutilize small shapes and possibly spill
  on large. Autotune / M-dependent config selection matters.
- H,W ∈ {14,28,56}; channel dims are always 128 / 512, i.e. **compile-time-friendly powers of 2**.
  This is ideal for Triton: C=128 fits a single 128-lane block; C4=512 = 4×128.
- The heavy-M workloads (12, 13, 6, 8, 10, 5) dominate wall-clock and therefore the geomean; the
  tiny ones (4, 9) are latency-bound (launch overhead / occupancy).

---

## 3. Where the reference is slow (opportunity analysis)

The reference is intentionally naive. Biggest costs, in likely order:

1. **`for g in range(C)` weight-gradient loop (128 Python iterations).** Each iteration does
   `F.pad` + double `.unfold(...,7,1)` (materializing a `(B,1,H,W,7,7)` tensor = `M·49` elements) +
   elementwise multiply + a 3-axis sum. For workload 12 that is `100352·49 ≈ 4.9M` temp elements ×
   128 channels, plus 128× kernel-launch/Python overhead. This single loop is almost certainly the
   dominant term and the largest single opportunity — a proper reduction kernel replaces it.
2. **Many materialized fp32 temporaries** across GRN/GELU/LN (`.clone()`, `.pow(3)`, repeated
   broadcasts). Each is a full `(M,C4)` or `(M,C)` read+write to HBM. Fusing these elementwise
   stages into a few passes cuts memory traffic several-fold.
3. **Two `(M,C)↔(M,C4)` GEMM pairs.** These are genuine FLOPs; cuBLAS is fast, but Triton can match
   and, more importantly, can be **fused with surrounding elementwise/epilogue work** (e.g. produce
   `grad_x_expanded` directly from the GEMM output plus the GELU/GRN epilogue).
4. **conv_transpose2d + conv2d (cuDNN)** are already reasonably fast; the input-gradient path is not
   the main win, but replacing it keeps everything in Triton (mandatory) and lets us fuse the
   `+ grad_residual` and the NHWC→NCHW permute.

Because the reference's weight-grad loop is so slow, even a *straightforwardly correct* full-Triton
implementation should yield a large geomean speedup; subsequent candidates then optimize the GEMMs
and fusion.

---

## 4. Constraints (hard rules that shape the design)

- **Triton-only compute.** No `torch.matmul` / `F.linear` / `F.conv*` as the computational op, no
  NumPy/CPU path, no CUDA extension. Torch is permitted only for `empty`/`zeros`/`view`/`permute`
  metadata and launch plumbing. => We must write Triton kernels for the two GEMM families, the
  depthwise conv input-grad, the depthwise weight-grad correlation, all reductions, and the
  elementwise chains.
- **A failed Triton kernel is invalid** — cannot silently fall back to Torch. Every candidate must
  pass or be discarded.
- **Immutable candidates**: `c001, c002, …`, one source version each; never reuse an ID for changed
  source; append one JSON record per eval to `candidates.jsonl` (never rewrite).
- **Budget**: 100 candidate evaluations; token soft 9M / normal 10M / hard 11M. Feedback run = all
  14 workloads = 1 evaluation.
- **Evaluation only** via `./scripts/evaluate_candidate.sh feedback cNNN`. **Profiling only** via
  `./scripts/ncu_profile.sh …` (ncu-report-skill workflow). Never profile and evaluate at once (a
  foreign process on the locked GPU → return code 3, wasted eval). `final` only on operator approval.
- No local/alternate correctness harness, no direct CUDA/`nvidia-smi`, no web/subagents/MCP.

---

## 5. Numerical risks & correctness subtleties

1. **The `grad_gf_mean` "quirk" must be reproduced verbatim.** The reference does *not* sum
   `grad_gf_mean` over C4 before dividing by C4 (mathematically the derivative of a mean should
   accumulate all C4 channels). We must match the reference output, so we replicate its exact
   per-channel formula `grad_global_features += (-grad_norm_features*global_features/(gf_mean+eps)^2)
   / C4`, **not** the textbook backward. Getting this "right" (textbook) would fail correctness.
2. **GELU tanh approximation.** Use the exact constants `a=0.7978845608028654`, `0.044715`. Compute
   `tanh` accurately (Triton `libdevice.tanh` / `tl.math.tanh`, or `2*sigmoid(2x)-1`). The derivative
   uses `1 - t^2`; ensure `t` computed in fp32. Cubic term `x^3` and quadratic `x^2` on modest-range
   `x_expanded` (~O(1–3)) → no overflow.
3. **Precision of GEMMs.** `max_rtol=0.001` with `match_ratio=0.98`. TF32 (~10-bit mantissa) gives
   ~5e-4 per-element relative error and could brush the rtol on the largest-magnitude outputs
   (`grad_pwconv*_weight`, summed over up to 100352 rows). Plan: **start with fp32 / `input_precision
   ="ieee"`** for safety; only later try `"tf32"`/`"tf32x3"` as a *separate* speed candidate and let
   the evaluator confirm it still passes. fp32 accumulation over 100k elements has relative error
   ~sqrt(M)·eps ≈ 2e-5, comfortably inside tolerance.
4. **LayerNorm backward stability.** `std=sqrt(var+eps)`; `var≥0`. The `sum_C(x_nhwc-mean)` term is
   ~0 by construction but is present in the reference — keep it (cheap, and dropping it changes
   rounding). Reduce over C=128 in fp32.
5. **GRN divisions.** `gf_mean+eps` and `global_features+eps` guard against zero; use the given
   `eps=1e-6`. `global_features` ~ sqrt(H·W)·O(1) (up to ~56), well away from zero here, but keep the
   `+eps` exactly where the reference has it (denominators differ: `gf_mean+eps` vs `(gf_mean+eps)^2`
   vs `global_features+eps`).
6. **Drop path scaling applies only to the projected branch**, not to `grad_residual`. Two distinct
   uses of `grad_output`: scaled (feeds pwconv2) and un-scaled (added to `grad_x`). Easy to conflate.
7. **Depthwise conv index arithmetic.** Input-grad is a *flipped* (transposed) correlation:
   `grad_x[b,c,p,q] = Σ_{i,j} W[c,0,i,j]·grad_x_dwconv[b,c,p+3-i, q+3-j]` with bounds mask on the
   fetched index. Weight-grad uses `residual[b,c,y+i-3,x+j-3]` (zero outside). Off-by-one/flip errors
   here are the most likely correctness bug — validate against the evaluator early.
8. **Reduction associativity vs. reference.** Match ratio is 0.98 and atol is generous, so summation
   order differences (Triton tree-reduce vs Torch) are fine; no need to bit-match.
9. **Tiny shapes (B=1).** Values small, atol tight (0.2 for #4). Reinforces the fp32-first choice.

---

## 6. Triton design space

Overall approach: reorganize into a small set of Triton kernels operating on NHWC `(M, C)`/`(M, C4)`
"row-major over channel" layout, with the two GEMM families and the two depthwise kernels bracketing
the elementwise/reduction chains. Candidate progression should start correct-and-simple, then fuse.

### 6.1 Layout / transpose
- `grad_output`, `residual`, `x_dwconv`, `grad_x` are **NCHW-contiguous**; nearly everything else is
  NHWC-contiguous `(B,H,W,·)`. The pwconv GEMMs want NHWC rows `(M, C/ C4)` with channel contiguous.
- Option A: a small Triton "load+scale+transpose" kernel converts `grad_output` (NCHW) → NHWC
  `grad_x_projected (M,C)` while folding the `drop_mask/keep` scale (per-batch scalar). Symmetrically,
  a final kernel writes `grad_x_nhwc (M,C)` back to NCHW `grad_x` and adds `grad_output` (residual).
- Option B: keep NCHW and use strided loads inside kernels. Simpler code but strided/uncoalesced for
  the channel-contiguous GEMMs. Prefer A for the GEMM-facing tensors.
- Because C=128 and C4=512 are constant powers of two, `BLOCK_C=128`, `BLOCK_C4∈{128,256,512}` are
  natural; the channel dim can be a single contiguous block for LN (C) and 1–4 blocks for C4.

### 6.2 GEMM kernels (4 GEMMs, two shapes)
- **Activation GEMMs** (`grad_x_grn = A(M,C)·W(C,C4)`, K=128; `grad_x_ln = A(M,C4)·W(C4,C)`, K=512):
  standard tiled Triton matmul, `BLOCK_M × BLOCK_N × BLOCK_K`, program over `(M-tiles, N-tiles)`.
  N is 512 or 128 (small), so grid is mostly along M. Autotune `BLOCK_M∈{64,128,256}`,
  `BLOCK_N∈{128,256,512}`, `BLOCK_K∈{32,64,128}`, warps, stages.
- **Weight GEMMs** (`grad_pwconv2_weight = Aᵀ(C,M)·B(M,C4)`; `grad_pwconv1_weight = Aᵀ(C4,M)·B(M,C)`):
  small output `(C,C4)=(128,512)` / `(C4,C)=(512,128)` but **large K=M** (up to 100352). This is a
  "tall-skinny reduction" GEMM: either a single tile with a long K-loop, or a **split-K** with atomic
  / two-pass accumulation for large M to expose parallelism. Split-K is likely important for the big
  workloads (12,13) where a single K-loop serializes 100k.
- **Fusion opportunity**: the activation-GEMM epilogue can directly emit the next elementwise stage.
  E.g. `grad_x_grn` GEMM epilogue can accumulate `grad_grn_weight`, `grad_grn_bias`,
  `grad_pwconv2_bias`, and the spatial partials for `grad_norm_features`. And the `grad_x_expanded`
  computation (GELU epilogue) can be fused onto the pwconv2 activation-GEMM path if we stage the
  norm_features correction first. Start unfused for correctness; fuse in later candidates.
- **Precision**: fp32/`ieee` first (see §5.3); TF32 as a later opt candidate.

### 6.3 Elementwise / reduction chains
- **GRN pass-1 reduction kernel** over `(M,C4)`: produce `grad_x_grn` consumers' reductions
  (`grad_grn_weight`, `grad_grn_bias`, `grad_pwconv2_bias` from `grad_x_projected`) and per-`(b,c)`
  spatial partials for `grad_norm_features`. `grad_norm_features` reduces over H·W within each batch —
  grid over `(B, C4-tiles)`, loop over spatial, or a two-stage reduction for large H·W.
- **Small kernel** for `grad_global_features` from `grad_norm_features` (size B×C4 ≤ 16384) — trivial.
- **GELU/GRN pass-2 elementwise** over `(M,C4)`: build `grad_x_gelu` (three contributions) then
  `grad_x_expanded = grad_x_gelu * gelu_grad`. Pure elementwise + broadcast of per-`(b,c)` vectors —
  one fused kernel.
- **LayerNorm backward kernel** over `(M,C)`: per-row (over C=128) reduce for `grad_var`,`grad_mean`,
  and the affine-grad partials (`grad_layernorm_weight/bias` accumulate across rows → either atomics
  or a second reduction pass / split over M with a small combine). One row = 128 elements fits a
  single block; classic Triton layernorm-bwd pattern.
- **Bias/weight reductions** (`grad_pwconv1_bias (C4,)`, `grad_pwconv2_bias (C,)`,
  `grad_layernorm_*`): column reductions over M. For large M use split-M partials + combine, or
  atomic-add into the small output. Careful: atomics in fp32 change rounding but tolerance allows it.

### 6.4 Depthwise conv kernels
- **Input-grad kernel** (`grad_x`): per `(b,c)` map, tile over `(H,W)`; each output pixel sums 49 taps
  of `dwconv_weight[c]` × shifted `grad_x_dwconv[b,c]` with a boundary mask, then `+ grad_output`
  (the residual branch), and writes NCHW. Weights per channel are 49 scalars — preload into regs/SMEM.
  Consider caching a haloed tile of `grad_x_dwconv` in shared memory for reuse across the 49 taps.
- **Weight-grad kernel** (`grad_dwconv_weight (C,1,7,7)` + `grad_dwconv_bias (C,)`): for each channel
  and each of 49 taps, reduce `Σ_{b,y,x} residual[...shifted...]·grad_x_dwconv[b,c,y,x]`. Parallelize
  over `(C, taps)` (=128×49=6272 outputs) with each program reducing over `B·H·W`; or over `(C, M-
  tiles)` with atomic/2-pass accumulation into the 49-wide weight and the scalar bias. This kernel is
  the replacement for the reference's Python loop — the single biggest win. `grad_dwconv_bias` can be
  fused into the same reduction (it's just the tap-independent sum of `grad_x_dwconv`).
- Depthwise index math (flip/pad) is the top correctness hazard (§5.7).

### 6.5 Kernel-count vs. fusion tradeoff
- **c001 (correctness baseline)**: maximal but simple decomposition — separate kernels for
  transpose-in, GEMM×4, GRN reductions, GELU pass, LN-bwd, depthwise input-grad, depthwise weight-
  grad, transpose-out. Prioritize matching the reference; accept extra HBM traffic. Verify all 14 pass.
- Later candidates: (a) fuse GEMM epilogues with adjacent elementwise/reductions; (b) split-K/split-M
  for the tall-skinny weight GEMMs and column reductions; (c) SMEM tiling for the depthwise kernels;
  (d) TF32/tf32x3 GEMMs; (e) per-workload (M-bucketed) autotune configs; (f) reduce temporaries by
  computing `grad_x_gelu` in-place.

### 6.6 Autotune knobs
`BLOCK_M/N/K`, `num_warps∈{2,4,8}`, `num_stages∈{2,3,4}`, split-K factor, spatial tile for depthwise,
`input_precision`. Because M ranges 500×, an M-bucketed config table (small/medium/large) is likely
better than one autotune cache; validate that autotuning doesn't blow the eval timing (autotune runs
during warmup, warmup=2 — so pre-declare a few good configs rather than a huge search).

---

## 7. Validation strategy & workflow

- **Correctness authority = the feedback evaluator only** (`evaluate_candidate.sh feedback cNNN`),
  which checks all 14 workloads at `rtol=1e-3`, per-workload `atol`, `match_ratio=0.98`. No local or
  alternate correctness harness is permitted, so we cannot pre-check against Torch on-box.
- Therefore **c001 must be conservatively correct** (fp32/ieee, verbatim reference arithmetic incl.
  the §5.1 quirk and the §5.6 two-branch drop-path) to avoid burning evaluations on avoidable bugs.
- **Highest-risk-first mental review** before each eval: (i) depthwise flip/pad indices, (ii) the
  grad_gf_mean non-sum quirk, (iii) drop-path applied only to projected branch, (iv) F.linear
  transpose conventions (which weight is transposed), (v) reduction axes for each `grad_*` output.
- **Performance diagnosis** only *between* evaluations via `ncu_profile.sh` on a small self-written
  `harness.py` (perf harness, not a correctness oracle). Never run profiling while an evaluation is
  in flight (return-code-3 hazard). Use ncu to find the dominant kernel and guide the next candidate.
- **Record per candidate** in `candidates.jsonl`: parent, source hash, hypothesis, validation
  (pass/fail per workload), per-workload speedup, geomean, decision, cumulative eval count, skill
  usage. Never rewrite earlier records.
- **Convergence / stop**: stop at budget or when successive candidates stop improving geomean
  meaningfully; then write `SEARCH_COMPLETE`. `final` only after explicit operator approval.

---

## 8. Open questions / things to confirm empirically

1. Actual reference wall-clock split — is the `for g in range(C)` weight-grad loop the dominant term
   as hypothesized? (Confirm via ncu/timing of a baseline; guides where to spend optimization budget.)
2. Does TF32 for the two GEMM families stay within `rtol=1e-3`/`match_ratio=0.98` on workloads 12/13
   (largest magnitudes)? Decide via a dedicated candidate, not by assumption.
3. Best structure for the tall-skinny weight GEMMs (single long-K vs split-K+atomics) at M≈100k.
4. Whether fusing the pwconv2 activation-GEMM epilogue with the GRN reductions is worth the added
   kernel complexity vs. a clean separate reduction kernel.
5. Depthwise input-grad: is a Triton gather-of-49-taps competitive with cuDNN conv_transpose (which we
   cannot use anyway)? SMEM haloed tiling likely needed for the 56×56 shapes.
6. Launch-overhead floor on tiny shapes (#4 B=1,H=14): with ~8–10 kernels, small-shape latency may cap
   speedup there; consider fusing more aggressively for small M or a single mega-kernel path.

---

## 9. Summary of the plan-shaping conclusions

- Full-Triton reimplementation is mandatory; the reference's Python per-channel weight-grad loop and
  its many temporaries make a large geomean speedup very likely even for a first correct version.
- Work in NHWC `(M, {C,C4})` with transpose-in/out kernels folding drop-path and residual.
- Four Triton GEMMs (two activation, two tall-skinny weight), fp32/ieee first.
- Elementwise/reduction chains for GRN, GELU, LayerNorm; two depthwise kernels (input-grad,
  weight+bias-grad).
- Replicate the reference arithmetic exactly — including the `grad_gf_mean` non-sum quirk and the
  drop-path-on-projected-branch-only detail.
- c001 = correct & simple; then fuse epilogues, split-K/M, SMEM-tile depthwise, and try TF32, using
  ncu between evals and the 14-workload feedback set as the sole correctness gate.
