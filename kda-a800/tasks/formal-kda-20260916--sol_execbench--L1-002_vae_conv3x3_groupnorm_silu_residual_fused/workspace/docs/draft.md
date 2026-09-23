# Draft: L1/002 — VAE fused residual block (Conv3x3 → GroupNorm → SiLU, ×2, + residual)

Task id: `sol_execbench / L1 / 002_vae_conv3x3_groupnorm_silu_residual_fused`
Target HW: **NVIDIA A800, `sm_80` (Ampere)**. FP32 ~19.5 TFLOP/s, TF32 tensor-core ~156 TFLOP/s, HBM ~1.5–2.0 TB/s.
Primary framework: **Triton** (PyTorch only for metadata/allocation/launch plumbing — no Torch/CPU/NumPy compute fallback).

This document is analysis only. No `plan.md` and no solution code are produced in this step.

---

## 1. Operation summary and signature

`run(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)` computes, for input `x : (B, C, H, W)` float32:

```
residual = x
out = conv2d(x,   conv1_weight, bias=None, stride=1, padding=1)   # 3x3, same spatial
out = group_norm(out, num_groups=32, weight=norm1_weight, bias=norm1_bias, eps)
out = silu(out)                                                    # out * sigmoid(out)
out = conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
out = group_norm(out, num_groups=32, weight=norm2_weight, bias=norm2_bias, eps)
out = silu(out)
out = out + residual
return out                                                         # (B, C, H, W) float32
```

Fixed constants (from `definition.json`): `C = 256` (in == out), `num_groups G = 32` ⇒ **channels per group = C/G = 8**, `kernel_size = 3`, `stride = 1`, `padding = 1` (spatial dims preserved). Variable axes: `B, H, W`. All tensors float32. `eps = 1e-6` (scalar) for every feedback workload.

Weight shapes: `conv{1,2}_weight : (256, 256, 3, 3)`; `norm{1,2}_weight/bias : (256,)`. Implicit-GEMM contraction dimension per conv is `K = C_in * 3 * 3 = 2304`; GEMM shape is `M = B*H*W` output positions × `N = C_out = 256` × `K = 2304`.

### Semantic details that MUST be matched exactly
- **GroupNorm reduces over (C/G channels = 8, H, W)** for each `(batch, group)` pair → `B*G` independent statistics. Not per-channel, not per-pixel.
- PyTorch `group_norm` uses the **biased (population) variance** (divide by `N = 8*H*W`, `unbiased=False`). Using the unbiased `1/(N-1)` estimator is a correctness bug.
- Affine is applied **per channel** (`norm_weight[c]`, `norm_bias[c]`), after normalizing by the per-group `(mean, rstd)`: `y = (x - mean_g) * rstd_g * weight[c] + bias[c]`.
- SiLU is `x * sigmoid(x)` (not ReLU/GELU); applied elementwise after each GroupNorm.
- Residual is the **original `x`** (pre-conv1), added only after the second SiLU.
- Padding is zero-padding of width 1 on H and W; output positions map 1:1 to input positions (`ih = oh + kh - 1`, `iw = ow + kw - 1`, valid when in `[0,H)×[0,W)`), with no cross-batch bleed.

---

## 2. Feedback workload analysis

Five fixed workloads (one immutable kernel over all five = one evaluation). Per-tensor element count `= B*C*H*W`; each float32 tensor is `4×` that in bytes.

| WL | uuid prefix | B  | H×W     | M = B·H·W | tensor elems | tensor bytes | FLOP/conv (2·M·N·K) | total 2-conv FLOP | atol   |
|----|-------------|----|---------|-----------|--------------|--------------|---------------------|-------------------|--------|
| 1  | 952fec71    | 4  | 256×256 | 262 144   | 67.1 M       | 268 MB       | 3.09e11             | 6.19e11           | 0.0030 |
| 2  | 6284bbd2    | 32 | 64×64   | 131 072   | 33.6 M       | 134 MB       | 1.55e11             | 3.09e11           | 0.0032 |
| 3  | a4c930bb    | 1  | 128×128 | 16 384    | 4.19 M       | 16.8 MB      | 1.93e10             | 3.87e10           | 0.0028 |
| 4  | 144fdf2b    | 64 | 64×64   | 262 144   | 67.1 M       | 268 MB       | 3.09e11             | 6.19e11           | 0.0032 |
| 5  | 8de1dc63    | 1  | 768×768 | 589 824   | 151.0 M      | 604 MB       | 6.96e11             | 1.39e12           | 0.0034 |

Observations:
- **Convolution dominates compute.** Two 3×3, 256→256 convs per call. GroupNorm + SiLU + residual are memory-bound elementwise/reduction passes whose cost is a small fraction of the conv FLOPs but non-trivial in memory traffic.
- Conv is **strongly compute-bound** (arithmetic intensity `≈ N·K/(bytes per output) = 256·2304·2 / (few·4)` FLOP/byte — hundreds), so the conv kernel's throughput (and whether TF32 tensor cores can be used) sets the ceiling.
- Rough FP32-only lower bound (at 19.5 TFLOP/s, 100% util): WL5 ≈ 71 ms, WL1/WL4 ≈ 32 ms, small ones sub-ms. With TF32 tensor cores (156 TFLOP/s) these drop ~8×. **The TF32-vs-FP32 decision is the single largest performance lever.**
- Two shape regimes: small-batch/large-spatial (WL1, WL5, WL3) and large-batch/small-spatial (WL2, WL4). Both have huge `M`; `N=256` is small (favors `BLOCK_N ∈ {64,128,256}`); `K=2304` is a modest reduction loop.
- `tolerance.max_rtol = 1e-5` for all; `max_atol` in `[0.0028, 0.0034]`. With rtol so tight, the **atol term dominates** for typical magnitudes; the pass test is essentially `|a-b| ≤ atol` for values of order ~1 and `|a-b| ≤ rtol·|b|` only helps at large magnitudes.

---

## 3. Constraints (hard requirements)

- **Triton is the primary and only compute path.** Conv must be implemented in Triton (cannot call `F.conv2d`); GroupNorm/SiLU/residual likewise. PyTorch usage limited to `.shape/.stride/.dtype`, `torch.empty`/`torch.empty_like` allocation, and kernel launch.
- **No fallback of any kind** (Torch compute, CPU, NumPy, CUDA C++ extension, alternate impl). A failing Triton kernel is an invalid candidate — do not patch it with a Torch path.
- **Immutable, sequential candidates** `c001, c002, …`; any meaningful source/config/launch change ⇒ new id; never reuse an id for changed source; never rewrite earlier `candidates.jsonl` records.
- **Evaluation only** via `./scripts/evaluate_candidate.sh feedback <id>`; the five workloads together = one evaluation. Budget: **100 evaluations**; token soft 1.0M / hard 1.2M. `final` (20 workloads) is operator-approval-only.
- **Cannot self-run** CUDA/Triton/torch, profilers, `nvidia-smi`, or any alternate correctness harness locally. All correctness/perf signal comes from the evaluator. (Confirmed: direct Python execution is blocked in this environment.)
- Primary ranking metric: **geometric-mean speedup across workloads, with every workload required to pass correctness.** A single correctness failure invalidates the candidate regardless of speed.

Skill usage note: `KernelWiki` (Blackwell/Hopper-specific) and `ncu-report-skill` (B200/sm_100 profiling) are **not applicable** to this Ampere `sm_80` target and are excluded; profiler access is prohibited anyway. This will be recorded as "no skill used" in candidate records.

---

## 4. Numerical risks

The tight atol (≈3e-3) against a compute-heavy, multi-stage pipeline makes numerics the primary correctness risk.

### 4.1 TF32 vs FP32 in the convolutions (central trade-off)
- `tl.dot` accumulates in FP32 regardless; the choice is whether **inputs are rounded to TF32** (10-bit mantissa, `input_precision="tf32"`/`allow_tf32=True`) or kept IEEE FP32 (`input_precision="ieee"`).
- Magnitude estimate with random N(0,1) inputs/weights: conv1 output σ ≈ `sqrt(K)·σ_w·σ_x ≈ sqrt(2304) ≈ 48`. TF32 rounding error per product ≈ `2^-11 ≈ 5e-4` (relative); summed over 2304 random terms → absolute conv error `≈ sqrt(2304)·5e-4·1 ≈ 0.024` on an output of magnitude ~48 → **relative ~5e-4**.
- **GroupNorm rescues precision:** normalization divides by the per-group std (~48), so the ~5e-4 *relative* conv error maps to ~5e-4 *absolute* error on normalized values of order ~1 — comfortably below atol 3e-3. SiLU is ~1-Lipschitz and does not amplify. The same argument repeats for conv2/GN2. Expected end-to-end TF32 error ≈ 1e-3, **plausibly within tolerance**.
- **Residual caveat:** the final `+ residual` re-injects `x` (magnitude ~1) exactly, so it adds no error; the output magnitude stays ~1–few, keeping the atol budget meaningful.
- **Reference-comparison uncertainty:** the ground truth is the evaluator running the PyTorch reference. If it uses cuDNN with default `allow_tf32=True`, the reference itself is TF32-grade and our TF32 error and cuDNN's TF32 error may partially add (up to ~1e-3). Still likely < atol, but not guaranteed. **Mitigation:** first land a correct **FP32/IEEE** candidate (`input_precision="ieee"`) to guarantee correctness and establish baseline speed; then test a TF32 candidate for speed and let the evaluator confirm it still passes all five. Keep both as immutable candidates for comparison. A middle option ("3xTF32"/error-corrected) is a fallback if plain TF32 fails but IEEE is too slow.

### 4.2 GroupNorm variance computation
- Use a numerically safe reduction. `Var = E[x^2] - E[x]^2` risks catastrophic cancellation when `mean^2 ≈ E[x^2]`; here conv output has mean ≈ 0 and large variance so cancellation is mild, but a **two-pass (mean, then Σ(x-mean)^2)** or **Welford** accumulation is safer and cheap enough. Prefer two-pass or Welford.
- **Biased variance** (divide by `N=8·H·W`) to match PyTorch. `rstd = rsqrt(var + eps)`. `eps=1e-6` is negligible vs var~O(10³) but include it exactly.
- Large-group reductions: WL5 group size `= 8·768·768 ≈ 4.7M` elements. Naive serial FP32 accumulation error ~`eps·sqrt(N)·magnitude`. Accumulate partials in FP32 with tiled/hierarchical reduction (per-tile local sum then combine), or Welford-combine across tiles, to keep variance relative error ~1e-4. If atomics are used to combine tile partials, ordering nondeterminism yields only ~ULP-level differences — acceptable.

### 4.3 Intermediate storage dtype
- Keep intermediates `t1` (conv1 out) and `t3` (conv2 out) in **float32**. Storing them in bf16/fp16 to save bandwidth would inject ~1e-2 relative error before normalization → risks exceeding atol. Do not downcast intermediates.

### 4.4 Elementwise / activation
- SiLU `x*sigmoid(x)`: compute `sigmoid` in FP32; large negative `x` underflows sigmoid to 0 gracefully. No special handling needed.
- `rsqrt`: `tl.math.rsqrt` FP32 is accurate to ~1 ULP; acceptable. Avoid computing `1/std` via low-precision fast-math.

### 4.5 Padding / masking correctness (not "numerical" but silent-error prone)
- Zero-pad taps must be **masked to 0** in the input load (out-of-range `ih/iw`), and batch index derived per output pixel so no wraparound across batch/rows. A masking bug shows as small localized errors near borders that may still slip under atol on some workloads but not others — treat any single-workload failure as a masking suspect.

---

## 5. Triton design space

### 5.1 Convolution as implicit GEMM (per 3×3 tap)
Preferred formulation: for output tile of `BLOCK_M` pixels × `BLOCK_N` channels, accumulate over the 9 taps:
```
acc[BLOCK_M, BLOCK_N] = 0
for (kh, kw) in 3x3:               # 9 taps
    Xt[BLOCK_M, K_cin] = load input at (b, cin, oh+kh-1, ow+kw-1), masked for padding
    Wt[K_cin, BLOCK_N] = weight[:, cin, kh, kw] arranged as [cin, cout]
    acc += dot(Xt, Wt)             # K_cin = 256 (full C_in) or tiled
```
- `K_cin = C_in = 256` fits as a single reduction chunk (or split into 2×128). No cross-tap split-K needed; 9 sequential dots of K=256 accumulate into one FP32 acc.
- `BLOCK_N ∈ {64,128,256}` (N=256 small); `BLOCK_M ∈ {32,64,128}`. Autotune over these with `num_warps ∈ {4,8}`, `num_stages ∈ {2,3,4}`.
- Padding handled by masking `Xt` loads where `ih/iw` out of range.

### 5.2 Memory layout: NCHW vs NHWC
- Inputs arrive **NCHW** (channels stride `H*W`, spatial contiguous). An implicit-GEMM `[pixel, cin]` tile then gathers `cin` with stride `H*W` (poorly coalesced) while pixels within a row are contiguous.
- **Option A (stay NCHW):** program the pointer math directly; simpler plumbing, but strided `cin` loads. Often acceptable because the conv is compute-bound and L2 caches the reused input.
- **Option B (NHWC / channels-last):** a `[pixel, cin]` tile becomes contiguous in `cin` → coalesced, faster dots. Requires a layout transform of `x`/intermediates. Doing `.contiguous(channels_last)` via torch is borderline "compute"; safer to write a **small Triton transpose kernel** (NCHW→NHWC) so all compute stays in Triton. Cost is a memory-bound pass; may pay off on the big workloads.
- Decision: start NCHW for a correct baseline; evaluate an NHWC variant as a perf candidate. Keep the winner.

### 5.3 Weight preparation
- For tap `(kh,kw)` we need `Wt[cin, cout] = weight[cout, cin, kh, kw]`. Native layout `(Cout,Cin,3,3)` gives strided access. A **one-time (per call) Triton permute** of weights to `(3,3,Cin,Cout)` (or `(3,3,Cout,Cin)`) makes tap tiles contiguous. Weights are tiny (256·256·9 = 590K elems, 2.4 MB) so the permute is cheap and reused across all `M` tiles. Consider for the optimized candidate.

### 5.4 GroupNorm: stats then apply
- **Stats kernel:** one program per `(b, g)` (there are `B*G` ≤ 64·32=2048 programs; WL5 has only 32). Loop spatial tiles over the group's 8 channels × H×W, accumulate Welford/two-pass `(mean, var)` in FP32 → write `mean[B,G], rstd[B,G]`. For very small `B*G` (WL5: 32 programs) parallelism is low; consider a **split-reduction** (multiple programs per group + atomic/second-stage combine) so the big-spatial workloads keep the GPU busy.
- **Apply kernel:** elementwise `y = silu((x - mean_g)*rstd_g*weight[c] + bias[c])`, fully parallel.

### 5.5 Fusion opportunities (memory-traffic savings)
Baseline pipeline is up to 6 kernels: `conv1 → gn1_stats → gn1_apply+silu → conv2 → gn2_stats → gn2_apply+silu+add`. Fusion candidates:
1. **Fuse GN-apply+SiLU into the *next* conv's input load.** conv2 reads `t2 = silu(gn1(t1))`; instead read raw `t1` and apply `(normalize·affine·silu)` on-the-fly during each tap load (stats precomputed). Saves a full write+read of a 256-channel feature map. Trade-off: `t1` is re-read ~9× (overlapping taps) so the activation is recomputed ~9×, adding ALU to a compute-bound kernel. **Test both** (materialize `t2` vs fuse).
2. **Fuse final GN2-apply + SiLU + residual add** into one elementwise kernel (already natural).
3. **Fuse conv1 epilogue into GN1 stats** via atomic accumulation of group sums during conv1 (saves reading `t1` for stats). Adds atomic complexity + minor nondeterminism; optional.
4. **Keep intermediates FP32** (see 4.3) — do not fuse-away precision.

### 5.6 Kernel-count / launch-overhead angle
For small workloads (WL3: 4.2M elems), kernel-launch overhead and low occupancy matter more than raw FLOPs; fewer, fused kernels help. For big workloads (WL1/4/5), conv throughput dominates. The autotuned config likely differs by regime — Triton autotune keyed on shape handles this.

### 5.7 What we are competing against
The reference is cuDNN conv (highly tuned on Ampere) + separate GN/SiLU/add kernels. **Beating cuDNN's raw conv with hand-written Triton is hard**; our edge must come from (a) TF32 tensor-core utilization matching cuDNN, plus (b) fusing away the reference's extra elementwise/reduction memory passes and kernel launches. Net geomean >1.0 is plausible but not guaranteed and is the main project risk. If Triton conv lands well below cuDNN, geomean may be <1.0 despite fusion — this must be measured, not assumed.

---

## 6. Performance model (order-of-magnitude)

- Conv is the bottleneck. At TF32 156 TFLOP/s and ~50–70% achieved util, WL5 (1.39 TFLOP) ≈ 13–18 ms; WL1/WL4 (0.62 TFLOP) ≈ 6–8 ms; WL2 (0.31 TFLOP) ≈ 3–4 ms; WL3 (0.039 TFLOP) sub-ms (launch/occupancy bound). FP32/IEEE path is ~8× slower on conv and will almost certainly lose to a TF32 cuDNN reference.
- GN/SiLU/residual traffic per call ≈ a handful of full-tensor passes (each `4·B·C·H·W` bytes). WL5 tensor = 604 MB; ~5 passes ≈ 3 GB ⇒ ~1.5–2 ms at HBM BW. Fusing these is the concrete, low-risk win over the reference.
- Implication: **prioritize a TF32 conv** (if it passes atol) and **minimize elementwise passes via fusion**; the FP32/IEEE variant exists mainly as a correctness anchor.

---

## 7. Validation strategy

Because local execution is prohibited, **the evaluator is the sole oracle** for both correctness and speed. Strategy:

1. **c001 — correctness anchor:** simplest correct Triton pipeline (implicit-GEMM conv, `input_precision="ieee"`, NCHW, two-pass/Welford GroupNorm with biased variance, un-fused or lightly-fused). Goal: pass correctness on all 5 workloads and record baseline geomean. This isolates *algorithmic* correctness before any perf trick.
2. **c002+ — one lever per candidate**, so each evaluation attributes cause→effect cleanly:
   - TF32 conv (`allow_tf32`/`input_precision="tf32"`) — confirms whether atol survives (biggest speed lever).
   - Autotune block sizes / `num_warps` / `num_stages`.
   - Weight prepermute; NHWC layout transform.
   - Fuse GN-apply+SiLU into conv2 input; fuse GN2+SiLU+residual.
   - Optional atomic-fused GN stats; split-reduction for low-`B*G` workloads (WL5/WL3).
3. **Correctness gating:** any single-workload correctness failure ⇒ candidate invalid; treat first as a numerical (TF32/variance/biased-var) or masking (padding/batch) issue per §4. Never widen scope by adding a Torch path.
4. **Metric tracking:** record per-workload pass/fail, per-workload speedup, and **geomean** in `candidates.jsonl` with parent, source hash, hypothesis, decision, cumulative eval count, and skill usage (none). Keep records append-only and immutable.
5. **Convergence / budget:** stop when geomean improvement stalls across a few candidates or on budget (100 evals / token limits); then write `SEARCH_COMPLETE` with the reason. Never run `final` without operator approval.
6. **Attribution discipline:** change exactly one meaningful thing per candidate id; if a change regresses, revert to the best-known parent for the next candidate rather than stacking speculative changes.

---

## 8. Open questions / decisions to resolve empirically

- Does plain **TF32 conv pass all five atol thresholds**? (Expected yes per §4.1, but reference may itself be TF32-cuDNN — must confirm.) If it fails only on the tightest (WL3 atol 0.0028), consider IEEE conv1 + TF32 conv2, or 3xTF32.
- **NCHW vs NHWC**: is the layout-transform pass worth it given conv is compute-bound and L2 may absorb strided reuse?
- **Materialize `t2` vs fuse GN+SiLU into conv2**: does saved bandwidth beat recomputed activation in a compute-bound kernel?
- Best **stats-kernel parallelization** for low-`B*G` big-spatial workloads (WL5 has only 32 groups): split-reduction vs single-program-per-group.
- Whether a **single autotune config** covers both shape regimes or shape-keyed autotune is needed.
- Realistically, **can Triton conv reach a geomean >1.0 vs cuDNN**? If not after the main levers, document and converge rather than overfit.

---

### Next step (separate turn)
Author `docs/plan.md` translating §5–§7 into a concrete, ordered candidate roadmap (starting with the c001 IEEE correctness anchor), then implement `solution/solution.py` for c001 and evaluate.
