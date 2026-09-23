# Draft — L2/036 ConvNextV2 Layer (NHWC persistence) Backward

Target HW: **NVIDIA A800, `sm_80` (Ampere)**. dtype: **float32** everywhere.
Submission: `solution/solution.py` exposing `run(...)` with the exact signature from
`task/definition.json`. Primary compute must be **Triton**; PyTorch only for metadata,
allocation, and launch plumbing. **No Torch/CPU/NumPy/cuDNN-extension compute fallback.**

> Skill applicability: `KernelWiki` targets Blackwell (sm_100) / Hopper (sm_90); `ncu-report-skill`
> targets B200 / sm_100. This task is Ampere (sm_80), so **neither skill applies**. No profiler /
> `nvidia-smi` / direct CUDA runs are permitted. Local ad-hoc execution (`Bash`) is denied in this
> environment, so **all functional feedback comes only from `./scripts/evaluate_candidate.sh feedback cNNN`**,
> each call costing one candidate evaluation against the 100-eval budget.

---

## 1. What the operation is

This is the **backward pass** of one ConvNextV2 block, written with NHWC "persistence" (activations
kept in `(B,H,W,C)` between stages). All forward intermediates are **provided as inputs** (saved from
the forward), so we never recompute the forward — we only propagate gradients in reverse order and emit
11 gradient tensors.

Constants: `C = 128` (channels), `C4 = 4*C = 512` (expanded channels), depthwise kernel `7x7`,
`padding = 3`, `groups = C` (true depthwise), `eps = 1e-6`, `drop_path_prob = 0.1`
(so `keep_prob = 0.9`, drop-path branch is always active in these workloads).

Forward block (for context; **not** recomputed):
`residual → dwconv(7x7,dw) → permute NCHW→NHWC → LayerNorm(affine) → pwconv1 (C→C4) → GELU(tanh) →
GRN → pwconv2 (C4→C) → permute NHWC→NCHW → drop_path → + residual`.

### 1.1 Notation / shapes
Let `M = B*H*W` be the number of spatial tokens (flattened `(B,H,W)`), `HW = H*W`.

| tensor | shape | role |
|---|---|---|
| grad_output, residual, x_dwconv | (B,C,H,W) NCHW | upstream grad / saved inputs |
| x_nhwc, x_normalized, x_ln | (B,H,W,C) | LN activations |
| mean, var | (B,H,W,1) | LN stats (per token, over C) |
| x_expanded, x_gelu, x_grn_scaled, x_grn | (B,H,W,C4) | pwconv1/GELU/GRN activations |
| global_features, norm_features | (B,1,1,C4) | GRN spatial L2 norm / normalized |
| gf_mean | (B,1,1,1) | channel mean of global_features |
| dwconv_weight | (C,1,7,7) | depthwise kernel |
| layernorm_weight | (C,) | LN gamma |
| pwconv1_weight | (C4,C) | expansion matrix |
| grn_weight | (1,1,1,C4) | GRN gamma |
| pwconv2_weight | (C,C4) | projection matrix |
| drop_mask | (B,1,1,1) | per-sample keep mask |

Outputs (11): `grad_x (B,C,H,W)`, `grad_dwconv_weight (C,1,7,7)`, `grad_dwconv_bias (C,)`,
`grad_layernorm_weight (C,)`, `grad_layernorm_bias (C,)`, `grad_pwconv1_weight (C4,C)`,
`grad_pwconv1_bias (C4,)`, `grad_grn_weight (1,1,1,C4)`, `grad_grn_bias (1,1,1,C4)`,
`grad_pwconv2_weight (C,C4)`, `grad_pwconv2_bias (C,)`.

### 1.2 Feedback workloads (fixed 5)
| id | B | H | W | M=B·H·W | HW | atol | rtol | match |
|----|---|---|---|--------|------|------|------|-------|
| W1 | 2 | 28 | 28 | 1,568  | 784  | 0.68 | 1e-3 | 0.98 |
| W2 | 1 | 56 | 56 | 3,136  | 3136 | 0.75 | 1e-3 | 0.98 |
| W3 | 16| 56 | 56 | 50,176 | 3136 | 5.2  | 1e-3 | 0.98 |
| W4 | 16| 14 | 14 | 3,136  | 196  | 0.96 | 1e-3 | 0.98 |
| W5 | 4 | 56 | 56 | 12,544 | 3136 | 1.3  | 1e-3 | 0.98 |

W3 is by far the largest (dominates the geomean cost); W1/W4 are small (latency/launch-overhead bound).
`atol` grows with problem size (`grad_dwconv_weight` and other reductions sum over `M`, so their
magnitude scales with `M` — the evaluator widens `atol` accordingly).

---

## 2. Exact reference math to replicate (bit-for-bit intent)

The evaluator compares against `run(...)` in `definition.json`. We must reproduce its **exact
arithmetic**, including one non-standard chain-rule step, not the "textbook" gradient. Reverse order:

1. **Residual + drop-path split.** `grad_residual = grad_output` (raw, *no* drop-path scaling).
   `grad_x_nchw = grad_output * drop_mask / keep_prob` (`keep_prob = 1 - drop_path_prob = 0.9`).
2. **Permute NCHW→NHWC.** `grad_x_projected[b,h,w,c] = grad_x_nchw[b,c,h,w]` → `(B,H,W,C)`.
3. **pwconv2 backward.** `F.linear(x, pwconv2_weight.t()) == x @ pwconv2_weight`.
   - `grad_x_grn = grad_x_projected @ pwconv2_weight`  → `(M,C)·(C,C4)=(M,C4)`.
   - `grad_pwconv2_weight = grad_x_projected_flatᵀ @ x_grn_flat` → `(C,M)·(M,C4)=(C,C4)`.
   - `grad_pwconv2_bias = grad_x_projected.sum(0,1,2)` → `(C,)`.
4. **GRN backward.** `x_grn = grn_weight * x_grn_scaled + x_gelu` (bias≡0 in fwd but grad still emitted);
   `x_grn_scaled = x_gelu * norm_features` (norm_features broadcast over H,W).
   - `grad_x_grn_scaled = grad_x_grn * grn_weight`.
   - `grad_grn_weight = (grad_x_grn * x_grn_scaled).sum(0,1,2, keepdim)` → `(1,1,1,C4)`.
   - `grad_grn_bias = grad_x_grn.sum(0,1,2, keepdim)` → `(1,1,1,C4)`.
   - `grad_norm_features = (grad_x_grn_scaled * x_gelu).sum(1,2, keepdim)` → `(B,1,1,C4)` (**spatial** reduction per (b,c)).
   - `grad_x_gelu = grad_x_grn (residual path) + grad_x_grn_scaled * norm_features (scaled path)`.
5. **norm_features = global_features / (gf_mean + eps).**
   - `grad_global_features = grad_norm_features / (gf_mean + eps)`.
   - `grad_gf_mean = -grad_norm_features * global_features / (gf_mean + eps)**2`  → kept **element-wise `(B,1,1,C4)`, NOT summed**.
   - `grad_global_features += grad_gf_mean / C4` (added element-wise, channel `c` gets only its own term).
   - ⚠️ **Non-standard**: the true gradient of `gf_mean = mean_c(global_features)` would sum
     `grad_gf_mean` over channels and broadcast `/C4`. The reference does **not** sum. We must copy this
     exactly (per-(b,c): `ggf[b,c] = grad_nf[b,c]/(gfm[b]+eps) - grad_nf[b,c]*gf[b,c]/((gfm[b]+eps)²·C4)`).
6. **global_features = ‖x_gelu‖₂ over spatial (dims 1,2).**
   `grad_x_gelu += x_gelu * grad_global_features / (global_features + eps)` (broadcast gf/ggf over H,W).
7. **GELU backward (tanh approx).** Constants `k0 = 0.7978845608028654 = sqrt(2/π)`, `k1 = 0.044715`.
   `inner = k0*(x_expanded + k1*x_expanded³)`; `t = tanh(inner)`; `cdf = 0.5*(1+t)`;
   `pdf = 0.5*(1-t²)*k0*(1 + 3*k1*x_expanded²)`; `gelu_grad = cdf + x_expanded*pdf`;
   `grad_x_expanded = grad_x_gelu * gelu_grad`. (Use `x_expanded` given; do not re-derive from x_gelu.)
8. **pwconv1 backward.** `F.linear(x, pwconv1_weight.t()) == x @ pwconv1_weight`.
   - `grad_x_ln = grad_x_expanded @ pwconv1_weight` → `(M,C4)·(C4,C)=(M,C)`.
   - `grad_pwconv1_weight = grad_x_expanded_flatᵀ @ x_ln_flat` → `(C4,M)·(M,C)=(C4,C)`.
   - `grad_pwconv1_bias = grad_x_expanded.sum(0,1,2)` → `(C4,)`.
9. **LayerNorm affine backward.**
   - `grad_x_normalized = grad_x_ln * layernorm_weight`.
   - `grad_layernorm_weight = (grad_x_ln * x_normalized).sum(0,1,2)` → `(C,)`.
   - `grad_layernorm_bias = grad_x_ln.sum(0,1,2)` → `(C,)`.
10. **LayerNorm normalization backward** (`N = C = 128`, reduction over channel axis per token):
    - `std = sqrt(var + eps)` (use given `var`).
    - `grad_x_nhwc = grad_x_normalized / std`
    - `grad_var = -(grad_x_normalized*(x_nhwc-mean)).sum(-1,keepdim) / (2*(var+eps)*std)`
    - `grad_mean = -(grad_x_normalized/std).sum(-1,keepdim)`
    - `grad_mean += grad_var * (-2*(x_nhwc-mean).sum(-1,keepdim)/N)`  (this `(x-mean).sum` ≈ 0 numerically but MUST be kept)
    - `grad_x_nhwc += grad_var * (2*(x_nhwc-mean)/N)`
    - `grad_x_nhwc += grad_mean / N`
11. **Permute NHWC→NCHW.** `grad_x_dwconv[b,c,h,w] = grad_x_nhwc[b,h,w,c]`.
12. **Depthwise conv backward** (kernel 7x7, pad 3, stride 1, groups C):
    - **Input grad**: `grad_x = conv_transpose2d(grad_x_dwconv, dwconv_weight, pad=3, groups=C) + grad_residual`.
      Explicit stencil: `grad_x[b,c,iy,ix] = grad_output[b,c,iy,ix] + Σ_{i,j∈0..6} grad_x_dwconv[b,c, iy-i+3, ix-j+3]·W[c,0,i,j]`
      (valid taps only, i.e. `0 ≤ iy-i+3 < H`, `0 ≤ ix-j+3 < W`). Note the **flip** of taps vs forward.
    - **Weight grad**: `grad_dwconv_weight[c,0,i,j] = Σ_{b,oy,ox} grad_x_dwconv[b,c,oy,ox] · residual_pad[b,c, oy+i, ox+j]`,
      `residual_pad` = residual zero-padded by 3. Equivalent: `Σ grad·residual[b,c, oy+i-3, ox+j-3]` over valid positions.
    - **Bias grad**: `grad_dwconv_bias[c] = grad_x_dwconv.sum(0,2,3)` → `(C,)`.

**Cross-check of the two conv directions (off-by-one is the top functional risk):** forward uses
`out[oy,ox] = Σ in_pad[oy+i, ox+j]·W[i,j]` (cross-correlation, pad 3). Therefore input-grad uses the
flipped kernel and weight-grad correlates grad with the padded input at offset `(+i,+j)`. Both must be
validated by construction and, if a candidate fails only on `grad_x` / `grad_dwconv_weight`, this
indexing is the first suspect.

---

## 3. Constraints & correctness invariants

- **Triton-only compute.** Every one of: 4 GEMMs (pwconv1/2 fwd-grad + weight-grad), all element-wise
  chains, all reductions (biases, LN weight/bias, GRN weight/bias, spatial L2-norm grad, LN stat grads),
  and both depthwise-conv gradients must be Triton. Torch may only `empty`/`zeros`/`reshape`/`view`/
  `stride`/`permute-metadata` and launch kernels.
- **No fallback.** A failing Triton path is invalid; do not substitute `F.linear`/`F.conv*`/`torch.tanh`.
- **Exact arithmetic** for the non-standard `gf_mean` step and the `(x-mean).sum` LN term (§2.5, §2.10).
- **Use provided intermediates** (`mean`, `var`, `global_features`, `gf_mean`, `norm_features`,
  `x_grn_scaled`, `x_normalized`, `x_ln`, `x_expanded`, `x_gelu`) directly — do not recompute from
  scratch, or rounding will drift from the reference.
- **Output dtypes/shapes** must match exactly, including keep-dim shapes `grad_grn_weight/bias (1,1,1,C4)`
  and `grad_pwconv2_bias (C,)`.
- **Immutability**: one source version per candidate id; new id for any source/config/launch change.

---

## 4. Numerical risk register

1. **TF32 vs IEEE fp32 in `tl.dot`.** `rtol = 1e-3` sits right at TF32's (~2⁻¹⁰ mantissa) error edge.
   Accumulation is fp32 regardless, but TF32 truncates multiply operands. For large-magnitude reduction
   outputs the wide `atol` likely absorbs it, but `grad_x` / LN-derived grads have modest magnitude where
   `rtol` bites. **Plan:** start with `input_precision="ieee"` (true fp32) for all `tl.dot` to guarantee
   correctness; later test `"tf32"` as a *speed* candidate and keep it only if all 5 workloads still pass.
2. **Non-standard `gf_mean` gradient (§2.5).** Copying the textbook (summed) version will fail the
   `grad_grn_*` / `grad_x` chain. Replicate the element-wise, unsummed form exactly.
3. **GELU tanh approx.** Reuse the exact constants and the *single* `tanh` evaluation reused in `cdf`
   and `pdf`; recomputing/reordering changes low bits. Use `tl.math.tanh` (or `libdevice.tanh`) in fp32.
4. **Divisions by (stat+eps).** `1/std`, `1/(gf_mean+eps)`, `1/(global_features+eps)`. `global_features`
   is an L2 norm ≥ 0; `+eps` guards zero. Use given values, not recomputed norms.
5. **Depthwise stencil boundaries (pad 3).** Off-by-one / flip errors in input-grad and weight-grad are
   the highest-probability functional bug (see §2.12 cross-check). Mask out-of-range taps to 0.
6. **Reduction associativity / atomics.** `grad_dwconv_weight`, biases, LN/GRN weight&bias sum over many
   elements. Atomic-add or multi-block split changes summation order → last-bit noise, acceptable under
   `atol`/`rtol`+98% match, but prefer deterministic (single-pass or tree) reductions where cheap to keep
   headroom. Accumulate in fp32.
7. **Match ratio 0.98.** Up to 2% of elements per tensor may miss; still aim for ~exact so no tensor
   drifts systematically (a wrong constant would fail *all* elements, which 0.98 does not save).
8. **Large-M accumulation error** (W3, M≈50k) for `grad_pwconv*_weight` (K=M reduction). fp32 accumulate;
   consider split-K with fp32 partials if a single accumulator loses precision (unlikely to matter given
   atol 5.2 on W3).
9. **Empty/degenerate tiles**: masks for M not divisible by BLOCK, C4=512/C=128 are clean multiples of
   common block sizes (32/64/128), which simplifies masking on the channel axes.

---

## 5. Where the reference is slow (optimization opportunity)

The reference `run(...)` has one catastrophic hotspot and several ordinary ops:

- 🔴 **`for g in range(C): ...unfold...`** — a **Python loop over 128 channels**, each doing
  `F.pad` + double `unfold` (materializing `(B,H,W,7,7)` patches) + multiply + 3-D sum. This is the
  dominant cost, especially on W3 (128 × huge unfold), and launch/host overhead on small W1/W4. A single
  fused Triton depthwise-weight-grad kernel replaces all 128 iterations → the primary speedup lever.
- 🟠 4 GEMMs via cuBLAS (fast baseline; our Triton GEMM must be competitive, not just correct).
- 🟠 `conv_transpose2d` via cuDNN for `grad_x` (fast baseline; depthwise stencil in Triton must match).
- 🟢 Many element-wise/reduction ops materializing full `(M,C4)` temporaries repeatedly → **fusion**
  reduces DRAM traffic (memory-bound).

**Thesis:** even a *modular, correct* Triton port should beat the reference purely by eliminating the
128-iteration Python loop; further gains come from (a) a well-tuned depthwise weight-grad kernel and
(b) fusing the element-wise chains into the GEMM epilogues / stencil prologues.

---

## 6. Triton design space

Pipeline temporaries kept in **NHWC-flat `(M, ·)`** contiguous buffers so GEMMs are simple row-major:
`grad_x_projected (M,C)`, `grad_x_grn (M,C4)`, `grad_x_expanded (M,C4)`, `grad_x_ln (M,C)`,
`grad_x_nhwc (M,C)`, `grad_x_dwconv (B,C,H,W)`.

### 6.1 Kernel decomposition (baseline modular plan for c001)
- **K1 — permute+drop-path+bias** (grid over (M tiles, C)): compute `grad_x_projected[b,h,w,c] =
  grad_output[b,c,h,w]*drop_mask[b]/keep_prob`; accumulate `grad_pwconv2_bias[c]` (reduction over M).
- **K2 — GEMM** `grad_x_grn = grad_x_projected @ pwconv2_weight` (M×C×C4), fp32 `tl.dot`.
- **K3 — GEMM** `grad_pwconv2_weight = grad_x_projectedᵀ @ x_grn` (C×C4, K=M).
- **K4 — GRN reduce (pass 1)**: per (b,c) spatial sums → `grad_norm_features (B,C4)`; global sums →
  `grad_grn_weight (C4)`, `grad_grn_bias (C4)`. Then compute `grad_global_features (B,C4)` from
  `grad_norm_features`, `gf_mean`, `global_features` (§2.5, non-standard).
- **K5 — GRN apply + GELU (pass 2)**: `grad_x_gelu` (residual + scaled + L2-norm paths) then GELU-grad →
  `grad_x_expanded (M,C4)`.
- **K6 — GEMM** `grad_x_ln = grad_x_expanded @ pwconv1_weight` (M×C4×C).
- **K7 — GEMM** `grad_pwconv1_weight = grad_x_expandedᵀ @ x_ln` (C4×C, K=M); `grad_pwconv1_bias (C4)`
  (reduce over M — can be epilogue of K5 or a small reduce).
- **K8 — LN affine + normalization backward**: per-token (row) load C=128 channels, compute
  `grad_layernorm_weight/bias` (reduce over M, accumulate) and `grad_x_nhwc (M,C)` via §2.9–2.10.
  C=128 fits one row-block → single-pass row kernel (like Triton LN-bwd).
- **K9 — depthwise input-grad stencil**: `grad_x = stencil(grad_x_nhwc→NCHW, W_flip) + grad_output`.
- **K10 — depthwise weight-grad + bias**: `grad_dwconv_weight (C,49)` and `grad_dwconv_bias (C)` via
  correlation of `grad_x_dwconv` with padded `residual`.

### 6.2 Depthwise weight-grad options (highest-value kernel)
- (a) **One program per channel**, serial grid-stride over `B×HW`, 49 fp32 register accumulators.
  Simplest/deterministic; risk: single program serializes all M for a channel (W3 heavy).
- (b) **Split over (channel, spatial-block)**, each program accumulates 49 partials for its block, then
  `atomic_add` into `grad_dwconv_weight[c]` (or a `(nblk,C,49)` scratch + reduce kernel). Better
  parallelism for W3; atomics add nondeterminism (tolerable).
- (c) **Tile over channels in the inner axis** (vectorized 49-tap over a channel block) to raise ILP.
  Start with (a) for c001 correctness; move to (b)/(c) for W3 throughput.

### 6.3 Depthwise input-grad options
- Per-`(b, c-block, spatial-tile)` stencil, 49 taps of flipped weight, boundary-masked, `+ grad_output`.
  Channels contiguous (NCHW) → coalesced along W. Consider caching the 49 weights in registers/shared.

### 6.4 GEMM tuning axes
- Block sizes `(BM, BN, BK)`, `num_warps ∈ {4,8}`, `num_stages ∈ {2,3,4}`, `input_precision`
  (`ieee` first, `tf32` later), grouped-M ordering for L2 reuse. K is 128/512/M.
- **Fusion candidates**: fold K1's drop-path/permute into K2's A-load; fold `grad_pwconv2_bias` /
  `grad_pwconv1_bias` into GEMM epilogues; fuse K4/K5 GELU into K6's A-load; fuse LN-affine into K8.
  Autotune once shapes are known; C4=512, C=128 are TensorCore-friendly multiples.

### 6.5 Persistence / occupancy
- Reuse `(M,C4)` buffers across stages to cap DRAM. Small workloads (W1/W4) are launch-bound → minimize
  kernel count / prefer fused kernels there; large W3 is compute/BW-bound → prioritize GEMM + weight-grad
  throughput. A single set of launch params must serve all 5 (immutable per candidate), so autotune keys
  on shape.

---

## 7. Implementation & candidate strategy (for plan.md later)

- **c001 — correctness-first modular port.** All 10 kernels above, `input_precision="ieee"`, deterministic
  reductions, conservative block sizes. Goal: pass all 5 workloads; likely already > reference (kills the
  Python loop). Establishes the correctness baseline and per-tensor sanity.
- **c002+ — targeted speed** once green: (i) depthwise weight-grad option (b) split+atomic for W3;
  (ii) GEMM autotune / grouped-M; (iii) fuse element-wise chains into GEMM epilogues & stencil prologues;
  (iv) try `tf32` (guard: must keep all 5 passing); (v) reduce kernel count for W1/W4 launch overhead.
- Change **one lever per candidate id**; re-evaluate; keep only monotone geomean improvements; record
  parent/hash/hypothesis/per-workload/decision in `candidates.jsonl`.

## 8. Validation strategy

- **No local execution available** (Bash denied; no CUDA/profiler/nvidia-smi allowed). The *only* trusted
  signal is `./scripts/evaluate_candidate.sh feedback cNNN` (5 workloads = 1 eval). Budget: 100 evals,
  tokens soft 1.0M / hard 1.2M — so **spend evals deliberately**, not as a debugger.
- **Pre-eval self-checks (by construction):** (1) shape/stride table for every kernel I/O matches §1.1
  outputs; (2) reverse-order math matches §2 line-by-line, including the two non-standard steps; (3)
  depthwise flip/offset cross-check (§2.12); (4) all `tl.dot` accumulate fp32; (5) no Torch compute path.
- **Failure triage plan:** the evaluator reports per-workload pass/fail (and, we expect, which output
  mismatches). Map failures to suspects — `grad_x`/`grad_dwconv_weight` → conv indexing; `grad_grn_*`/
  `grad_x_gelu` → §2.5 non-standard step; broad small-magnitude misses on many tensors → TF32; single
  tensor all-fail → a wrong constant/transpose.
- **Metric:** geometric-mean speedup across the 5 feedback workloads, gated on every workload passing
  correctness. Final 14-workload eval is operator-approval-only; never auto-run `final`.
- **Convergence / stop:** stop when geomean plateaus across successive candidates, or on budget; then
  write `SEARCH_COMPLETE` with the reason.

## 9. Open questions to resolve during implementation
- Exact evaluator match rule (assumed `|a-b| ≤ atol + rtol·|ref|`, ≥98% elements). Confirm empirically on c001.
- Whether `tf32` passes `rtol=1e-3` on the small workloads (W1/W2/W4) — test as a dedicated candidate.
- Best depthwise weight-grad parallelization for the wide range M∈[1568, 50176] under one immutable config.
