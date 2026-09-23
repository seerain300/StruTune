# Draft — L2/040 `altup_predict_correction_cycle_backward`

Status: analysis only. No code, no `plan.md` yet. This document establishes the
operation semantics, constraints, numerical risks, the Triton design space, and
a validation strategy so that the executable plan (`docs/plan.md`) can be written
with full confidence next.

---

## 1. Operation summary

This is the **backward pass** of the Gemma-3n "AltUp" predict→correct cycle. The
forward pass (not given, but recomputed inside `run`) does, for a stack of
`N = altup_num_inputs = 3` hidden variants over `H = hidden_size = 2304`:

1. **Predict step** — RMSNorm the active hidden variant, a router linear
   (`H→3`), `tanh`, a coefficient linear (`3→9`), reshape/permute to a per-token
   `3×3` mixing matrix, mix the `N` hidden variants along the channel axis
   (a per-token `[H,3]·[3,3]` matmul), add a residual.
2. **Correct step** — RMSNorm the `activated` tensor, router linear, `tanh`,
   correction-coefficient linear (`3→3`) `+1`, form the innovation
   `activated − predictions[idx]`, broadcast-scale it across `N`, add to
   predictions.

The task's `run` **recomputes the whole forward** (activations were not
stashed) and then backpropagates to produce six gradients.

### Constants (from `definition.json`)
- `N = altup_num_inputs = 3` (fixed).
- `H = hidden_size = 2304` (fixed). `2304 = 2^8 · 3^2 = 256·9 = 768·3`. **Not a
  power of two**; `next_pow2(2304) = 4096`. Divisible by 128/256/384/768 (useful
  for clean H-tiling), not by 512.
- `router_scale = H^-1 = 1/2304` (exact fp32 constant `1.0/2304.0`).
- `rms_norm_eps = 1e-6` (workload scalar).
- `altup_active_idx = 0` in **all five** feedback workloads (but the kernel must
  treat it as a runtime scalar for generality on the hidden 16-workload set).

### Tensor inventory

Inputs (all `bf16` unless noted):
| name | shape | note |
|---|---|---|
| `grad_corrected` | `[N,B,S,H]` | upstream grad of corrected output |
| `hidden_states` | `[N,B,S,H]` | predict-step input |
| `activated` | `[B,S,H]` | correct-step input (MLP output) |
| `prediction_coef_weight` | `[N*N=9, N=3]` | predict coef linear weight |
| `correction_coef_weight` | `[3,3]` | correct coef linear weight |
| `router_weight` | `[3, H]` | shared router linear weight |
| `norm_weight` | `[H]` | shared RMSNorm weight |
| `altup_active_idx` | scalar int32 | active variant index |
| `rms_norm_eps` | scalar fp32 | RMSNorm epsilon |

Outputs:
| name | shape | dtype |
|---|---|---|
| `grad_hidden_states` | `[N,B,S,H]` | `bf16` |
| `grad_activated` | `[B,S,H]` | `bf16` |
| `grad_prediction_coef_weight` | `[9,3]` | `fp32` |
| `grad_correction_coef_weight` | `[3,3]` | `fp32` |
| `grad_router_weight` | `[3,H]` | `fp32` |
| `grad_norm_weight` | `[H]` | `fp32` |

Note the router and norm weights are **shared** between the predict and correct
steps; their gradients are the **sum** of the two contributions. The coef
weights are step-specific.

### Feedback workloads (fixed, 5 count as one evaluation)
| uuid short | B | S | rows = B·S | atol | rtol | match |
|---|---|---|---|---|---|---|
| fcd64c92 | 64 | 613 | 39232 | 0.015 | 0.05 | 0.98 |
| 5834489e | 64 | 128 | 8192 | 0.0075 | 0.05 | 0.98 |
| f2508211 | 4 | 256 | 1024 | 0.0029 | 0.05 | 0.98 |
| e9c4303d | 64 | 256 | 16384 | 0.012 | 0.05 | 0.98 |
| a6c812e5 | 8 | 373 | 2984 | 0.0045 | 0.05 | 0.98 |

`altup_active_idx = 0`, `eps = 1e-6` for all. All inputs random. Observations:
- Row counts span **1024 → 39232**; both small- and large-grid regimes must be
  fast. 1024 rows still saturates a 108-SM A800 comfortably with a per-row grid.
- The **tightest absolute tolerance is 0.0029** (small B). `match_ratio=0.98`
  means up to 2% of elements per output may exceed tolerance — headroom for
  bf16 rounding on the `bf16` outputs but no excuse for a wrong formula.

---

## 2. Exact math the kernel must reproduce (per-token derivation)

Everything below is **per token row** `r = (b,s)` with a length-`H` feature axis
and a tiny `N=3` channel axis. Let subscripts `k,j,q ∈ {0,1,2}`, `p ∈ {0..8}`,
`h ∈ {0..H-1}`. All compute is in **fp32** (the reference upcasts every input).

### 2.1 Forward recomputation

Shared RMSNorm-router-tanh helper, applied to a length-`H` fp32 vector `x`:
```
var   = mean_h(x[h]^2)                    # /H exactly
rstd  = rsqrt(var + eps)
nrm   = x * rstd                          # "normalized"
nmd   = nrm * norm_weight                 # "normed"
scl   = nmd * router_scale                # "scaled" (router_scale = 1/H)
routed[k] = sum_h scl[h] * router_weight[k,h]      # k=0,1,2
mod[k]    = tanh(routed[k])               # "modalities"
```
returning `(rstd, nrm, scl, mod)`.

**Predict:** apply helper to `x = hidden_states[idx, b, s, :]` →
`(rstd_P, nrm_P, scl_P, mod_P)`.
```
flat[p]   = sum_q mod_P[q] * prediction_coef_weight[p,q]     # p=0..8   (F.linear, W [9,3])
C[k,j]    = flat[3*j + k]                                    # per-token 3x3 mixing matrix
pred[j,h] = sum_k hidden_states[k,b,s,h] * C[k,j] + hidden_states[j,b,s,h]   # +residual
```
(The reshape→`[3,3]`→`permute(0,1,3,2)` collapses to `C[k,j]=flat[3j+k]`; the
`h_permuted@all_coefs` matmul collapses to the length-3 channel contraction above.)

**Correct:** apply helper to `x = activated[b,s,:]` →
`(rstd_C, nrm_C, scl_C, mod_C)`.
```
acoef[k]  = sum_q mod_C[q] * correction_coef_weight[k,q] + 1.0   # F.linear, W [3,3], then +1
innov[h]  = activated[b,s,h] - pred[idx,h]
```

### 2.2 Backward — correct step
```
gpred[k,h] = grad_corrected[k,b,s,h]                              # clone (residual path)
# corrected = innov_repeated * acoef_expanded + predictions
g_innov[h]      = sum_k grad_corrected[k,h] * acoef[k]            # sum over repeated N axis
g_acoef[k]      = sum_h grad_corrected[k,h] * innov[h]            # reduce over H  -> 3-vec
# accumulate correction-coef weight grad (outer product, summed over rows)
grad_correction_coef_weight[p,q] += g_acoef[p] * mod_C[q]        # [3,3]
g_mod_C[q]      = sum_p g_acoef[p] * correction_coef_weight[p,q]  # grad into modalities
grad_activated[h]  = g_innov[h]                                  # (part 1)
gpred[idx,h]      -= g_innov[h]                                  # innovation subtracts pred[idx]
# router/RMSNorm backward for correct step (see 2.4), adds to grad_activated
```

### 2.3 Backward — predict step
```
grad_hidden_states[k,h] = gpred[k,h]                             # clone of (possibly idx-modified) gpred
# channel matmul backward
grad_hidden_states[k,h] += sum_j gpred[j,h] * C[k,j]             # from grad_h_permuted
g_flat[3*i + j] = sum_h hidden_states[j,b,s,h] * gpred[i,b,s,h]  # reduce over H -> 9-vec
grad_prediction_coef_weight[p,q] += g_flat[p] * mod_P[q]         # [9,3], summed over rows
g_mod_P[q]      = sum_p g_flat[p] * prediction_coef_weight[p,q]
# router/RMSNorm backward for predict step (see 2.4), adds to grad_hidden_states[idx,:]
```
Note the index transpose: `g_flat[3*i+j] = Σ_h hidden[j]·gpred[i]` (i outer, j
inner) — comes from `grad_all_coefs_matmul.permute(0,1,3,2).reshape`. This must
be reproduced exactly.

### 2.4 Router + RMSNorm backward (shared shape, two instances)

Given `g_mod` (3-vec), the pre-tanh `mod` (3-vec), `scl` (H-vec), `rstd` scalar,
`nrm` (H-vec), source `x` (H-vec):
```
g_routed[k]     = g_mod[k] * (1 - mod[k]^2)                      # tanh'
grad_router_weight[k,h] += g_routed[k] * scl[h]                 # [3,H], summed over rows
g_scl[h]        = sum_k g_routed[k] * router_weight[k,h]
g_nmd[h]        = g_scl[h] * router_scale
grad_norm_weight[h] += g_nmd[h] * nrm[h]                        # [H], summed over rows
g_nrm[h]        = g_nmd[h] * norm_weight[h]
mbar            = mean_h( g_nrm[h] * x[h] )                      # /H exactly
g_x[h]          = g_nrm[h] * rstd - x[h] * rstd^3 * mbar        # RMSNorm input grad
```
- Predict instance (`x = hidden_states[idx]`): `grad_hidden_states[idx,h] += g_x[h]`.
- Correct instance (`x = activated`): `grad_activated[h] += g_x[h]`.
- `grad_router_weight` / `grad_norm_weight` accumulate **both** instances.

### 2.5 Data-flow ordering (critical)
`gpred[idx]` is decremented by `g_innov` **before** it is used in the predict
channel-matmul backward and in `g_flat`. So the per-row pipeline must be:
build `gpred[k,h] = grad_corrected[k,h] - (k==idx)·g_innov[h]` **first**, then
compute `grad_hidden_states` and `g_flat` from that modified `gpred`. Getting
this order wrong silently corrupts `grad_hidden_states[idx]`, `g_flat`, and
`grad_prediction_coef_weight`.

---

## 3. Structural insight → why this is a fusion win

- The op is **memory-bound**. The heavy tensors are the `[N,B,S,H]` /
  `[B,S,H]` ones. Ideal traffic per row: read `hidden(3H) + grad_corrected(3H)
  + activated(H) = 7H`, write `grad_hidden(3H) + grad_activated(H) = 4H`
  ⇒ ≈`11·H·2 B ≈ 50 KB/row` of unavoidable bf16 traffic, plus the tiny
  weight/coef grads. (`router_weight[3,H]` and `norm_weight[H]` are re-read per
  row but are small and L2-resident.)
- The reference materializes **~20+ full-size intermediates** (`predictions`,
  `grad_predictions`, `grad_innovation_repeated`, all `normalized/normed/scaled`
  for both steps, several `[3,B,S,H]` permutes/reshapes) and launches many
  kernels (RMSNorm ×2, linears ×6, tanh ×2, batched matmuls ×3, plus elementwise
  and reductions). Each is a full HBM round-trip. A single fused kernel that
  reads each input **once** and writes each output **once** should give a large
  multiplicative speedup; the ceiling is the ~11H bf16 traffic.
- All "matmuls" contract only over `N=3` (channel) or reduce over `H`. There is
  **no large GEMM** — no Tensor-Core tile needed. This is a fused elementwise +
  small-reduction kernel, not a GEMM kernel.

---

## 4. Triton design space

### 4.1 Parallelization axis
Everything is per-token; `H` reductions (variance, `mbar`, `g_acoef`, `g_flat`,
`grad_norm_weight`, `grad_router_weight`) couple the whole `H` axis of a row. So
**one row = one unit of work**, with the entire `H` axis handled inside the
program. This is the FlashAttention/RMSNorm "row-resident feature axis" pattern
and reads every input exactly once.

**Design A — one program per row (grid = `rows = B·S`).** `BLOCK_H` covers all
of `H` in a single block. Options for `H=2304`:
- `BLOCK_H = 4096` masked (`h < 2304`): simplest, ~44% lane waste.
- Inner loop over exact tiles (`768×3`, `384×6`, `256×9`): no waste but must keep
  all needed H-vectors resident across the pass boundary (variance→router→grads),
  i.e. reload or keep in SRAM. Since values are reused across ~3 sub-phases,
  keeping the whole row resident (single block) avoids re-reading global memory.
- Tradeoff: single big block risks register spilling (see §4.4); tiling reduces
  live registers but complicates the multi-pass reductions.

**Design B — one program per block of rows (`BLOCK_ROWS>1`), loop over rows.**
Same per-row math, but amortizes the **weight-gradient reduction**: keep a local
fp32 accumulator for `grad_router_weight[3,H]`, `grad_norm_weight[H]`,
`grad_prediction_coef_weight[9,3]`, `grad_correction_coef_weight[3,3]` in
SRAM/registers, and flush **once per program** instead of once per row. Also lets
`router_weight`/`norm_weight` be loaded once and reused. Cuts atomic contention
by `BLOCK_ROWS×`. Cost: `[3,H]` accumulator ≈ 27 KB fp32 + `[H]` ≈ 9 KB — fits in
SRAM; register pressure grows.

**Design C — split weight-grad reduction into a second kernel.** Main kernel
writes per-program partials to a `[num_programs, ...]` buffer; a tiny second
kernel reduces them. Avoids atomics entirely and is deterministic, at the cost of
an extra buffer + launch. Viable if atomics prove to be the bottleneck.

Decision lean: start with **Design A + atomics** (simplest correct baseline,
`c001`), then measure and move to **Design B** (row-blocking) to cut atomic
pressure, and keep **Design C** in reserve.

### 4.2 The weight-gradient reductions (the hard part)
`grad_router_weight[3,H]`, `grad_norm_weight[H]`, `grad_prediction_coef_weight[9,3]`,
`grad_correction_coef_weight[3,3]` all sum over **all rows**. Cross-program
accumulation options:
- **`tl.atomic_add` to fp32 output buffers.** For per-row Design A this is
  `rows` atomic passes over `H` for norm + `3H` for router → high contention on
  the same addresses (all rows hit the same `H` columns). Could bottleneck on
  A800 fp32 atomics. Mitigate with Design B (fewer flushes) or Design C.
- The coef-weight grads (`[9,3]`, `[3,3]`) are tiny (27/9 floats) — atomic-add
  once per program is negligible regardless of design.
- **Outputs must be pre-zeroed** since we accumulate with `+=`. The wrapper will
  allocate with `torch.zeros` (fp32) before launch. `grad_hidden_states` and
  `grad_activated` are fully written (not accumulated) so may use `empty`.

### 4.3 Numerical policy
- Load every bf16 input and immediately `.to(tl.float32)`; do **all** arithmetic
  in fp32; cast only the two `bf16` outputs at store time. This mirrors the
  reference's global `.float()`.
- `mean` must divide by exactly `H=2304` (both `var` and `mbar`). Use fp32 sum
  then `* (1.0/2304.0)` or `/2304.0` consistently.
- `router_scale` and the `1/H` mean factor are the same constant; keep it exact.
- `tanh`: use `libdevice.tanh` (accurate) over `x`; reuse the stored `mod` for
  the derivative `(1-mod^2)` rather than recomputing.
- `rsqrt`: `tl.rsqrt(var+eps)` or `1/tl.sqrt`. `rstd^3` via `rstd*rstd*rstd`.
- Reduction order (tree vs sequential) differs from PyTorch; with fp32
  accumulation the delta is ≪ the bf16 output ULP, comfortably inside the
  tolerances (tightest atol 0.0029, `match_ratio 0.98`).

### 4.4 Resource budget (Design A, `BLOCK_H≈2304/4096`)
Live fp32 H-vectors needed roughly simultaneously: `h0,h1,h2` (3),
`grad_corrected 0,1,2` (3), `activated` (1), `norm_weight` (1), plus a handful of
derived vectors (`nrm_P, scl_P, nrm_C, scl_C, gpred[idx], g_x, ...`). ≈ 10–16
arrays × 2304 fp32. Across `num_warps=8` (256 threads) that is ≈ 9 elems/thread ×
16 ≈ 140 registers/thread — tight; `num_warps=16` halves it. Expect spilling to
matter → **autotune `num_warps ∈ {4,8,16}`** and consider recomputing cheap
vectors (e.g. `scl` from `nrm`) instead of holding them. `router_weight[3,H]`
adds 3 more H-vectors if held; can be streamed/reloaded from L2.

### 4.5 Candidate ladder (tentative — finalized in `plan.md`)
- `c001`: Design A, single fused kernel, `BLOCK_H=4096` masked, atomics for all
  four weight grads, `num_warps=8`. Correctness-first baseline.
- `c002+`: autotune `num_warps`/`num_stages`; exact H-tiling to drop masked
  waste; hoist `router_weight` handling.
- `c00x`: Design B row-blocking to cut atomic contention (biggest expected win on
  large-row workloads).
- `c00y`: Design C two-stage reduction if atomics dominate; possibly `bf16`
  vectorized loads (`ld.global.v` alignment) and `tl.multiple_of`/`max_contiguous`
  hints.
- Consider whether `grad_activated`/`grad_hidden` stores can be pipelined with
  the reduction flush.

### 4.6 Layout / indexing notes
- `[N,B,S,H]` is contiguous; row `r=(b,s)` for variant `k` starts at
  `((k*B + b)*S + s)*H = (k*rows + r)*H`. So a program with `row_id = b*S+s`
  addresses variant `k` at base `(k*rows + row_id)*H`, stride 1 over `h`.
  `activated`/`grad_activated` at `row_id*H`. Clean, coalesced.
- `router_weight[k,h]` at `k*H + h`; `norm_weight[h]` at `h`.
- `altup_active_idx` passed as a scalar kernel arg; apply variant-specific terms
  branchlessly via `tl.where(k == idx, ...)` over the 3 explicit variants (N is a
  compile-time 3, so unroll k=0,1,2 rather than a masked N-axis block).

---

## 5. Constraints & compliance
- **Triton primary**, PyTorch only for metadata/allocation/launch. No Torch/CPU/
  NumPy/CUDA-extension computational fallback (a failed Triton kernel is invalid).
- Target A800 `sm_80`: fp32 SIMT throughput, no fp8/TF32-matmul tricks relevant
  here (no large GEMM); bf16 loads/stores + fp32 math.
- Output dtypes exactly: two `bf16`, four `fp32`. Return **tuple in the reference
  order**: `(grad_hidden_states, grad_activated, grad_prediction_coef_weight,
  grad_correction_coef_weight, grad_router_weight, grad_norm_weight)`.
- Pre-zero the four accumulated fp32 outputs before launch.
- Keep each candidate immutable; new source ⇒ new `cNNN`. Only evaluate via
  `./scripts/evaluate_candidate.sh feedback cNNN`. Budget: 100 evals; token soft
  1.0M / normal 1.5M / hard 1.65M. `final` only with operator approval.
- Handle `altup_active_idx` generally (feedback fixes it to 0, hidden set may not).

## 6. Numerical-risk register
1. **idx-subtraction ordering** (§2.5) — corrupts `grad_hidden_states[idx]`,
   `g_flat`, `grad_prediction_coef_weight` if applied late. *Mitigation:* build
   modified `gpred` before any predict-backward use.
2. **`g_flat` index transpose** `3*i+j` with `Σ_h hidden[j]·gpred[i]` — easy to
   swap i/j. *Mitigation:* derived explicitly in §2.3; unit-check against a hand
   example mentally before eval.
3. **Shared router/norm weight-grad accumulation** must sum predict+correct.
   *Mitigation:* single accumulator per output, both instances add in.
4. **Atomic accumulation into non-zeroed buffers** → garbage. *Mitigation:*
   `torch.zeros` for the four accumulated outputs.
5. **bf16 rounding on `grad_hidden_states`/`grad_activated`** near tight atol
   (0.0029). *Mitigation:* fp32 math throughout, cast only at store; rely on
   `match_ratio 0.98` headroom, not on it.
6. **Register spilling** with large `BLOCK_H` → slow, not wrong. *Mitigation:*
   autotune `num_warps`, recompute cheap vectors, consider tiling / Design B.
7. **`mean` divisor / `router_scale` constant drift** — must be exact `1/2304`.
8. **Coalescing** — ensure stride-1 `h` access; add `tl.max_contiguous`/
   `multiple_of` hints in later candidates.

## 7. Validation strategy
- **Execution is disabled in this environment** (Bash is deny-mode) and the rules
  forbid running CUDA / profilers / the external evaluator / any alternate
  correctness harness directly. Therefore the *only* execution-based signal is
  `./scripts/evaluate_candidate.sh feedback cNNN`, which runs all five workloads
  as **one** evaluation.
- Pre-evaluation correctness rests on **analytic derivation** (§2, cross-checked
  line-by-line against the reference in `definition.json`), **shape/stride
  bookkeeping** (§4.6), and the **risk register** (§6). I will re-read the
  reference term-by-term against the kernel before spending each evaluation.
- Use the feedback evaluator sparingly and purposefully: `c001` establishes
  correctness + a speed baseline; subsequent candidates change **one** thing at a
  time so each eval attributes cause cleanly. Record per-workload pass/fail,
  atol/rtol margins, geomean speedup, decision, cumulative eval count, and skill
  usage in `candidates.jsonl` (append-only).
- Watch the **tightest-tolerance small-B workloads** (f2508211 atol 0.0029,
  a6c812e5 0.0045) as the correctness canaries; watch the **largest-row**
  workload (fcd64c92, 39232 rows) as the atomic-contention / perf canary.
- Convergence: stop when successive candidates no longer improve geomean
  meaningfully (or budget), then write `SEARCH_COMPLETE` with the reason. Never
  run `final` without operator approval.

## 8. Open questions for `plan.md`
- Design A vs B for `c001`: lean A (simplest correct), then B for atomics.
- `BLOCK_H` masked-4096 vs exact-tile (768/384/256) — measure waste vs spilling.
- Whether to hold `router_weight[3,H]` resident or stream from L2.
- Whether coef/weight-grad atomics need Design C (two-stage) at 39232 rows.
- Skill check: consult `KernelWiki` for A800/`sm_80` RMSNorm-backward fusion and
  atomic-reduction patterns before finalizing the plan.
