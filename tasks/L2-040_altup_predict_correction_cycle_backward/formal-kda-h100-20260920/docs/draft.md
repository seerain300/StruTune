# Draft — L2/040 `altup_predict_correction_cycle_backward`

Target: NVIDIA H100 (`sm_90`). Primary implementation in Triton; PyTorch allowed only
for metadata / launch plumbing. No Torch/CPU/NumPy/CUDA-extension computational fallback.
Submission entry point: `solution/solution.py::run(...)` with the exact reference signature.

This document analyzes the operation, its constraints, numerical risks, the Triton design
space, and a validation strategy. No code and no plan yet (per turn instructions).

---

## 1. Operation overview

This is the **backward pass** of the Gemma-3n AltUp "predict → correct" cycle. It recomputes
the forward intermediates (there is no saved activation stash) and then propagates gradients
back through, in reverse order:

1. **Correct step backward** (uses recomputed `activated` RMSNorm/router/tanh path and the
   recomputed `predictions[active_idx]`).
2. **Predict step backward** (uses recomputed `hidden_states[active_idx]` RMSNorm/router/tanh
   path and the per-token 3×3 prediction-coefficient matmul).

It returns six gradients:

| output | shape | dtype |
|---|---|---|
| `grad_hidden_states` | `[N, B, S, H]` | bf16 |
| `grad_activated` | `[B, S, H]` | bf16 |
| `grad_prediction_coef_weight` | `[N*N=9, N=3]` | **fp32** |
| `grad_correction_coef_weight` | `[N=3, N=3]` | **fp32** |
| `grad_router_weight` | `[N=3, H]` | **fp32** |
| `grad_norm_weight` | `[H]` | **fp32** |

### Fixed constants (from `definition.json`)
- `altup_num_inputs` **N = 3** (const)
- `hidden_size` **H = 2304** (const)
- `altup_num_inputs_sq` = 9
- `router_scale = H**-1 = 1/2304`
- `altup_active_idx = 0` in **all 16** feedback workloads (int32 scalar input).
- `rms_norm_eps = 1e-6` in all 16 workloads (fp32 scalar input).

### Variable axes (the 16 feedback workloads)
`(batch_size, seq_len)` pairs → let `M = B*S` (total tokens):

| # | B | S | M = B·S | atol |
|---|---|---|---|---|
| 1 | 64 | 256 | 16384 | 0.012 |
| 2 | 1 | 1024 | 1024 | 0.0029 |
| 3 | 4 | 293 | 1172 | 0.0035 |
| 4 | 1 | 256 | 256 | 0.0014 |
| 5 | 8 | 373 | 2984 | 0.0045 |
| 6 | 1 | 512 | 512 | 0.0026 |
| 7 | 4 | 256 | 1024 | 0.0029 |
| 8 | 8 | 256 | 2048 | 0.0031 |
| 9 | 16 | 512 | 8192 | 0.0075 |
| 10 | 16 | 256 | 4096 | 0.0072 |
| 11 | 4 | 1024 | 4096 | 0.0072 |
| 12 | 8 | 512 | 4096 | 0.0072 |
| 13 | 64 | 613 | 39232 | 0.015 |
| 14 | 16 | 449 | 7184 | 0.010 |
| 15 | 2 | 512 | 1024 | 0.0029 |
| 16 | 64 | 128 | 8192 | 0.0075 |

All: `max_rtol = 0.05`, `required_match_ratio = 0.98`.

Observations:
- `M` spans **256 → 39232**. Boundary/small-M cases (4, 6, 2) have the **tightest** atol
  and the least parallelism → launch/occupancy matters there. Large-M cases (13, 1, 16, 9)
  dominate absolute runtime and set the throughput ceiling.
- `S` values are frequently **non-power-of-two and odd** (293, 373, 613, 449). Since we will
  flatten to `M = B*S`, that is fine as long as the kernel masks the `M` tail. `H = 2304`
  is also **not** a power of two → the `H` dimension needs masking or fixed-tile looping.

---

## 2. Exact math, simplified for a tiny-N fused kernel

The key structural fact: **N = 3 is tiny and H = 2304 is the only large inner dimension.**
Every "matmul" in the reference is therefore one of:
- a contraction over **H** producing an N-vector (router forward) or consuming an N-vector
  (router backward) — a *matvec*, memory-bound, no tensor-core benefit for width-3 output;
- a contraction over the tiny **3 / 9** dims (coef projections) — trivial scalar work;
- a contraction over **M** (token reduction) producing the small weight gradients — a
  cross-token reduction, best done by accumulation, not GEMM.

So there is **no genuine large GEMM**. The whole op is a **bandwidth-bound fused
elementwise + reduction** kernel. This is the central design insight.

Below, per token row `t = (b,s)`, I collapse the reference into scalar/vector-per-row form.
`idx = altup_active_idx = 0`. Let `rs = router_scale`.

### Forward recompute (per row)
RMSNorm helper `RN(x[H]) → (rstd, normalized[H], scaled[H])`:
- `var = mean_h(x^2)`, `rstd = rsqrt(var + eps)`, `normalized = x*rstd`,
  `scaled = normalized * norm_weight * rs`.  *(note `norm_weight`, then `*rs`)*

Predict path (on `x_p = hidden_states[idx, t, :]`, fp32):
- `(rstd_p, normalized_p, scaled_p) = RN(x_p)`
- `routed_p[n] = Σ_h scaled_p[h]·router_weight[n,h]`, `n=0..2`
- `mod_p[n] = tanh(routed_p[n])`
- `coefs_flat[c] = Σ_n mod_p[n]·pred_coef_weight[c,n]`, `c=0..8`  *(F.linear with `[9,3]` weight)*

Correct path (on `x_c = activated[t, :]`, fp32):
- `(rstd_c, normalized_c, scaled_c) = RN(x_c)`
- `routed_c[n] = Σ_h scaled_c[h]·router_weight[n,h]`, `mod_c[n] = tanh(routed_c[n])`
- `coefs_correct[n] = Σ_m mod_c[m]·correction_coef_weight[n,m] + 1`, `n=0..2`

Recompute `predictions[idx]` (only column `idx` is ever consumed downstream):
- `all_coefs[i,j] = coefs_flat[3*j + i]` (from `reshape[B,S,3,3].permute(...,3,2)`).
- `predictions_idx[h] = Σ_k hidden_states[k,t,h]·all_coefs[k,idx] + hidden_states[idx,t,h]`
  `= Σ_k hidden_states[k,t,h]·coefs_flat[3*idx + k] + hidden_states[idx,t,h]`.
  For `idx=0`: uses `coefs_flat[0..2]`.
- `innovation[h] = activated[t,h] − predictions_idx[h]`.

### Correct-step backward (per row)
`gc_k[h] = grad_corrected[k,t,h]` (fp32), `k=0..2`.
- `grad_innovation[h] = Σ_k coefs_correct[k]·gc_k[h]`  (weighted sum of the 3 grad rows).
- `grad_coefs_correct[k] = Σ_h gc_k[h]·innovation[h]`  (3 scalars; `innovation_repeated`
  is the *same* `innovation` for all k).
- `grad_correction_coef_weight[n,m] += grad_coefs_correct[n]·mod_c[m]`  → **[3,3] reduce over M**.
- `grad_mod_c[m] = Σ_n grad_coefs_correct[n]·correction_coef_weight[n,m]`  (3 scalars).
- `grad_routed_c[m] = grad_mod_c[m]·(1 − mod_c[m]^2)`  (tanh′).
- `grad_router_weight[m,h] += grad_routed_c[m]·scaled_c[h]`  → **[3,H] reduce over M** (shared buffer).
- `grad_scaled_c[h] = Σ_m grad_routed_c[m]·router_weight[m,h]`.
- `grad_normed_c[h] = grad_scaled_c[h]·rs`.
- `grad_norm_weight[h] += grad_normed_c[h]·normalized_c[h]`  → **[H] reduce over M** (shared buffer).
- `grad_normalized_c[h] = grad_normed_c[h]·norm_weight[h]`.
- `mean_c = mean_h(grad_normalized_c[h]·x_c[h])`.
- `grad_act_router[h] = grad_normalized_c[h]·rstd_c − x_c[h]·rstd_c^3·mean_c`.
- `grad_activated[t,h] = grad_innovation[h] + grad_act_router[h]`  → **write bf16**.
- `grad_predictions[k,h] = gc_k[h]` for `k≠idx`; `grad_predictions[idx,h] = gc_idx[h] − grad_innovation[h]`.

### Predict-step backward (per row)
- `grad_h_permuted[h,i] = Σ_j grad_predictions[j,h]·all_coefs[i,j]`
  `= Σ_j grad_predictions[j,h]·coefs_flat[3*j + i]`.
- `grad_all_coefs_flat[c] = Σ_h hidden_states[(c%3),t,h]·grad_predictions[(c//3),h]`, `c=0..8`
  (derived from `matmul(h_permutedᵀ, grad_predictions_permuted)` then `permute(...,3,2)`+reshape).
- `grad_prediction_coef_weight[c,n] += grad_all_coefs_flat[c]·mod_p[n]` → **[9,3] reduce over M**.
- `grad_mod_p[n] = Σ_c grad_all_coefs_flat[c]·pred_coef_weight[c,n]`  (3 scalars).
- `grad_routed_p[n] = grad_mod_p[n]·(1 − mod_p[n]^2)`.
- `grad_router_weight[n,h] += grad_routed_p[n]·scaled_p[h]` → **same [3,H] buffer** (predict+correct combined).
- `grad_scaled_p[h] = Σ_n grad_routed_p[n]·router_weight[n,h]`.
- `grad_normed_p[h] = grad_scaled_p[h]·rs`.
- `grad_norm_weight[h] += grad_normed_p[h]·normalized_p[h]` → **same [H] buffer** (combined).
- `grad_normalized_p[h] = grad_normed_p[h]·norm_weight[h]`.
- `mean_p = mean_h(grad_normalized_p[h]·x_p[h])`.
- `grad_active_input[h] = grad_normalized_p[h]·rstd_p − x_p[h]·rstd_p^3·mean_p`.
- `grad_hidden_states[k,t,h] = grad_predictions[k,h] + grad_h_permuted[h,k]`, and for `k=idx`
  additionally `+ grad_active_input[h]`.  → **write bf16** (3 rows).

`grad_router_weight` and `grad_norm_weight` are single fp32 buffers that receive **both** the
correct-path and predict-path contributions (norm_weight and router_weight are shared params).

### Per-row I/O footprint (the bandwidth floor)
Big reads per row: `hidden_states[0..2]` (3×H), `activated` (1×H), `grad_corrected[0..2]`
(3×H) = **7 H-vectors**. Big writes: `grad_hidden_states[0..2]` (3×H) + `grad_activated`
(1×H) = **4 H-vectors**. Shared small reads (`router_weight` 3×H, `norm_weight` H,
`pred_coef` 9×3, `corr_coef` 3×3) are reused across all rows → cache-resident.
⇒ Ideal traffic ≈ **11·M·H bf16 elements**. The reference executes dozens of separate
CUDA ops over `[3,M,H]`/`[M,H]` tensors (many full passes + several temporaries the size of
the input) → large speedup headroom for a single fused kernel.

---

## 3. Constraints

- **Triton-only compute.** Torch permitted for allocation, dtype/shape handling, launch grid,
  and returning the tuple. No torch math on the hot path, no fallback of any kind.
- **Exact signature / return order.** `run(grad_corrected, hidden_states, activated,
  prediction_coef_weight, correction_coef_weight, router_weight, norm_weight,
  altup_active_idx, rms_norm_eps)` → 6-tuple in the documented order and dtypes.
- **Output dtypes are load-bearing:** two bf16 tensors, four **fp32** weight-gradient tensors.
  Getting a weight-grad dtype wrong will fail the harness even if values match.
- **`altup_active_idx`**: 0 in every workload, but it is a runtime input. Implement it
  parametrically (pass as kernel arg or `tl.constexpr` specialized to the observed value)
  so a non-zero index would still be correct; do **not** silently hardcode in a way that
  breaks generality of the selection of the RMSNorm-predict row and the residual injection row.
- **`rms_norm_eps`** must be added to variance in fp32 exactly as the reference does.
- **Non-power-of-two dims.** `H=2304` and odd `S` → the flattened `M` tail and the `H`
  dimension both need masking (or fixed-size `H` tiling with a boundary block). `2304 = 2^8·9
  = 256·9 = 512·4.5`; convenient exact tilings: `9×256`, `18×128`, `4×576`, `6×384`, `3×768`.
- **Isolation / process rules.** Only `KernelWiki` and `ncu-report-skill` external knowledge.
  Evaluate exclusively via `./scripts/evaluate_candidate.sh feedback cNNN`. Profiling only via
  `./scripts/ncu_profile.sh` under the `ncu-report-skill` workflow. **Never** profile and
  evaluate concurrently (foreign process on the locked GPU ⇒ discarded measurement, wasted
  budget). Do not run CUDA/`nvidia-smi`/the evaluator/any alternate correctness harness directly.
- **Budget.** 100 candidate evals; token soft limit 9M / normal 10M / hard 11M. The full
  16-workload feedback set = **one** candidate evaluation. Each meaningful source/config/launch
  change ⇒ a new immutable `cNNN` id; never reuse an id for changed source.

---

## 4. Numerical risks

1. **fp32 internal math is mandatory.** The reference casts *every* input to `.float()` and
   does all arithmetic in fp32, casting only the two `grad_*` activation outputs back to bf16
   at the very end. We must accumulate in fp32 throughout (variances, matvecs, reductions,
   tanh, RMSNorm-backward). Doing intermediate math in bf16 will blow atol on the tight cases
   (workload 4: atol 0.0014).

2. **Avoid `tl.dot` tf32 on the H-contractions.** The only large contractions are over H with
   width-3 outputs. `tl.dot` on fp32 inputs defaults to **tf32** (~10-bit mantissa), which can
   exceed the tight atol on small-M workloads and is *slower* than a plain masked
   multiply-reduce for a width-3 result anyway. Plan: implement `routed_*` and `grad_scaled_*`
   as explicit fp32 `tl.sum` over H (load the 3 `router_weight` rows once). This keeps full
   fp32 precision **and** avoids the tensor-core setup overhead for a degenerate GEMM shape.

3. **Cross-token reduction (weight grads) accumulation order.** `grad_prediction_coef_weight`,
   `grad_correction_coef_weight`, `grad_router_weight`, `grad_norm_weight` are sums over up to
   ~39k tokens. Options: (a) `tl.atomic_add` into fp32 global buffers; (b) block-local fp32
   accumulation then one atomic per program; (c) partial buffers + a second reduction kernel.
   Atomic-add order is nondeterministic → low-bit differences vs torch's fp32 tree reduction,
   but with `rtol=0.05` and `match_ratio=0.98` this is comfortably safe. Prefer **(b)** to cut
   atomic traffic by `BLOCK_M`. Watch: buffers must be **zero-initialized** before the kernel
   (use `torch.zeros`), because we accumulate.

4. **`rsqrt` / `rstd^3`.** RMSNorm-backward uses `rstd^3`. Compute `rstd` once in fp32 via
   `tl.rsqrt`/`1/tl.sqrt`; reuse `rstd*rstd*rstd`. `eps=1e-6` keeps `var+eps` well away from 0.

5. **`tanh` fidelity.** Use `tl.extra.libdevice.tanh` (or `tl.math.tanh`) in fp32; matches
   `torch.tanh` to well within tolerance. The `tanh′ = 1 − tanh²` uses the *recomputed*
   `mod` (the reference reuses `modalities`, i.e. tanh(routed), not a separately stored value).

6. **`mean` over H uses the true H, not the padded tile.** Both `variance` and
   `mean_grad_normalized_x` divide by `H=2304`. When masking the `H` tail, masked lanes must
   contribute 0 to the sum and the divisor must remain the exact `H` — not the padded width.

7. **Residual + selection bookkeeping.** `grad_hidden_states[idx]` receives four additive
   terms (`gc_idx − grad_innovation + grad_h_permuted[:,idx] + grad_active_input`); the other
   two rows receive two terms. `grad_predictions[idx]` must include the `−grad_innovation` term
   *before* it feeds both `grad_all_coefs_flat` and `grad_h_permuted`. Getting this ordering
   wrong silently corrupts the predict-backward matvec.

8. **bf16 rounding on outputs.** Final cast to bf16 for the two activation grads is the same
   lossy step the reference performs, so it does not *add* error relative to the reference — as
   long as the pre-cast fp32 values match.

---

## 5. Triton design space

### 5.1 Primary design — single fused token-parallel kernel
- **Grid:** over token blocks; program `pid` owns `BLOCK_M` rows of the flattened `M = B*S`
  axis, full `H` handled internally. Mask the `M` tail.
- **Per row work:** load the 7 big H-vectors (fp32-upcast), run the forward recompute + both
  backward paths from §2, write the 4 big H-vectors, and accumulate the four weight-grad
  partials in registers. One `tl.atomic_add` (or block-reduced atomic) per program per weight
  buffer.
- **H handling — two sub-options:**
  - **(A) Whole-row-in-SRAM:** one program = a few rows, `BLOCK_H ≥ 2304` (pad to 2560/4096
    with mask, or loop fixed `H`-tiles). Because several H-reductions feed later H-elementwise
    steps (variance→scaled→routed; grad_routed→grad_scaled→mean→grad_x), holding the row lets
    us reduce then reuse without re-reading global memory. Register/SMEM pressure is the limit:
    ~10 live fp32 H-vectors ≈ 90 KB — pushes occupancy down; keep `BLOCK_M` small (1–2).
  - **(B) Streaming multi-pass over H-tiles:** loop H-tiles once to get all H-reductions
    (`var_p`, `var_c`, `routed_p[3]`, `routed_c[3]` — but note `routed` needs `scaled` which
    needs `rstd` which needs `var`, a dependency chain), then a second loop to compute
    `grad_scaled`/outputs (needs `mean`, itself an H-reduction of grad-side quantities). This
    implies **2–3 sequential H passes** re-reading the row from L2. Lower register pressure,
    higher L2 traffic. Given `H=2304` rows are L2-resident per program, this is attractive.
- **Tiling for `H=2304`:** exact factor tiles avoid wasted masked lanes — e.g. `9` iterations
  of `BLOCK_H=256`, or a single `BLOCK_H=2304`-with-mask if the compiler accepts a non-pow2
  `tl.arange` (else pad to 4096 + mask, or use `256×9`).
- **Autotune knobs:** `BLOCK_M ∈ {1,2,4,8}`, `BLOCK_H ∈ {256,512,768,2304}`,
  `num_warps ∈ {2,4,8}`, `num_stages`. Small-M workloads (256–1024) want small `BLOCK_M` /
  more programs for occupancy; large-M (39k) wants throughput.

### 5.2 Alternative — split into 2–3 kernels
- **K1 (forward recompute):** per row, compute `rstd_p, rstd_c`, `mod_p[3], mod_c[3]`,
  `coefs_flat[9]`, `coefs_correct[3]`, and `innovation`/`predictions_idx`. Store the *small*
  per-row tensors (`mod_p [M,3]`, `mod_c [M,3]`, `coefs_flat [M,9]`, `coefs_correct [M,3]`,
  `rstd_p/rstd_c [M]`) — these are cheap. **Do not** store `scaled_*`/`normalized_*` ([M,H],
  input-sized) — recompute them in K2 to stay bandwidth-bound.
- **K2 (backward + writes + weight-grad partials):** recompute the H-vectors and finish.
- Trade-off: cleaner register budget and simpler autotuning per kernel, at the cost of
  re-reading the big inputs across kernels (roughly doubles input reads). Kept as a fallback if
  the single fused kernel spills badly or fails to hit occupancy on small-M cases.

### 5.3 What to deliberately avoid
- Tensor-core `tl.dot` for the width-3/9 contractions (tf32 precision loss + degenerate shape).
- Storing any `[M,H]` or `[3,M,H]` intermediate (defeats the fusion; that is exactly what the
  slow reference does).
- Per-element atomics for weight grads (use block-local accumulation first).
- Any bf16 accumulation in reductions/means.

### 5.4 Reference / prior-art to consult
Use the **KernelWiki** skill for H100 (SM90) fused RMSNorm-backward and warp-specialization
patterns, non-pow2 `H` tiling, and `tl.atomic_add` reduction idioms before finalizing the
plan. Use **ncu-report-skill** (via `./scripts/ncu_profile.sh`) only *after* a correct
candidate exists, to confirm the kernel is DRAM-bandwidth-bound and to tune `BLOCK_*`/warps —
never while an evaluation is running.

---

## 6. Validation strategy

1. **Correctness by construction.** Mirror the §2 formulas exactly, in the reference's fp32
   order of operations, especially the residual/selection bookkeeping (risk 5.7) and the
   `H`-divisor for means (risk 5.6). Cross-check every index against `definition.json`'s
   reference string, not from memory.
2. **Sole correctness oracle = the provided evaluator.** Rules forbid running CUDA, the
   evaluator internals, or any alternate correctness harness directly. Validate a candidate
   only with `./scripts/evaluate_candidate.sh feedback cNNN`, which runs the **full 16-workload
   set** (warmup 2 / 10 iters) and checks `atol/rtol/match_ratio` per workload. That single
   invocation counts as one of the 100 evaluations.
3. **First candidate = correctness-first.** `c001` should be the straightforward fused kernel
   (option 5.1B or the simplest that compiles), fp32 throughout, prioritizing passing all 16
   workloads — including the tight-atol boundary cases (4, 6, 2) and the largest case (13) —
   before any perf tuning. A candidate that fails any selected workload's correctness is
   invalid regardless of speed.
4. **Watch the tight cases specifically.** Workload 4 (atol 0.0014, M=256) is the precision
   canary; workloads with odd `S` (293/373/613/449) are the `M`-tail-masking canary; large-M
   13/16 are the atomic-accumulation and throughput canaries. Inspect the per-workload lines in
   the evaluator output, not just the geomean.
5. **Iterate perf only on a correct base.** Once green, use `ncu_profile.sh` to check DRAM
   throughput vs roofline (target: approach the 11·M·H bandwidth floor) and autotune
   `BLOCK_M/BLOCK_H/num_warps`. Each meaningful change ⇒ new `cNNN`, appended as one complete
   JSON record to `candidates.jsonl` (parent, source hash, hypothesis, validation, per-workload
   result, geomean, decision, cumulative eval count, skill usage). Never rewrite earlier records.
6. **Convergence / stop.** Stop at the token or evaluation budget, or when the geomean speedup
   has genuinely converged; then write `SEARCH_COMPLETE` with the reason. `final` only with
   explicit operator approval.

---

## 7. Open questions to resolve in the plan (next step)

- Single fused kernel (5.1) vs 2-kernel split (5.2) for `c001` — lean single-fused, streaming
  H-tiles (5.1B) for a manageable register budget while staying at the bandwidth floor.
- Weight-grad reduction: block-local fp32 accumulate + one atomic per program (preferred) vs
  partial-buffer + reduction kernel — decide based on `num_programs` and observed atomic cost.
- Exact `H` tiling (`256×9` masked-free vs single masked `2304`) and whether the installed
  Triton accepts a non-pow2 block; confirm empirically in `c001`.
- Whether to specialize `altup_active_idx=0` as `constexpr` (all workloads use 0) while keeping
  a correct general path.
