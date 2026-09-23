# Draft — L1/070 Mamba2 Fused Intra-Chunk Diagonal Computation

Task: optimize SOL-ExecBench `L1/070_mamba2_fused_intra_chunk_diagonal_computation` on
NVIDIA H100 (`sm_90`). Submission is `solution/solution.py` exposing `run(hidden_states,
A_cumsum, B, C) -> Y_diag`. Primary implementation must be Triton; no Torch/CPU/NumPy/CUDA
computational fallback. This document is the analysis-only draft; no plan or code yet.

---

## 1. Operation — what the reference actually computes

### 1.1 Inputs / outputs (all `bfloat16`)

| Tensor | Shape | Layout note (last dim fastest / contiguous) |
|---|---|---|
| `hidden_states` (X) | `[b, nc, L, H, P]` = `[b, nc, 128, 32, 128]` | X[b,c,j,h,d]; stride over j = H·P = 4096 |
| `A_cumsum` (a) | `[b, H, nc, L]` = `[b, 32, nc, 128]` | a[b,h,c,l]; contiguous in l (stride 1) |
| `B` | `[b, nc, L, Gp, N]` = `[b, nc, 128, 8, 128]` | B[b,c,j,g,n]; stride over j = Gp·N = 1024 |
| `C` | `[b, nc, L, Gp, N]` = `[b, nc, 128, 8, 128]` | C[b,c,i,g,n]; stride over i = Gp·N = 1024 |
| `Y_diag` (out) | `[b, nc, L, H, P]` = `[b, nc, 128, 32, 128]` | same layout as X |

Constant axes (fixed for **every** workload, feedback and final): `chunk_size L = 128`,
`num_heads H = 32`, `head_dim P = 128`, `n_groups Gp = 8`, `state_size N = 128`. Only
`batch_size b` and `num_chunks nc` vary. Heads-per-group = `H/Gp = 4`; head `h` uses group
`g = h // 4`.

### 1.2 Math, reduced to its essence

The reference builds several giant intermediates, but the operation factorizes cleanly
**per `(b, c, h)` tile** into an attention-like block. Fix `(b, c, h)` and let `g = h // 4`:

- `C_h = C[b, c, :, g, :]`  → shape `[L, N] = [128, 128]`, indexed `[i, n]`
- `B_h = B[b, c, :, g, :]`  → shape `[L, N] = [128, 128]`, indexed `[j, n]`
- `X_h = hidden_states[b, c, :, h, :]` → shape `[L, P] = [128, 128]`, indexed `[j, d]`
- `a   = A_cumsum[b, h, c, :]` → length-`L` vector `[128]`

Then:

1. **Content weights (G):** `G[i,j] = Σ_n C_h[i,n]·B_h[j,n] = (C_h · B_hᵀ)[i,j]`  → `[128,128]`.
2. **Decay mask (L):** with the inclusive prefix sum `S[i] = Σ_{i'=0..i} a[i']`,
   `Ldec[i,j] = exp(S[i] − S[j])` for `i ≥ j`, and `0` for `i < j` (strict causal).
3. **Masked weights:** `M[i,j] = G[i,j] · Ldec[i,j]`  → `[128,128]`, lower-triangular.
4. **Output:** `Y[i,d] = Σ_j M[i,j]·X_h[j,d] = (M · X_h)[i,d]`  → `[128,128]`, cast to bf16.

So each `(b,c,h)` tile = **two 128×128×128 matmuls** (`C·Bᵀ` and `M·X`) plus a decay-mask
elementwise stage. This is exactly the "diagonal block" of chunked linear attention / the SSD
scan: `G` plays `QKᵀ`, `Ldec` the causal-decay mask, `X` the values, `Y` the output.

### 1.3 Deriving the decay identity (must match the reference exactly)

The reference computes, for column `j`, `cumsum_i( a[i']·1[i'>j] )`, i.e. for `i > j`:
`Σ_{i'=j+1}^{i} a[i']`. Using the inclusive prefix sum `S`, `Σ_{i'=j+1}^{i} a[i'] = S[i] − S[j]`.
For `i = j` the sum is empty → `0 = S[i] − S[j]`. The reference then `masked_fill(~tril(diag=0),
-inf)` (so `i < j → −inf`) and applies `exp`, giving:

- `i > j`: `exp(S[i] − S[j])`
- `i = j`: `exp(0) = 1`
- `i < j`: `exp(−inf) = 0`

**Key subtlety:** the reference's per-column cumsum and the prefix-difference `S[i]−S[j]` are
mathematically identical but *not* bitwise identical in fp32 (different summation groupings).
This is a negligible rounding difference given `rtol = 0.05`; I will use the prefix-difference
form (one `tl.cumsum` over the 128-vector, then broadcast `S[:,None] − S[None,:]`).

### 1.4 Group sharing (algorithmic reuse the reference throws away)

`G` depends on the head **only through the group** `g = h//4` (it is built from `C_h`, `B_h`,
both group-level). Hence the 4 heads in a group share **the same `G`**; only `Ldec` (per-head
`A_cumsum`), `X_h`, and `Y` differ per head. The reference `repeat_interleave`s B/C to 32 heads
and recomputes everything 4× redundantly. Fusing at the group level lets us compute `C·Bᵀ`
once and reuse it for 4 heads → up to a 2× cut in matmul work (one of the two matmuls shared
4-way). Noted as an optimization axis, not the first candidate.

---

## 2. Why this is a large SOL opportunity (roofline intuition)

The reference materializes, in fp32 HBM, `G` `[b,nc,128,128,H]`, `Ldec` `[b,H,nc,128,128]`,
`M` `[b,nc,128,128,H]`, and — worst of all — the broadcast product
`M_expanded * hidden_states_expanded` of shape `[b,nc,128,128,H,P]` before summing over `j`.
For the largest feedback workload (`b=4, nc=16`): `G/L/M` are ~134 MB each, and the broadcast
product is `4·16·128·128·32·128·4 B ≈ 17 GB` written+read. That single temp dwarfs everything.

A fused kernel never materializes any `[L,L]` or `[L,L,·]` tensor in HBM. Its HBM traffic per
workload is just: read `X` + read `B,C` + read `a` + write `Y`. For `b=4, nc=16`:
`X`≈268 MB, `Y`≈268 MB, `B,C`≈67 MB each, `a` negligible → ~670 MB total vs the reference's tens
of GB. At ~3 TB/s that is a few hundred µs of unavoidable traffic; the arithmetic
(`b·nc·H·2·128³·2 ≈ 17 GFLOP` for the big case) is light on tensor cores. **Fusion alone is
expected to be the dominant win (order 5–20×).** Later tuning targets occupancy, block shape,
group-sharing, and the causal-triangle skip.

---

## 3. Constraints & evaluation rules (operational)

- **Triton primary**, Torch only for metadata/launch; no Torch/CPU/NumPy/CUDA-ext/alternate
  fallback. A failed Triton impl is invalid and must not be replaced by a Torch path.
- **Tolerance:** `max_atol = 1e-5`, `max_rtol = 0.05` for every workload; output bf16. With
  bf16 outputs of O(1)–O(large) magnitude, the `1e-5` atol is essentially unreachable, so the
  **relative 5%** gate is the effective correctness criterion (assuming the usual
  `|Δ| ≤ atol + rtol·|ref|` combination). This is forgiving enough to permit bf16 tensor-core
  matmuls with fp32 accumulation.
- **Feedback set = 14 workloads** (the `task/feedback_workloads.jsonl` has 14 lines; the README
  mentions "five fixed" but the file is authoritative — I will treat all 14 as the feedback
  set). One immutable kernel over the full set = one candidate evaluation. Budget: 100
  evaluations; token soft/normal/absolute limits 9M/10M/11M.
- Evaluate **only** via `./scripts/evaluate_candidate.sh feedback cNNN`. `final` is
  operator-only. Each candidate is immutable with a fresh ID; never reuse an ID for changed
  source; append one JSON record per eval to `candidates.jsonl`, never rewrite.
- **Profiling** only through `./scripts/ncu_profile.sh ...` (ncu-report-skill workflow); never
  run `ncu`/CUDA/`nvidia-smi`/the evaluator directly. **Never profile and evaluate
  concurrently** (a foreign process on the locked GPU makes the controller discard the timing,
  return code 3, and burns one evaluation).
- Correctness authority is the official evaluator; I will **not** build a private
  Torch-on-GPU comparison harness (that would be an "alternate correctness harness").

### 3.1 Feedback workload shapes (`(batch, num_chunks)`)

`(4,7) (1,8) (4,1) (2,8) (1,4) (1,2) (1,3) (16,1) (2,2) (4,16) (1,32) (8,4) (4,4) (2,4)`.

Number of `(b,c,h)` tiles = `b·nc·32`. Range: min `(1,2)→64` tiles, max `(4,16)→2048` tiles.
Many workloads have small `b·nc` (≤ 8 → ≤ 256 tiles), so with 132 SMs on H100 **occupancy for
the small cases is the key perf risk** — the kernel must expose enough parallelism (e.g. split
the query rows into 2 blocks of 64, or grid over `(b, nc, h, i_block)`) so small workloads still
fill the machine, while the big `(4,16)/(1,32)` cases stay compute/BW efficient.

---

## 4. Numerical risks

1. **`exp` of `S[i]−S[j]` with random `A_cumsum`.** Inputs are "random" bf16. If `a` behaves
   like `N(0,1)`, a partial sum over up to 128 terms has std ~`√128 ≈ 11`, so `S[i]−S[j]` can
   reach ±30 → `exp(±30) ≈ 10^±13`. `M = G·Ldec` then spans a huge dynamic range, and `Y` is
   dominated by the largest-`Ldec` terms. This is inherent to the reference; I must **replicate
   its exact formula** (difference form + `−inf` on the strict-upper triangle before `exp`), and
   must **not** introduce a flash-style row-max subtraction — factoring `exp(S[i])·exp(−S[j])`
   or subtracting a per-row constant changes rounding/overflow behavior and is unnecessary here
   (single-pass, no online softmax). Set masked entries to `−inf` *before* `exp` so overflow on
   the discarded upper triangle can't leak.
2. **Possible `inf/nan` in the reference itself.** If some workload's `a` yields `S[i]−S[j]`
   large enough to overflow fp32 `exp`, the reference output contains `inf`, and any correct
   replica would still fail a `nan/inf`-sensitive comparison. If the evaluator reports failures
   that look overflow-driven, I will inspect via ncu / small analysis rather than assume a bug.
   Most likely the generator keeps `A_cumsum` bounded/negative (real Mamba `A_cumsum` is a
   cumsum of negative decays, so `S` is decreasing and `S[i]−S[j] ≤ 0` for `i ≥ j`, giving
   `Ldec ∈ (0,1]`); I'll confirm empirically through eval pass/fail.
3. **bf16 matmul rounding.** Reference does both matmuls in fp32. Triton `tl.dot` on bf16 inputs
   with fp32 accumulation reproduces `C·Bᵀ` essentially exactly (bf16→fp32 is exact; the
   tensor-core product of two bf16 values is computed to fp32). For the second matmul `M·X`,
   `M` is fp32; casting `M` to bf16 to use tensor cores injects ~0.4% relative error per element,
   which fp32-accumulated over 128 terms should stay within the 5% rtol gate. **Fallback if it
   fails:** keep `M` in fp32 and do the second dot in fp32/TF32-off (slower) — a correctness
   lever to hold in reserve.
4. **cumsum associativity** vs the reference's per-column cumsum: negligible at fp32 magnitudes
   involved (see §1.3).
5. **Strict-vs-inclusive diagonal.** Diagonal `i=j` contributes `Ldec = 1` (included). The mask
   is `i ≥ j` (tril diag=0), *not* `i > j`. Off-by-one here silently drops/keeps the diagonal
   term — must use `>=`.

---

## 5. Triton design space

### 5.1 Baseline candidate (c001 target): per-head fused block

- **Grid:** flat `b·nc·H` programs; decode `pid → (b, c, h)`, `g = h//4`.
- Load `C_h[128,128]`, `B_h[128,128]` (bf16) with block pointers over the group-strided layout.
- `G = tl.dot(C_h, tl.trans(B_h))` (or `tl.dot` with a transposed block-ptr for `B`),
  fp32 accumulate → `[128,128]`.
- Load `a[128]`; `S = tl.cumsum(a_fp32, axis=0)` (inclusive); `diff = S[:,None] − S[None,:]`;
  `diff = tl.where(i>=j, diff, -inf)`; `Ldec = tl.exp(diff)`.
- `M = G * Ldec` (fp32).
- Load `X_h[128,128]` (bf16, head-strided); `Y = tl.dot(M.to(bf16), X_h)`, fp32 accumulate.
- Store `Y.to(bf16)` to `Y_diag[b,c,:,h,:]`.
- Single `j`-pass (chunk = 128 = one `BLOCK_N`), so **no online softmax / running max** — much
  simpler than FlashAttention. Whole `[128,128]` `G`/`M` live in SRAM/registers.

Goal of c001: correctness on all 14 workloads + capture the fusion win. Establish the geomean
baseline before micro-tuning.

### 5.2 Optimization axes (subsequent candidates, one change per ID)

1. **Query-row split (`BLOCK_M ∈ {128, 64, 32}`), grid `(b,nc,h,i_block)`.** Raises occupancy for
   small `b·nc` workloads (the majority) and enables the **causal-triangle skip**: an `i`-block
   only needs `j`-blocks `≤` its own, cutting ~25–40% of both matmuls and all masked-`exp` work
   above the diagonal. Requires a `j`-loop (return of a mini accumulation), but still no online
   softmax (no normalization).
2. **`num_warps` / `num_stages` sweep** (`num_warps ∈ {4,8}`, stages `∈ {2,3,4}`) — classic
   matmul occupancy/pipelining trade; autotune or hand-pick per shape class.
3. **Group-level fusion (§1.4):** grid `(b,nc,Gp)`, compute `G` once, loop the 4 heads (each with
   its own `a`, `X`, `Y`). Cuts the `C·Bᵀ` matmul 4× and the redundant `B/C` HBM reads 4×, at the
   cost of 4× per-program work and lower parallelism (max 512 programs) — likely a win for the
   large workloads, possibly a loss for tiny ones; evaluate both.
4. **Decay-vector caching:** `Ldec` depends only on `(b,c,h)` and is reused across the entire
   `M·X` — already computed once per tile; ensure `S` is computed once and the `exp` is not
   recomputed per `d`-tile if `P` is ever tiled.
5. **Second-matmul precision lever:** default bf16-cast `M`; fp32/TF32-off fallback only if
   rtol fails (§4.3).
6. **Layout / vectorization:** `X` and `Y` load/store are head-strided (stride `H·P = 4096`
   between `j`); inner 128 (`d`) is contiguous → 128-wide contiguous vectors, good. `C`/`B` inner
   128 (`n`) contiguous, `i/j` stride `Gp·N = 1024`. Use `tl.make_block_ptr` for clean, coalesced
   access.
7. **Const specialization:** pass `L,N,P,H,Gp` as `tl.constexpr` (all fixed) to unlock unrolling
   and compile-time block shapes, while still deriving `b,nc` from the runtime tensor shapes for
   robustness. Never hardcode `b`/`nc`.

### 5.3 Kernel-count decision

Whether to split C·Bᵀ and M·X into two kernels (materializing `M` in HBM) or keep one fused
kernel: **one fused kernel** is the whole point (avoids the GB-scale intermediates of §2). Only
consider splitting if register/SRAM pressure forces it; but two 128×128 fp32 accumulators
(`G`, `Y`) across `num_warps=8` is within the flash-attention envelope, so a single kernel is
expected to hold.

---

## 6. Validation strategy

1. **Correctness = official evaluator only.** Run `./scripts/evaluate_candidate.sh feedback cNNN`;
   read per-workload pass/fail + speedup + geomean. No private GPU comparison harness (§3).
2. **Paper equivalence** (this doc §1.2–§1.3) is the primary pre-eval correctness argument:
   index algebra, the `S[i]−S[j]` identity, the `i≥j` diagonal-inclusive mask, group index
   `g=h//4`, and the two matmul orientations (`C·trans(B)`, `M·X`) are all pinned down here so
   the first candidate is written against a verified derivation, not guesswork.
3. **Correctness-first ordering:** c001 prioritizes matching the reference (bf16 matmuls, fp32
   accumulate, exact decay formula). Only after all 14 workloads pass do I pursue perf axes; any
   perf change that breaks a workload is reverted (new ID).
4. **Performance = ncu-report-skill via `./scripts/ncu_profile.sh`**, never concurrent with an
   evaluation. Use it to confirm the kernel is BW/occupancy-bound as predicted and to guide the
   block-shape / warps / group-fusion choices. Profile the representative large case
   (`b=4,nc=16`) and a small case (`b=1,nc=2`) to see both regimes.
5. **Convergence:** stop when geomean improvement flattens across successive candidates or at the
   budget; write `SEARCH_COMPLETE` with the reason. `final` only on explicit operator approval.

---

## 7. Open questions to resolve during search

- Actual distribution/scale of `A_cumsum` (bounds the `exp` dynamic range and whether `Ldec ≤ 1`).
  Inferred from eval pass/fail rather than assumed.
- Does bf16-cast of `M` in the second matmul stay within rtol on the high-dynamic-range workloads,
  or is the fp32 second-dot fallback needed?
- Crossover point where group-level fusion (§5.2.3) beats per-head for large vs small workloads.
- Best `BLOCK_M` / grid factorization to satisfy both the tiny (occupancy-starved) and large
  (BW-bound) workloads with a single kernel + autotune config.
