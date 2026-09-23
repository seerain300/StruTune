# Draft Analysis — `mla_paged_decode_h16_ckv512_kpe64_ps1`

Target HW: **NVIDIA A800 (`sm_80`, Ampere)**. Framework entry point: `solution/solution.py::run(...)`.
Primary implementation must be **Triton**; PyTorch only for metadata / launch plumbing. No Torch/CPU/NumPy/CUDA-ext computational fallback.

---

## 1. Operation semantics

This is **Multi-head Latent Attention (MLA) paged decode** as used in DeepSeek-V3/R1 (TP=8), in its
*weight-absorbed* form. A single query position per sequence (decode) attends to a paged KV cache in
which each cached token is stored as one **compressed latent vector** `ckv` (dim 512) plus a separate
**RoPE key** `kpe` (dim 64). Crucially the *value* is the same compressed latent `ckv` (dim 512) — there
is no separate V after absorption.

### Per-batch math (reference, computed in fp32)

For batch element `b`, let the token index list be
`tok = kv_indices[kv_indptr[b] : kv_indptr[b+1]]`, with `L = kv_indptr[b+1] - kv_indptr[b]` tokens
(page_size = 1 ⇒ page index == token index).

```
Kc = ckv_cache[tok, 0, :]            # [L, 512]  (also serves as V)
Kp = kpe_cache[tok, 0, :]            # [L, 64]
qn = q_nope[b]                       # [16, 512]
qp = q_pe[b]                         # [16, 64]

logits  = qn @ Kc.T + qp @ Kp.T      # [16, L]      (effective key dim = 576)
s       = logits * sm_scale          # fp32
lse[b]  = logsumexp(s, dim=-1) / ln(2)     # [16]   base-2 log-sum-exp
attn    = softmax(s, dim=-1)         # [16, L]  fp32
out[b]  = (attn @ Kc).to(bf16)       # [16, 512]  value == ckv latent
```

Key structural facts that drive the kernel design:
- The **QK score** is a fused dot over a 576-dim key (`[qn|qp]·[Kc|Kp]`), but the **PV/output**
  contracts only over the 512-dim `ckv` latent. So `ckv` (Kc) is consumed twice (as K and as V),
  `kpe` (Kp) only once (as K).
- **All 16 query heads share the same KV** for a given batch element. KV read cost is therefore
  amortized across all heads if we batch heads together in one pass.
- The output value dim is 512 ⇒ the FlashAttention accumulator is `[16, 512]` fp32 (large; register-pressure driver, see §6).

### Empty-sequence semantics (must replicate exactly)
Reference pre-fills `output = zeros`, `lse = -inf`. If `L == 0` for a batch element, it stays
`output[b] = 0`, `lse[b] = -inf`. Our kernel must produce **zeros + (-inf)** for empty rows (guard the
`1/l` division when the running denominator is 0).

### LSE base convention
Output `lse` is **base-2** log-sum-exp (`logsumexp/ln2 = log2(Σ exp(s))`). Cleanest match: run the online
softmax in base-2 by scaling scores with `log2(e) = 1.4426950408889634` and using `exp2`:
`s' = s·log2(e)`, `p = exp2(s' − m)`, and then `lse = m + log2(l)` directly yields the base-2 result
with no extra `/ln2` step. This is also the fastest path on Ampere (`ex2.approx`).

---

## 2. Shapes, dtypes, constants, workload profile

Constants: `num_qo_heads = 16`, `head_dim_ckv = 512`, `head_dim_kpe = 64`, `page_size = 1`.

| Tensor | Shape | Dtype |
|---|---|---|
| `q_nope` | `[B, 16, 512]` | bf16 |
| `q_pe` | `[B, 16, 64]` | bf16 |
| `ckv_cache` | `[num_pages, 1, 512]` | bf16 |
| `kpe_cache` | `[num_pages, 1, 64]` | bf16 |
| `kv_indptr` | `[B+1]` | int32 |
| `kv_indices` | `[num_kv_indices]` | int32 |
| `sm_scale` | scalar | fp32 |
| **out** `output` | `[B, 16, 512]` | bf16 |
| **out** `lse` | `[B, 16]` | fp32 |

Constraints: `len_indptr == B+1`, `num_kv_indices == kv_indptr[-1]`.

### Fixed feedback workloads (5)

| # | batch B | num_kv_indices | avg L = idx/B | notes |
|---|---|---|---|---|
| 1 | 16 | 10857 | ~679 | mid batch, long seq |
| 2 | 16 | 12857 | ~804 | mid batch, longest avg |
| 3 | 1  | 208   | 208  | **single batch** → GPU underfill risk |
| 4 | 16 | 1857  | ~116 | mid batch, short seq |
| 5 | 64 | 30745 | ~480 | **large batch**, most total work |

`num_pages = 989669` in all cases (large cache pool; only gathered rows are touched). Per-batch `L`
varies within a workload (indptr is non-uniform), so the kernel must read the actual per-batch indptr
rather than assume equal splits.

Observations:
- Batch spans **1 → 64**; sequence length spans **~116 → ~800**. A single grid strategy that only
  parallelizes over batch (`grid = B`) starves the 108-SM A800 for workload 3 (B=1) and is thin for
  workloads 1/2/4 (16 CTAs). **Split-KV (FlashDecoding) is needed** to fill the machine at low batch.
- `sm_scale = 0.13523…` is supplied per workload (note: not the `1/sqrt(192)` default — we simply use
  the provided value; it is a runtime scalar).

---

## 3. Performance model (roofline on A800)

Per batch element, with 16 heads sharing KV read once:
- FLOPs ≈ `16 · L · (576 + 512) · 2 ≈ 16·L·2176`.
- KV bytes (read once) ≈ `L · (512 + 64) · 2 = L·1152`.
- Arithmetic intensity ≈ `16·2176 / 1152 ≈ 30 FLOP/byte`.

A800 ridge point ≈ `312 TFLOP/s (bf16) / ~2.0 TB/s ≈ 156 FLOP/byte`. Since `30 ≪ 156`, the kernel is
firmly **memory-bandwidth bound**. Implications:
- The dominant, irreducible cost is streaming the gathered KV: total ≈ `num_kv_indices · 1152` bytes.
  Workload 5: `30745·1152 ≈ 35 MB` → ideal ≈ **~18 µs** at 2 TB/s. All workloads are microsecond-scale.
- Optimization priorities: (a) read each KV token **exactly once** and reuse across all 16 heads and
  for both K and V; (b) saturate memory bandwidth with enough concurrent CTAs (split-KV); (c) minimize
  launch/overhead and avoid redundant passes; (d) keep the `[16,512]` accumulator resident to avoid a
  second KV read.
- Split-KV does **not** increase KV traffic (each token belongs to exactly one split); it only adds tiny
  partial-output writes + a cheap combine. So it is essentially free bandwidth-wise and strictly helps
  occupancy at low batch.

### Baseline (what we beat)
The reference is a **Python `for b in range(B)` loop** issuing, per batch, several eager kernels
(gather, two matmuls, softmax, logsumexp, matmul, casts) plus fp32 materialization of gathered
`Kc/Kp`. This is launch- and memory-traffic-heavy and serialized on the host. A single fused Triton
kernel that streams KV once and keeps everything on-chip should yield a large geomean speedup
(plausibly 10–100×), especially at small batch where the Python loop overhead dominates.

---

## 4. Numerical risks & correctness requirements

1. **QK precision** — inputs are bf16; reference upcasts bf16→fp32 then matmuls. Bf16 values are exactly
   representable, so Ampere `mma` (bf16×bf16→fp32 via `tl.dot`) reproduces the reference dot to fp32
   accuracy. Low risk. Accumulate scores in fp32.
2. **PV precision** — reference computes `attn @ Kc` with **fp32 `attn`** and fp32 `Kc`. For tensor-core
   PV we would cast `P` to bf16 (`P·Kc` bf16×bf16→fp32), losing ~2⁻⁸ on the weights. Since `output` is
   **bf16** (≈2⁻⁸ resolution), this is usually within tolerance, but it is the **main accuracy risk**.
   Mitigations, in order of preference if validation fails:
   - keep fp32 accumulation of the PV result (already planned);
   - do PV in fp32 `tl.dot` (TF32 path) — cheap here since memory-bound;
   - split `P` into bf16 hi/lo (error-compensated) for a near-fp32 PV.
3. **Online softmax / base-2 exp** — `exp2` uses `ex2.approx` (~2⁻²³ rel err); summed over ≤~800 terms in
   fp32 the error is negligible vs bf16 output tolerance. `lse` is fp32 output; base-2 formulation
   `lse = m + log2(l)` avoids an extra division. Track `m`,`l` in fp32.
4. **Empty sequences** — must emit `output = 0`, `lse = -inf`. Guard `acc / l` when `l == 0`; initialize
   `m = -inf`, `l = 0`, `acc = 0`. Confirm combine across splits preserves `-inf`.
5. **Split-KV combine** — recombining partial `(m_i, l_i, acc_i)` must use the standard log-sum-exp
   merge (rescale by `2^(m_i − m)`), and the **final** `lse = m + log2(l)` computed only after merge.
   Avoid double-scaling `sm_scale` (apply once, in the partial kernel).
6. **Index dtype / bounds** — `kv_indices`, `kv_indptr` are int32; token offsets into a 989k-row cache
   need int64 pointer arithmetic (or careful int32) to avoid overflow: `page_idx · 512` can exceed 2³¹
   only if `page_idx·512 > 2.1e9` ⇒ `page_idx > 4.19M`; `num_pages=989669` < that, so int32 element
   offset is safe, but I will use int64 offsets defensively for the gather base pointer.
7. **Tolerance is not stated explicitly** in `definition.json`; the evaluator applies its own. Strategy:
   match the reference fp32 compute path as closely as practical and validate empirically; tighten PV
   precision only if a candidate fails correctness.
8. **Determinism / NaN** — masked/out-of-range token lanes in the last partial `BLOCK_N` block must
   contribute `p = 0` (set masked scores to `-inf` before `exp2`), never NaN.

---

## 5. Reference behavior to reproduce (checklist)
- `output` bf16, `[B,16,512]`; `lse` fp32, `[B,16]`, **base-2**.
- Empty batch → zeros + `-inf`.
- `sm_scale` applied to logits **before** softmax and lse.
- Value contraction uses `Kc` (ckv) only, not `Kp`.
- Score contraction uses both `Kc` (512) and `Kp` (64).

---

## 6. Triton design space (A800 / sm_80)

### Grid / parallelization
- **Head batching:** process all 16 heads together as the `M=16` tile of every `tl.dot`, so KV is read
  once and reused across heads and across K/V roles. `M=16` maps cleanly onto Ampere `mma` (m16nNk16).
- **Split-KV (FlashDecoding):** two-kernel scheme.
  - *Partial kernel:* `grid = (B, num_splits)`. Each CTA owns batch `b` and a contiguous token sub-range
    `[kv_indptr[b] + s·chunk, min(kv_indptr[b+1], …))`, streams that range with online softmax, and
    writes partial `acc [16,512] fp32`, `m [16]`, `l [16]` to scratch.
  - *Combine kernel:* `grid = (B,)` (or `(B, head-block)`), reduces the `num_splits` partials via
    log-sum-exp merge, divides by `l`, casts to bf16 `output`, writes base-2 `lse`.
  - When `num_splits == 1`, the partial kernel can write the final result directly and skip combine.
- **Adaptive `num_splits`:** choose on the host from `B`, target CTA count (~2–4× 108 SMs), and
  `max_L` so each split has a reasonable token count (e.g. ≥ BLOCK_N). Rough policy:
  `num_splits = clamp(round(target_ctas / B), 1, ceil(max_L / BLOCK_N))`. Examples:
  B=1 (wl3) → many splits; B=64 (wl5) → 1–2; B=16 → 4–8.

### Inner loop (partial kernel)
```
init m[16]=-inf, l[16]=0, acc[16,512]=0
load qn[16,512] bf16, qp[16,64] bf16 (in regs/smem)
for n0 in range(split_start, split_end, BLOCK_N):
    idx = kv_indices[n0 : n0+BLOCK_N]          # gather token ids, masked tail
    Kc  = ckv_cache[idx, :]  # [BN,512] bf16   gathered contiguous rows
    Kp  = kpe_cache[idx, :]  # [BN,64]  bf16
    S   = tl.dot(qn, Kc.T) + tl.dot(qp, Kp.T)  # [16,BN] fp32
    S   = S * sm_scale * log2e
    S   = where(mask, S, -inf)
    m_new = max(m, max(S,axis=1))
    p   = exp2(S - m_new[:,None])              # [16,BN] fp32
    alpha = exp2(m - m_new)
    l   = l*alpha + sum(p,axis=1)
    acc = acc*alpha[:,None] + tl.dot(p.to(bf16), Kc)   # PV, [16,512]
    m   = m_new
```
Notes:
- `Kc` gather is a **row gather**: each of `BLOCK_N` token ids selects a contiguous 512-vector, so
  loads are coalesced within a row; pointer = `ckv_base + idx[:,None]*512 + arange(512)[None,:]`.
- Reuse `Kc` for both the score dot (`qn·Kcᵀ`) and PV (`p·Kc`) — one load, two uses.
- `acc [16,512] fp32` = 32 KB/CTA. With more warps (e.g. `num_warps=8` → 256 threads) that is ~32
  fp32/thread for `acc` — feasible within the 64K-reg/SM file but tight; watch spills. `num_warps=4`
  halves occupancy pressure elsewhere but doubles per-thread `acc`. This is the main tuning tension.

### Block-size / config space to explore
- `BLOCK_N ∈ {32, 64, 128}` (token block). 64 is a good starting point; SMEM for `Kc` tile
  `64×512×2 = 64 KB` + `Kp 64×64×2 = 8 KB` fits A800’s ≤164 KB/SM (single block/SM).
- `num_warps ∈ {4, 8}`, `num_stages ∈ {2, 3, 4}` (software pipelining of the gather+K load).
- PV `P` dtype: bf16 (fast) vs fp32/TF32 (accurate) — pick per numerical validation.
- Optionally tile the 512 output dim if register pressure forces it (loop `D` in chunks of 128/256),
  at the cost of re-reading `Kc` or holding it in SMEM.

### Alternative / fallback designs
- **Single fused kernel, `grid = (B,)`** (no split): simplest, correct; good for high batch, weak for B=1.
  Good **c001** baseline to lock correctness before adding complexity.
- **`grid = (B, num_splits)` split-KV + combine**: the performant target (**c002+**).
- **Head-block split** (e.g. 8+8 heads) only if register pressure from `[16,512]` acc is prohibitive;
  costs a second KV read, so avoid unless forced.

---

## 7. Planned candidate sequence (to be detailed in `plan.md`)
1. **c001 — correctness baseline.** Single fused Triton kernel, `grid=(B,)`, all 16 heads, online
   base-2 softmax, fp32 acc, bf16 PV, empty-seq guard, direct write. Establish correctness + baseline
   speed on all 5 feedback workloads.
2. **c002 — split-KV (FlashDecoding).** Partial + combine kernels, adaptive `num_splits`, to fill the
   GPU at low batch (esp. wl3 B=1, wl1/2/4). Expect the biggest low-batch gains.
3. **c003+ — tuning.** Sweep `BLOCK_N`, `num_warps`, `num_stages`, split policy; try SMEM-resident
   `Kc`, `num_stages` pipelining, and (only if needed) output-dim tiling or head-blocking.
4. **cNNN — numerical hardening** if any correctness failure: fp32/TF32 PV or error-compensated `P`.

Each candidate is immutable, evaluated once over the 5 workloads via
`./scripts/evaluate_candidate.sh feedback cNNN`, recorded in `candidates.jsonl`. Stop at convergence /
budget; write `SEARCH_COMPLETE` when converged. `final` only on explicit operator approval.

---

## 8. Validation strategy
- **Correctness gate:** the official evaluator over the 5 fixed feedback workloads is the sole
  correctness/perf oracle (I do not run CUDA/profiler/nvidia-smi/alternate harness directly).
- **Self-checks before submitting each candidate** (reasoning-level, mirroring the reference in the
  draft/plan, not an alternate numeric harness): verify (a) base-2 lse formula, (b) empty-seq → 0/−inf,
  (c) value uses `Kc` only, (d) score uses `Kc`+`Kp`, (e) single `sm_scale` application, (f) split
  combine merges `(m,l,acc)` correctly, (g) masked tail lanes contribute 0.
- **Numerical guardrails:** start with fp32 accumulation everywhere except the two `tl.dot` inputs;
  if a candidate fails correctness, escalate PV precision (bf16 → TF32/fp32 → hi/lo split) rather than
  loosening the algorithm.
- **Performance triage between candidates:** compare per-workload speedups and geomean from
  `candidates.jsonl`; attribute low-batch (wl3) gains to split-KV, high-batch (wl5) to bandwidth/occupancy.
  Keep the change per candidate ID small enough to attribute cause.
- **Regression safety:** never rewrite earlier `candidates.jsonl` records; every meaningful source/config/
  launch change gets a new candidate ID.

---

## 9. Skill applicability note
- `KernelWiki` is scoped to **Blackwell (SM100)/Hopper (SM90)** (tcgen05/TMEM/WGMMA/CLC/NVFP4, FA-4,
  DeepGEMM, 2-SM). This task is **A800 / sm_80 (Ampere)** — none of those primitives exist here — so the
  skill's PR-level techniques do **not** transfer. Not invoked.
- `ncu-report-skill` profiles on **B200 / sm_100**, and CLAUDE.md forbids running a profiler directly.
  Not applicable / not invoked.
- Skill usage for this run: **none applicable**; will be recorded as such in `candidates.jsonl`.

---

## 10. Open questions / risks to watch
- Exact evaluator tolerances (unstated) → mitigate by matching the fp32 reference path; escalate PV
  precision only on failure.
- Register spill from the `[16,512]` fp32 accumulator at `num_warps=8` → may need `num_warps=4`,
  output-dim tiling, or SMEM staging.
- Split policy vs per-batch `L` non-uniformity → use per-batch indptr in-kernel; empty splits exit cheaply.
- Whether any feedback workload contains a zero-length batch row → handle unconditionally.
