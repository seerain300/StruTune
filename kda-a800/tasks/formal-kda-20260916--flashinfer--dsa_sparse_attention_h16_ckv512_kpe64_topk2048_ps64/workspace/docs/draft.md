# Draft: `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`

Target: NVIDIA **A800 (sm_80, Ampere)**. Entry point: `solution/solution.py::run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)`.
Primary implementation must be **Triton**; PyTorch allowed only for metadata/launch plumbing. No Torch/CPU/NumPy/CUDA-extension computational fallback.

---

## 1. Operation semantics (what we must reproduce)

This is **DeepSeek-V3.2 DSA (paged MLA) sparse decode/prefill attention**. For each query token `t` we attend over a *sparse, per-token-selected* set of up to `topk=2048` KV-cache entries. All heads of a token share the same selected KV set.

### 1.1 Shapes and dtypes (from `definition.json`)

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `q_nope` | `[num_tokens, H=16, ckv=512]` | bf16 | query "no positional-encoding" part |
| `q_pe` | `[num_tokens, H=16, kpe=64]` | bf16 | query positional-encoding part |
| `ckv_cache` | `[num_pages, page_size=64, ckv=512]` | bf16 | compressed KV (acts as **both K and V**) |
| `kpe_cache` | `[num_pages, page_size=64, kpe=64]` | bf16 | key positional-encoding cache |
| `sparse_indices` | `[num_tokens, topk=2048]` | int32 | selected KV token indices; `-1` = padding/invalid |
| `sm_scale` | scalar | fp32 | `= 1/sqrt(128+64) = 1/sqrt(192) ≈ 0.135233...` (fixed for all feedback workloads) |
| **out** `output` | `[num_tokens, H=16, ckv=512]` | bf16 | attention output |
| **out** `lse` | `[num_tokens, H=16]` | fp32 | **base-2** log-sum-exp of scaled logits |

Constants are guaranteed: `H=16`, `ckv=512`, `kpe=64`, `page_size=64`, `topk=2048`. We may hard-code them as `tl.constexpr` for best codegen.

### 1.2 Reference math (per token `t`, per head `h`)

1. `indices = sparse_indices[t]` (length `topk`). `valid = indices != -1`.
2. If **no** valid index: `output[t] = 0`, `lse[t] = -inf` (row stays at its init).
3. Otherwise gather from the **flattened** cache:
   - `Kc_all = ckv_cache.reshape(num_pages*page_size, 512)`; `Kc = Kc_all[valid_indices]`.
   - `Kp_all = kpe_cache.reshape(num_pages*page_size, 64)`; `Kp = Kp_all[valid_indices]`.
   - **Index semantics:** a value `idx` (page-64 encoding `page*64+offset`) directly indexes the *flattened* cache row. So we do **not** need any page/offset arithmetic — `flat_row = idx` exactly. This is confirmed by the reference (`Kc_all[valid_indices]`).
4. Logits: `logits[h, j] = (q_nope[t,h]·Kc[j]) + (q_pe[t,h]·Kp[j])`, then `logits_scaled = logits * sm_scale`.
5. `lse[t,h] = logsumexp(logits_scaled, dim=-1) / ln(2)`  (i.e. base-2 LSE).
6. `attn = softmax(logits_scaled, dim=-1)`; `out = attn @ Kc`; `output[t,h] = out.to(bf16)`.

All reference math is in **fp32** (bf16 inputs are widened to fp32 before any matmul).

### 1.3 Structure = "MLA / FlashDecoding with a gather front-end"

Equivalent to a standard flash-attention where the K/V sequence is the **gathered** rows and the "V" tensor equals the **K's ckv part** (`Kc`). QK contraction dim = `ckv+kpe = 576`; PV contraction dim (softmax length) = up to `2048`; output dim = `ckv=512`. `q_pe/kpe` only affect logits, never the output.

---

## 2. Constraints, invariants, and the baseline inefficiency (the real speedup lever)

- Verified constraints: `sparse_indices.shape[0]==num_tokens`, `sparse_indices.shape[-1]==topk`, `ckv_cache.shape[1]==page_size`.
- `num_tokens` in the 5 feedback workloads is **tiny**: `{8, 6, 2, 2, 8}`. `num_pages=8462` for all five (⇒ `flat_kv = 8462*64 = 541,568` rows). `sm_scale` identical across all five.
- Because `num_tokens ≤ 8` and `H=16`, a naive "one program per token" launches only 2–8 CTAs on a 108-SM GPU → **severe under-occupancy**. Parallelization strategy is the dominant design question (see §4).

**Why the reference is slow (our opportunity):**
The reference does `ckv_cache.reshape(-1,512).to(float32)` and `kpe_cache.reshape(-1,64).to(float32)` — it **materializes the entire cache in fp32 up front**, independent of `topk`:
- `Kc_all` fp32 = `541,568 × 512 × 4 B ≈ 1.11 GiB`
- `Kp_all` fp32 = `541,568 × 64 × 4 B ≈ 139 MiB`

So the baseline reads + writes >1.25 GiB of HBM regardless of how few tokens/indices are actually used, plus a Python `for t in range(num_tokens)` loop of small gathers/matmuls (many kernel launches, poor utilization). Our Triton kernel only needs to touch the **selected `topk` rows in bf16** — see §3. This asymmetry (touch ~18 MiB vs >1.25 GiB) is where the geomean speedup comes from; micro-optimizing the flash inner loop is secondary.

---

## 3. Performance characterization (roofline)

Per query token, each selected K row is read **once** and reused by all 16 heads and reused as V:
- `Kc` bytes = `2048 × 512 × 2 = 2.00 MiB`; `Kp` bytes = `2048 × 64 × 2 = 0.25 MiB` ⇒ **≈ 2.25 MiB/token** of essential HBM gather traffic.
- Largest feedback (`num_tokens=8`): `≈ 18 MiB` essential reads. At A800 HBM ≈ 2.0 TB/s that is **≈ 9 µs** ideal — the kernel is *tiny* and effectively **launch-/latency-/occupancy-bound**, not throughput-bound.

FLOPs per token: QK `16·2048·576·2 ≈ 37.7 MFLOP` + PV `16·2048·512·2 ≈ 33.6 MFLOP` ≈ **71.3 MFLOP/token**. Eight tokens ≈ 0.57 GFLOP → ≈ 1.8 µs at 312 bf16 TFLOP/s. Compute < memory.

Arithmetic intensity ≈ `71.3e6 / 2.36e6 ≈ 30 FLOP/byte`, well below the A800 ridge (`≈ 312e12/2.0e12 ≈ 156 FLOP/byte`) ⇒ **memory/gather bound**. Consequences:
- Gathers of 512-dim bf16 rows are 1 KiB contiguous chunks (good intra-row coalescing); indices are scattered across 541k rows (poor inter-row locality, minimal L2 reuse). Prioritize wide, aligned vector loads and enough in-flight CTAs to hide HBM latency.
- Given only 2–8 tokens, we **must split the KV dimension across CTAs** (FlashDecoding/split-K) to fill 108 SMs and hide latency.

---

## 4. Triton design space

### 4.1 Parallelization axis (the central decision)

Tension: **head reuse** (load each K row once for all 16 heads → keep heads together in one program) vs **occupancy** (only 2–8 tokens → few programs). Options:

- **A. One program per token, loop all 2048 indices, all 16 heads.** Max K reuse, min traffic, but only 2–8 CTAs ⇒ terrible occupancy. Rejected as the primary design (may still be a useful baseline `c001`).
- **B. Split-K / FlashDecoding (preferred).** Grid `(num_tokens, num_kv_splits)`. Each CTA handles **all 16 heads** for one contiguous chunk of the 2048 indices, doing online-softmax over its chunk and emitting partial `(acc[16,512], m[16], l[16])` to scratch. A small **combine kernel** reduces across splits → final `output` + base-2 `lse`. With `num_kv_splits ≈ 16–32`, program count `= num_tokens × splits ≈ 128–256` ⇒ fills the GPU while keeping full per-CTA head reuse (each K row still read once per split). This is the standard MLA-decode pattern and the main candidate family.
- **C. Program per (token, head) or (token, head-group).** `8×16=128` CTAs, but each of the 128 re-reads the same 2.25 MiB gather (16× redundant HBM traffic) ⇒ bad for a memory-bound kernel. Rejected.
- **D. Persistent / fused single-pass split-K** (atomics or a flag-based reduction to avoid the second kernel). More complex; consider only if the two-pass launch overhead is material for these micro-sized problems.

### 4.2 Inner tiling (flash / online softmax)

- Tile the index axis in `BLOCK_N` (candidates 64/128/256). For a chunk: `idx = sparse_indices[t, n0:n0+BLOCK_N]`; mask `idx == -1` (and out-of-range) → set those logits to `-inf`, gather with `mask, other=0`.
- Load `Kc_blk [BLOCK_N, 512]` and `Kp_blk [BLOCK_N, 64]` via gathered row pointers (`base + idx[:,None]*stride_row + arange(dim)`).
- `S = tl.dot(q_nope[16,512], Kc_blk.T) + tl.dot(q_pe[16,64], Kp_blk.T)` → `[16, BLOCK_N]`, scale by `sm_scale`.
- Online-softmax update of running `m[16]`, `l[16]`, `acc[16,512]`; second matmul `acc += P[16,BLOCK_N] @ Kc_blk[BLOCK_N,512]`.
- **`M=16` is the minimum tensor-core tile** — fine but a bit wasteful; `N/K` dims are large enough to keep MMA busy. Consider whether processing 2 tokens per CTA (M=32) improves MMA efficiency vs. hurting occupancy.

### 4.3 exp2 vs exp / base-2 LSE bookkeeping

Fold `sm_scale` and `log2(e)` together and use **`exp2`** (hardware-fast): work in a base-2 logit domain `s2 = logits * (sm_scale * log2e)`. Track `m2 = max s2`, `l2 = Σ 2^(s2 - m2)`. Then:
- `attn = 2^(s2 - m2) / l2` (probabilities identical to the natural-domain softmax; the `log2e` factor cancels).
- **Base-2 LSE** required by the spec: `lse = m2 + log2(l2)` (already base-2, no `/ln2` needed). Verify sign/scaling carefully — spec wants `logsumexp(logits_scaled)/ln2`, and `m2 + log2(l2) = (m_nat + ln(l_nat))·log2e` = exactly that. 

### 4.4 Precision / dtype choices for MMA (numerical risk hotspot)

Reference casts bf16→fp32 then matmuls in **full fp32**. On A800 our options for `tl.dot`:
- **QK:** inputs are the raw bf16 `q`/`K` → `tl.dot(bf16, bf16, out=fp32)` matches the reference well (bf16 values widened exactly; TC does high-precision product + fp32 accumulate). Preferred. Must ensure TF32 path is **not** silently used on fp32 operands.
- **PV:** `P` is an fp32 softmax probability; `Kc` is bf16. Flash kernels usually cast `P→bf16` for the second MMA. That drops `P` to 8-bit mantissa and is the **main accuracy risk**. Mitigations, in order of preference: (a) `bf16` `P` and accept error (validate against tolerance); (b) fp32 accumulate with `tl.dot(..., allow_tf32=False)` (slower, but kernel is memory-bound so compute cost is hidden); (c) split-K reduces per-chunk sums, limiting catastrophic cancellation.
- **Accumulator** `acc` and softmax stats always **fp32**. Final cast to bf16 only at store.

### 4.5 Memory / occupancy budget (A800 = A100 class: 192 KiB smem/SM, ≤ ~163 KiB/CTA, 108 SMs)

Rough per-CTA working set for `BLOCK_N=64`, all 16 heads:
- `Kc_blk` bf16 `64×512×2 = 64 KiB`, `Kp_blk` bf16 `64×64×2 = 8 KiB`, `q_nope` `16×512×2 = 16 KiB`, `q_pe` `2 KiB`, `acc` fp32 `16×512×4 = 32 KiB`.
- `acc` (32 KiB) really wants to live in **registers**, not smem. Watch register pressure (16×512 fp32 accumulator is large). Smaller `BLOCK_N`, fewer heads-per-CTA, or splitting the 512 output dim may be needed. Tune `num_warps ∈ {4,8}`, `num_stages ∈ {2,3,4}` for gather latency hiding.

### 4.6 Tunable / autotune space

`BLOCK_N ∈ {64,128,256}`, `num_kv_splits ∈ {8,16,32,64}` (or derived from `ceil(topk/chunk)`), `num_warps ∈ {4,8}`, `num_stages ∈ {2,3,4}`, tokens-per-CTA `∈ {1,2}`, `P` dtype `∈ {bf16, fp32/tf32-off}`, single-pass vs two-pass reduction. Keep the *first* candidate simple and correct; only autotune once correctness is locked.

### 4.7 Edge cases the kernel must handle

- **`-1` padding**: mask before gathering (avoid OOB row pointers via `mask`/`other=0`) *and* set masked logits to `-inf` so they contribute zero probability.
- **Fully-invalid row** (all `-1`, or a split with no valid index): `l==0`. Must yield `output=0` and `lse=-inf` without `0/0` NaNs. Guard the final normalization (`out = acc / l` only if `l>0`, else `0`), and set `lse=-inf` when `l==0`. In split-K combine, splits with `m=-inf`/`l=0` must be skipped safely.
- **Init of outputs**: `output` init to 0, `lse` init to `-inf` (matches reference), so tokens/heads never written still satisfy the spec.
- **Alignment**: `ckv=512`, `kpe=64` are nice powers of two; row strides are contiguous ⇒ vectorized loads OK.

---

## 5. Numerical-correctness risks (summary)

1. **PV `P→bf16` cast** — largest error source; may need fp32/`allow_tf32=False` PV path or bf16 with tolerance headroom.
2. **TF32 contamination** on any fp32 `tl.dot` — must be explicitly disabled where fp32 accuracy is intended.
3. **Base-2 LSE**: easy to get the `log2e`/`ln2` factor or the `m + log(l)` reconstruction wrong. Cross-check formula (§4.3).
4. **Split-K online-softmax combine**: max/rescale across partials; mishandling `-inf` partials produces NaNs.
5. **Empty/invalid rows**: `0/0`, `-inf` propagation, OOB gather.
6. **bf16 vs fp32 accumulation order**: reference sums over up to 2048 terms in fp32; our chunked online sum reorders additions → small ULP differences, expected within bf16 tolerance.
7. **Index dtype**: `sparse_indices` is int32; ensure 64-bit pointer arithmetic (`idx.to(int64)` or int32*stride within range — `541,568×512 ≈ 2.77e8 < 2^31`, so int32 offsets for `ckv` are borderline-safe but int64 is safer).

---

## 6. Validation strategy (given the isolation rules)

We **cannot** run CUDA/profiler/`nvidia-smi`/evaluator directly, nor any alternate correctness harness. The only sanctioned signal is:
```bash
./scripts/evaluate_candidate.sh feedback <candidate-id>
```
which runs the official evaluator over the 5 fixed workloads (correctness gate + geomean speedup). Plan:

1. **Static/Manual correctness reasoning** before each eval: re-derive the LSE formula, the softmax/probability equivalence under `exp2`, and the edge-case guards on paper; keep the kernel minimal for `c001`.
2. **`c001` = simplest correct design**: FlashDecoding split-K (design B) or even single-CTA-per-token (design A) purely to establish a *correct, valid, Triton-only* baseline and confirm the harness/plumbing, tolerances, and geomean reporting. Do not over-engineer before the first green run.
3. **One change per candidate ID** (immutable): after `c001` passes correctness, iterate on parallelization (splits), `BLOCK_N`, dtype of `P`, num_warps/stages, single- vs two-pass. Record parent, source hash, hypothesis, per-workload pass/latency, geomean, decision, cumulative eval count, skill usage in `candidates.jsonl`.
4. **Correctness is a hard gate**: any workload that fails correctness invalidates the candidate regardless of speed. Never introduce a Torch/CPU/NumPy fallback to "pass".
5. **Convergence/stop**: stop when geomean improvement plateaus, or at the eval/token budget. Write `SEARCH_COMPLETE` with the reason. Never run `final` without operator approval.

**Debugging aids allowed within rules**: I can inspect `feedback_workloads.jsonl` (already done) for shapes/`sm_scale`, and reason about the reference. I cannot open the safetensors indices content path (outside workspace `./blob/...` — treat as evaluator-owned); handle `-1` defensively regardless of what the data contains.

### Skill usage note
`KernelWiki` and `ncu-report-skill` target **Blackwell (SM100) / Hopper (SM90)**; this task is **Ampere sm_80 (A800)**, so their tcgen05/TMEM/CLC/wgmma-specific guidance largely does **not** transfer, and I cannot run `ncu` here anyway. I will consult `KernelWiki` only for portable flash-decoding/online-softmax/split-K patterns if useful, and will note any consultation in `candidates.jsonl`. The A800-specific reasoning (Ampere `cp.async` pipelining, bf16 MMA, smem budget, occupancy) is captured above.

---

## 7. Provisional direction (to be turned into `docs/plan.md` next)

- **Primary family:** Triton **FlashDecoding split-K** — grid `(num_tokens, num_kv_splits)`, per-CTA all-16-heads online softmax over a KV chunk with masked `-1` gathers and `exp2`, fp32 `acc`/stats, then a **combine kernel** producing bf16 `output` and base-2 `lse`.
- **First candidate `c001`:** the simplest correct variant to establish validity and the geomean baseline.
- **Then iterate:** number of splits (occupancy), `BLOCK_N`, `P` dtype (accuracy vs speed), `num_warps`/`num_stages`, tokens-per-CTA, possibly single-pass reduction — one immutable candidate per change.
- **Expected big lever:** avoiding the reference's full-cache fp32 materialization by touching only the selected `topk` bf16 rows.
