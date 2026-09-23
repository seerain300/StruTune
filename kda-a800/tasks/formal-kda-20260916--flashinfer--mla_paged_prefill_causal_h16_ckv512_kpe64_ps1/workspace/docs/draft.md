# Draft: `mla_paged_prefill_causal_h16_ckv512_kpe64_ps1`

Target: NVIDIA **A800 / `sm_80` (Ampere)**. Implementation must be **Triton** (PyTorch only for
metadata/launch plumbing). No Torch/CPU/NumPy/CUDA-extension fallback. Ranking metric: geometric
mean speedup over the reference across selected workloads, with **every** selected workload required
to pass correctness.

> Skill note: `KernelWiki` is scoped to Blackwell (SM100) / Hopper (SM90) primitives (tcgen05,
> TMEM, warp-specialized WGMMA, FP8/NVFP4, 2-SM). The target here is Ampere `sm_80`, so none of
> those primitives exist and that skill is **not** applicable. `ncu-report-skill` targets B200; also
> out of scope for A800 (and profiling is disallowed by the task rules). Optimization here relies on
> classic Ampere tensor-core (BF16/TF32) FlashAttention techniques.

---

## 1. What the operation computes

Batched **Multi-head Latent Attention (MLA)** *prefill* with a **paged KV cache** and a **causal**
mask, captured from DeepSeek-V3 incremental prefill at TP=8.

Fixed constants for this task:

| Constant | Value | Meaning |
|---|---|---|
| `num_qo_heads` | 16 | query heads after TP split (128/8) |
| `head_dim_ckv` | 512 | compressed KV latent dim (this is also the **value** width) |
| `head_dim_kpe` | 64 | rotary/positional key dim (score-only, not a value) |
| `page_size` | 1 | one token per page ⇒ `kv_indices` are **token indices** |

### Inputs (dtypes)
- `q_nope`  `[total_q, 16, 512]` **bf16** — query, latent part
- `q_pe`    `[total_q, 16, 64]`  **bf16** — query, positional part
- `ckv_cache` `[num_pages, 1, 512]` **bf16** — compressed KV latent (serves as both **K-latent** and **V**)
- `kpe_cache` `[num_pages, 1, 64]`  **bf16** — positional key
- `qo_indptr` `[batch+1]` int32 — per-sequence query offsets
- `kv_indptr` `[batch+1]` int32 — per-sequence offsets into `kv_indices`
- `kv_indices` `[num_kv_indices]` int32 — token indices into the cache (page_size=1)
- `sm_scale` scalar **float32** — softmax scale (given per workload; **do not hardcode**)

### Outputs
- `output` `[total_q, 16, 512]` **bf16**
- `lse` `[total_q, 16]` **float32** — **base-2** log-sum-exp of the scaled logits

### Per-(sequence `b`, local query `i`, all 16 heads) math
Let `page_beg=kv_indptr[b]`, `page_end=kv_indptr[b+1]`, `kv_len=page_end-page_beg`,
`tok = kv_indices[page_beg:page_end]`, `Kc = ckv_cache[tok]` `[kv_len,512]`,
`Kp = kpe_cache[tok]` `[kv_len,64]`, `q_len = qo_indptr[b+1]-qo_indptr[b]`.

```
logits[h, j] = (q_nope[i,h,:] · Kc[j,:]) + (q_pe[i,h,:] · Kp[j,:])      # j = 0..kv_len-1
s[h, j]      = logits[h, j] * sm_scale
prefix_len   = kv_len - q_len
query_abs    = prefix_len + i                    # absolute position of query i
s[h, j]      = -inf   where  j > query_abs       # causal mask (per-row cutoff)
lse[q,h]     = logsumexp_j( s[h,:] ) / ln(2)     # base-2
p[h, j]      = softmax_j( s[h,:] )               # fp32
out[q,h,:]   = sum_j p[h,j] * Kc[j,:]            # [512], then cast bf16
```
If a sequence has **no KV** (`page_beg >= page_end`) or **no queries**, the reference `continue`s,
leaving `output = 0` and `lse = -inf` for those rows. This must be reproduced exactly.

### Structural observations (drive the whole design)
1. **This is MQA-shaped.** All 16 query heads at a given token share the *same* gathered `Kc/Kp`.
   The expensive paged gather is **amortized across all 16 heads** (and across query tokens in the
   same sequence). Treat the 16 heads as the "M" (row) dimension of both matmuls.
2. **V equals the latent K.** The value used in `out = p @ Kc` is exactly the ckv latent (512 wide).
   The `kpe` (64) contributes to scores only; there is no separate V tensor. Effective attention
   dims: Q/K contract = 512+64 = **576**, output/V width = **512**.
3. **`page_size == 1`** ⇒ each KV row is an independent gather of one cache row (512+64 bf16 ≈ 1.15 KB),
   scattered across `num_pages ≈ 9.9e5`. KV access is a **random gather**, essentially
   bandwidth/latency-bound with poor spatial locality across the page dimension.
4. **Causal with a prefix offset.** Because `prefix_len = kv_len - q_len` can be large (decode-like),
   the causal frontier for a query block can be exploited to *skip fully-masked KV tiles* and *skip
   masking on fully-unmasked KV tiles*.

---

## 2. Workload analysis (five fixed feedback workloads)

`num_pages = 989669` and `sm_scale = 0.1352337747812271` are constant across all five.
`batch = len_indptr - 1`. (Indptr contents come from safetensors; exact per-seq splits unknown until
runtime, so averages below are indicative.)

| # | total_q | batch | num_kv_idx | avg q_len | avg kv_len | Regime |
|---|--:|--:|--:|--:|--:|---|
| 1 | 6053 | 11 | 6091 | ~550 | ~554 | **Large prefill** (compute + KV bandwidth heavy) |
| 2 | 43   | 1  | 46   | 43   | 46   | Tiny single-sequence prefill |
| 3 | 17   | 1  | 19   | 17   | 19   | Tiny single-sequence prefill |
| 4 | 22   | 22 | 17759| **~1**| ~807 | **Decode-like**: 1 query/seq, long context |
| 5 | 3842 | 20 | 3916 | ~192 | ~196 | Medium prefill |

Key implications:
- **Workload 4 is the parallelism trap.** ~22 query tokens × 16 heads = ~352 output rows total, but
  ~17.8k KV tokens to stream. Without splitting the KV dimension, only a handful of CTAs launch and
  the A800's ~108 SMs are starved → this is where a naive "one CTA per (seq, query-block)" kernel
  will be slowest relative to a good baseline. A **flash-decoding / split-KV** path is essential here.
- **Workloads 1 & 5** have plenty of query rows → tiling the query dimension gives ample CTAs; KV
  reuse across a query tile is the lever (bigger query tiles ⇒ fewer KV re-reads).
- **Workloads 2 & 3** are tiny; kernel-launch overhead and single-CTA latency dominate. Correctness
  and low fixed overhead matter more than throughput here.
- Because `avg kv_len ≈ avg q_len` in 1/5, the causal triangle means roughly half the score work is
  masked away — causal tile-skipping is worth ~2×.

A single kernel must be *correct* everywhere, but a **regime-aware launch heuristic** (q-tiling vs
split-KV) is likely needed to be *fast* everywhere, especially to avoid regressing on #4.

---

## 3. Correctness requirements & exact-match hazards

1. **base-2 LSE.** `lse = logsumexp(scaled)/ln2`. In FlashAttention we track running max `m` and sum
   `l`; final natural LSE `= m + ln(l)`, then divide by `ln2`. If the online softmax runs in base-2
   (via `exp2`), keep the base bookkeeping consistent: `lse2 = (m + ln l)/ln2 = m/ln2 + log2(l)`.
   Easiest safe path: keep `m` and `l` in the **natural** domain (`exp`), emit `(m+log(l))/log(2)`.
   Alternatively fold `sm_scale*log2(e)` into the QK scale and use `exp2` throughout, then
   `lse2 = m2 + log2(l2)` where `m2` is the running max of `scaled*log2(e)`. Must be verified against
   the fp32 reference — LSE is fp32-compared and less forgiving than the bf16 output.
2. **sm_scale applied before both** softmax and LSE. Use the provided fp32 scalar; the definition's
   "default 1/sqrt(192)" is *not* the workload value (0.1352…). Never hardcode.
3. **Causal cutoff is per-row with a prefix offset.** `valid(j) ⟺ j ≤ (kv_len - q_len) + local_i`.
   Compute `local_i = global_q - qo_indptr[b]` and use that batch's `kv_len`, `q_len`. Mask sets
   `-inf` (equivalently `exp=0`) — never contributes to `l`, `m`, or `out`.
4. **Empty / skipped sequences** must yield `output = 0` and `lse = -inf`. If `kv_len == 0` (or the
   query's sequence is skipped), write zeros and `-inf` and do not divide by an empty sum.
   Guaranteed non-empty in these five workloads, but the kernel must not NaN if it happens.
5. **No fully-masked rows in normal case.** `query_abs = kv_len - q_len + i ∈ [kv_len-q_len, kv_len-1]`
   is always a valid index ≥ 0 (assuming `kv_len ≥ q_len`), so every query attends to ≥1 token and
   `l > 0`. Still guard `l == 0` to avoid `1/0`.
6. **Output dtype/layout.** `output` bf16 `[total_q,16,512]`, `lse` fp32 `[total_q,16]`. Cast to bf16
   only at the final store (accumulate fp32 internally).
7. **Tolerance is not stated in `definition.json`.** No explicit rtol/atol/cosine field is present.
   Treat this as a correctness risk: design to match fp32 reference as tightly as feasible (fp32
   accumulation everywhere), so we pass regardless of whether the harness uses bf16-scale tolerance
   or something tighter, and confirm empirically on the feedback set.

---

## 4. Numerical risk analysis (Ampere `sm_80`)

- **QK matmul is effectively exact vs the reference.** The reference casts the bf16 `q`/`K` to fp32
  then multiplies. bf16×bf16 products are *exact* in fp32 (8+8 mantissa bits ⇒ ≤16, fits fp32's 23),
  and Ampere BF16 tensor cores accumulate in fp32. So a BF16-MMA `q@Kᵀ` equals the reference fp32
  matmul up to accumulation order — **no precision concern for QK**. This is the strongest reason to
  keep inputs bf16 and use BF16 tensor cores rather than up-casting.
- **Avoid TF32 traps.** If we do fp32 `tl.dot`, Triton/Ampere may lower it to **TF32** (10-bit
  mantissa) which is *worse* than bf16-exact-product accumulation and would diverge from the fp32
  reference. Prefer explicit bf16 operands with fp32 accumulator; if any fp32 dot is used, set
  `allow_tf32=False` and understand it will be slow. Net: keep QK in bf16 MMA.
- **PV matmul is the real precision knob.** `out = p @ Kc` with `p` a fp32 softmax probability. To use
  tensor cores we cast `p → bf16` (standard FlashAttention). `p ∈ [0,1]` rounded to 8 mantissa bits
  introduces ~2⁻⁸ relative error per weight; summed over `kv_len` weights it stays small and the
  output is stored as **bf16** anyway (≈3 significant digits). This is expected to pass typical
  bf16-output tolerances. Fallbacks if it does not: (a) split `p` into hi/bf16 + lo/bf16 error-
  compensated dot (2× MMAs, ~fp32 accuracy), or (b) fp32 dot with `allow_tf32=False` (slow, exact-
  ish). Start with bf16 `p`; escalate only if the feedback eval shows correctness failure.
- **Softmax stability.** Standard online max-subtraction (`exp(s - m)`), rescale `l` and `acc` on max
  update. `s` magnitude ~ `sm_scale * (q·k)` with 576-dim bf16 dots — modest; `-inf` masked lanes use
  `exp = 0`. No overflow expected. Use `exp2` for speed only if LSE base bookkeeping stays consistent.
- **Accumulator width.** `acc[M,512]` and `l`,`m` kept fp32 throughout; only final cast to bf16.
- **LSE fp32 comparison** is the tightest check (see §3.1). Get the base-2 conversion and the
  `m+log(l)` combination exactly right; a wrong `/ln2` or a base-2/e mismatch is the most likely
  correctness bug.

---

## 5. Triton design space

### 5.1 Core kernel shape (shared by all variants)
Because MLA is MQA-shaped, put the **16 heads on the M (row) axis** so one paged gather feeds all
heads:
- QK: `logits[M, BLOCK_N] = q_nope[M,512] @ Kcᵀ[512,BLOCK_N] + q_pe[M,64] @ Kpᵀ[64,BLOCK_N]`
  (two `tl.dot`s accumulated into the same fp32 logits tile).
- Online softmax over KV tiles (running `m`,`l`, rescale `acc`).
- PV: `acc[M,512] += p[M,BLOCK_N] @ Kc[BLOCK_N,512]` (bf16 `p`).
- `M = 16` (single query token) or `M = BLOCK_Q_tok * 16` (a few query tokens stacked as rows, all
  sharing the same sequence's KV, each row carrying its own causal cutoff).

`acc` sizing: `[16,512]` fp32 = 32 KB/CTA; `[32,512]` = 64 KB; `[64,512]` = 128 KB. On `sm_80`
(64K 32-bit regs, 164 KB smem per SM) the 512-wide fp32 accumulator is the dominant resource and
caps `M` and occupancy. Likely sweet spot: `M ∈ {16, 32}` (i.e. 1–2 query tokens × 16 heads), 4 warps.

### 5.2 Paged gather
`kv_indices` are token indices (page_size=1). For a KV tile, load `tok = kv_indices[base + n]`
(`n = arange(BLOCK_N)`), then gather `Kc = ckv_cache_ptr + tok[:,None]*512 + arange(512)`
and `Kp` similarly (64). Mask tail lanes where `n ≥ kv_len`. This is a gather-block load; with
`num_pages ≈ 1e6` there is little cross-tile locality, so KV traffic ≈ bandwidth cost. Minimizing KV
**re-reads** (via larger query tiles / split-KV combine) is the main bandwidth lever.

### 5.3 Grid / parallelization strategies (candidate axes)
1. **Prefill q-tiling (workloads 1, 5, 2, 3).** Grid over `(query-tile across total_q)`; each CTA maps
   its tile to a sequence `b` (host-precomputed `tile → b` table, or binary search on `qo_indptr`),
   loads its Q rows once, streams KV tiles with online softmax, exploits causal skipping. Larger
   `BLOCK_Q` ⇒ fewer KV re-reads but bigger `acc`/less occupancy — tune per regime.
2. **Split-KV / flash-decoding (workload 4).** For few-query, long-KV sequences, add a `split`
   dimension to the grid: each CTA handles `(seq, query, kv-chunk)` producing partial `acc`, `m`, `l`;
   a lightweight **combine** kernel merges partials (rescale by global max) and writes final `out`/
   `lse`. This is the only way to fill ~108 SMs when there are ~22 query rows. Number of splits chosen
   from `kv_len` and available parallelism.
3. **Unified persistent kernel** with a host-side scheduler that emits work items of both kinds. More
   complex; defer unless the two-path approach leaves perf on the table.

A host heuristic picks the path per workload, e.g. split-KV when `total_q_rows` is small relative to
SM count and `kv_len` is large; otherwise q-tiling. The *kernel source stays immutable per candidate*;
only launch config differs — but note per KDA rules **any launch change requires a new candidate ID**.

### 5.4 Causal-aware tiling
For a query tile with local rows `[i_lo, i_hi]`, the max valid KV index is `kv_len - q_len + i_hi`.
- Skip KV tiles whose start `> max valid index` (fully masked) — big win for #1/#5 (~2×).
- For KV tiles entirely `≤ (kv_len - q_len + i_lo)` skip mask computation (fully unmasked).
- Only boundary tiles pay the per-element mask compare.

### 5.5 Block-size / tuning axes
- `BLOCK_N` (KV tile): 32 / 64 / 128 — trade gather granularity vs smem/regs.
- `BLOCK_Q_tok`: 1 / 2 / 4 — query tokens per CTA (prefill KV-reuse vs `acc` size).
- `num_warps`: 4 / 8; `num_stages`: 2 / 3 (software pipelining of gather + MMA).
- Split count for decode path.
- Whether the 512-contraction QK uses one `tl.dot(K=512)` or is chunked.
- `exp` vs `exp2` in the online softmax (perf only; must preserve LSE base-2 correctness).

### 5.6 Data-movement / bottleneck model
- **Prefill (1,5):** compute is `~ q_len·kv_len·576·2 (QK) + q_len·kv_len·512·2 (PV)` FLOPs/seq over
  16 heads; also KV bytes `~ (q_len/BLOCK_Q)·kv_len·576·2`. Balanced compute+bandwidth; causal halves
  it. Levers: BF16 tensor cores, causal skipping, KV reuse via query tiling.
- **Decode (4):** ~1 query/seq ⇒ almost no compute reuse; dominated by streaming ~17.8k KV rows ×
  576 bf16 ≈ 20 MB of gathers. Pure bandwidth/latency; split-KV for occupancy is the lever.
- **Tiny (2,3):** launch + single-CTA latency bound; keep the kernel light, avoid extra passes.

---

## 6. Baseline & speedup expectations
The reference is a **Python double `for`-loop over batch and over every query token**, doing fp32
matmuls and full materialized `[16,kv_len]` logits and softmax per token — extremely slow and
launch-heavy. A correct fused Triton FlashAttention-MLA kernel should beat it by a large margin on the
larger workloads (1, 5) and the decode workload (4, with split-KV), and still win on the tiny ones by
collapsing the Python loop into one launch. The geomean is therefore mostly protected as long as we
(a) never regress #4 into a starved single-CTA kernel and (b) keep tiny-workload overhead minimal.

---

## 7. Validation strategy (evaluation-budget aware)
Local execution of CUDA/Triton, profilers, `nvidia-smi`, the evaluator, or any alternate correctness
harness is **prohibited**; the *only* empirical signal is
`./scripts/evaluate_candidate.sh feedback <cNNN>` over the five fixed workloads (one candidate =
all five). Budget: 100 evals; token soft/hard 1.0M/1.2M. Therefore:

1. **Correctness by construction first.** Mirror the reference semantics exactly: sm_scale placement,
   per-row causal-with-prefix cutoff, empty-seq zeros/`-inf`, base-2 LSE, fp32 accumulation, final
   bf16 cast. Re-derive the LSE base-2 math on paper before coding.
2. **c001 = simplest fully-correct fused kernel** (q-tiling, `M=16`, one query/CTA, bf16 QK + bf16 PV,
   correct masking/LSE). Establish a correct, immutable baseline and read its per-workload speedups —
   especially whether #4 is starved.
3. **Then optimize in small, attributable steps**, each a new candidate ID: add causal tile-skipping;
   add split-KV/decode path for #4; tune `BLOCK_N`/`BLOCK_Q`/warps/stages; stack query tokens for KV
   reuse. Change one lever per candidate so the feedback delta is attributable.
4. **Escalate PV precision only if a correctness failure appears** (bf16-`p` → compensated/hi-lo or
   fp32 dot). Don't pay precision cost pre-emptively.
5. **Record** per-candidate: parent, source hash, hypothesis, per-workload result + geomean, decision,
   cumulative eval count, skill usage — append-only to `candidates.jsonl`.
6. **Converge & stop** when deltas flatten; write `SEARCH_COMPLETE`. `final` only on operator approval.

## 8. Open questions / risks to watch
- **Exact tolerance** used by the harness (unstated) — governs whether bf16-`p` PV suffices. Watch the
  first feedback result closely.
- **LSE base-2** conversion is the most likely subtle correctness bug (fp32-compared).
- **Workload #4 occupancy** — the main perf risk; validate that the decode/split-KV path actually
  lands more CTAs.
- **`tile → sequence` mapping**: host-side precompute (from indptrs) vs in-kernel binary search — pick
  the cheaper/robust option; keep it in launch plumbing (PyTorch) which is allowed.
- **Register/occupancy pressure** from the 512-wide fp32 accumulator caps `M` and split sizes; may
  force `M=16` and modest tiles on `sm_80`.
- **TF32 leakage** into any accidental fp32 dot would silently hurt accuracy — keep dots in bf16.
