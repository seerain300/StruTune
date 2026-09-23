# Draft — `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`

Task: optimize the official FlashInfer DSA (DeepSeek‑V3.2) sparse paged MLA
attention op on **H100 (sm_90)**. Primary implementation must be Triton; the
`run(...)` entry point lives in `solution/solution.py`. PyTorch is allowed only
for metadata / launch plumbing. No Torch/CPU/NumPy/CUDA‑extension computational
fallback. Ranking metric = geomean speedup over the reference, with every
selected workload required to pass correctness.

This document is analysis only. No `docs/plan.md` and no solution code are
produced this turn.

---

## 1. What the operation computes

### 1.1 Shapes and dtypes (from `task/definition.json`)

Constants for this variant:

| name          | value | meaning                                   |
|---------------|-------|-------------------------------------------|
| `num_qo_heads`| 16    | query heads after TP=8 split (128/8)      |
| `head_dim_ckv`| 512   | compressed KV / latent dim (also = V dim) |
| `head_dim_kpe`| 64    | key positional‑encoding dim               |
| `page_size`   | 64    | tokens per KV page                        |
| `topk`        | 2048  | sparse entries selected per token         |

Variables across the workload set:

- `num_tokens` ∈ {1, 2, 6, 7, 8} (decode‑like; tiny query dimension).
- `num_pages` = 8462 for every feedback workload → KV pool =
  `8462 * 64 = 541,568` tokens. topk/pool ≈ **0.38 % sparsity**.

Inputs:

- `q_nope`  `[num_tokens, 16, 512]` bf16
- `q_pe`    `[num_tokens, 16, 64]`  bf16
- `ckv_cache` `[num_pages, 64, 512]` bf16  (paged latent KV)
- `kpe_cache` `[num_pages, 64, 64]`  bf16  (paged key PE)
- `sparse_indices` `[num_tokens, 2048]` int32 — **loaded from safetensors**
  (real captured patterns, fixed per workload). `-1` = padding/invalid.
  A valid value `v` addresses the *flattened* token
  `page_idx*64 + offset`, i.e. it directly indexes
  `ckv_cache.reshape(-1, 512)[v]` and `kpe_cache.reshape(-1, 64)[v]`.
- `sm_scale` scalar fp32 = `0.1352337788608801`.

Outputs:

- `output` `[num_tokens, 16, 512]` bf16
- `lse`    `[num_tokens, 16]` fp32 — **base‑2** log‑sum‑exp of the scaled
  logits (`logsumexp(scaled)/ln 2`).

### 1.2 Reference math (per token `t`, per head `h`)

```
indices  = sparse_indices[t]                    # [2048], may contain -1
valid    = indices != -1
Kc       = Kc_all[indices[valid]]               # [nv, 512]  (from ckv_cache)
Kp       = Kp_all[indices[valid]]               # [nv, 64]   (from kpe_cache)
logits   = q_nope[t] @ Kc.T + q_pe[t] @ Kp.T    # [16, nv], done in fp32
scaled   = logits * sm_scale
lse[t]   = logsumexp(scaled, -1) / ln2          # [16]
attn     = softmax(scaled, -1)                  # [16, nv]
out[t]   = (attn @ Kc).to(bf16)                 # [16, 512], V == Kc
```

### 1.3 The key structural observation — this is sparse **MLA** decode

The "value" matrix **is** `Kc` (the same compressed latent tensor used for the
score's nope part). `kpe` contributes to the *score only*, never to the output.
So, defining an effective query/key of width `512+64 = 576`:

- `Q_eff = concat(q_nope, q_pe)` → `[16, 576]`
- `K_eff = concat(Kc, Kp)`       → `[nv, 576]`
- scores `= (Q_eff @ K_eff.T) * sm_scale`
- `V = Kc` → `[nv, 512]`, `out = softmax(scores) @ V`

This is exactly the FlashMLA / DeepSeek‑V3.2 sparse decode kernel shape
`h16_ckv512_kpe64_topk2048_ps64` described in KernelWiki
(`wiki/kernels/sparse-mla.md`, `contest-flashinfer-track-b`,
`wiki/kernels/flashmla.md`). It is an MQA‑style pattern: all 16 heads of a
token share the *same* gathered `K_eff`/`V`, so the gather is amortized across
heads. Note our task receives `sparse_indices` **already computed** — only
Stage‑2 (sparse MLA attention) is in scope; the Lightning Indexer is not.

### 1.4 Arithmetic intensity / roofline (why it is memory‑bound)

Per token (assuming ~2048 valid):
- QK flops: `16 * 2048 * 576 * 2 ≈ 37.7 MFLOP`
- PV flops: `16 * 2048 * 512 * 2 ≈ 33.6 MFLOP`
- total ≈ **71 MFLOP/token** → ≤ ~0.57 GFLOP for the largest (8‑token) case.

Gathered bytes per token (bf16): `Kc = 2048*512*2 = 2.0 MiB`,
`Kp = 2048*64*2 = 0.25 MiB` → ≈ **2.25 MiB/token**. For 8 tokens ≈ 18 MiB.
H100 HBM ≈ 3.35 TB/s → ideal gather ≈ **~5–6 µs**; compute at even 100 TFLOP
effective is ~6 µs but tensor‑core‑friendly so lower. The op is firmly
**HBM‑bandwidth / gather‑latency bound**, and the gather (`ckv_cache` reads
through arbitrary `sparse_indices`) dominates. This shapes every optimization
decision below.

---

## 2. Constraints and correctness contract

- Constants asserted in the reference (`h=16, ckv=512, kpe=64, ps=64,
  topk=2048`) — we may hard‑code them as `tl.constexpr` and specialize.
- `sparse_indices.shape[0] == num_tokens`, `[-1] == topk`,
  `ckv_cache.shape[1] == page_size`.
- Output dtype bf16; lse dtype fp32 and **base‑2**.
- `-1` padding must be excluded from softmax (contributes `-inf` logit, `0`
  weight) and from the output sum.
- **Empty token** (all indices `-1`, `valid_indices.numel()==0`): reference
  writes `output[t] = 0` and **leaves `lse[t] = -inf`** (its initialized value;
  the loop `continue`s before assigning lse). Our kernel must reproduce both:
  zero output row AND `lse = -inf`, not `0` and not NaN.
- Indices are `int32`; the reference casts to `long` for gather. Real captured
  patterns for a 8462‑page pool are well within int32 range. We still clamp
  before any masked load to avoid OOB address computation.
- The evaluator compares against the reference with its own tolerance (not
  disclosed here); given bf16 outputs the effective target is roughly bf16
  relative precision. We design for fp32 logit/softmax accumulation to stay
  safely inside that.

---

## 3. Numerical risks and how to control them

1. **Softmax stability.** Use the standard flash online‑softmax with running
   max `m` and denominator `l`; never exponentiate raw logits. Fold `sm_scale`
   into the logits, and fold `log2(e)` so we can use `exp2` (matches base‑2 LSE
   naturally and is faster on HW). `scaled = logits*sm_scale`; work in the
   base‑2 domain `p = exp2((s - m) * log2e_scaled)`.

2. **LSE definition.** `lse = logsumexp(scaled)/ln2 = log2(sum exp(scaled))`.
   In flash base‑2 form: `lse = m2 + log2(l2)` where `m2` and `l2` are the
   running max and sum expressed in base‑2 units (verified numerically to match
   the reference to fp64 precision). For empty tokens force `lse = -inf`.

3. **QK precision.** `q` and `K` are already bf16 in memory; the reference
   upcasts to fp32 then multiplies. A bf16×bf16 tensor‑core matmul with fp32
   accumulate computes the exact same products (product of two 8‑bit‑mantissa
   values is representable in fp32) up to accumulation order → high fidelity.
   Accumulate logits in fp32.

4. **PV precision.** The probabilities `P` are fp32. To use tensor cores for
   `P @ V` we must down‑cast `P` to bf16, which loses ~8 mantissa bits. Because
   the final `output` is bf16 anyway, this is expected to sit within tolerance,
   but it is the single biggest numerical risk. Mitigations, in order of
   preference to try: (a) bf16 `P` with fp32 accumulate (fastest); (b) split‑K
   partials combined in fp32; (c) if correctness fails, fp32 FMA PV (no tensor
   core) — compute is trivial so the cost may be acceptable. Decide via the
   evaluator.

5. **Split‑K / flash‑decoding reduction.** Because `num_tokens` is tiny (see
   §4), we must split the 2048 KV entries across multiple CTAs and combine
   partial `(m, l, acc)`. The combine must use the standard log‑sum‑exp merge
   in fp32 to avoid cancellation; guard the all‑`-inf` (empty) case so the
   rescale `exp2(m_i - m)` never produces NaN (`0 * inf`).

6. **`-inf` handling in masked tiles.** For masked/padding lanes set the logit
   to `-inf` *before* the max so they never contribute. Ensure a fully masked
   tile yields `m = -inf, l = 0` and contributes nothing to the merge.

7. **Gather OOB safety.** Compute the flattened row from the index, mask the
   `-1` lanes, and clamp the address for masked lanes to a valid row (e.g. 0)
   with `other=0.0` on the load so no illegal address is formed.

---

## 4. Parallelization analysis (the crux on H100)

H100 has 132 SMs. The natural "one program per (token, head‑tile)" gives only
`num_tokens * head_tiles` CTAs:

- `num_tokens = 1` → as few as 1–2 CTAs. Catastrophic under‑occupancy.
- `num_tokens = 8`, one CTA/token → 8 CTAs. Still ~6 % of the machine.

Since all 16 heads share the same gather and 16 rows is a natural `BLOCK_M`,
the query side alone cannot fill the GPU. The dominant, well‑known remedy is
**flash‑decoding / split‑K over the KV (topk) dimension**:

- Stage‑1 grid = `(num_tokens, num_kv_splits)` (optionally × head‑tiles). Each
  program consumes a contiguous slice of the 2048 indices, gathers that slice,
  and produces a partial `(m, l, acc[16,512])`.
- Stage‑2 combine kernel reduces the `num_kv_splits` partials per (token, head)
  into final `output` + `lse`.

Choosing `num_kv_splits` so that `num_tokens * num_kv_splits (* head_tiles)`
comfortably exceeds ~132 (aim a few‑× for latency hiding) is the primary
occupancy lever. For `num_tokens=1` we need ~64–128 splits; for `num_tokens=8`
fewer. A launch‑time heuristic based on `num_tokens` (and topk/split size) is
warranted. This mirrors vLLM/SGLang Triton MLA decode which use split‑K
reductions (`pr-vllm-12528`, `pr-vllm-34597`, `pr-sglang-18442`).

### 4.1 Accumulator sizing

`acc[BLOCK_M=16, 512]` fp32 = 8192 floats = 32 KiB of register/local state per
program — large but feasible with few warps. Options if register pressure
hurts: tile the 512 output dim (e.g. two passes of 256, or a `BLOCK_DV`), or
reduce warps. The 512 K‑contraction for QK and the 512 V dim are the heavy
tiles; `kpe=64` is a cheap appendage to the score. Autotuning
`BLOCK_N (KV tile)`, `num_kv_splits`, `num_warps`, `num_stages` is expected.

### 4.2 Memory‑access / gather strategy

- The gather over `sparse_indices` is the bottleneck. `page_size=64` means 64
  consecutive valid indices *could* be contiguous in the cache, but captured
  patterns are arbitrary, so assume random 512‑element (1 KiB bf16) row reads.
- Load `q_nope`/`q_pe` once per (token, head‑tile) into SMEM/registers and
  reuse across all KV tiles (amortized; q is tiny: `16*576*2 ≈ 18 KiB`).
- Use `tl.load` with an index vector (`ckv_ptr + idx[:,None]*512 + arange(512)`)
  and `mask` for `-1`. `other=0.0`.
- Consider reading `Kc` once and using it for *both* the QK (nope part) and the
  PV, avoiding a second HBM read of the same rows (major bandwidth saving,
  since Kc is 2 MiB/token vs Kp 0.25 MiB). Keeping the KV tile resident in SMEM
  across the QK→softmax→PV steps is the classic flash pattern and directly
  cuts the dominant traffic.

---

## 5. Triton design space

Candidate axes to explore (later, in `plan.md` / candidates):

1. **Baseline single‑pass kernel** (`c001`): grid `(num_tokens,)`, `BLOCK_M=16`
   (all heads), loop over KV tiles with online softmax, `V=Kc` reused from the
   same tile, fp32 accumulate, bf16 `P@V`. Establishes correctness + a
   reference speed. Likely occupancy‑limited but validates the math/edge cases.
2. **Split‑K flash‑decoding** (`c002`+): add `num_kv_splits`, stage‑2 combine.
   Primary expected win for the tiny‑`num_tokens` regime.
3. **KV‑tile resident in SMEM**, single load of `Kc` reused for QK and PV.
4. **Autotune** `BLOCK_N`, `num_kv_splits`, `num_warps`, `num_stages`;
   `num_kv_splits` chosen by a host heuristic from `num_tokens`.
5. **Output‑dim tiling** (`BLOCK_DV` over the 512) if register pressure caps
   occupancy or `num_stages`.
6. **PV precision variants** — bf16‑`P` fast path vs fp32 PV fallback, chosen if
   correctness margin is tight.
7. **Index/pointer layout** — precompute valid counts / compact indices on host
   (allowed as metadata plumbing) vs. masking inside the kernel; weigh the host
   launch overhead against kernel simplicity. Prefer in‑kernel masking to avoid
   extra passes unless profiling shows the masked lanes waste real bandwidth.
8. **Handling variable valid counts** — since padding trails as `-1`, an early
   exit / dynamic loop bound per token can skip fully‑padded tail tiles.

Design principles: hard‑code the constexpr shapes (16/512/64/64/2048); keep the
gather resident and reused; fill the SMs via split‑K; accumulate in fp32.

---

## 6. Validation strategy

- **Correctness gate = the evaluator only.** Per the rules I must not run CUDA,
  `nvidia-smi`, the external evaluator directly, or any alternate correctness
  harness. Kernel correctness is judged by
  `./scripts/evaluate_candidate.sh feedback <cid>` over the full 23‑workload
  feedback set (that counts as one evaluation), which does the allclose vs the
  reference and returns per‑workload pass/fail + timing.
- **Offline (CPU, no GPU) math checks** are used only to lock down formulas
  *before* coding: I already verified the base‑2 flash LSE identity matches
  `logsumexp(scaled)/ln2` to fp64 precision, and confirmed `sm_scale`
  (0.13523…) is the given scalar (note it is ~1.874× `1/sqrt(192)`, so use the
  passed value verbatim — do **not** recompute the scale). These checks touch
  no GPU and no evaluator.
- **Edge‑case matrix to confirm through the evaluator:** tokens with trailing
  `-1` padding (partial valid count), any fully‑empty token
  (`lse=-inf`, zero output), `num_tokens=1` (occupancy stress), and the largest
  `num_tokens=8`.
- **Performance profiling** only via `./scripts/ncu_profile.sh` following the
  `ncu-report-skill` workflow, and **never concurrently with an evaluation**
  (a foreign process on the locked GPU during timing → return code 3, wasted
  budget). Use ncu to confirm the memory‑bound diagnosis: HBM throughput,
  achieved occupancy, split‑K effectiveness, and gather (L2/DRAM) behavior.
- **Candidate discipline:** each meaningful source/config/launch change gets a
  new immutable id `cNNN`; results appended to `candidates.jsonl` with parent,
  source hash, hypothesis, per‑workload result, geomean, decision, cumulative
  eval count, and skill usage. Stop / write `SEARCH_COMPLETE` when converged;
  never run `final` without operator approval.

---

## 7. Risks, unknowns, and mitigations

| risk | impact | mitigation |
|------|--------|-----------|
| Under‑occupancy at `num_tokens=1` | dominates geomean | split‑K flash‑decoding with host‑chosen `num_kv_splits` |
| bf16 `P@V` precision | correctness fail | fp32 combine; fp32 PV fallback if needed |
| Empty‑token `lse=-inf` mismatch | correctness fail | explicit `-inf` init + guarded combine |
| Gather OOB from `-1` | illegal memory | mask + clamp + `other=0` |
| Register pressure from `acc[16,512]` | low occupancy / spills | tile `BLOCK_DV`, tune warps/stages |
| Redundant double read of `Kc` | wastes dominant bandwidth | keep KV tile resident, reuse for QK & PV |
| Profiling/eval overlap | rc=3, wasted budget | strictly serialize; never background profile |

---

## 8. Baseline understanding of the reference cost

The reference is a pure‑PyTorch Python `for t in range(num_tokens)` loop with a
boolean‑mask gather, two fp32 matmuls, a `logsumexp`, a `softmax`, and a matmul
per token — many small eager ops and host‑side control flow. It is expected to
be dramatically launch/overhead bound at these tiny token counts, so a single
fused, well‑occupied Triton kernel (even the baseline `c001`) should already
beat it substantially, with split‑K providing the headroom to fill H100 and
lift the geomean further.

---

## 9. KernelWiki references consulted

- `wiki/kernels/sparse-mla.md` (`kernel-sparse-mla`) — DeepSeek V3.2 sparse MLA
  two‑stage structure; confirms our op is Stage‑2 with V==compressed KV.
- `sources/contests/flashinfer-mlsys26/track-b-sparse-attention.md`
  (`contest-flashinfer-track-b`) — exact benchmark id
  `h16_ckv512_kpe64_topk2048_ps64`, sparse‑gather + on‑the‑fly attention,
  padding‑to‑64 head waste caution, index‑layout locality notes.
- `wiki/kernels/flashmla.md` (`kernel-flashmla`) — paged block_size=64, online
  softmax with LSE, sparse token‑index selection, decode is memory‑bound.
- `pr-vllm-12528`, `pr-vllm-34597`, `pr-sglang-18442` — Triton MLA decode /
  split‑K decode patterns to model the flash‑decoding reduction on.

(Note: several KernelWiki pages target B200/FP8 KV. Our task is **H100/sm_90**
with **bf16** caches and precomputed indices, so FP8 dequant and tcgen05/TMEM
specifics do not apply; the transferable ideas are the MLA structure, online
LSE, sparse gather locality, and split‑K occupancy.)
