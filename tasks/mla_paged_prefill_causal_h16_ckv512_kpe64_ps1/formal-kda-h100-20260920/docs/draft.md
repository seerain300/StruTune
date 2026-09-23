# Draft — `mla_paged_prefill_causal_h16_ckv512_kpe64_ps1`

Target: NVIDIA H100 (`sm_90`), Triton primary implementation. Metric: geomean speedup
over the reference across the full 38-workload feedback set; every selected workload must
pass correctness. Budget: 100 candidate evaluations, ~9–11M tokens.

## 1. What the operation computes

Batched **Multi-head Latent Attention (MLA) prefill** with a paged KV cache
(page_size = 1), causal mask, captured from DeepSeek-V3 incremental prefill (TP=8, so 16 heads).

Fixed constants (from `task/definition.json`):
- `num_qo_heads = 16`
- `head_dim_ckv = 512`  (the compressed latent / "nope" dimension)
- `head_dim_kpe = 64`   (the RoPE / positional dimension)
- `page_size = 1`  ⇒ each entry of `kv_indices` is a **direct token index** into the cache.

Inputs (all bf16 except indptr/indices int32 and `sm_scale` fp32 scalar):
- `q_nope`  `[total_q, 16, 512]`
- `q_pe`    `[total_q, 16, 64]`
- `ckv_cache` `[num_pages, 1, 512]`  (compressed KV; used as **both** key-content and value)
- `kpe_cache` `[num_pages, 1, 64]`   (positional key; contributes to logits only)
- `qo_indptr[len_indptr]`, `kv_indptr[len_indptr]`, `kv_indices[num_kv_indices]` (int32)
- `sm_scale` = 1/sqrt(192) ≈ 0.1352337747812271 (constant in every workload)

Outputs:
- `output` `[total_q, 16, 512]` bf16
- `lse`    `[total_q, 16]` fp32 — the **base-2** log-sum-exp of the scaled logits.

### Per-sequence math (from the reference)

For sequence `b` with `q_start:q_end` queries (`q_len`) and `kv_indptr[b]:kv_indptr[b+1]`
page range (`kv_len` tokens), gather from the cache using `kv_indices` (page_size=1):
- `Kc = ckv_cache[tok_idx]` → `[kv_len, 512]`  (key-content **and** value)
- `Kp = kpe_cache[tok_idx]` → `[kv_len, 64]`   (positional key)

For each query row `i` and each head `h`:
- `logits[h,:] = q_nope[i,h] @ Kc.T + q_pe[i,h] @ Kp.T`  → `[kv_len]`
- `logits_scaled = logits * sm_scale`
- **Causal mask**: `prefix_len = kv_len - q_len`; `query_abs_pos = prefix_len + i`;
  keep token `j` iff `j <= query_abs_pos` (mask `j > query_abs_pos` to `-inf`).
- `lse[i,h] = logsumexp(logits_scaled) / ln(2)`  (natural LSE converted to base 2)
- `attn = softmax(logits_scaled)`; `out[i,h] = attn @ Kc` → `[512]`

### Key structural observations

1. **V ≡ Kc.** The value matrix is exactly the compressed-KV content used for the "nope"
   part of the score. So the key is a 576-wide vector `[Kc(512) | Kp(64)]`, the query is
   576-wide `[q_nope(512) | q_pe(64)]`, but the value is only the first 512 (`Kc`). The
   `kpe` part affects logits only, never the output. This is the classic "matrix-absorbed"
   MLA form.
2. **All 16 heads share the same K and V** (`ckv_cache`/`kpe_cache` have no head axis). So
   `kv_group_num = 16`. A naive per-head kernel re-reads the entire latent KV 16× — the
   single biggest avoidable memory cost for the memory-bound (small) workloads.
3. **Causal with a prefix.** `kv_len = prefix_len + q_len` per sequence: the first
   `prefix_len` cached tokens are attended by every query; the last `q_len` tokens are the
   extend tokens and are causal among themselves. Equivalent single test: keep token `j`
   iff `j <= prefix_len + i`. This is exactly the SGLang "extend / prefill-with-KV-cache"
   pattern with a unified KV buffer.
4. **sm_scale is a constant** (0.13523…) across all 38 workloads — but it is passed as a
   runtime scalar; treat it as a kernel arg (do not bake it in — a config change would need
   a new candidate anyway, and correctness must not rely on the literal).

## 2. The feedback workload distribution (38 workloads)

`num_pages` is always 989669 (huge cache; only affects index range, not work).
`len_indptr-1 = batch_size`. Grouping by regime:

| Regime | Examples (total_q / kv / batch) | Character |
|---|---|---|
| Tiny single-seq | 1/34/1, 3/5/1, 5/7/1, 8/12/1, 10/12/1, 13/14/1, 15/18/1, 17/19/1, 29/34/1, 33/34/1, 43/46/1, 58/60/1, 96/98/1 | launch/latency-bound, tiny compute |
| Short-Q multi-seq (decode-like) | 2/53/2, 4/121/4, 6/109/6, **22/17759/22** | ~1 query/seq, memory-bound; #17 has ~807 kv/seq |
| Medium prefill | 123/185/2, 138/151/4, 199/203/1, 287/288/1, 376/381/1, 473/491/5, 805/814/4 | mixed |
| Large single/few-seq prefill | 1028/1038/1, 1187/1205/3, 3024/3029/3, 6053/6091/11, **16384/16387/1**, 10870/10875/2 | compute-bound |
| Large multi-seq prefill | 1954/2044/28, 3842/3916/20, 8987/14390/56, 15092/15187/26, 15883/15937/18 | compute-bound, many seqs |

Implications:
- The geomean rewards **both** the tiny/latency-bound cases (where launch overhead, grid
  occupancy, and avoiding 16× KV reload dominate) **and** the big compute-bound cases
  (where tensor-core utilization, causal early-exit, and tiling dominate). No single block
  config is optimal everywhere; heuristics keyed on `q_len` regime and/or `@triton.autotune`
  are likely needed.
- Workload #17 (22 queries, 22 seqs, ~807 kv each) and the short-Q multi-seq cases are the
  strongest argument for **folding the 16 heads into the matmul M dimension** so the shared
  latent KV is read once per query rather than 16×.
- The 16384/16387 single-sequence case is the dominant compute case and needs **causal
  early-exit** (a query block only needs KV up to `prefix_len + block_end`) plus good
  `BLOCK_M/BLOCK_N`.

## 3. Constraints and rules that shape the design

- Triton must do the actual attention math; PyTorch only for metadata/launch. **No Torch/
  CPU/NumPy/CUDA-extension fallback** — a failing Triton kernel is invalid, not replaceable.
- One immutable source version = one candidate = one full 38-workload evaluation. Any
  meaningful source/config/launch change ⇒ new candidate id. Never reuse ids.
- Evaluate only via `./scripts/evaluate_candidate.sh feedback cNNN`. Profiling only via
  `./scripts/ncu_profile.sh` and **never** concurrently with an evaluation (return code 3
  wastes a budget slot).
- `final` only with explicit operator approval.
- Output must be exactly `solution/solution.py` exposing `run(q_nope, q_pe, ckv_cache,
  kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)` returning `(output, lse)`.

## 4. Numerical analysis and risks

- **Accumulation.** Reference upcasts q and K to fp32 and matmuls in fp32. bf16 inputs are
  exactly representable in fp32, and Hopper bf16 tensor-core matmul (`tl.dot`, bf16×bf16 →
  fp32 accumulate) computes each product exactly (bf16 8-bit mantissa product fits in fp32)
  and accumulates in fp32. So `tl.dot` on bf16 with fp32 accumulation matches the reference
  QK/PV to ~fp32 precision. **Do all reductions/softmax/accumulators in fp32.**
- **P@V cast.** Tensor cores need bf16/fp16 operands, so the softmax probabilities `p`
  (fp32) must be cast to bf16 for `p @ Kc`. Reference keeps `p` in fp32. bf16 `p` introduces
  ~2^-8 relative error per element; summed it stays ~0.4%, well within typical bf16-output
  tolerance, and the output is bf16 anyway. Keep the online-softmax rescale + accumulator in
  fp32; only the `tl.dot` operand is bf16. (If correctness is marginal, an fp32
  `sum(p[:,:,None]*v)` fallback exists but is much slower — treat as last resort, not a
  Torch fallback.)
- **Base-2 LSE.** Reference: `logsumexp(logits*sm_scale)/ln(2)`. Cleanest and fastest to
  compute the whole softmax in base-2: multiply logits by `sm_scale * log2(e)` and use
  `exp2`. Then `p = exp2(s2 - m2)` are the identical probabilities, and
  `lse = m2 + log2(e_sum)` is already base-2 — no extra `/ln2`. This both matches the
  reference LSE and is faster than natural `exp`. Verify the `log2(e)` constant
  (1.4426950408889634) is applied exactly once.
- **All-masked / empty rows.**
  - Reference guards `if q_start>=q_end or page_beg>=page_end: continue`, leaving `output=0`
    and `lse=-inf` for empty sequences. A block kernel that hits `kv_len==0` would compute
    `0/0 = NaN` and `-inf + log(0) = NaN`. **Must** guard: initialize output to zeros / lse
    to `-inf` and skip or clamp when `deno==0`.
  - For valid workloads `kv_len >= q_len` per sequence, so every real query row attends to
    at least token `prefix_len+i >= 0` (≥1 token) — no fully-masked valid row, no NaN. But
    padding rows in a `BLOCK_M` tile (beyond `q_len`) still produce `0/0`; guard with the
    store mask, and use the SGLang `row_max_fixed = where(row_max==-inf, -1e20, row_max)`
    trick so an all-`-inf` KV tile does not poison the running max/rescale.
- **Tolerance unknown.** `definition.json` exposes no explicit rtol/atol; the evaluator uses
  its defaults for a bf16 attention output + fp32 lse. Be conservative: fp32 everywhere
  except the two `tl.dot` operands, exact causal boundary (`<=`, matching `j > pos` masks),
  and exact base-2 LSE.
- **Masked logits value.** Use `-inf` (or a large negative) consistently before the max, so
  `exp2(masked - m) = 0`. Ensure masked entries never dominate the max.

## 5. Triton design space

The task maps almost exactly onto two verified upstream Triton kernels found via the
KernelWiki skill:
- **SGLang `extend_attention` `_fwd_kernel` / `_fwd_kernel_unified`** (PR-22079, verbatim in
  the wiki) — prefill-with-KV-cache, with the *exact* `Lq==576 → BLOCK_DMODEL=512,
  BLOCK_DPE=64, BLOCK_DV=512` split this task needs, Hopper block-size heuristics
  (`Lq>256 → BLOCK_M=32, BLOCK_N=64`, `num_stages=1`), transposed-K load for QK, causal
  early-exit (`cur_block_m_end = min(q_len, (m+1)*BLOCK_M)`), and the unified causal mask.
- **vLLM/SGLang grouped MLA decode `_fwd_grouped_kernel_stage1`** (PR-34597) — folds
  `BLOCK_H` heads into the M dimension so shared KV is read once, plus `NUM_KV_SPLITS`
  flash-decoding. This is the template for the decode-like / short-Q workloads.

Reference performance context (KernelWiki `kernel-flashmla`): dense MLA on SM90 reaches
~660 TFLOPS / ~3000 GB/s in hand-written CUDA; Triton will not match that, but the goal here
is beating the pure-PyTorch python-loop reference, which is enormous (it loops over batch and
over every query token in Python) — so a single fused Triton kernel should already be a very
large speedup, and the search is about maximizing geomean across regimes.

### Axis A — parallelization / M-dimension layout
1. **Per-(batch, head, q-block)** (SGLang extend baseline): grid `(batch, 16, cdiv(q_len,
   BLOCK_M))`, each program handles `BLOCK_M` query rows of one head, loops all needed KV.
   Simplest, proven, correct. Downside: reads the latent KV 16× (once per head).
2. **Head-folded** grid `(batch, cdiv(q_len,BLOCK_M?), ...)` with the 16 heads placed in the
   matmul M dimension for a single query token (decode style): KV read once for all heads.
   Best for short-Q / decode-heavy workloads (#2,4,6,15,17,28,32).
3. **Combined (head × q-tile) M**: rows = `(q_tile × 16)` for a shared KV block. Maximizes
   both tensor-core M utilization (e.g. q_tile=4 → M=64, a full wgmma tile) and KV reuse.
   Most general but most complex index math (output row `(t,h)` → `output[q_base+t, h, :]`).
   Causal mask depends only on `t`, identical across the 16 heads of a tile.

### Axis B — KV loop
- Causal **early-exit**: only iterate KV up to `prefix_len + (block's max query pos)`.
- **Prefix vs triangle split** (SGLang two-loop) vs **unified single loop** with a `<=`
  mask. Unified is simpler and page_size=1 makes indexing trivial; keep it unless profiling
  shows the split wins.
- Optional **flash-decoding KV split** (`NUM_KV_SPLITS` + a tiny stage-2 reduction) for the
  short-Q-long-KV cases (esp. #17: 807 kv/seq, 22 seqs) to raise occupancy.

### Axis C — memory layout of the latent KV
- QK needs `Kc` as `[512, BLOCK_N]` (contract on 512); PV needs `Kc` as `[BLOCK_N, 512]`.
  SGLang loads K (transposed) and V (normal) separately — here that means reading the same
  `ckv_cache` **twice** (2×`BLOCK_N`×1KB SMEM/traffic). Alternative: load `Kc` once and use
  `tl.trans` for the QK matmul → halves latent-KV traffic and SMEM, letting us grow blocks
  or add a pipeline stage. Worth an explicit A/B candidate.

### Axis D — block sizes / occupancy / SMEM
- SMEM is tight: `Kc` tile at `[512,BLOCK_N]` bf16 = `BLOCK_N`×1KB; with q, kpe, and V that
  is ~150–170KB for `BLOCK_M=32, BLOCK_N=64` single-buffered — hence SGLang's `num_stages=1`
  for the 576 case on Hopper (228KB SMEM). De-duplicating Kc (Axis C) frees ~64KB.
- Heuristic block sizes by regime: small `BLOCK_M` (16/32) for tiny/decode, larger
  `BLOCK_N`, and possibly `@triton.autotune` keyed on `Lq`/`q_len` bucket. Tune `num_warps`
  (4–8) and `num_stages` (1–2).

### Axis E — launch/overhead reduction for tiny workloads
- Precompute `prefix_len`, per-seq q/kv lengths, and (if needed) a flattened work list on
  the host in torch, minimizing python and `.item()` syncs. Avoid per-sequence python loops.
- Consider a single grid over all (seq, head, block) using indptr lookups inside the kernel
  (no host loop), matching the SGLang launch.

### Candidate ladder (tentative, to be finalized in plan.md)
- `c001`: correctness-first single fused kernel, per-(batch,head,q-block), unified causal
  loop with `<=` mask + early-exit, `BLOCK_DMODEL=512/BLOCK_DPE=64/BLOCK_DV=512`, exp2 +
  base-2 LSE, fp32 accumulators, empty-seq guards. Establish a valid baseline & speedup.
- `c00x`: de-duplicate `Kc` load via `tl.trans` (Axis C).
- `c00x`: head-folded / combined (head×q-tile) M dimension (Axis A) for KV reuse.
- `c00x`: regime heuristics / autotune for block sizes (Axis D) and `num_warps`/`num_stages`.
- `c00x`: optional flash-decoding split for short-Q-long-KV (Axis B) if profiling justifies.
Each is a separate immutable candidate id.

## 6. Validation strategy

1. **Correctness gate = the evaluator.** Only `./scripts/evaluate_candidate.sh feedback cNNN`
   determines pass/fail across all 38 workloads; every workload must pass. Do not build an
   alternate correctness harness or run CUDA directly.
2. **Pre-eval self-checks** (design/mental, since we cannot run the evaluator's dataset):
   re-derive causal boundary against the reference (`j <= prefix_len + i`), the base-2 LSE
   identity, the V≡Kc mapping, and empty-sequence handling before writing each candidate.
3. **Confirm baseline validity first.** `c001` must pass all 38 before optimizing; if any
   workload fails (esp. empty-seq, tiny, or the 16384 case), fix correctness before touching
   performance. A perf change that breaks any workload is worse than a slower correct one.
4. **Attribute changes to ids.** Each source/config/launch change → new id; record parent,
   source hash, hypothesis, per-workload result, geomean, decision, cumulative eval count,
   and skill usage in `candidates.jsonl` (append-only).
5. **Profiling** only through `./scripts/ncu_profile.sh`, never overlapping an evaluation,
   used to decide between the optimization axes (e.g., is the tiny-case cost launch or
   memory; is the 16384 case tensor-core or memory bound; does Kc de-dup help).
6. **Convergence / stop.** Stop at budget or when geomean improvement genuinely plateaus,
   then write `SEARCH_COMPLETE`. Never run `final` without operator approval.

## 7. Skill usage
- **KernelWiki**: `kernel-flashmla` (MLA KV layout, V≡latent-KV, SM90 perf context),
  `lang-triton` → verbatim `pr-vllm-34597` grouped MLA decode kernel (head-folding +
  KV-splits) and `pr-sglang-22079` extend/prefill kernel (the `Lq==576`
  512+64 split, Hopper block heuristics, transposed-K load, causal early-exit, unified
  mask). These directly inform Axes A–D above.
- **ncu-report-skill**: to be used (via `./scripts/ncu_profile.sh`) during the search to
  diagnose per-regime bottlenecks before committing block-size/layout candidates.
