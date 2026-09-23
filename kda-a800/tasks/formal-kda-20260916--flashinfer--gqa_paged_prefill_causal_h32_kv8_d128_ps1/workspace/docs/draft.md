# Draft — `gqa_paged_prefill_causal_h32_kv8_d128_ps1`

Target: NVIDIA A800 (`sm_80`, ~2.0 TB/s HBM, no FP8/tcgen05, BF16 tensor cores).
Goal: a Triton implementation of `run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)`
that returns `(output, lse)` matching the reference within tolerance, maximizing geomean speedup
over the reference across the fixed feedback workloads (and, on approval, the 38‑workload final set).

This is the analysis‑only draft. No `plan.md`, no solution code yet.

---

## 1. Operation semantics (exact, from `task/definition.json` reference)

Batched **Grouped‑Query Attention prefill** over a **paged KV cache** with `page_size = 1` and a
causal mask, captured from Llama‑3.1‑8B incremental prefill.

Fixed constants: `num_qo_heads = 32`, `num_kv_heads = 8`, `head_dim = 128`, `page_size = 1`,
`gqa_ratio = 32/8 = 4`.

Inputs / outputs:
- `q`: `[total_q, 32, 128]` bf16.
- `k_cache`, `v_cache`: `[num_pages, 1, 8, 128]` bf16. Because `page_size == 1`, page id == KV token id.
- `qo_indptr`, `kv_indptr`: `[batch+1]` int32 (batch = `len_indptr - 1`).
- `kv_indices`: `[num_kv_indices]` int32 — page ids to gather per sequence.
- `sm_scale`: f32 scalar (default `1/sqrt(128) ≈ 0.0883883`).
- `output`: `[total_q, 32, 128]` bf16 (init `zeros`).
- `lse`: `[total_q, 32]` f32 (init `-inf`), **base‑2** log‑sum‑exp.

Per‑sequence loop (`b = 0..batch-1`):
- `q_start,q_end = qo_indptr[b],qo_indptr[b+1]`; `kv_start,kv_end = kv_indptr[b],kv_indptr[b+1]`.
- `num_q = q_end-q_start`, `num_kv = kv_end-kv_start`. If `num_q==0` or `num_kv==0` → sequence skipped
  (its rows stay `zeros` / `-inf`).
- `page_ids = kv_indices[kv_start:kv_end]`; gather `k = k_cache[page_ids,0,:,:]`, `v` likewise.
- `delta = num_kv - num_q`.
- For each local query `q_idx` (global row `q_start+q_idx`):
  - `max_kv_idx = min(q_idx + 1 + delta, num_kv)`. If `max_kv_idx <= 0` → query skipped
    (row stays `zeros`, `lse` stays `-inf`).
  - For each head `h` (kv head = `h // 4`), over keys `j in [0, max_kv_idx)`:
    - `logits_j = (q_h · k_{j,kv_head})` computed **in float32**, then `× sm_scale`.
    - `lse[row,h] = logsumexp_j(logits_scaled) / ln(2) = log2( Σ_j exp(logits_scaled_j) )`.
    - `attn = softmax(logits_scaled)`, `out = Σ_j attn_j · v_{j,kv_head}`, cast to bf16.

Key structural facts that drive the whole design:
1. **Causal alignment is end‑anchored.** Key `j` is valid for query `q_idx` iff `j <= q_idx + delta`
   (and `j < num_kv`). The last query attends to all `num_kv` keys; earlier queries attend to a
   shrinking prefix. Standard prefill causal mask, valid for `num_kv < num_q` too.
2. **A query with `q_idx < num_q - num_kv` produces no output** (`max_kv_idx <= 0`). The first active
   local query is `active_start = max(0, num_q - num_kv)`; number of active queries per sequence is
   `n_active = min(num_q, num_kv)` (0 if `num_kv==0`).
3. Reference upcasts `q,k,v` to **float32** before all matmuls; accumulation is f32; only the final
   `output` write is cast to bf16. `lse` stays f32.

---

## 2. Feedback workload analysis (the decisive observation)

Hand‑derived from `task/feedback_workloads.jsonl` (all share `sm_scale = 0.0883883`):

| WL | seqs | total_q | num_kv_indices | num_pages | q/seq | kv/seq | out size | active q (≈) |
|----|-----:|--------:|---------------:|----------:|------:|-------:|---------:|-------------:|
| W1 | 38   | 7,140   | 97             | 399,096   | ~188  | ~2.6   | ~58.5 MB | ≤ 97 |
| W2 | 3    | 123     | 12             | 135,698   | ~41   | ~4     | ~1.0 MB  | ≤ 12 |
| W3 | 28   | 13,515  | 28             | 2         | ~483  | ~1     | ~110.7 MB| ≤ 28 |
| W4 | 1    | 16,384  | 3              | 25,634    | 16384 | 3      | ~134.2 MB| 3    |
| W5 | 10   | 4,950   | 25             | 296,392   | ~495  | ~2.5   | ~40.6 MB | ≤ 25 |

(out size = `total_q·32·128·2` bytes; `lse` adds `total_q·32·4` bytes, ~1/64 of out.)

**This is a degenerate/near‑empty‑KV regime: `num_kv << num_q` in every workload.** Consequences:

- **The attention compute is negligible.** Total active `(query, seq)` pairs across all five WLs is
  `≈ Σ num_kv_indices = 97+12+28+3+25 = 165`, each × 32 heads, each attending to ≤ a few KV tokens.
  This is a few thousand tiny dot products of length 128 — microseconds of FLOPs.
- **The cost is materializing the (mostly‑zero) `output`.** `output` is `[total_q,32,128]` bf16 and is
  overwhelmingly zeros (only the last `min(num_q,num_kv)` rows per sequence are non‑zero). W3/W4 must
  still write ~110–134 MB. At ~2 TB/s write that is a **~55–70 µs bandwidth floor per large WL** —
  this dominates everything else.
- **KV traffic is tiny.** We must **never** touch full `k_cache`/`v_cache` (up to ~817 MB for W1);
  we gather only the `num_kv_indices` referenced pages (≤ 97). The reference already does this via
  `k_cache_flat[page_ids]`.

**Strategic headline:** this is essentially a **memset‑bound** problem with a trivial compute tail.
The reference is a pure‑Python quadruple loop in f32 (e.g. W3: 13,515 q × 32 heads ≈ 432k Python inner
iterations, each doing torch matmuls) — extremely slow. Any correct fused Triton kernel that (a) uses a
fast memset for the zero region and (b) does compute only on the active tail will be dramatically faster.
The engineering risk is **correctness**, not raw throughput.

---

## 3. Constraints (task + platform)

- **Triton‑only compute.** PyTorch permitted only for metadata / launch plumbing. No Torch/CPU/NumPy/
  CUDA‑extension/alternate computational fallback. A failing Triton kernel is invalid and must not be
  swapped for a Torch path.
- **No local correctness harness / profiler / raw CUDA / nvidia‑smi / direct evaluator.** Validation is
  *reasoning + the official feedback evaluator only* (`./scripts/evaluate_candidate.sh feedback cNNN`).
  Bash scratch execution is also disabled here. ⇒ Each candidate must be derived to be correct *before*
  spending an evaluation; c001 should be the conservative, exactly‑matching version.
- Immutable, sequential candidates `c001, c002, …`; one kernel version over all 5 feedback WLs = one
  evaluation. Budget: 100 evals; token soft 1.0M / hard 1.2M.
- `sm_80`: BF16 tensor cores available; f32 (`ieee`) and `tf32` `tl.dot` paths available; no FP8, no
  `tcgen05`/TMEM/WGMMA. `tl.exp2`, `tl.log2` available.
- Allocation note: creating `output = torch.zeros(...)` and `lse = torch.full(..., -inf)` reproduces the
  reference's *initialization* (not the attention math) and is standard plumbing for Triton attention
  kernels; the attention itself is fully in Triton. I treat this as permitted plumbing and flag it as a
  decision point (§8) with a kernel‑write fallback if the evaluator objects.

---

## 4. Numerical risks & how to neutralize them

1. **Base‑2 LSE.** Reference: `lse = logsumexp(sm_scale·qk) / ln2 = log2(Σ exp(sm_scale·qk))`.
   Flash/exp2 identity: fold `qk_scale = sm_scale · LOG2E` (`LOG2E = 1.4426950408889634`); with
   `s'_j = qk_j · qk_scale`, `M' = max_j s'_j`, `p_j = exp2(s'_j − M')`, `L' = Σ p_j`:
   `lse = M' + log2(L')`, `output = (Σ p_j v_j) / L'`. Verified on the single‑KV case: `L'=1`,
   `lse = M' = sm_scale·qk·LOG2E` = reference `sm_scale·qk / ln2`. ✅ Use `tl.exp2`/`tl.log2`.
2. **Do NOT write skipped rows.** Queries with `max_kv_idx <= 0` and whole `num_kv==0` sequences must
   keep `output=0`, `lse=-inf`. The kernel writes *only* active rows; the memset supplies the rest.
   Getting the active predicate exactly right (`active_start = max(0, num_q-num_kv)`) is essential.
3. **Per‑row causal prefix mask.** Within the active tail each query attends to prefix
   `j <= q_idx + delta` (`delta = num_kv - num_q`), `j < num_kv`. Masked logits → `-inf` before exp2
   (i.e. `p_j = 0`). Must reproduce the shrinking triangle, not a flat `num_kv` window.
4. **f32 accumulation to match reference.** Reference multiplies f32 operands. Two safe options:
   (a) load `q,k` as f32 and do `tl.dot(..., input_precision="ieee")` (true f32, no tensor core —
   fine given tiny sizes); (b) for the tiny‑KV regime, an explicit f32 reduction. **Avoid** default
   bf16×bf16 tensor‑core dot (bf16 mantissa in the multiply) as a first cut — over a length‑128 sum it
   can drift ~1e‑2 and risk the bf16 tolerance. Start with `ieee`; consider `tf32x3`/tensor cores only
   if a larger‑KV final workload makes compute matter (feedback shows it never does).
5. **`exp2` vs torch `exp`/`softmax` path.** Small ULP differences, well inside bf16 output tolerance;
   `lse` is f32 but only a log2 of a sum of ≤ a few terms — safe.
6. **Empty‑tile / boundary indexing.** Guard gathers with masks (`j < num_kv`), guard the query row
   range (`row < q_end`), and handle `num_kv==0` (skip). Duplicate page ids (W3 has only 2 pages for 28
   indices) are fine — plain gather.
7. **`lse` dtype/shape.** f32 `[total_q,32]`; `-inf` sentinel preserved for inactive rows.
8. **No global `q.to(f32)`.** Reference does it for convenience; we upcast per‑tile inside the kernel to
   avoid an extra `total_q·32·128·4`‑byte pass.

---

## 5. Triton design space

### 5.1 Fusion & memory strategy
- One fused kernel: gather K/V for the tile's sequence, QKᵀ (f32), online/exp2 softmax with causal
  prefix mask, PV, write `output` (bf16) and `lse` (f32). Single pass; KV is tiny so no multi‑block
  streaming is needed in practice, but the online‑softmax form generalizes to larger KV in the final set.
- `output`/`lse` allocated via `torch.zeros` / `torch.full(-inf)` (fast memset); kernel writes only
  active rows. This is the main perf lever — we skip *computing* the zero bulk entirely.

### 5.2 GQA packing
- For each `kv_head` (8), the 4 qo heads share the same K/V. Pack the `gqa_ratio=4` heads into the row
  (M) dimension à la FlashInfer, so a program handles `BLOCK_Q` queries × 4 heads = up to `4·BLOCK_Q`
  rows against one gathered K/V set. Amortizes the KV gather across 4 heads.

### 5.3 Grid / scheduling (tail‑only)
Because only `n_active = min(num_q,num_kv)` queries per sequence do work, launching a block per full
query range wastes almost all blocks (W4: 16,384 queries, 3 active). Options:
- **Option S (preferred): host‑built compact schedule.** Read `qo_indptr`/`kv_indptr` (already a host
  sync the reference itself does) to compute per‑seq `active_start`, `n_active`; build a small list of
  `(batch_id, q_tile_start)` tiles covering only `[active_start, num_q)`. Grid = `(num_active_tiles,
  num_kv_heads)`. `num_active_tiles` is tiny (Σ active ≤ ~100 ⇒ a handful of tiles), so launch overhead
  is negligible and there are *no* dead blocks. Schedule build = torch ops + one `.cpu()` sync (plumbing).
- **Option N (fallback, simpler): dense grid + in‑kernel skip.** Grid over all `ceil(num_q/BLOCK_Q)`
  tiles per seq × kv_heads; each program early‑exits if its tile is entirely below `active_start`. Robust
  and easy to get correct, but launches many dead blocks (W4: ~128 tiles × 8, mostly no‑ops). Cheap
  per‑block (a few int loads) but not ideal. Good candidate for c001; migrate to Option S once correct.
- **Batch lookup.** Under Option N, map program→batch either by launching a 3D grid `(batch, q_tile,
  kv_head)` with a conservative `max_q_tiles` and masking, or by binary‑searching `qo_indptr` in‑kernel.
  A `(batch, q_tile, kv_head)` grid with per‑seq tile masking avoids straddling sequences.

### 5.4 Block sizes / dtypes
- `BLOCK_D = 128` (= head_dim, whole vector in registers).
- `BLOCK_KV`: cover full `num_kv` in one block for the feedback regime (tiny); parameterize (e.g. 16/32/
  64) with online softmax for generality.
- `BLOCK_Q`: small (e.g. 16) since active tails are short; packed with `×4` heads.
- Accumulator f32; K/V/Q upcast to f32 in‑kernel; `output` stored bf16, `lse` stored f32.
- `num_warps`/`num_stages`: modest (e.g. 4 warps, 2–3 stages); autotune later — compute is not the
  bottleneck, so keep it simple first.

### 5.5 What *not* to do
- Don't read full `k_cache`/`v_cache`. Gather only referenced pages.
- Don't materialize `q_f32` globally.
- Don't launch a block per (query,head) — memset already covers the zero bulk.

---

## 6. Performance model

Per‑WL lower bound ≈ `output` write + `lse` write + tiny KV gather + tiny compute:
- W4 (worst): ~134 MB write ⇒ ~65–70 µs floor (memset‑dominated). W3: ~110 MB ⇒ ~55 µs.
- W1/W5: ~40–58 MB ⇒ ~20–30 µs. W2: ~1 MB ⇒ launch/overhead‑dominated (a few µs).
Reference is a Python‑loop f32 implementation (hundreds of thousands of interpreter iterations for
W3/W4/W5 plus `q.to(f32)` and `torch.zeros`), i.e. milliseconds–seconds. Expected speedups are large
(often ≫10×). The realistic ceiling is set by the unavoidable `output` memset, so beyond a correct fused
tail kernel the remaining levers are: (a) efficient memset (torch.zeros), (b) zero dead blocks
(Option S), (c) minimal host syncs, (d) not touching cold KV. Diminishing returns arrive quickly once
those are in place → convergence likely within a few candidates.

---

## 7. Validation strategy (no local harness)

Because no local correctness harness/profiler is permitted and each feedback run consumes an evaluation:
1. **Derive c001 to be exactly correct** (conservative): f32 (`ieee`) accumulation, explicit base‑2 LSE
   via `M' + log2(L')`, exact `active_start`/`max_kv_idx`/prefix mask, write only active rows, memset for
   the rest. Prefer the simplest robust scheduling (Option N dense grid + in‑kernel skip) to minimize
   logic that can be wrong.
2. **Confirm correctness on the 5 feedback WLs** via `./scripts/evaluate_candidate.sh feedback c001`
   before any perf tuning. Correctness on all five (esp. W4 single‑seq extreme tail, W3 duplicate pages /
   `num_kv==0` handling, W2 tiny) is the gate.
3. **Only then optimize**: c002+ swap to Option S compact schedule, GQA packing, block‑size/warp tuning,
   reduced host syncs. Each change = new candidate id; keep earlier records immutable in
   `candidates.jsonl`.
4. **Cross‑check reasoning** against reference edge cases enumerated in §4 before every submission
   (single KV; `max_kv_idx<=0`; `num_kv==0`; duplicate pages; last‑query full window). Log hypothesis +
   expected behavior per candidate.
5. Stop / `SEARCH_COMPLETE` when speedup converges near the memset floor. Never run `final` without
   operator approval.

---

## 8. Open questions / risks to resolve in the plan

- **Is `torch.zeros`/`torch.full` allocation accepted as plumbing?** Believed yes (mirrors reference
  init; attention math is fully Triton). Fallback: have the kernel additionally write zeros/`-inf` to
  inactive rows (same bandwidth, more launched blocks) if the evaluator flags the memset. Decide in plan.
- **Exact tolerance** is not in `definition.json`; assume bf16‑attention‑grade (roughly `atol~2e‑2`).
  f32/`ieee` accumulation should clear it comfortably; revisit only if a candidate fails.
- **Host sync cost** of reading `indptr` for Option S — negligible vs memset, and the reference already
  syncs; but batch to a single `.cpu()` transfer.
- **Final 38‑workload generalization:** feedback is uniformly `num_kv << num_q`. The final set *might*
  include larger‑KV cases; keep the online‑softmax/`BLOCK_KV` streaming path general so the kernel stays
  correct (and reasonably fast) if KV grows, without over‑fitting to the near‑empty regime.
- **`num_warps`/`num_stages`/block sizes**: defer autotuning; compute is not the bottleneck, so avoid
  spending evaluations chasing compute throughput that the memset floor hides.

---

### Summary
This op is a paged causal GQA prefill whose feedback workloads are in a degenerate `num_kv << num_q`
regime, making it **memset/bandwidth‑bound on the mostly‑zero `output`** with a negligible compute tail.
The winning approach is a single fused Triton kernel with exact base‑2 LSE and f32 accumulation that
**writes only the active query tail** while `torch.zeros`/`torch.full` supply the zero/`-inf` bulk, and a
**tail‑only schedule** so no work is spent on the inactive majority. Correctness (base‑2 LSE,
end‑anchored causal prefix, `-inf`/zero preservation, paged gather) is the primary risk and must be
nailed in c001 because validation is reasoning + the official feedback evaluator only.
