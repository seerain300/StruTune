# Draft — `gqa_ragged_prefill_causal_h32_kv8_d128`

Target: NVIDIA H100 (`sm_90`, Hopper). Primary implementation must be Triton.
Ranking metric: geometric-mean speedup over the reference across the full workload
set, subject to every selected workload passing correctness.

---

## 1. Operation semantics

The op is a **batched Grouped-Query-Attention prefill with ragged (variable-length)
sequences and a causal mask**, captured from Llama-3.1-8B during full prefill.

### 1.1 Shapes and dtypes (from `task/definition.json`)

Constants (fixed for this task):
- `num_qo_heads = 32`
- `num_kv_heads = 8`  →  `gqa_ratio = 32 / 8 = 4`
- `head_dim = 128`

Variable axes:
- `len_indptr = batch_size + 1`
- `total_q` = total query tokens across all sequences
- `total_kv` = total key/value tokens across all sequences

Tensors:
| name | shape | dtype |
|------|-------|-------|
| `q` | `[total_q, 32, 128]` | bf16 |
| `k` | `[total_kv, 8, 128]` | bf16 |
| `v` | `[total_kv, 8, 128]` | bf16 |
| `qo_indptr` | `[len_indptr]` | int32 |
| `kv_indptr` | `[len_indptr]` | int32 |
| `sm_scale` | scalar | fp32 (≈ 0.08838834764831843 = 1/√128) |
| **out** `output` | `[total_q, 32, 128]` | bf16 |
| **out** `lse` | `[total_q, 32]` | fp32 |

Layout is **token-major, head-second, dim-last** (a "ragged"/varlen packed layout,
*not* `[B, H, S, D]`). Rows of `q`/`output` are indexed by a global token id;
sequence boundaries come from `qo_indptr` / `kv_indptr`.

### 1.2 Reference math (per sequence `b`)

For sequence `b` with query rows `q_start:q_end` and kv rows `kv_start:kv_end`:
- `num_q  = q_end - q_start`, `num_kv = kv_end - kv_start`, `delta = num_kv - num_q`.
- If `q_start >= q_end` or `kv_start >= kv_end`: skip (output stays 0, lse stays −inf).
- K and V are expanded from 8 → 32 heads by `repeat_interleave(gqa_ratio=4, dim=1)`,
  i.e. query head `h` (0..31) uses kv head `h // 4`.
- `logits[q,h,k] = (Σ_d q[q,h,d]·k_exp[k,h,d]) · sm_scale`, **computed in fp32**
  (reference casts q,k,v to fp32 *before* the matmul).
- **Causal mask:** query row `i` (0-based within the sequence) may attend to kv
  columns `j` with `j < i + 1 + delta`. Invalid entries set to `−inf`.
- `lse[q,h] = logsumexp_k(logits[q,h,:]) / ln(2)`  (natural LSE converted to **base-2**).
- `attn = softmax_k(logits)`; `output[q,h,:] = Σ_k attn[q,h,k]·v_exp[k,h,:]`, cast to bf16.

### 1.3 Consequence of `total_q == total_kv` in every feedback workload

All 21 feedback workloads have `total_q == total_kv`. For a Llama full-prefill
capture this means **`q_len == kv_len` per sequence ⇒ `delta = 0`**, so the mask is
the ordinary lower-triangular causal mask `j <= i`. With `delta ≥ 0`, row 0 always
attends to at least `1+delta ≥ 1` keys, so **no query row is ever fully masked** →
no `−inf`-only rows, no `0/0` in softmax, no NaNs. This removes the nastiest
numerical edge case for the observed data. I will nonetheless implement the
general `delta` boundary (`j < i + 1 + delta`) so the kernel is correct if a
sequence ever had `kv_len > q_len`; I will *not* rely on empty-row NaN behavior
(see §3.4).

---

## 2. Workload distribution (drives the optimization priorities)

Parsed from `task/feedback_workloads.jsonl` (21 workloads):

| regime | workloads | `total_q` (=`total_kv`) | `len_indptr` (=B+1) |
|--------|-----------|--------------------------|---------------------|
| micro  | #2,8,14 | 1 | 2 (single seq) |
| tiny   | #1,7,13 | 7 | 2 |
| tiny   | #11,12 | 18 | 2 |
| tiny   | #3,9,10,15,16 | 35 | 2 |
| tiny   | #18,19 | 71 | 2 |
| small  | #4 | 81 | 3 (2 seq) |
| small  | #21 | 92 | 6 (5 seq) |
| small  | #17 | 982 | 16 (15 seq) |
| large  | #20 | 12571 | 11 (10 seq) |
| large  | #6 | 12845 | 37 (36 seq) |
| large  | #5 | 13557 | 26 (25 seq) |

**Key takeaways**

1. **~17 of 21 workloads are tiny** (`total_q ≤ 92`). Geomean weights each workload
   equally, so the tiny cases dominate the score. There the arithmetic is trivial and
   the win comes almost entirely from **launch/host-overhead reduction**, not GPU FLOP
   efficiency.
2. The reference is *very* slow on the tiny cases for reasons unrelated to compute:
   a Python `for b in range(B)` loop, repeated `int(indptr[i].item())` calls
   (**host↔device syncs**), full-tensor `q.to(fp32)` / `k.to(fp32)` / `v.to(fp32)`
   casts, `repeat_interleave` allocating an expanded 32-head K/V, and full
   materialization of `logits` + `softmax` before the second matmul. A single fused
   Triton kernel with **zero host syncs** should beat it by a large margin on the
   tiny cases just by eliminating Python/allocation overhead.
3. The **3 large cases (~12.5k–13.5k tokens)** are where genuine GPU efficiency
   (tiling, causal skip of the upper triangle, GQA K/V reuse, pipelining) matters, and
   where the reference’s O(S²·H) `logits` materialization is genuinely expensive.
   These set the ceiling for a fused flash-attention kernel.
4. Sequences within a batch vary in length (ragged) → **load imbalance** is a real
   concern for the large multi-seq workloads (#5,#6 have 25/36 sequences).

Optimization priority: (a) a correct, low-overhead single fused kernel that wins
everywhere; (b) causal upper-triangle skipping + GQA reuse for the large cases;
(c) ragged load-balancing so short sequences in a mixed batch don’t waste blocks.

---

## 3. Constraints and numerical risks

### 3.1 Precision of the QKᵀ and PV matmuls
The reference performs the matmuls in **true fp32** (it upcasts q,k,v first). A Triton
`tl.dot` on **bf16** operands with fp32 accumulation multiplies in bf16 (≈8-bit
mantissa) — less precise than fp32. On Hopper, `tl.dot` on fp32 operands would use
**TF32** (≈10-bit mantissa), still not true fp32. Risks:
- QKᵀ in bf16 loses mantissa bits per product; with `head_dim=128` the dot has 128
  terms, so rounding accumulates but fp32 accumulation bounds it.
- Attention is generally tolerant (softmax normalizes; V-weighted average smooths).
Mitigation ladder if correctness fails:
1. Keep bf16 inputs + fp32 accumulator (default `tl.dot` behavior) — try first.
2. Force higher precision on the QKᵀ dot via `tl.dot(..., input_precision="ieee")`
   or `"tf32x3"` (3-pass) if the base bf16 path is out of tolerance, accepting some
   perf loss on the large cases only.
3. Softmax and rescale math strictly in fp32 (never bf16) — mandatory regardless.

The exact evaluator tolerance is not stated in `definition.json`; treat it as a
standard bf16-attention tolerance and confirm empirically via the first candidate’s
feedback run. **Do not** tune away correctness margin for speed.

### 3.2 Base-2 LSE — must match exactly (in tolerance)
`lse` is the **base-2** log-sum-exp: `lse = logsumexp(logits)/ln(2)`. In an online
(flash) softmax I track a running max `m` and running sum `l` of `exp(logit − m)`.
Two equivalent formulations:
- Natural: accumulate with `exp`, then `lse2 = (m + ln(l)) / ln(2)`.
- Base-2: fold `log2(e)` into the scale, use `exp2`, track `m2`,`l2`, then
  `lse2 = m2 + log2(l2)` directly. This is the flash-attention-native form and avoids
  a final `ln`. Either is fine numerically in fp32; I’ll prefer the base-2 form
  (matches `lse` output naturally) and verify the constant `log2(e)` folding against
  the reference on the feedback run.
- `logits` include `sm_scale` **before** the max/exp; fold `sm_scale · log2(e)` into
  one scale applied to the raw QKᵀ so `exp2` sees the scaled score.

### 3.3 Softmax numerical stability
Standard flash online-softmax with running max subtraction. Guard the running max
init at `−inf` and the running denom at 0; use `exp2`/`exp` on `(score − m)` which is
`≤ 0`, so no overflow. For a row where a KV tile is entirely masked, `tl.where(mask,
score, −inf)` yields `−inf`; the tile’s `max` becomes `−inf` → handle via a
`row_max_fixed = where(row_max==−inf, −large, row_max)` trick (as in the SGLang
reference kernel) so a fully-masked *tile* (not row) doesn’t poison the running max.

### 3.4 Empty / fully-masked rows
As established in §1.3, with `delta ≥ 0` (all feedback data has `delta = 0`) every
query row sees `≥ 1` valid key, so no fully-masked rows occur and no NaNs arise. I
will *not* emit NaN outputs; for defensive robustness I’ll guard the final divide
(`acc / max(deno, tiny)`), which is a no-op on the actual data but avoids a `0/0`
should a degenerate row ever appear. The reference leaves `output=0`, `lse=−inf` for
*skipped* sequences (`q_start>=q_end`); with ragged indptr this can happen only for a
zero-length sequence — handled naturally because that sequence contributes no query
rows / no program instances.

### 3.5 Output dtype and rounding
`output` must be **bf16**; compute the weighted V-sum in fp32 then cast once at store
(matches reference `output_batch.to(bf16)`). `lse` stays fp32. Storing bf16 from a
correctly-normalized fp32 accumulator is the last rounding step; keep it single.

### 3.6 GQA head mapping
Query head `h` → kv head `h // 4` (from `repeat_interleave(4)`). The kernel must not
actually materialize an expanded K/V; instead index kv head `cur_kv_head = cur_head //
kv_group_num` when loading K/V (as in the SGLang extend kernel). This both matches the
reference and is the main memory-traffic win over the reference’s `repeat_interleave`.

### 3.7 Ragged indexing / int32
`qo_indptr`, `kv_indptr` are int32 and live on device. Reading sequence bounds inside
the kernel (`tl.load(qo_indptr + pid)`) avoids all host syncs. Any host-side quantity
needed for the launch grid (e.g. `max_seqlen` or a precomputed block schedule) must be
computed with a **single** cheap device→host transfer, not a per-batch `.item()` loop.

### 3.8 Constants baked as `constexpr`
`head_dim=128`, `num_qo_heads=32`, `num_kv_heads=8`, `gqa_ratio=4` are compile-time
constants — pass as `tl.constexpr` so the compiler can fully unroll the D dimension
(BLOCK_D=128) and specialize the kernel. This is safe because the task fixes them.

---

## 4. Triton design space

### 4.1 Baseline kernel structure (flash-attention forward, ragged, causal)
A single `@triton.jit` fused kernel per query tile:
- Program over `(seq, q_head, m_block)` (see §4.3 for grid variants).
- Compute `cur_kv_head = q_head // 4`.
- Load a `[BLOCK_M, 128]` Q tile for `(seq, q_head)` (mask rows `≥ seqlen`).
- Loop over KV tiles `[BLOCK_N, 128]` for this sequence:
  - `qk = tl.dot(q, kᵀ) * (sm_scale)` in fp32.
  - Apply causal mask `j <= i + delta` (or general `< i+1+delta`) + row/col bounds.
  - Online softmax: running `m`, `l`, rescale `acc`.
  - `acc += tl.dot(p.to(bf16), v)`.
- Epilogue: `out = acc / l` → cast bf16 → store; `lse = m2 + log2(l2)` → store fp32.
This directly mirrors the verified SGLang `extend_attention._fwd_kernel` structure
(KernelWiki `pr-sglang-22079`, `lang-triton`), specialized to: no prefix/paged KV
(pure ragged prefill, `kv == q` region), no custom mask, no sliding window, no sink,
`Lq=Lv=128`, `BLOCK_DPE=0`.

### 4.2 Causal upper-triangle skipping
For the causal region, tile `m_block` only needs KV tiles up to
`min(seqlen, (m_block+1)·BLOCK_M)` (+`delta`), i.e. stop the KV loop early. This
roughly halves QKᵀ/PV work on the large cases (#5,#6,#20) vs a full rectangular loop.
Cheap and high-value.

### 4.3 Grid / scheduling strategies (the main ragged design axis)
- **V1 — dense 3D grid** `(num_seq, num_qo_heads, cdiv(max_seqlen, BLOCK_M))`.
  Simple; wastes programs when a batch mixes one long and many short sequences
  (blocks past a short seq’s length early-exit). Needs one host sync for `max_seqlen`.
  Good enough for single-seq tiny cases (the majority) and a fine first candidate.
- **V2 — flattened (seq, m_block) schedule.** Precompute on host (torch, allowed for
  plumbing) an array mapping a linear tile id → `(seq, m_block)`, so no wasted blocks
  for ragged batches; grid = `(num_tiles, num_qo_heads)` or fold head in. Removes the
  load-imbalance waste on #5/#6. One device→host transfer of indptr to build it.
- **V3 — persistent / grouped over heads.** Since `gqa_ratio=4`, a program could load
  a K/V tile once and compute all 4 query heads that share it, cutting K/V global
  traffic 4×. On the large, memory-heavy cases this is attractive; adds register
  pressure (4× the P/acc state) — must watch occupancy on `sm_90`.
Plan the search to start at V1 (correctness + big overhead win) and escalate to V2/V3
only where profiling shows load imbalance / KV-bandwidth as the bottleneck.

### 4.4 Tile sizes and launch params (Hopper `sm_90`)
From the SGLang reference, Hopper with `Lq ≤ 256` uses `BLOCK_M=128, BLOCK_N=64,
num_warps=8`. With `head_dim=128` a `[128,128]` Q tile + `[64,128]` K/V tiles fit
Hopper SMEM comfortably. Candidate sweep axes:
- `BLOCK_M ∈ {64, 128}`, `BLOCK_N ∈ {32, 64, 128}`.
- `num_warps ∈ {4, 8}`, `num_stages ∈ {1, 2, 3}` (multi-stage pipelining overlaps TMA
  loads with `wgmma`; KernelWiki `technique-pipeline-stages` shows large gains from
  3-stage overlap on Hopper — but only helps the large cases; tiny cases are launch-
  bound and prefer minimal stages/warps to cut overhead).
- For the tiny cases (`total_q ≤ 71`, single tile), large BLOCK_M just adds masking;
  a smaller/cheaper configuration may reduce per-launch cost. Consider a size-adaptive
  launch (choose block/warps/stages from `total_q`) — still one immutable kernel with
  host-side config selection (a launch change ⇒ new candidate id per the rules).
- `triton.autotune` is possible but risks recomp*/*measurement noise inside the eval;
  prefer a small hand-picked config table keyed on a size bucket for determinism.

### 4.5 What NOT to do
- No `repeat_interleave` / no materialized expanded K/V (defeats the memory win).
- No full `logits` materialization (defeats flash-attention).
- No Torch/CPU/NumPy computational fallback (task rule: Triton must do the math).
- No autotune that changes timing nondeterministically across the eval’s 10 iters.
- Blackwell-specific paths (tcgen05/TMEM/2-SM, FA-4 ping-pong, software-exp) from
  KernelWiki are **not applicable** — target is Hopper `sm_90`; `tl.dot` lowers to
  `wgmma` here. FA-4’s software-`exp2` motivation (SFU scarcity) is a Blackwell issue;
  on Hopper the hardware `exp2` path is fine.

---

## 5. Correctness-first construction and validation strategy

### 5.1 Reference behaviors the kernel must reproduce
1. Per-sequence causal boundary `j < i + 1 + delta` (delta=0 ⇒ `j <= i`).
2. GQA head map `kv_head = q_head // 4`.
3. `output` bf16 = fp32 accumulator cast once at the end.
4. `lse` fp32 = base-2 logsumexp of the *scaled, masked* logits.
5. `sm_scale` applied before softmax; two different `sm_scale` values appear in the
   feedback set (0.0883883461356163 and 0.08838834764831843 — both ≈ 1/√128 but not
   bit-identical). Read `sm_scale` as a runtime scalar; never hardcode it.
6. Skipped/zero-length sequences leave output=0, lse=−inf (naturally handled).

### 5.2 Validation via the trusted evaluator only
- Correctness + timing come exclusively from
  `./scripts/evaluate_candidate.sh feedback <cNNN>`, which runs the **full** feedback
  set (warmup 2 / 10 iters) and counts as **one** candidate evaluation. Budget = 100
  evaluations; token soft limit 9M.
- I may **not** run CUDA/torch/nvidia-smi or any alternate correctness harness
  directly. All correctness reasoning is static + confirmed by the evaluator’s
  per-workload pass/fail.
- **Profiling** (only if needed to diagnose the large cases) goes strictly through
  `./scripts/ncu_profile.sh` per the `ncu-report-skill` workflow, and **never**
  concurrently with an evaluation (a foreign process on the locked GPU during timing
  → controller discards the measurement, return code 3, and burns one eval). Serialize:
  finish eval, then profile, then next eval.

### 5.3 Candidate sequencing (records appended to `candidates.jsonl`)
- `c001`: minimal correct fused flash kernel, V1 dense grid, `BLOCK_M=128,
  BLOCK_N=64, num_warps=8, num_stages=2`, bf16 dot + fp32 accum, base-2 LSE, causal
  skip. Goal: establish correctness across all 21 + baseline speedup. If any workload
  fails correctness, first suspect: LSE base/const, mask off-by-one, GQA mapping,
  bf16-dot tolerance (escalate precision per §3.1).
- Subsequent candidates change **one** axis at a time (tile size, stages/warps, grid
  V1→V2, GQA-reuse V3, size-bucketed launch config, precision knob), each with a new
  immutable id, hypothesis, and full-set result recorded (parent, source hash,
  per-workload result, geomean, decision, cumulative eval count, skill usage).
- Stop when geomean improvement converges or budget nears; then write
  `SEARCH_COMPLETE`. Never run `final` without operator approval.

### 5.4 Metrics to record per candidate
Per-workload correctness (pass/fail) and speedup, plus the geometric mean; note which
regime (tiny vs large) each change helps, since the geomean is tiny-case dominated but
the large cases are where GPU-efficiency changes show up. Watch for a change that helps
the large cases but regresses the launch-bound tiny cases (net geomean loss).

---

## 6. Open questions to resolve empirically (via feedback runs / profiling)
1. Is the bf16 `tl.dot` path within tolerance on all 21 workloads, or is an
   `input_precision` bump needed (and only on large cases)?
2. Exact evaluator tolerance and whether `lse` is checked as tightly as `output`.
3. For the large ragged cases (#5,#6), does V1’s dense grid waste enough on
   load-imbalance to justify V2’s flattened schedule? (profile SM utilization / tail).
4. Does GQA K/V reuse (V3) pay off on Hopper given the added register pressure, or is
   L2 reuse across the 4 same-kv-head programs already sufficient?
5. Optimal `num_stages` per regime — tiny (launch-bound, minimal) vs large
   (bandwidth/compute, 2–3 stage pipeline).

---

## 7. KernelWiki sources consulted
- `wiki/languages/triton-blackwell.md` (`lang-triton`) — Triton `tl.dot` matmul on
  Hopper/Blackwell; confirms Triton is appropriate for varlen/prefill attention and
  that on `sm_90` `tl.dot` lowers to `wgmma` (fp32-accum).
- `artifacts/prs/sglang/PR-22079/.../extend_attention.py` (`pr-sglang-22079`) —
  verbatim upstream ragged/extend prefill Triton attention kernel: online-softmax
  structure, causal two-stage loop, GQA `kv_head = head // kv_group_num`, Hopper block
  sizing (`BLOCK_M=128, BLOCK_N=64, num_warps=8`, `num_stages=1`), and the
  `row_max_fixed` masked-tile guard. This is the closest verified template.
- `wiki/kernels/flash-attention-4.md` (`kernel-flash-attention-4`) — flash-attention
  online-softmax + LSE accounting; its Blackwell-specific techniques (tcgen05/TMEM,
  ping-pong, software-exp2, 2-CTA backward) are **out of scope** for Hopper and noted
  as such.
- `wiki/techniques/pipeline-stages.md` (`technique-pipeline-stages`) — multi-stage
  TMA/MMA overlap motivating the `num_stages` sweep for the large cases.
- `queries/by-problem.md` — load-imbalance / tail-effect patterns motivating the
  ragged scheduling variants (V2/V3) for the multi-sequence large workloads.
