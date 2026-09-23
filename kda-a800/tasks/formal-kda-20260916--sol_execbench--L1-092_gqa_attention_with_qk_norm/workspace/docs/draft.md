# Draft — L1/092 GQA Attention with QK Norm (GLM-4.5-Air)

Target GPU: **NVIDIA A800, `sm_80` (Ampere)**. Primary implementation must be **Triton**;
PyTorch permitted only for tensor metadata / launch plumbing (no Torch/CPU/NumPy/CUDA-extension
compute fallback). Entry point: `solution/solution.py:run(...)` with the exact signature from
`task/definition.json`.

---

## 1. Operation analysis

### 1.1 Fixed configuration (from `definition.json`)
- `hidden_size = 4096`
- `num_attention_heads = 96`, `head_dim = 128` → `q_out_features = 96*128 = 12288`
- `num_key_value_heads = 8` → `kv_out_features = 8*128 = 1024`
- `num_key_value_groups = 96 / 8 = 12` (each KV head is shared by 12 query heads)
- `head_dim/2 = 64` (RoPE rotate-half split point)
- `scaling = head_dim**-0.5 = 128**-0.5 ≈ 0.0883883476`
- Variable axes: `batch_size (B)`, `seq_len (S)`. All tensors `bfloat16` except `rms_norm_eps` (fp32 scalar).

### 1.2 Reference dataflow (must be numerically reproduced)
Given `hidden_states [B,S,4096]`:

1. **Projections** (`F.linear`, i.e. `x @ W^T + b`):
   - `q = hidden @ q_proj_weight^T + q_proj_bias` → `[B,S,12288]`
   - `k = hidden @ k_proj_weight^T + k_proj_bias` → `[B,S,1024]`
   - `v = hidden @ v_proj_weight^T + v_proj_bias` → `[B,S,1024]`
2. **Reshape to heads**: q→`[B,S,96,128]`, k,v→`[B,S,8,128]`.
3. **QK RMSNorm** over the last axis (`head_dim=128`), computed in **fp32**:
   `x_f32 = x.float(); var = mean(x_f32^2, -1, keepdim); x = x_f32 * rsqrt(var + eps);
    out = (weight * x).to(bf16)`. Applied to q (with `q_norm_weight`) and k (with `k_norm_weight`).
   **Result is cast back to bf16 *before* RoPE.**
4. **Transpose** to `[B, H, S, D]` (q: H=96, k/v: H=8).
5. **RoPE** (`cos`,`sin` are `[B,S,128]`, `unsqueeze(1)`→`[B,1,S,128]`):
   `rotate_half(x) = cat(-x[...,64:], x[...,:64])`;
   `x = x*cos + rotate_half(x)*sin`. Applied to q and k in **bf16** (inputs are already bf16).
   NOTE: `cos`/`sin` are arbitrary random bf16 tensors of full width 128 in the test harness — they are
   **not** guaranteed to be true cos/sin nor half-duplicated; apply the formula literally element-wise.
6. **GQA expand**: k,v repeated 8→96 heads (12× each).
7. **Attention**: `scores = (q @ k^T) * scaling` → `[B,96,S,S]`;
   add causal mask (`triu(diagonal=1) = -inf`, i.e. key index `> query index` masked);
   `p = softmax(scores, dim=-1, dtype=fp32).to(bf16)`; `attn = p @ v` → `[B,96,S,128]`.
8. **Merge heads**: transpose→`[B,S,96,128]`→reshape `[B,S,12288]`.
9. **Output projection** (no bias): `out = attn @ o_proj_weight^T` → `[B,S,4096]`.

### 1.3 Feedback workloads (five fixed; one eval covers all five)
| uuid short | B | S | M=B·S | atol | rtol |
|---|---|---|---|---|---|
| ccea3f00 | 1 | 128 | 128 | 0.0034 | 0.05 |
| 1e96cc88 | 4 | 293 | 1172 | 0.0037 | 0.05 |
| d0ce8c45 | 1 | 1024 | 1024 | 0.0030 | 0.05 |
| c92bb0a5 | 1 | 512 | 512 | 0.0025 | 0.05 |
| 65a1864d | 8 | 373 | 2984 | 0.0038 | 0.05 |

Observations:
- `S ∈ {128, 293, 373, 512, 1024}` — **two are non-power-of-two (293, 373)**, so every kernel
  that tiles over `S` (projection M-dim, attention seq-dim) must mask boundary tiles.
- `M = B·S ∈ {128 … 2984}` — GEMM M-dim is modest; the projection GEMMs are "tall-skinny→wide"
  (`N`=12288 for Q, `K`=4096) and the output GEMM is `N`=4096, `K`=12288.
- Small S means the `O(S²)` attention is cheap **in FLOPs** but the reference pays a large
  **memory** cost (materialized `[B,96,S,S]` scores + fp32 softmax buffer + 12× KV expansion).

### 1.4 Roofline / where time goes (rough FLOP accounting, per workload, ×2 for MAC)
For the largest case B=8,S=373 (M=2984):
- Q proj: `M·12288·4096·2 ≈ 300 GFLOP`; K proj + V proj: `M·1024·4096·2·2 ≈ 50 GFLOP`.
- Attention (QK^T + PV): `B·96·S²·128·2·2 ≈ 8·96·373²·128·4 ≈ 55 GFLOP`.
- Output proj: `M·4096·12288·2 ≈ 300 GFLOP`.

⇒ **GEMMs dominate FLOPs** (~650 GFLOP vs ~55 GFLOP attention). The reference runs GEMMs on
cuBLAS (hard to beat) but runs attention as a **naive materialized path with a 12× KV blow-up**.
Therefore the most reliable speedup source is **fusing/streaming attention (flash-style, no S²
materialization, no KV expansion)**, while the projection/output GEMMs must be re-implemented in
Triton *at least competitively* with cuBLAS so they don't erase the attention win.

---

## 2. Constraints (from CLAUDE.md / TASK.md)

- **Triton-only compute.** No `torch.matmul`/`F.linear`/SDPA in the compute path — those count as a
  Torch computational fallback → invalid. The three projections, attention, and output projection
  must all be Triton kernels. PyTorch may only allocate/reshape/stride and launch.
- **A failed Triton kernel is invalid** — no silent Torch fallback allowed.
- **Immutability**: candidates `c001, c002, …` are append-only; new source ⇒ new ID.
- **Evaluation only** via `./scripts/evaluate_candidate.sh feedback cNNN` (5 workloads = 1 eval).
- **No direct CUDA/profiler/nvidia-smi/evaluator/alternate correctness harness.** This means I
  **cannot run the kernel locally** to sanity-check numerics; correctness feedback comes *only*
  from the sanctioned feedback evaluation. ⇒ Design conservatively; spend evals deliberately.
- Budget: 100 candidate evals; token soft/hard 1.0M/1.2M. Ranking = geomean speedup; **every
  selected workload must pass correctness** (a fast-but-wrong candidate is worthless).
- Final 16-workload eval is operator-approved only; never auto-run `final`.

### 2.1 Skill applicability (documented, per isolation rule)
- `KernelWiki`: scoped to Blackwell SM100 / Hopper SM90 (tcgen05, TMEM, TMA, wgmma, 2-SM, NVFP4…).
  Target is **Ampere sm_80** — none of those primitives exist. **Not applicable**; not invoked.
- `ncu-report-skill`: profiles B200/sm_100 and requires running Nsight Compute, which is prohibited
  here and targets the wrong arch. **Not applicable**; not invoked.
- Conclusion: rely on first-principles Ampere Triton design + feedback-eval evidence.

---

## 3. Numerical risk analysis

### 3.1 Tolerance model
Assume standard `torch.allclose`-style test: `|a-b| ≤ atol + rtol·|b|`, with `rtol = 0.05` and
`atol ∈ [0.0025, 0.0038]`. Reasoning about output magnitude:
- After RMSNorm each Q/K entry is `O(1)` (unit RMS × weight). RoPE keeps it `O(1)`.
- `scores` after `·scaling` are `O(1)`; softmax weights sum to 1.
- `V` entries inherit projection magnitude (`std ≈ sqrt(4096)` if inputs ~N(0,1); smaller if inputs
  are pre-scaled). `attn = p@v` is a convex combination of V rows ⇒ same order as V.
- `out = attn @ Wo^T` sums 12288 terms ⇒ potentially **large magnitude**.

⇒ For large-magnitude outputs the **relative** term `rtol·|b| = 0.05·|b|` dominates and is very
generous versus bf16 rounding (~0.4% relative). For near-zero outputs, `atol ≈ 0.003` governs.
Either way, a **fp32-accumulating** implementation should pass comfortably. The tolerance is loose
enough that flash-style attention (fp32 online softmax) will not be a correctness problem.

### 3.2 Specific precision-matching risks (ordered by importance)
1. **fp32 accumulation everywhere.** All GEMMs and the P·V accumulation must accumulate in fp32
   (bf16 accumulation would drift, especially the K=12288 output GEMM). Triton `tl.dot` with fp32
   accumulator matches cuBLAS behavior.
2. **RMSNorm in fp32.** Compute `mean(x²)` and `rsqrt` in fp32 (reference upcasts). Casting the
   *result* back to bf16 before RoPE is the reference behavior; storing normalized Q/K as bf16 is
   naturally what the attention kernel consumes, so this matches by construction.
3. **RoPE precision.** Reference does RoPE in **bf16**. Doing RoPE in fp32 then rounding to bf16 is
   *more* accurate and differs only by sub-ULP; expected to pass. Low risk. Keep note in case a
   workload is marginal — can force bf16-intermediate RoPE to match exactly.
4. **Softmax path divergence.** Reference materializes `p` in fp32 then **rounds p to bf16** before
   `p@v`. Flash attention keeps `p` in fp32 through the accumulation ⇒ slightly *more* accurate.
   The difference is bounded by per-weight bf16 rounding over the reduction; within tolerance. This
   is the main intentional numeric deviation and is safe given `rtol=0.05`.
5. **Scaling placement.** Apply `scaling` to scores (or fold into Q). `q·k` sum over 128 in fp32,
   multiply by `scaling` in fp32 — matches reference `matmul(...)·scaling` up to fp32 assoc. Safe.
6. **Causal mask edge.** `diagonal=1` ⇒ key `j > i` masked; the diagonal (`j=i`) is **kept**, so no
   row is fully masked (every query attends to at least itself) ⇒ no NaN from all `-inf` rows.
   Flash kernel must include `j ≤ i` (not `j < i`).
7. **RoPE literal semantics.** Because `cos`/`sin` are random 128-wide tensors (not real angles),
   do **not** assume `cos[:64]==cos[64:]`; index the full 128 width. `rotate_half` uses the *pre-*
   scaled x for the rotated term (standard: `out = x*cos + rotate_half(x)*sin`).
8. **Non-pow2 S masking.** Boundary tiles in both projection (M) and attention (S) must mask OOB
   rows/keys with `-inf` (scores) / `0` (loads) to avoid contaminating softmax or accumulation.
9. **bf16 weight load, fp32 math.** Load bf16 operands, cast to fp32 (or let `tl.dot` handle) — same
   as cuBLAS bf16-in/fp32-accum.

### 3.3 Failure modes to guard against
- NaN/Inf from unmasked OOB keys in the last attention tile.
- Wrong GQA head mapping (`kv_head = q_head // 12`) → silently wrong but may still be "close" on
  random data; must get the integer mapping exactly right.
- RoPE half-swap sign error (`-x[64:]` for the low half, `+x[:64]` for the high half).
- Reshape/transpose stride mistakes when writing `[B,H,S,D]` vs `[B,S,H,D]`.

---

## 4. Triton design space

### 4.1 Kernel decomposition options
- **A. Fully separate (baseline-first):** four stages —
  (1) three projection GEMMs (+bias epilogue), (2) a fused RMSNorm+RoPE elementwise kernel writing
  Q/K directly into `[B,H,S,D]` layout, (3) flash-attention (GQA, causal) writing `[B,S,H·D]`,
  (4) output GEMM. Simplest to reason about ⇒ best first correctness candidate.
- **B. Norm+RoPE fused into flash-attention prologue:** attention kernel loads raw projected Q/K
  tiles and applies norm+RoPE on the fly. Saves one Q/K memory round-trip. More complex (K tile is
  re-loaded across M-blocks ⇒ recomputes norm/RoPE); usually still a net win because K/V are small.
- **C. Fused QKV projection:** one GEMM with vertically-concatenated weights `[14336,4096]`. Needs a
  one-time weight concat (copy) — amortized only if weights were reused across calls (they are not,
  per-call inputs) ⇒ likely not worth the concat cost. Keep K and V possibly fused (`[2048,4096]`)
  since they share M-tiling; Q separate.
- **D. Fuse output-projection reshape:** attention writes `attn` already in `[M,12288]` row-major so
  the output GEMM reads it directly (no extra transpose kernel).

**Plan:** start with **A** for a correct, evaluable baseline; then layer B/C/D and tiling as
separate candidates guided by feedback speedups.

### 4.2 Projection GEMM design (`x[M,4096] @ W[N,4096]^T + bias`)
- Standard tiled Triton matmul over `K=4096`; `W` is `[N,K]` so we compute `x·W^T` by loading W with
  a transposed access pattern (K contiguous → good for `tl.dot(x_tile, w_tile.T)` or load W as
  `[K,N]` view via strides). Bias added in fp32 epilogue, cast to bf16.
- Autotune space: `BLOCK_M ∈ {32,64,128}`, `BLOCK_N ∈ {64,128,256}`, `BLOCK_K ∈ {32,64}`,
  `num_warps ∈ {4,8}`, `num_stages ∈ {2,3,4}`. Ampere sm_80 ⇒ software-pipelined `cp.async`
  (Triton handles via `num_stages`). Q proj (N=12288) benefits from larger BLOCK_N; K/V (N=1024)
  from smaller. Consider separate autotune per GEMM shape.
- M is small (128–2984) ⇒ grid can be M-light; ensure enough N-tiles × programs to fill the A800's
  SMs. Possibly split-K only if M very small (M=128 case) — evaluate later.

### 4.3 RMSNorm + RoPE kernel
- One program per (b, s, head) or per (b, s) processing a block of heads; load the 128-wide head
  vector, compute `var=mean(x²)` fp32 reduction, `x*rsqrt(var+eps)*weight`, cast bf16.
- Apply RoPE with `cos/sin[b,s,:]` (broadcast across heads): gather `x_rot` via the ±64 half-swap.
- Write directly to Q/K buffers laid out `[B,H,S,D]` (attention-friendly, contiguous in D then S).
- V needs no norm/RoPE; either pass through a copy into `[B,H,S,D]` or have attention read V from the
  projected `[B,S,8,128]` layout directly (preferred: avoid a copy — attention indexes kv head).

### 4.4 Flash-attention kernel (GQA, causal)
- Grid: `(num_M_blocks, B·num_heads)` or `(B, H, M_blocks)`. Program computes one `[BLOCK_M,128]`
  output tile for query head `h`, batch `b`.
- GQA: `kv_head = h // 12`; K/V loaded from that KV head only (no 12× expansion, no extra memory).
- Inner loop over key blocks `j`; **causal early-exit**: only iterate `key_block ≤ query_block`, and
  apply triangular mask (`j ≤ i`) on the diagonal block. Online softmax (running max `m`, running
  sum `l`, rescale accumulator) in fp32. Fold `scaling` into the score before max.
- `BLOCK_M ∈ {64,128}`, `BLOCK_N ∈ {32,64,128}`, `head_dim=128` fits registers, `num_warps ∈ {4,8}`,
  `num_stages ∈ {2,3,4}`. Boundary masking for non-pow2 S on both M and N.
- **KV-load reuse optimization (later):** since 12 query heads share one KV head, a program that
  handles a *block of query heads* from the same group amortizes K/V loads (batch the Q of several
  group-mates). Strong candidate for A800 memory savings; adds register pressure.
- Output written to `[B,S,H,128]`→`[M,12288]` row-major for the output GEMM.

### 4.5 Output-projection GEMM (`attn[M,12288] @ Wo[4096,12288]^T`)
- Same tiled-matmul template as projections but `K=12288` (large reduction ⇒ fp32 accum essential),
  `N=4096`, no bias. Autotune separately; large K favors `BLOCK_K=64`, `num_stages≥3`.

### 4.6 Layout / plumbing decisions
- Keep everything bf16 in DRAM; fp32 only in registers/accumulators.
- Precompute strides in Python; pass as kernel args. Avoid `.contiguous()` copies unless a layout is
  strictly required (attention output already row-major `[M,12288]`).
- One preallocated output `[B,S,4096]` bf16.

---

## 5. Validation strategy

Because local execution / alternate correctness harnesses are prohibited, validation is
**feedback-eval-driven** with heavy up-front paper reasoning:

1. **Static correctness review before each candidate:** re-derive shapes, strides, GQA mapping,
   RoPE sign pattern, causal condition (`j ≤ i`), OOB masking, fp32 accumulation — against §1.2.
2. **c001 = correctness-first baseline** (decomposition A, conservative single tiling, no autotune
   fireworks). Goal: **pass all five workloads** and record a baseline geomean speedup. A correct
   but only-modestly-fast baseline is the anchor; do not chase speed before correctness is proven.
3. **Interpret feedback per workload**, not just geomean: a single failing workload (likely the
   non-pow2 S=293/373 masking cases, or the tiny M=128 case) localizes the bug class.
4. **One change per candidate ID** (immutable). Sequence of hypotheses:
   - c001: separate kernels, correct math (establish PASS + baseline).
   - c00x: autotune / tile the projection + output GEMMs (biggest FLOP share).
   - c00x: flash-attention tile tuning + causal early-exit.
   - c00x: fuse norm+RoPE into attention prologue (option B); KV-load reuse across the group of 12.
   - c00x: K/V fused projection (option C) if profiled-by-feedback beneficial.
   Each step keeps prior correctness invariants; roll back if a change regresses geomean or breaks a
   workload.
5. **Numeric-margin guard:** if any workload is *correct but marginal* (near tolerance), prefer the
   more-precise variant (bf16-intermediate RoPE / match reference softmax rounding) even at small
   perf cost — a correctness fail zeroes the workload's contribution.
6. **Budget discipline:** spend evals on hypotheses with clear expected deltas; stop when geomean
   converges (diminishing returns across ≥2–3 consecutive candidates) and write `SEARCH_COMPLETE`.
   Never run `final` without operator approval.

### 5.1 Definition of done for the search
- All five feedback workloads PASS correctness on the chosen best candidate.
- Geomean speedup > 1.0 and converged (no candidate improving it materially).
- `candidates.jsonl` has one append-only record per eval (parent, source hash, hypothesis,
  validation, per-workload results, geomean, decision, cumulative eval count, skill usage).

---

## 6. Open questions / to confirm via first eval
- Actual magnitude/scale of "random" inputs (governs whether atol or rtol dominates) — inferred from
  the first feedback correctness result.
- Whether Triton projection GEMMs on A800 land close enough to cuBLAS that the attention-fusion win
  yields net geomean > 1 (the central performance bet).
- Whether the tiny M=128 workload needs split-K or a different tiling to avoid SM under-utilization.
