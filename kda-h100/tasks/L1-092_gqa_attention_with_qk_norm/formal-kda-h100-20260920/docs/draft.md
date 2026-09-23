# Draft — L1/092 GQA Attention with QK-Norm (GLM-4.5-Air block), H100 / sm_90

## 0. Purpose & scope of this document

This is the analysis draft required by the KDA workflow. It does **not** contain a
plan or any solution code. It fixes the exact operation, the constraints of this
run, the numerical hazards, the Triton design space, and how each candidate will
be validated given that this workspace can only run the two whitelisted launcher
scripts (`scripts/evaluate_candidate.sh`, `scripts/ncu_profile.sh`). All other
shell commands are blocked, so **there is no local ad-hoc correctness harness** and
every arithmetic sanity check below was done by hand.

---

## 1. The operation

`run(...)` is one full GLM-4 attention block (prefill, no KV cache) over
`hidden_states [B, S, 4096]` in bf16. The reference (`task/definition.json`)
decomposes into five stages:

1. **Q/K/V projections** (`F.linear`, with bias on all three):
   - `q = hidden @ q_proj_weightᵀ + q_proj_bias` → `[B, S, 12288]` (96 heads × 128)
   - `k = hidden @ k_proj_weightᵀ + k_proj_bias` → `[B, S, 1024]`  (8 heads × 128)
   - `v = hidden @ v_proj_weightᵀ + v_proj_bias` → `[B, S, 1024]`  (8 heads × 128)
   - reshape to `[B, S, H, 128]` (H=96 for q, 8 for k/v).

2. **QK RMSNorm** per (token, head) over the last dim (128), in **fp32**:
   ```
   x32 = x.float()
   var = mean(x32², axis=-1, keepdim)
   xn  = x32 * rsqrt(var + eps)
   out = (weight * xn).to(bf16)          # weight is [128] bf16
   ```
   Applied to q (with `q_norm_weight`) and k (with `k_norm_weight`). `eps = 1e-5`
   (fp32 scalar, identical for all 16 workloads). V is **not** normed.

3. **Transpose** to `[B, H, S, 128]`, then **RoPE** (GPT-NeoX "rotate_half",
   split at 64), computed in **bf16** in the reference:
   ```
   rot(x)[d]   = -x[d+64]   for d in [0,64)
   rot(x)[d]   =  x[d-64]   for d in [64,128)
   x'          = x*cos + rot(x)*sin
   ```
   `cos`, `sin` are `[B, S, 128]` bf16, broadcast across heads (`unsqueeze(1)`),
   i.e. indexed only by `(batch, position)`. Note the tensors are the **full 128**
   dim already (not the 64-length base) — use them verbatim, do **not** assume
   `cos[d]==cos[d+64]`. Applied to q and k only.

4. **GQA scaled-dot-product attention with causal mask** (num_key_value_groups=12):
   - repeat K,V from 8 → 96 heads (each kv head serves 12 q heads),
   - `scores = (q @ kᵀ) * scaling`, `scaling = 128**-0.5 ≈ 0.0883883476`,
   - add causal mask (`-inf` for key j > query i, dtype bf16),
   - `softmax(dim=-1, dtype=float32)` → cast to bf16,
   - `out = softmax @ v` → `[B, H, S, 128]`.

5. **Output projection** (no bias): transpose+reshape to `[B, S, 12288]`, then
   `output = attn @ o_proj_weightᵀ` → `[B, S, 4096]` bf16.

### Fixed constants
| symbol | value |
|---|---|
| hidden_size | 4096 |
| num_attention_heads H | 96 |
| num_key_value_heads | 8 |
| groups (H/kv) | 12 |
| head_dim d | 128 |
| q_out_features | 12288 |
| kv_out_features | 1024 |
| rotate split | 64 |
| scaling | 1/√128 ≈ 0.08838834764 |
| eps | 1e-5 |

Only `batch_size (B)` and `seq_len (S)` vary. All inputs/outputs bf16 except
`rms_norm_eps` (fp32 scalar).

### Entry point
`solution/solution.py` must expose `run(hidden_states, q_proj_weight, q_proj_bias,
k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight,
q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps)` with this exact 13-arg
signature and return the bf16 `[B, S, 4096]` output.

---

## 2. Workloads & roofline

The feedback set is the **full** official set = **16 workloads** (TASK.md line 8
and the "16-workload" final line). `M = B·S` = number of tokens:

| B | S | M=B·S | notes |
|---|---|---|---|
| 1 | 128 | 128 | tiny, memory-bound projections |
| 4 | 128 | 512 | |
| 1 | 512 | 512 | |
| 2 | 256 | 512 | |
| 2 | 512 | 1024 | |
| 1 | 1024 | 1024 | |
| 4 | 293 | 1172 | non-pow2 S |
| 1 | 2048 | 2048 | long S, small B |
| 16 | 128 | 2048 | |
| 4 | 512 | 2048 | |
| 8 | 373 | 2984 | non-pow2 S |
| 16 | 256 | 4096 | |
| 2 | 2048 | 4096 | long S — biggest naive-attn blowup |
| 8 | 512 | 4096 | |
| 4 | 1024 | 4096 | |
| 32 | 256 | 8192 | largest M |

**Projection roofline.** Total projection weight bytes (bf16) read once per call:
q 12288·4096·2 ≈ 100.7 MB, k 1024·4096·2 ≈ 8.4 MB, v ≈ 8.4 MB, o 4096·12288·2 ≈
100.7 MB → **≈ 218 MB**. At H100 HBM ≈ 3.35 TB/s that is a **≈ 65 µs floor** just
to stream the weights, independent of batch. Projection FLOPs ≈ `M · 2.18e8`, so
arithmetic intensity (FLOP per weight byte) ≈ `M`. Ridge point for bf16 (≈ 990
TF/s ÷ 3.35 TB/s ≈ 296) means: workloads with **M ≲ 300 are memory-bound** on the
projections (weight streaming dominates), workloads with **M ≳ 2k–8k are
compute-bound**. So for small M the win is launch/traffic reduction; for large M
the win is GEMM efficiency + attention fusion.

**Attention roofline (why this task is winnable).** The reference materializes the
full `attn_weights` tensor in **fp32**: `B·H·S·S·4` bytes, written by the matmul,
re-read/written by the mask add, and again by the fp32 softmax:
- (1,512): ≈ 100 MB, (16,256): ≈ 402 MB, (32,256): ≈ 805 MB,
- (4,1024): ≈ 1.6 GB, (1,2048): ≈ 3.2 GB, **(2,2048): ≈ 6.4 GB**.

That tensor is touched several times → multiple GB of HBM traffic on the long-S
cases, which is pure overhead a fused **FlashAttention** kernel eliminates entirely
(scores stay in registers/SMEM, only O is written). This is the single largest
speedup lever, and it grows with S. Attention compute itself (causal) ≈
`2·B·H·S²·d` — e.g. (2,2048) ≈ 206 GFLOP vs ≈ 893 GFLOP of projections there, so
projections dominate compute at large M but naive attention dominates *memory
traffic* at large S.

**Takeaway:** the geomean win comes from (a) eliminating the fp32 score-tensor
traffic via flash attention (dominant on high-S), and (b) fusing
RMSNorm+RoPE+reshape so the QKV tensors don't round-trip through HBM repeatedly,
and (c) getting the four GEMMs to at least cuBLAS-class efficiency. Small-M cases
are bounded by weight streaming and kernel-launch count; fewer, fused launches
help there.

---

## 3. Constraints (this run)

- **Triton is the primary compute.** PyTorch only for metadata / launch plumbing
  (shapes, empty allocation, contiguity, weight `cat` at setup). **No** Torch
  computational fallback (`F.linear`, `torch.matmul`, `torch.softmax` on the hot
  path all count as forbidden fallbacks), no CPU/NumPy, no CUDA-extension, no
  alternate implementation. A failed Triton kernel is invalid — it may **not** be
  replaced with a Torch path.
- **Immutable candidates.** `c001, c002, …` sequential; any meaningful source /
  config / launch change ⇒ new ID; never reuse an ID for changed source; never
  rewrite earlier `candidates.jsonl` records.
- **Evaluation.** Only via `./scripts/evaluate_candidate.sh feedback <id>`; the
  full 16-workload feedback run = **one** evaluation. Budget: **100** evaluations.
  Token soft/normal/hard limits 9M / 10M / 11M.
- **Profiling.** Only through `scripts/ncu_profile.sh` (ncu-report-skill workflow).
  **Never overlap profiling with evaluation** — a foreign process on the locked GPU
  during timing → controller discards the run (rc 3) and burns one eval. Serialize
  strictly.
- **`final`** only with explicit operator approval.
- **Isolation.** Stay in this workspace; only `KernelWiki` + `ncu-report-skill`
  external knowledge.

---

## 4. Numerical risks & tolerance analysis

Per-workload tolerances: `max_atol ∈ [0.0025, 0.0051]`, `max_rtol = 0.05`. Output
is bf16 (ulp near 1.0 ≈ 0.0078, so the atol alone is sub-ulp — the harness almost
certainly passes on `|Δ| ≤ atol + rtol·|ref|`, i.e. the generous rtol carries most
elements). rtol 0.05 is comfortable, but the composition through softmax + two
GEMMs means small biases can accumulate; treat each of the following as a
must-match-the-reference item:

1. **RMSNorm in fp32.** Cast to fp32, reduce `mean(x²)` over 128 in fp32,
   `rsqrt(var+eps)`, multiply by bf16 weight (promote weight to fp32 for the
   product, as `weight * xn` in the ref happens in fp32 before `.to(bf16)`), then
   round to bf16 **before** RoPE. Doing the reduction in bf16 would drift; keep it
   fp32.

2. **RoPE precision & ordering.** Reference does RoPE in **bf16** arithmetic
   (`x` is already bf16, `cos/sin` bf16). We may compute RoPE in fp32 and round
   once — that is *more* accurate than the reference, and within rtol, but it is a
   deliberate deviation to note and validate. RoPE must be applied **after**
   RMSNorm, and uses `cos/sin` indexed by `(batch, position)` only, broadcast over
   heads. Verify the rotate_half split is at 64 with the exact sign pattern in §1.3
   and that provided `cos/sin` are used verbatim across all 128 lanes.

3. **Softmax / scores.** Reference computes scores in bf16 (`q@kᵀ` bf16) *then*
   softmax in fp32. Flash attention computes `q@kᵀ` with fp32 accumulation (tl.dot),
   online max-subtracted softmax in fp32, and PV with fp32 accumulation → strictly
   more accurate than the reference. Fold `scaling` into scores (or pre-scale q)
   before the running-max. Cast the softmax→PV product path so PV matches the
   reference's "softmax cast to bf16, then bf16 matmul with fp32 accum". Small
   probability-vs-bf16-prob rounding differences are well inside rtol.

4. **Causal mask + non-power-of-2 S.** For query i, keys j>i masked to `-inf`.
   S ∈ {293, 373} are not multiples of any block size, so the flash kernel must
   (a) mask key columns `j ≥ S` (padding within the last K block) and (b) mask
   query rows `i ≥ S`, and (c) apply the triangular `j>i` mask on the diagonal
   block. A fully-masked row (shouldn't occur — row i always sees key i) would
   produce `−inf` max → NaN; guard the running-max init and the final
   normalization to avoid `0/0`. Diagonal blocks need per-element `i,j` compare;
   off-diagonal lower blocks skip the compare (all valid); upper blocks are skipped
   entirely.

5. **GQA repetition correctness.** q head `h` uses kv head `h // 12` (contiguous
   grouping matches the reference `expand(...).reshape(...)` — group index is the
   fast-repeat axis). Getting the head→kv-head mapping wrong is a silent
   correctness bug that still "runs".

6. **Bias.** q/k/v projections have bias; o_proj has none. Bias added in the GEMM
   epilogue in fp32 then rounded (bias is bf16). Forgetting bias, or adding it
   after RMSNorm, is wrong — bias is added *before* RMSNorm (it is part of the
   linear output that gets normed).

7. **Accumulation dtype.** All tl.dot must accumulate in fp32. O-projection sums
   over 12288 terms — bf16 accumulation there would lose precision; fp32 accum is
   required and sufficient.

8. **Layout / stride hazards.** `.transpose(1,2)` in the reference is logical; our
   kernels choose their own physical layout. Ensure the attention kernel reads q/k/v
   from whatever layout the projection epilogue writes, and that the final output is
   the contiguous `[B,S,4096]` the harness expects. Non-contiguous inputs (cos/sin,
   weights) should be handled defensively (they are contiguous as generated, but
   assert/΅contiguous() in plumbing is cheap insurance).

Because there is **no local reference to diff against**, correctness is judged only
by the evaluator. Mis-estimating any of the above wastes a full evaluation, so each
candidate must be argued structurally-equivalent to the reference before spending
an eval.

---

## 5. Triton design space

### 5.1 Kernel decomposition options
- **Option A (staged, 3–4 kernels):**
  1. QKV projection GEMM(s) → q/k/v in HBM,
  2. fused RMSNorm+RoPE+reshape (elementwise, per token·head),
  3. FlashAttention (causal, GQA) → attn_out `[B,S,12288]`,
  4. O-projection GEMM → output.
  Simplest to get correct first; extra HBM round-trips for q/k/v.
- **Option B (epilogue-fused QKV):** fold RMSNorm+RoPE into the QKV GEMM epilogue.
  Because head_dim=128 and out-features are head-contiguous, choosing `BLOCK_N=128`
  aligned to head boundaries gives one full head per output tile → RMSNorm (reduce
  over the 128 lanes the tile owns) and RoPE (needs cos/sin for the tile's
  positions) can run in the epilogue with no extra HBM pass. Saves the stage-2
  kernel and one q/k/v round-trip.
- **Option C (fuse attention output with O-GEMM):** possible but O-GEMM needs the
  full `[*,12288]` row assembled across all 96 heads; hard to fuse cleanly with a
  per-(head,q-tile) flash kernel. Lower priority.
- **QKV fusion:** the three input projections share the same `hidden` LHS; a single
  concatenated weight `[12288+1024+1024, 4096]` (built once at setup via `cat` —
  plumbing) lets one GEMM produce all of q/k/v, improving LHS reuse and cutting
  launches. Downstream norm/rope differs per slice but is handled by column range.

Plan: start with a **correct staged Option A** (c001) to lock down numerics with
the evaluator, then move toward B (epilogue fusion) and QKV-concat for speed.

### 5.2 GEMM design (H100, bf16, tl.dot ⇒ wgmma)
- Classic tiled GEMM: `BLOCK_M=128`, `BLOCK_N ∈ {128,256}`, `BLOCK_K ∈ {32,64}`,
  `num_stages ∈ {3,4}`, `num_warps ∈ {4,8}`, fp32 accumulate. Autotune over a small
  key on `(M, N, K)`. On sm_90 `tl.dot` lowers to `wgmma.mma_async`
  (per KernelWiki `lang-triton`, Triton 3.6 adds tcgen05/TMEM on **sm_100** only;
  on sm_90 we stay on the mature wgmma path).
- **K=4096** (all four GEMMs share the reduction dim 4096) → good for deep-K
  pipelining; `BLOCK_K=64`, `num_stages=3–4` to overlap TMA-ish global loads with
  MMA. Weights are `[N, K]` (row-major out×in), i.e. the standard `Wᵀ` linear
  layout — load with the transpose-friendly access pattern.
- Beating cuBLAS on pure bf16 GEMM is hard; realistic target is ≈ 0.85–1.0× cuBLAS
  per GEMM. Net win still comes from fusion + attention. For small-M GEMMs
  (memory-bound), a **split-K** or a thin-tile config may help utilization; for
  large-M use big N tiles. Autotune must include small-M-friendly configs
  (`BLOCK_M=64`) because M ranges 128 → 8192.
- Epilogue: bias add (fp32) + optional RMSNorm/RoPE (Option B) + bf16 store.

### 5.3 FlashAttention design (causal, GQA, d=128, bf16)
- Grid over `(B, H=96, S/BLOCK_M)`; each program handles a `BLOCK_M×128` query tile
  for one q-head, streaming K/V blocks of the mapped kv-head with online softmax
  (running max `m`, running sum `l`, accumulator `acc` fp32).
- head_dim=128 fits a single MMA K-dim tile; typical config `BLOCK_M=64/128`,
  `BLOCK_N=64/128`, `num_warps=4/8`, `num_stages=2–3` (mirror the Triton
  fused-attention tutorial, which is well-tuned for sm_90 d=128).
- **Causal early-exit:** for query tile starting at row `q0`, iterate key blocks
  only up to `q0+BLOCK_M`; the last (diagonal) block applies the `j>i` compare, the
  rest are unmasked. Saves ~½ the work.
- **GQA reuse:** naive mapping reads each kv-head's K/V 12× (once per q-head).
  Optimization: process the 12 q-heads of a group together (pack group into the M
  dimension or loop heads inside one program) so K/V blocks load once per kv-head —
  cuts K/V HBM traffic 12×. Consider after the correct baseline; measure with ncu.
- **RoPE-in-attention alternative:** RoPE could be applied inside the flash kernel
  when loading q/k (fusing stage 2 and 3), but that recomputes rope per key block —
  usually better to pre-apply once in the projection epilogue (Option B).
- **Boundary masking** for S∈{128,256,293,373,512,1024,2048}: mask key columns
  `≥S` with `-inf` before the max; mask/skip query rows `≥S` on store.

### 5.4 Autotuning & compile cost
- Autotune keys must bucket on `M` (and maybe `S`) so the 16 shapes don't each
  trigger a fresh long autotune during a single evaluation (warmup=2, iters=10 is
  coarse; excessive autotuning/compilation can dominate and even risk timeouts).
  Prefer a **small hand-curated config list** over a broad autotune grid, and warm
  the JIT cache in `run` setup if needed. Cache compiled kernels across the 16
  workloads within one process (they share the same constants; only M/S differ).

---

## 6. Validation strategy

Because no local correctness harness is permitted and Bash is locked to the two
launchers, validation is a disciplined loop:

1. **Structural equivalence proof (offline, by hand).** Before every eval, walk the
   candidate against the §1 stage list and the §4 hazard checklist; confirm dtypes,
   reduction precision, RoPE sign/split, GQA head mapping, bias placement, causal
   masking, and output layout. Only spend an eval when the argument is airtight.
2. **First candidate = correctness anchor (c001).** Prioritize a
   provably-equivalent staged implementation (Option A) even if not fastest, to
   confirm the numerics pass all 16 tolerances. This de-risks every later perf
   candidate (they inherit the same math).
3. **One variable per candidate.** After c001, change one thing at a time (fuse
   norm/rope epilogue; QKV-concat; GQA K/V reuse; GEMM configs; flash block sizes)
   so a pass/fail and a speedup delta map to a single cause. New ID each time;
   append one JSON record per eval to `candidates.jsonl` with parent, source hash,
   hypothesis, per-workload result, geomean, decision, cumulative eval count, skill
   usage.
4. **Profiling discipline.** Use `scripts/ncu_profile.sh` (ncu-report-skill) to find
   bottlenecks (GEMM MMA utilization, flash-attn occupancy/HBM, launch overhead on
   small-M), **never overlapping** an evaluation. Profile a representative subset
   (e.g. one small-M memory-bound shape like (1,128), one balanced (4,512), one
   long-S (2,2048)) rather than all 16.
5. **Boundary focus.** Explicitly reason about and, where cheap, target the
   non-pow2 shapes (293, 373) and the tiny (1,128) and huge (32,256)/(2,2048) cases
   — these are the most likely to expose masking or autotune-bucketing bugs.
6. **Budget guard.** 100 evals but correctness bugs are the expensive failure mode;
   converge by (a) locking numerics early, (b) profiling to pick the next lever, (c)
   stopping when geomean improvement plateaus, then writing `SEARCH_COMPLETE`.
   Never run `final` without operator approval.

### Correctness pitfalls that "run but are wrong" (highest-value to pre-check)
- GQA head→kv mapping (`h//12` vs `h%8`).
- RoPE applied before RMSNorm, wrong split, or assuming cos/sin duplication.
- Bias omitted or placed after norm.
- Softmax/rescale numerics (max init, `0/0` on masked rows).
- Autotune config that silently picks a shape-invalid block for S∈{293,373}.

---

## 7. Preliminary risk register (feeds the plan, not the plan)

| risk | impact | mitigation |
|---|---|---|
| Triton GEMM slower than cuBLAS at large M | net slowdown on M≥4k shapes | autotune curated configs; lean on attention+fusion win; profile MMA util |
| Autotune/JIT compile cost under coarse 10-iter timing | inflated measured time / timeout | small config lists, cache across workloads, bucket keys by M |
| Non-pow2 S masking bug | correctness fail (burns eval) | explicit row/col masks; pre-check c001 on 293/373 logic |
| Long-S register/SMEM pressure (d=128, big BLOCK) | occupancy drop / spill | moderate BLOCK_M/N, num_stages≤3, ncu-guided |
| RoPE/RMSNorm precision drift | rtol/atol borderline | fp32 reductions, single bf16 rounding, verify via evaluator on c001 |
| Over-fusing early | hard-to-debug correctness | stage first (Option A), fuse incrementally |

---

## 8. Immediate next step (after this draft)

Write `docs/plan.md` (separate turn) turning §5–§6 into an ordered candidate
roadmap: c001 = correct staged Triton baseline (Option A) to anchor numerics; then
epilogue-fused RMSNorm+RoPE, QKV-concat GEMM, GQA K/V reuse, and GEMM/flash block
autotuning as one-variable-at-a-time candidates, each gated by the structural
equivalence check and ncu profiling. No code until the plan is written.

### Skills consulted
- **KernelWiki**: `lang-triton` (Triton on sm_90 uses the wgmma path; tcgen05/TMEM
  is sm_100-only, so no Blackwell-specific lowering applies here) and
  `kernel-flash-attention-4` (confirms d=128 flash design intent; FA-4's
  tcgen05/2-CTA/software-exp tricks are B200-only and not usable on H100 — the
  standard Triton fused-attention pattern is the right reference here).
- **ncu-report-skill**: to be used for bottleneck profiling during the search
  (never concurrently with evaluation).
