# Draft — `gqa_ragged_prefill_causal_h32_kv8_d128`

Target GPU: **NVIDIA A800 (`sm_80`, Ampere)**. Benchmark family: FlashInfer.
Entry point to build: `solution/solution.py` exposing `run(q, k, v, qo_indptr, kv_indptr, sm_scale)`.
Primary implementation must be **Triton**; PyTorch allowed only for metadata / launch plumbing.
No Torch/CPU/NumPy/CUDA-extension computational fallback.

---

## 1. Operation semantics

Batched **Grouped-Query Attention (GQA)** **prefill** with **ragged** (variable-length) sequences and a
**causal** mask. Captured from Llama-3.1-8B prefill.

Fixed constants (asserted in the reference):
- `num_qo_heads = 32`
- `num_kv_heads = 8`
- `head_dim = 128`
- `gqa_ratio = num_qo_heads // num_kv_heads = 4`  → query head `h` uses kv head `h // 4`.

Inputs:
- `q`  : `[total_q, 32, 128]` bf16
- `k`  : `[total_kv, 8, 128]` bf16
- `v`  : `[total_kv, 8, 128]` bf16
- `qo_indptr` : `[len_indptr]` int32, prefix-sum query offsets (`len_indptr = batch+1`)
- `kv_indptr` : `[len_indptr]` int32, prefix-sum kv offsets
- `sm_scale`  : fp32 scalar, `1/sqrt(128) ≈ 0.08838834764831843`

Outputs:
- `output` : `[total_q, 32, 128]` bf16
- `lse`    : `[total_q, 32]` fp32 — **base-2** log-sum-exp of the (scaled) logits.

Per batch element `b`, with `q_start=qo_indptr[b]`, `q_end=qo_indptr[b+1]`,
`kv_start=kv_indptr[b]`, `kv_end=kv_indptr[b+1]`:
- `Nq = q_end - q_start`, `Nk = kv_end - kv_start`, `delta = Nk - Nq`.
- Query local index `i∈[0,Nq)` attends kv local index `j∈[0,Nk)` iff `j < i + 1 + delta`
  (equivalently `j <= i + delta`). This is the **bottom-right–aligned** causal mask: when `Nk==Nq`
  (`delta=0`) it is the standard causal mask (`j<=i`); when `Nk>Nq` the query block is aligned to the
  *end* of the kv, so the last query sees the whole kv.
- `logits = (q_batch @ k_expanded^T) * sm_scale` computed in **fp32** in the reference
  (q/k/v are upcast to fp32 first).
- `lse = logsumexp(logits, dim=-1) / ln(2)` = `log2(sum exp(logits))`.
- `output = softmax(logits) @ v_expanded`, cast to bf16.
- Sequences with `q_start>=q_end` or `kv_start>=kv_end` are skipped → their `output` stays `0`,
  `lse` stays `-inf`.

`k_expanded/v_expanded` = `repeat_interleave(gqa_ratio, dim=1)`, i.e. kv head `g` feeds q heads
`4g, 4g+1, 4g+2, 4g+3`.

---

## 2. Feedback workload characterization

The five fixed feedback workloads (all `head=32/kv=8/d=128`, `sm_scale≈1/sqrt(128)`):

| # | uuid (prefix) | len_indptr | total_q | total_kv | shape note |
|---|---------------|-----------:|--------:|---------:|------------|
| 1 | b8f94163 | 2  | 7   | 7   | single seq, 7 tokens |
| 2 | bdc49f9e | 2  | 1   | 1   | single seq, 1 token |
| 3 | f4c23a33 | 2  | 1   | 1   | single seq, 1 token (scale slightly diff fp32 rounding) |
| 4 | 32f5e961 | 2  | 35  | 35  | single seq, 35 tokens |
| 5 | 2328b031 | 16 | 982 | 982 | 15 sequences, avg ~65 tokens each |

Key observations:
- **All workloads are tiny.** Four of five are a single sequence with ≤35 query tokens; the largest is
  982 tokens spread over 15 sequences. Per-sequence `total_q == total_kv` → **delta = 0 is expected**
  (self-attention prefill), i.e. standard causal mask. We still implement general `delta>=0`.
- Compute is negligible (a 1×1 attention up to a ~65×65 attention per sequence). The workloads are
  **launch/overhead-bound**, not FLOP-bound. There is essentially no arithmetic intensity to exploit;
  occupancy, tensor-core utilization, and cp.async pipelining are largely irrelevant here.
- Therefore the dominant cost of the reference is: fp32 upcast of q/k/v, a **Python `for`-loop over
  batch** (1 iter for wl1-4, 15 iters for wl5), repeated `.item()` host syncs (4 per batch),
  `repeat_interleave` materialization, full `[Nq, 32, Nk]` logits materialization, `masked_fill`,
  separate `logsumexp` + `softmax` + two `einsum`s, and a bf16 cast — i.e. **many small kernel launches
  and host round-trips**.
- **Primary optimization lever = collapse everything into a single fused Triton kernel launch with
  minimal host-side Python and zero (or one) host↔device sync.** Secondary levers (tile sizes, warps)
  matter little given the size, but we tune them to avoid pathological under/over-provisioning.

Ranking metric = geometric mean speedup over the 5 (all must pass correctness). Because 4/5 workloads
are minuscule, the geomean is heavily driven by **fixed overhead reduction**; a single launch that beats
the reference's ~dozens of launches on the tiny cases will dominate, and the 982-token case rewards a
proper fused varlen flash kernel over the Python batch loop.

---

## 3. Exact numerics — base-2 flash formulation

We use the standard base-2 flash-attention online-softmax recurrence, which reproduces the reference
mathematically:

Let `L2E = log2(e) = 1.4426950408889634`. Define per (query, kv) natural logit
`s = (q·k) * sm_scale`. Work in the base-2 scaled domain `s2 = s * L2E`, so `exp(s) = exp2(s2)`.

Online over kv tiles, tracking running max `m` and running denom `l` in the `s2` domain:
- `m_new = max(m, rowmax(s2_tile_masked))`
- `p = exp2(s2_tile - m_new)`  (masked entries → `-inf` → `p=0`)
- `alpha = exp2(m - m_new)`  (rescale factor)
- `l = l*alpha + rowsum(p)`
- `acc = acc*alpha + p @ v_tile`   (fp32 accumulator, `[BLOCK_M, 128]`)

Final:
- `output = acc / l` → cast to bf16.
- `lse = m + log2(l)` in the **s2 domain**. Proof: `lse_out = log2(sum exp(s)) = log2(sum exp2(s2)) =
  m + log2(sum exp2(s2 - m)) = m + log2(l)`. This equals the reference `logsumexp(logits)/ln2`.

Matmul precision: reference upcasts q/k/v to fp32 then does fp32 einsum. In Triton we load bf16 and use
`tl.dot(bf16, bf16) → fp32` (Ampere HMMA, fp32 accumulate). The bf16 inputs are the *exact same values*
as the reference's `q_f32` (which is a lossless bf16→fp32 widening); the only difference is intermediate
product rounding inside the MMA vs an fp32 FMA reduction over 128 elements. This is the standard flash
approach and is well within bf16 output tolerance. If accuracy is marginal we can cast tiles to fp32 and
use `tl.dot(..., input_precision=...)` / an fp32 dot, but bf16-in/fp32-acc is the expected first choice.

---

## 4. Correctness constraints & numerical risks

1. **Base-2 LSE.** Easy to get wrong. `lse = m + log2(l)` with `m,l` in the `sm_scale*L2E`-scaled domain.
   Must divide the natural logsumexp by `ln2` — equivalently use `exp2`/`log2` throughout. Validate the
   1×1 case analytically: for `Nq=Nk=1`, single logit `s`, `lse = log2(exp(s)) = s*L2E`. Good closed-form
   check.
2. **Causal boundary with delta.** Allowed iff `kv_local <= q_local + delta`. Implement as effective
   query position `q_eff = q_local + delta` and compare `kv_local <= q_eff`. With `delta=0` this is
   `kv_local <= q_local`. Off-by-one is the classic bug (`<` vs `<=`, and the `+1` in the reference).
3. **GQA head mapping.** kv head = `q_head // 4`. Getting the stride/offset wrong silently mixes heads.
4. **Ragged offsets / no straddle.** Each program must operate strictly within one sequence's
   `[q_start,q_end)` × `[kv_start,kv_end)`. Tile masking on both M (query tail) and N (kv tail) is
   required since `Nq, Nk` are rarely multiples of the tile.
5. **Empty / degenerate sequences.** If `Nq>0` but `Nk==0`, reference leaves `output=0`, `lse=-inf`.
   Kernel must emit `0` output and `-inf` lse for such q blocks rather than dividing by `l=0`
   (→ NaN). Not present in feedback (all `Nk>0`) but needed for robustness on the 21-workload final.
6. **All-masked row (`delta<0`).** If `q_eff<0`, a query has no valid kv → reference yields NaN
   (softmax of all `-inf`). NaN cannot be matched under tolerance, so the evaluator presumably never
   generates it; we assume `delta>=0`. We will *not* try to fabricate a NaN. (Guarding to `0`/`-inf`
   is a safer default and won't hurt the delta>=0 cases.)
7. **Output/LSE init.** Reference zero-inits `output` and `-inf`-inits `lse`. If our grid covers all q
   rows we can write every row directly (use `torch.empty`). If any q row could be uncovered (e.g. a
   skipped sequence), we must pre-fill (`zeros`/`full(-inf)`) or write in-kernel. Given the fused
   single-launch design writes every real q row, `empty` + full in-kernel coverage is fine; a
   defensive `zeros`/`-inf` prefill is cheap for the tiny sizes if needed.
8. **`sm_scale` as fp32 scalar.** Passed straight through; fold `sm_scale*L2E` into a single constant
   applied to the raw `q·k` (either scale q before the dot, or scale logits after). Scaling after the
   dot avoids extra bf16 rounding of q.
9. **dtype of outputs.** `output` bf16, `lse` fp32 — must match exactly or shape/dtype check fails.

---

## 5. Triton design space

Core structure: a **fused varlen (ragged) causal flash-attention prefill** kernel, one launch total.

Program identity: `(q_block, q_head, batch)` — a 3D grid. Each program computes one `BLOCK_M`-row tile
of queries, for one query head, within one sequence, looping over kv tiles.

Design axes to explore in the plan:
- **Grid `dim0` sizing (host overhead vs over-provision).**
  - (A) *No-sync over-provision*: `dim0 = cdiv(total_q, BLOCK_M)` (upper bound since any single seqlen
    ≤ `total_q`); programs past a sequence's own length early-exit after reading two int32 offsets.
    Zero host↔device sync. Wastes grid rows for the multi-seq case (`982` bound vs true max seqlen),
    but each waste is a ~free early-exit. **Preferred default** because it removes all `.item()` syncs.
  - (B) *One-sync tight*: compute `max_seqlen = (qo_indptr[1:]-qo_indptr[:-1]).max().item()` and set
    `dim0 = cdiv(max_seqlen, BLOCK_M)`. One sync; tighter grid for wl5. Compare against (A).
- **Q-head grouping for GQA reuse.** Either one program per q head (simplest, 32 heads → more programs,
  more kv reloads) or one program per **kv-head group** processing all 4 sibling q-heads together
  (loads K/V once per kv tile, packs `4*BLOCK_M` or a `[4,BLOCK_M,·]` M dimension). Given kv is tiny,
  reuse savings are small; grouping mainly reduces program count for the 1-token cases. Keep as a
  tuning variant, default to per-head for simplicity/correctness first.
- **Tile sizes.** `BLOCK_M ∈ {16,32,64}`, `BLOCK_N ∈ {32,64,128}`, `BLOCK_D = 128` (no D tiling; full
  head dim in registers/shared). For the ≤35-token cases, small `BLOCK_M` (16/32) and a single kv tile
  (`BLOCK_N` ≥ seqlen) minimize masked waste. For wl5 (~65/seq), `BLOCK_M=32/64`, `BLOCK_N=64` is
  reasonable.
- **Causal skip.** Skip kv tiles entirely above the causal boundary for the block
  (`kv_tile_start > max_q_eff_in_block`) to cut work; marginal here but standard and free.
- **`num_warps ∈ {2,4,8}`, `num_stages ∈ {1,2,3}`.** With head_dim=128 and tiny kv, `num_warps=4`,
  `num_stages=2` is a sane default; low kv depth means deep pipelining gives little.
- **Autotune vs fixed config.** Autotune keyed on a coarse `max_seqlen` bucket could help across the
  size spread, but adds first-call benchmarking overhead and nondeterminism; a couple of hand-picked
  configs selected by a cheap host heuristic (e.g. tiny vs larger) is more predictable. Explore both;
  prefer a small fixed config for the overhead-bound tiny cases.
- **Single kernel vs split-KV.** Split-KV (flash-decoding style) is pointless here (kv tiny, prefill,
  plenty of query parallelism). Single kernel only.
- **Load path.** `q/k/v` are `[tokens, heads, 128]` contiguous → head-major inner stride 128. Load a
  `[BLOCK_M,128]` q tile and `[BLOCK_N,128]` k/v tiles with masks. Consider `tl.load` with
  `boundary_check`/`other=0` or explicit masks.

Anti-goals: no Torch attention fallback, no `repeat_interleave` materialization, no full-logits
materialization, no per-batch Python loop inside the kernel launch path (the single kernel handles all
batches via the grid).

---

## 6. Host-side / launch design (minimize overhead)

Because the tiny workloads are overhead-bound, the Python `run(...)` must be lean:
- Read shapes from tensors only (no `.item()` on data-dependent values in the preferred path A).
- Allocate `output = torch.empty([total_q,32,128], bf16)` and `lse = torch.empty([total_q,32], fp32)`
  (or pre-init if defensive coverage is needed).
- Compute grid from static shapes + chosen `BLOCK_M` (path A) → single `kernel[grid](...)` launch.
- Pass `qo_indptr`, `kv_indptr` as pointers; pass `num_qo_heads`, `num_kv_heads`, `gqa_ratio`,
  `head_dim`, `sm_scale*L2E` as (mostly `constexpr`) args; strides passed explicitly.
- Avoid `.contiguous()` copies if inputs are already contiguous (they are, freshly generated).
- Return `(output, lse)` in the exact declared order/dtypes.

Every avoided host sync and avoided intermediate kernel is direct geomean gain on wl1–4.

---

## 7. Performance hypotheses

- **H1 (dominant):** A single fused Triton launch replacing the reference's per-batch Python loop and
  ~dozen small torch kernels/syncs yields a large speedup on all five, largest multiplicative gains on
  the 1-token cases (wl2, wl3) where the reference is almost pure overhead.
- **H2:** Removing all host↔device syncs (grid path A) beats the one-sync path (B) on the tiny cases;
  the extra early-exit grid rows for wl5 are cheaper than a `.max().item()` sync. To be confirmed by
  A/B candidates.
- **H3:** Tile-size tuning has second-order impact; a small fixed config (`BLOCK_M=32`,
  `BLOCK_N=64`, `num_warps=4`, `num_stages=2`) is a good starting default. GQA head-grouping is a
  minor, later refinement.
- **H4:** bf16-in/fp32-acc matmul passes tolerance; fp32-dot is a fallback only if correctness is
  marginal (unlikely).

Non-hypotheses (explicitly not worth pursuing on `sm_80`, tiny sizes): tensor-core occupancy tuning,
cp.async multi-stage depth, warp specialization, persistent kernels — these are irrelevant at this
scale. (Note: the `KernelWiki` and `ncu-report-skill` skills target Blackwell/Hopper and B200 profiling
respectively; **neither applies to A800/`sm_80`**, and profiling tools cannot be run here anyway, so
they will not be used.)

---

## 8. Validation strategy

Constraints: we may **not** run CUDA, a profiler, `nvidia-smi`, the external evaluator directly, or any
alternate correctness harness. The only sanctioned signal is
`./scripts/evaluate_candidate.sh feedback <cid>`, which runs all five feedback workloads (= one
candidate evaluation) and reports per-workload correctness + speedup.

Offline (reasoning only, before each eval):
- Re-derive the base-2 LSE and the causal-with-delta boundary by hand; check the closed-form 1×1 case
  (`lse = s*L2E`, `output = v`).
- Trace tile masking for a small non-power-of-2 seqlen (e.g. 7, 35) to confirm M/N tail masks and the
  causal skip.
- Confirm GQA head→kv-head index arithmetic and strides.
- Confirm output/lse dtypes, shapes, and return order.

On each candidate:
- Run the feedback eval, record per-workload pass/fail + speedup and geomean.
- If a workload fails correctness, diagnose from the failure signature (which workload / magnitude) and
  spin a new candidate ID (never mutate an evaluated one).

Candidate roadmap (to be detailed in `docs/plan.md`, not now):
- `c001`: minimal correct fused varlen causal flash kernel, per-head program, grid path A, fixed small
  config, bf16-in/fp32-acc. Establish correctness + baseline speedup.
- Subsequent candidates: A/B the grid-sizing sync tradeoff, tile sizes/warps/stages, optional GQA
  head-grouping, optional autotune — each an immutable new ID, kept only if geomean improves and all
  workloads still pass. Stop when converged; then `SEARCH_COMPLETE`.

Budget: 100 candidate evals; token soft 1.0M / hard 1.2M. Given the small design space and
overhead-bound regime, expect convergence in far fewer than 100 evals. Final 21-workload eval is
operator-approval-only.

---

## 9. Risks & mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Base-2 LSE wrong (÷ln2 / exp2) | wl correctness fail | closed-form 1×1 check; `lse=m+log2(l)` in scaled domain |
| Causal off-by-one / delta sign | fail on wl1/4/5 | `kv<=q_local+delta`; hand-trace seqlen 7 |
| Grid over-provision straddling seqs | wrong results | tile stays inside one sequence; early-exit on `pid_m*BLOCK_M>=Nq` |
| Non-pow2 seqlen tails | NaN/garbage in tail | explicit M and N masks, `other=0`, masked logits `-inf` |
| `Nk==0` sequence | div-by-zero NaN | emit `0`/`-inf` for such q blocks (guard `l==0`) |
| Host sync overhead on tiny wls | lost speedup | grid path A (no `.item()`), lean `run` |
| bf16 matmul accuracy | marginal tolerance | fp32-dot fallback if needed |
| Triton compile/autotune overhead counted | slower first run | prefer fixed config; if autotune, coarse keying + warmup reliance |
| Accidental Torch fallback | invalid submission | single Triton kernel only; no torch attention math |

---

## 10. Summary decision for the plan

Build one fused, ragged, causal GQA flash-attention Triton kernel launched exactly once, with a
lean host wrapper and (preferably) zero host↔device syncs. Get correctness first (`c001`), then tune
the small set of levers (grid-sync tradeoff, tiles, warps/stages, optional GQA grouping) since the
regime is overhead-bound. Validate exclusively through `evaluate_candidate.sh feedback`. Do not use the
Blackwell/Hopper skills (wrong architecture; profiling unavailable). Proceed to `docs/plan.md` next.
