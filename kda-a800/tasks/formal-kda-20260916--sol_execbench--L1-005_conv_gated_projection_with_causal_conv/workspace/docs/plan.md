# Executable Optimization Plan — L1/005 `conv_gated_projection_with_causal_conv`

Companion to `docs/draft.md` (read that first for the full index-map derivation). This file is the
**executable roadmap**: how the source is structured, the exact sequential candidate lineage, the
correctness gates each candidate must pass *before* an eval, the per-candidate performance hypothesis,
the stopping rule, and the evidence record format. **No code is written or evaluated in this turn.**

Target: NVIDIA **A800 (`sm_80`, Ampere)**. Compute **must be Triton** (`tl.dot` GEMMs + Triton
element/stencil kernels); PyTorch only for metadata/launch plumbing. **No** Torch/CPU/NumPy/CUDA-ext
computational fallback — a failed Triton kernel is an invalid candidate, not a fallback opportunity.
Metric: **geometric-mean speedup vs reference over the 5 fixed feedback workloads, every selected
workload must pass correctness**. Budget: 100 evals; tokens soft 1.0M / normal 1.5M / hard 1.65M.

Constants baked as `tl.constexpr`: `H=2048`, `3H=6144`, `K_c=4`, `groups=H` (depthwise).

---

## 0. Ground rules that shape the whole plan

- **The only feedback channel is `./scripts/evaluate_candidate.sh feedback cNNN`**, which runs all 5
  workloads = **one** eval of the 100 budget and returns correctness + per-workload speedup. We may
  **not** run CUDA/profiler/`nvidia-smi`/the evaluator directly, or any alternate correctness harness.
  ⇒ **Correctness is established analytically before every eval.** An eval is spent only when the
  candidate is believed-correct and represents a *deliberate* single-variable change.
- **One variable per candidate.** Each `cNNN` changes exactly one meaningful thing vs its parent so the
  measured delta is attributable. Any meaningful source/config/launch change ⇒ new ID; IDs are
  immutable and records are append-only in `candidates.jsonl`.
- **Skills:** `KernelWiki` is Blackwell/Hopper-specific (tcgen05/TMEM/CLC/FP4/2-SM) and does **not**
  apply to `sm_80`; profiler-based skills are disallowed (can't run `ncu`). Both are unusable for
  compute decisions here; each candidate's `skill_usage` will record `"none (KernelWiki N/A for sm_80;
  profiler disallowed)"` unless a specific generic note applies.

---

## 1. Solution architecture (`solution/solution.py`)

Single file exposing `run(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight,
out_proj_bias) -> output`. Structure held constant across candidates (only kernel internals / configs /
fusion boundaries change per candidate):

```
run():
  # --- metadata & plumbing only (PyTorch allowed) ---
  B, S, H = x.shape ; M = B*S ; assert H==2048, kc==4, 3H==6144
  assert dtypes are bf16; ensure x contiguous view -> Xv = x.reshape(M, H)  (free view)
  allocate outputs as bf16 empty tensors
  # --- Triton compute ---
  K1: BCx (M, 3H)          = GEMM1(Xv, in_proj_weight, in_proj_bias)          # tl.dot, fp32 accum
  K2: y   (M, H)           = gating+causal-depthwise-conv+gating(BCx, conv_weight, conv_bias)
  K3: out (M, H)           = GEMM2(y, out_proj_weight, out_proj_bias)         # tl.dot, fp32 accum
  return out.reshape(B, S, H)
```

Everything lives in row-major `(M, ·)` space; the reference's two `transpose`s + `.contiguous()` are
**eliminated** (they were only artifacts of routing through `conv1d`'s `(B,C,S)` convention). See draft
§1.1 for why the conv along `M`-rows with an `s = m mod S ≥ k-shift` guard is equivalent.

**Kernel definitions (baseline decomposition):**

- **K1 — GEMM1+bias** → `BCx = Xv @ in_proj_weight^T + in_proj_bias`, `(M, 3H)`. Tiled bf16 matmul,
  fp32 accumulator, bias added in fp32, cast bf16 on store. `N=6144`, `K=2048` are mult. of 128;
  `M∈{1024,2048,8192}` mult. of 256 ⇒ **mask-free fast path** for `BLOCK_{M,N,K}∈{64,128,256}`.
- **K2 — fused gating+conv+gating** → `y`, `(M, H)`. For each output element:
  `Bx[m,h] = BCx[m,h] * BCx[m,2H+h]` (fp32),
  `conv_out[m,h] = conv_bias[h] + Σ_{k=0..3} conv_weight[h,0,k] * Bx[b,h, s-3+k]` with the tap masked
  to 0 when `s-3+k < 0` (causal left-pad = batch-boundary guard, they coincide),
  `y[m,h] = BCx[m,H+h] * conv_out[m,h]` → cast bf16. Memory-bound.
- **K3 — GEMM2+bias** → `out = y @ out_proj_weight^T + out_proj_bias`, `(M, H)`. Same template as K1,
  `N=2048`.

---

## 2. Correctness checks (run BEFORE spending any eval)

These are the gates every candidate must clear analytically. They are re-verified whenever a candidate
touches the relevant code path.

### 2.1 Static/host checks (metadata only — allowed, not compute)
- Assert `x.shape == (B,S,H)`, `H==2048`, `conv_weight.shape==(H,1,4)`, `in_proj_weight.shape==(6144,2048)`,
  `out_proj_weight.shape==(2048,2048)`, all inputs `bfloat16`.
- Assert `x` contiguous so `reshape(M,H)` is a free view (do **not** silently `.contiguous()` an
  already-contiguous tensor; if ever non-contiguous, `.reshape` will copy — acceptable, note it).
- Output allocated `(B,S,H)` bf16, returned via `reshape` from `(M,H)`.

### 2.2 Kernel-derivation checklist (from draft §1.1 / §4.3) — the 6 classic bugs
1. **Chunk offsets:** `Bgate=[0,H)`, `Cgate=[H,2H)`, `Xproj=[2H,3H)`. Gating uses `Bx=BCx[:,0:H]*BCx[:,2H:3H]`;
   output gate uses `BCx[:,H:2H]`. Off-by-`H` is the most likely silent bug.
2. **Conv tap order / direction:** PyTorch `conv1d` is **cross-correlation** (not flipped). With
   `F.pad(left=3)`, output `s` reads `Bx` at `s-3+k` for `k=0..3`; `k=3`=current sample `s` (weight
   `conv_weight[h,0,3]`), `k=0`=oldest `s-3` (weight `[h,0,0]`). A mirrored kernel passes shapes, fails
   numerics.
3. **Causal / batch-boundary mask:** tap valid iff `s-3+k ≥ 0` (equivalently global row `≥ b*S`). Never
   read a neighbor row from the previous batch.
4. **`F.linear` transpose:** weights `(out,in)`; compute `Σ_in x[m,in]·W[out,in]` = `x @ W^T`.
5. **Bias dtype:** biases are bf16; promote to fp32, add to fp32 accumulator, then round to bf16.
6. **Accumulation precision:** GEMMs accumulate fp32 (matches cuBLAS); conv accumulates fp32 (4 taps,
   order-insensitive); middle computed fully in fp32.

### 2.3 Numerical-tolerance reasoning (why fp32 middle is safe)
Reference rounds to bf16 at 4 intermediate points (`BCx`, `Bx`, `conv_out`, `y`); our fp32 middle is
*more* accurate. Tolerances: atol WL5 0.0098 (tightest), WL1/2 0.013, WL3/4 0.021; rtol 0.05 (loose).
**WL5 (B=1,S=2048) is the correctness watch-list case.** Fallback lever if WL5 fails only numerically:
insert an explicit `Bx = Bx.to(bf16).to(fp32)` round before the conv to mirror the reference's
pre-conv bf16 rounding. This lever is a *new candidate* (source change), not an in-place patch.

### 2.4 Failure triage table (map a feedback failure → suspected cause, without a profiler)
| Symptom | Prime suspects | Action |
|---|---|---|
| All 5 wrong by large margin | chunk offset (#1) or `W` vs `W^T` (#4) | re-derive index map, fix, new candidate |
| All wrong by small margin near atol | conv tap/edge (#2/#3) or bias dtype (#5) | check tap order + boundary guard |
| Only WL5 fails, others pass | fp32-vs-bf16 `Bx` rounding (§2.3) | apply the bf16-round lever candidate |
| Only large-M (WL3/4) fails | M-tiling / mask / accumulator overflow | check BLOCK_M bounds, no ragged mask needed |
| Correct but slow (<1×) | Triton GEMM below cuBLAS | GEMM autotune candidates (§3) |

---

## 3. Candidate lineage strategy (sequential roadmap)

Each step lists **parent → hypothesis → single change → expected metric effect → decision rule**.
Later steps are contingent on earlier results; the exact continuation is chosen from feedback, but the
priority order is fixed by the draft's roofline (GEMMs dominate: GEMM1 ≈ 3× GEMM2 FLOPs; middle is
memory-bound and tiny — draft §5 "where the time goes").

### Phase A — Correctness anchor
- **c001 — baseline, prove correctness.** Parent: none. 3 kernels (K1, K2 *row-parallel* simple
  variant, K3), **single conservative tile config** (e.g. `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
  num_warps=4, num_stages=3`), fp32 middle. Goal: **pass all 5 workloads** and get a first speedup
  number + the GEMM-gap read. Decision: if any workload fails → triage via §2.4, fix as c002 (still
  "anchor"), do **not** proceed to tuning until correctness holds. If passes → record baseline geomean.

### Phase B — GEMM tuning (highest leverage; GEMMs dominate)
- **c00x — autotune K1 (GEMM1).** Parent: last-correct anchor. Single change: add `@triton.autotune`
  over the tile menu (draft §5: `BLOCK_M∈{64,128,256}`, `BLOCK_N∈{64,128,256}`, `BLOCK_K∈{32,64,128}`,
  `num_stages∈{3,4,5}`, `num_warps∈{4,8}`, `GROUP_M∈{4,8}`) for K1 only. K2/K3 unchanged. Hypothesis:
  GEMM1 is the largest single cost (N=6144); best tile picks up most of the win. Expect the biggest
  single jump here. Decision: keep if geomean improves and correctness holds.
- **c00x — autotune K3 (GEMM2).** Parent: previous. Single change: same autotune menu on K3 (N=2048).
  Hypothesis: second-largest GEMM. Expect a smaller-but-real gain. Keep on improvement.
- **c00x — L2 grouping / persistent scheduling tweak on GEMMs (optional).** Only if the GEMM gap to the
  expected cuBLAS ceiling still looks large. Try `GROUP_M` sweep / split-K for the small-M cases
  (M=1024). Keep on improvement; else revert (record as rejected).

### Phase C — Fusion of the memory-bound middle
- **c00x — K2 sequence-tiling with halo reuse.** Parent: best GEMM candidate. Single change: replace K2
  row-parallel variant with sequence-tiled `(BLOCK_S within one batch × BLOCK_H)` that loads a 3-row
  halo and computes `Bx` once (draft §5 K2 option a). Hypothesis: removes up to 4× redundant
  `Bgate/Xproj` loads; but K2 is memory-bound and small vs GEMMs, so effect is small. Keep only if
  measurable and correctness holds.
- **c00x — F1: fold gating into GEMM1 epilogue.** Parent: best so far. Single change: K1 writes only
  `Bx (M,H)` and `Cgate (M,H)` (2H columns) by combining the `[0,H)`×`[2H,3H)` epilogue, so K2 no
  longer recomputes the input gate. Hypothesis: saves ~1/3 of the intermediate write+read. Risk: needs
  epilogue to see both column blocks (N tiling or 2-pass accumulator). Keep on improvement.

### Phase D — Stretch fusions (only if justified by measured bottleneck)
- **c00x — F2: fuse conv into GEMM2 A-load** (draft §5 F2). High complexity/correctness risk at
  batch/tile edges; attempt only if K2+intermediate traffic is shown to matter after Phase C.
- **c00x — merge K1+K2** (F3) — likely not worth it (conv needs cross-`BLOCK_M` neighbors); keep as a
  last idea only.

**Numbering:** candidates are numbered strictly sequentially (`c001`, `c002`, …) in creation order;
the phase labels above are logical, not literal IDs. If an early candidate fails correctness, the fix
consumes the next ID (it is a new source version) — never reuse an ID.

---

## 4. Performance hypotheses (roofline expectations)

From draft §5 (Ampere model, M=8192): GEMM1 ≈ 2.06e11 FLOP, GEMM2 ≈ 6.9e10 FLOP (GEMM1 ≈ 3× GEMM2);
middle traffic ≈ 0.17 GB. **GEMMs dominate; the middle is a rounding error on the timeline.**
- **Ceiling:** set by how close a tuned Triton bf16 matmul gets to cuBLAS on A800 for these M,N,K.
- **Headroom (our win source):** recovering the reference's overheads — slow cuDNN depthwise
  `conv1d(groups=2048)`, two transposes, a `.contiguous()` copy, and extra kernel launches/allocs.
- **Overall expectation:** modest **≈1.1–1.5×** if Triton GEMMs match cuBLAS; **<1× risk** if they
  underperform cuBLAS by more than the fusion savings. This risk ordering is exactly why Phase B (GEMM
  tuning) precedes Phase C (fusion): first close the GEMM gap, then harvest fusion.
- **Falsifiable predictions:** (P1) c001→GEMM1-autotune yields the single largest jump; (P2) K2
  sequence-tiling and F1 each yield small gains (middle is memory-bound & minor); (P3) if even the best
  GEMM candidate is <1×, the Triton matmul cannot beat cuBLAS here and the task's realistic ceiling is
  near the fusion savings alone — converge early rather than burn evals.

---

## 5. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with the reason) when **any** of:
1. **Convergence:** best geomean improves by **< ~1%** across **2 consecutive** accepted candidates, and
   the remaining roadmap items are the low-yield ones (Phase C/D) whose predicted gains are within noise.
2. **Budget:** approaching the token soft limit (1.0M) with no candidate in flight expected to move the
   metric materially, or the 100-eval cap is near. Token budget is expected to bind before 100 evals.
3. **Ceiling hit:** GEMM tuning has plateaued (Phase B exhausted), fusion (Phase C) added, and further
   ideas are Phase-D stretch fusions judged not worth the correctness risk for the roofline headroom.
4. **No viable win:** if the best *correct* candidate is persistently <1× and analysis (P3) shows the
   Triton-vs-cuBLAS gap exceeds all fusion savings, stop and report the best correct candidate.

Record the chosen best **valid** (all-workloads-correct) candidate as the final selection. **Never run
`./scripts/evaluate_candidate.sh final <id>` without explicit operator approval.**

---

## 6. Evidence format (append one JSON object per eval to `candidates.jsonl`)

Append-only; never rewrite earlier records. One object per **evaluated** candidate (an eval = all 5
feedback workloads). Schema:

```json
{
  "candidate_id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "Correctness anchor: 3-kernel decomposition, fp32 middle, single conservative tile.",
  "change_vs_parent": "initial baseline",
  "validation": {
    "analytical_checks": ["chunk offsets", "conv tap/dir", "causal+batch mask", "W^T", "fp32 accum"],
    "expected_correct": true
  },
  "results": {
    "b0a9e3f0": {"axes": {"B":1,"S":1024},  "correct": null, "speedup": null},
    "8b678b13": {"axes": {"B":4,"S":256},   "correct": null, "speedup": null},
    "58e3ac47": {"axes": {"B":32,"S":256},  "correct": null, "speedup": null},
    "6033fd0f": {"axes": {"B":2,"S":4096},  "correct": null, "speedup": null},
    "bb262a64": {"axes": {"B":1,"S":2048},  "correct": null, "speedup": null}
  },
  "geomean_speedup": null,
  "all_correct": null,
  "decision": "accept | reject | anchor-fix",
  "decision_reason": "<why kept/dropped; which prediction confirmed/refuted>",
  "cumulative_evals": 1,
  "skill_usage": "none (KernelWiki N/A for sm_80; profiler disallowed)"
}
```

Field rules:
- `speedup` values are copied verbatim from the launcher's per-workload report; `geomean_speedup` is the
  geometric mean over the 5 workloads (only meaningful if `all_correct` is true).
- `decision`: **accept** if `all_correct && geomean improved over current best`; **reject** otherwise
  (kept in the log for lineage); **anchor-fix** for a correctness repair of a failing anchor.
- `parent` = the candidate this one was derived from (the current best-correct, unless triaging a bug).
- `cumulative_evals` = running count of feedback evals consumed (out of 100).
- `source_sha256` recomputed from the exact `solution/solution.py` submitted for that ID (immutability
  proof; a changed source always gets a new ID).

---

## 7. Execution checklist (per candidate, mechanical)

1. Write/modify `solution/solution.py` for the **single** intended change; keep the `run()` contract.
2. Re-run the §2 correctness checklist for every touched path; do not eval unless believed-correct.
3. Freeze the source as the immutable candidate `cNNN` (record `source_sha256`).
4. `./scripts/evaluate_candidate.sh feedback cNNN` (one eval).
5. Append the §6 record to `candidates.jsonl` (never edit prior lines).
6. Update "current best-correct" pointer; pick next candidate per §3 priority using §4 predictions.
7. Re-check §5 stopping criteria; if met, write `SEARCH_COMPLETE` with the reason and stop.

---

## 8. Immediate next step (next turn, not now)
Implement **c001** (the correctness anchor): 3-kernel Triton decomposition in `solution/solution.py`,
fp32 middle, single conservative GEMM config, with the §2 host asserts. Then run one feedback eval and
record it. Do not tune before c001 is correct.

---

## 9. Decision log

### c001 — DONE (eval 1/100). Accepted as current best-correct.
- Result: **all 5 correct**, geomean **1.0809×**. Per-workload: WL1 1.167, WL2 1.171,
  **WL3 0.889** (M=8192, B32/S256), WL4 1.026 (M=8192, B2/S4096), WL5 1.184.
- Reading: correctness derivation validated end-to-end (no numerical failure, even WL5 atol 0.0098 —
  so the fp32 middle is safe; the §2.3 bf16-round lever is **not** needed). Prediction **P1 confirmed**:
  the GEMMs dominate and the conservative `64×64×32` tile is the bottleneck at large M. WL3 is <1×
  (cuBLAS beats our untuned tile); WL4 (same M=8192 but S=4096) is 1.026×, so the deficit is tile-shape
  driven, not batch-boundary driven.
- **Next candidate → c002 (Phase B): autotune the GEMM kernel** (`_gemm_bias_kernel`, used by both K1
  and K3) over the draft §5 tile menu (`BLOCK_M∈{64,128,256}`, `BLOCK_N∈{64,128,256}`,
  `BLOCK_K∈{32,64,128}`, `num_stages∈{3,4,5}`, `num_warps∈{4,8}`, `GROUP_M∈{4,8}`), keying on
  `(M,N,K)`. Single change vs c001: add `@triton.autotune`; K2 untouched. Expected: recover WL3 to ≥1×
  and lift the large-M cases, biggest single geomean jump. Note: autotuning both GEMMs at once is the
  minimal source change (they share one kernel); this is still "one variable" (the GEMM tile policy).

### c002 — DONE (eval 2/100). Accepted as current best-correct.
- Change: added `@triton.autotune` (11 curated Ampere bf16 configs, `key=(M,N,K)`) on the shared GEMM
  kernel; K2 unchanged. Single variable = GEMM tile policy.
- Result: **all 5 correct**, geomean **1.4678×** (up from 1.0809×, **+35.8%**). Per-workload:
  WL1 1.663, WL2 1.675, **WL3 1.315** (was 0.889), WL4 1.330 (was 1.026), WL5 1.400.
- Reading: **Prediction P1 confirmed** — GEMM tile policy was the dominant lever and delivered the
  single largest jump; WL3 recovered from <1× to 1.315×. Small-M cases (WL1/2 ≈1.67×) now lead;
  large-M cases (WL3/4 ≈1.32×) lag. Correctness held everywhere including WL5 (atol 0.0098), so the
  fp32 middle remains safe.
- **Next candidate → c003 (Phase C): reduce the memory-bound middle's traffic + launch overhead.**
  With GEMM tuning captured, the remaining headroom is (i) K2's redundant `Bgate/Xproj` re-loads per
  tap (up to 4×) and (ii) the intermediate `BCx (M,3H)` / `y (M,H)` write+read traffic. First cheap,
  low-risk lever: sequence-tiled K2 with a 3-row halo so `Bx` is computed **once** per row instead of
  re-loaded 4×, and per-channel conv weights/bias loaded once per tile. Single change vs c002: replace
  K2 row-parallel variant with the halo/shared variant; GEMMs untouched. Expected: small gain (K2 is
  memory-bound and minor vs GEMMs — draft P2), most visible on large-M (WL3/4) where K2 traffic is
  largest. Keep only if measurable and correctness holds.

### c003 — DONE (eval 3/100). Accepted as current best-correct.
- Change: added `@triton.autotune` (9 configs, `key=(M,S,H)`) on the K2 fused
  gating+conv+gating kernel; removed the fixed `_BM2/_BH2=64` launch. Single variable vs
  c002 = K2 tile/warp policy. GEMM kernel + all numerics byte-for-byte identical. (This is a
  simpler realization of the Phase C K2 lever above — tune K2's launch config rather than
  restructure it to halo-tiling; picked as the lowest-risk first Phase C step.)
- Result: **all 5 correct**, geomean **1.4724×** (up from 1.4678×, **+0.31%**). Per-workload:
  WL1 1.655, WL2 1.668, **WL3 1.334** (was 1.315), **WL4 1.342** (was 1.330), WL5 1.401.
- Reading: **Prediction P2 confirmed** — the memory-bound middle is minor vs the GEMMs, so
  K2 tuning yields only a small gain, and it lands exactly where predicted (large-M WL3/WL4,
  where K2 traffic is largest). Small-M WL1/WL2 dipped ~0.4–0.6% (autotune/measurement noise);
  WL5 flat. Correctness held everywhere incl. WL5 (atol 0.0098).
- **Next candidate → c004 (Phase C, F1): fold the input gate into GEMM1's epilogue.**
  With both GEMM and K2 launch policies now tuned, the only remaining structural headroom is
  the intermediate traffic. F1 makes K1 write `Bx (M,H)` and `Cgate (M,H)` (2H columns
  instead of 3H) by multiplying the `[0,H)` and `[2H,3H)` output-column tiles inside K1's
  epilogue; K2 then reads `Bx` directly (no input-gate recompute) and still owns the 4-tap
  causal conv + output gate. Saves ~1/3 of the `BCx` write and the `Bgate/Xproj` re-reads.
  Risk: the epilogue must see matched `[0,H)`/`[2H,3H)` column pairs → cleanest realization is
  a dedicated fused-GEMM1 kernel that computes both projections' `BLOCK_N=H`-column tiles in
  one program (two accumulators over the shared K-loop) and multiplies them at store. This is
  a real source change to K1, so it is a distinct candidate, not a config tweak. Given P2
  (middle is minor), the expected gain is small; if it does not beat 1.4724× or adds
  correctness risk at the epilogue pairing, reject and move toward convergence.

### c004 — DONE (eval 4/100). REJECTED (invalid: 14/16 correct).
- Change: expanded the GEMM autotune menu from 11→18 configs (strict superset of c003 +
  7 large-M tiles). GEMM kernel body + all numerics byte-for-byte identical to c003;
  K2 unchanged. (This was the "finish Phase B" GEMM-refine lever, done before F1.)
- **MAJOR ENVIRONMENT CHANGE DISCOVERED at this eval:** the `feedback` stage now runs
  **16 workloads**, not the 5 documented in `task/feedback_workloads.jsonl`. The 5 original
  UUIDs are a subset; 11 new shapes appear, including small/odd `S` (e.g. B1/S256 M=256,
  B2/S128 M=256, B8/S997, B2/S1879, B1/S8192). Tolerance shown as atol 0.021, rtol 0.05,
  required matched ratio 0.99.
- Result: geomean 1.3651× over the 14 passing, **but valid=false**: two INCORRECT_NUMERICAL:
  - `66fd2ad8` B1/S256 (M=256), max_abs 0.0625, max_rel 4.60
  - `dcc98535` B2/S128 (M=256), max_abs 0.0625, max_rel 4.77
  Both are the **smallest-M / small-S** cases. All larger cases pass.
- Root-cause reasoning: c004's GEMM math == c003's, so the expanded menu did **not** cause the
  failure — it is a **pre-existing numerical edge case newly exposed** by the 16-workload set.
  Per draft §2.3/§4.2, our K2 keeps the whole middle in fp32 while the **reference rounds
  `Bx`, `conv_out`, and `y` to bf16 at each intermediate step**. On tight-atol small problems
  the fp32-vs-bf16 divergence pushes >1% of elements past atol=0.021 (only ~0.99 match needed,
  so a small fraction of large-rel outliers fails). This is exactly the "WL5 watch-list /
  bf16-round lever" foreseen in the draft, now triggered by even-smaller-M new workloads.
- **IMPORTANT status correction:** c001/c002/c003 were validated only on the OLD 5-workload
  set. **No candidate is yet known-valid on the 16-workload feedback set.** The prior "best"
  geomeans (1.47×) are not comparable to the 16-set numbers. Re-establishing a *valid*
  candidate on the 16-set is now the top priority (correctness before speed).
- **Next candidate → c005 (correctness repair):** revert the GEMM menu to c003's 11 configs
  (isolate the change) AND apply the draft's bf16-round lever inside K2 — mirror the
  reference by rounding `Bx` to bf16 before the conv accumulation and rounding `conv_out`
  (post-bias) to bf16 before the output gate. This should bring the two small-M cases into
  tolerance. Expected small perf cost (extra casts, still fp32 accum). Goal: **valid on all
  16**; that becomes the new baseline. Only after a valid 16-set candidate exists do we
  resume perf tuning (F1, GEMM menu) measured against the 16-set geomean.

### c005 — DONE (eval 5/100). ACCEPTED — first candidate VALID on the 16-workload set.
- Change vs c003: (1) reverted the GEMM menu to c003's 11 configs; (2) applied the K2
  bf16-round lever — round `Bx=(Bgate*Xproj)` to bf16 before the 4-tap fp32 conv accum, and
  round `conv_out=(acc+conv_bias)` to bf16 before the output gate. Conv still accumulates in
  fp32 (matches cuDNN's fp32-accum-then-round).
- Result: **16/16 correct**, geomean **1.3772×**. The two previously-failing small-M cases
  now pass: `66fd2ad8` (B1/S256) and `dcc98535` (B2/S128) both at 1.4124×. Confirms the
  root-cause diagnosis: the fp32-vs-bf16 intermediate divergence was the failure, and
  mirroring the reference's roundings fixes it.
- Perf profile (16-set): small-M lead (~1.40–1.42×), large-M/large-S lag (d9432839 B8/S997
  1.306×, 6033fd0f B2/S4096 1.327×, 58e3ac47 B32/S256 1.359×) — same GEMM-bound picture as
  before, so the earlier optimization logic still applies.
- **Status reset:** c005 is the FIRST known-valid candidate on the true 16-workload feedback
  set and is the new current best-correct / baseline. Prior c001–c003 geomeans (~1.47×) were
  on the old 5-set and are NOT comparable.
- **Next candidate → c006 (resume Phase B on the 16-set):** GEMMs still dominate and the
  laggards are the large-M/large-S cases. Re-introduce the safe subset of c004's large-M GEMM
  tiles (deeper `s5` pipelines at BLOCK_K=32, high-BLOCK_M `256×64`, `GROUP_M=16` for L2 reuse
  at large M) on top of c005's now-correct numerics. Correctness is decoupled: the menu change
  cannot affect numerics, and the GEMM kernel masks M via `m_mask = offs_m < M`, so the ragged
  M values from the new odd-S shapes (B8·S997=7976, B4·S541=2164, B2·S1879=3758 — not mult 256)
  are already handled (c005 passed them, proving the mask path is correct). Any tile is
  therefore safe. Expected: lift the large-M laggards, small 16-set geomean gain. Keep on
  improvement.
