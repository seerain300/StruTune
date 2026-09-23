# Plan — L1/092 GQA Attention with QK-Norm (GLM-4.5-Air block), H100 / sm_90

## 0. Scope

Executable, sequential candidate roadmap derived from `docs/draft.md`. This turn
writes the plan only — no code, no evaluation. Everything here is subordinate to
`CLAUDE.md` (isolation, Triton-only compute, immutable sequential candidates,
one-eval-per-full-16-workload-set, 100-eval budget, token limits 9M/10M/11M,
profiling never overlapping evaluation, `final` operator-only).

Key facts carried from the draft:
- One call = QKV proj (+bias) → fp32 QK-RMSNorm(d=128) → NeoX RoPE(split 64) →
  GQA causal SDPA (96 q / 8 kv, 12× repeat, scaling 1/√128) → O-proj (no bias).
- 16 feedback workloads, M=B·S ∈ [128, 8192]; non-pow2 S ∈ {293, 373}.
- Dominant win: kill the reference's fp32 `attn_weights` HBM traffic (up to ≈6.4 GB
  on (2,2048)) via fused FlashAttention; secondary wins: fuse RMSNorm+RoPE, QKV
  concat, GQA K/V reuse, competitive bf16 GEMMs.
- All four matmuls **must** be Triton (`tl.dot`, fp32 accum). sm_90 ⇒ wgmma path.

---

## 1. Guiding principles

1. **Correctness before speed.** c001 is a provably-equivalent baseline whose only
   job is to pass all 16 tolerances. Every later candidate inherits its math, so a
   numerics bug found late is cheap to localize.
2. **One variable per candidate.** Each new ID changes exactly one thing (a fusion,
   a mapping, a config set). A pass/fail + speedup delta then maps to one cause.
3. **Structural-equivalence proof gates every eval.** Before spending an eval, walk
   the candidate against draft §1 stages + §4 hazard checklist (reproduced in §4
   below). No eval on an un-argued candidate — a wasted eval is the expensive
   failure mode, not a slow kernel.
4. **Profile to choose the next lever, don't guess.** After the baseline passes,
   use `ncu_profile.sh` on a 3-shape subset to decide which optimization candidate
   comes next. Never profile while an eval runs.
5. **Keep a warm JIT / autotune cache within the process** and bucket autotune keys
   by M so the 16 shapes don't each trigger fresh long autotunes inside one eval.
6. **Never introduce a Torch/CPU/NumPy compute fallback.** If a Triton kernel fails,
   fix the kernel or revert to the last good candidate; do not paper over it.

---

## 2. Candidate lineage (sequential roadmap)

Each entry lists: parent, the single change, the hypothesis, and the pass/keep gate.
IDs are allocated only when implemented; later IDs may be re-planned based on
evidence. The roadmap is a priority order, not a fixed commitment past c001–c003.

### Phase A — correctness anchor

- **c001 — staged, correct Triton baseline (parent: none).**
  - QKV: Triton tiled bf16 GEMM (fp32 accum) + bias epilogue → q `[B,S,12288]`,
    k/v `[B,S,1024]`. Simple, small curated autotune config list keyed on M.
  - Norm+RoPE: one elementwise Triton kernel per tensor, grid over (token, head);
    fp32 RMSNorm reduction over 128, bf16 weight promoted to fp32, single bf16
    round, then NeoX rotate-half RoPE using cos/sin[(b, pos), :128] verbatim.
  - Attention: standard Triton FlashAttention (online softmax, fp32 accum),
    causal early-exit, diagonal-block `j>i` compare, key-col mask `j≥S`, query-row
    mask `i≥S`, GQA mapping kv_head = q_head // 12, scaling folded into scores,
    `0/0`/`-inf`-row guards.
  - O-proj: Triton tiled bf16 GEMM (fp32 accum), no bias, over K=12288.
  - **Hypothesis:** passes all 16 tolerances; likely already ≥1× on high-S shapes
    (flash removes fp32 score traffic) even before tuning.
  - **Gate:** all 16 pass. If any fail, next ID fixes only the failing math
    (do not add speed changes until green).

### Phase B — structural fusions (largest expected wins)

- **c002 — QKV-concat GEMM (parent: c001).** Build one concatenated weight
  `[12288+1024+1024, 4096]` and bias once at setup (Torch `cat`, plumbing only);
  single GEMM produces q|k|v, then slice by column range for norm/rope.
  - **Hypothesis:** fewer launches + LHS (`hidden`) reused across all out-features
    → faster projections, esp. small-M/memory-bound shapes.
  - **Gate:** all 16 pass; geomean ≥ c001 (accept if ≥, keep best).
- **c003 — fuse RMSNorm+RoPE into the QKV epilogue (parent: best of c001/c002).**
  Choose BLOCK_N=128 aligned to head boundaries so each output tile owns one full
  head; do the fp32 RMSNorm reduction and RoPE in the GEMM epilogue, storing
  already-normed-and-roped q/k. Eliminates stage-2 kernel and a q/k HBM round-trip.
  - **Hypothesis:** removes 2 elementwise passes + their HBM traffic; net speedup
    on all M, biggest on small-M.
  - **Gate:** all 16 pass; geomean ≥ parent.
- **c004 — GQA K/V reuse in FlashAttention (parent: best so far).** Process the 12
  q-heads of a kv-group cooperatively (pack group into M or loop q-heads inside one
  program) so each kv-head's K/V blocks load once instead of 12×.
  - **Hypothesis:** cuts attention K/V HBM traffic ~12×; largest win on high-S
    shapes (2048) where K/V streaming dominates.
  - **Gate:** all 16 pass; geomean ≥ parent (watch register/SMEM pressure).

### Phase C — profile-guided tuning (evidence-driven, order set by ncu)

- **c005 — GEMM config/tuning** (block sizes, num_stages, num_warps, split-K for
  small-M) informed by ncu MMA-utilization + memory analysis.
- **c006 — FlashAttention config/tuning** (BLOCK_M/N, num_stages, num_warps) for
  d=128 causal, informed by ncu occupancy/HBM.
- **c007 — boundary-shape specialization** if 293/373 or tiny (1,128) / huge
  (32,256)/(2,2048) underperform (e.g. dedicated small-M path, launch-overhead
  reduction, autotune bucket split).
- **c008+ — deeper fusion** (attention output → O-GEMM), persistent/stream-K GEMM,
  or reverting a regressed change — each as its own single-variable candidate.

Phases B/C are re-prioritized after each eval + profile; the roadmap is a ranked
backlog, not a fixed sequence past c001.

---

## 3. Lineage & bookkeeping rules

- IDs strictly sequential `c001, c002, …`; **new ID for any meaningful source,
  config, or launch change**; never reuse an ID for changed source.
- Exactly one `solution/solution.py` version per candidate; record its source hash
  (e.g. `sha256` of the file) in the candidate record before evaluating.
- Append **one** JSON object per evaluated candidate to `candidates.jsonl`; never
  edit or rewrite an earlier record.
- `parent` links each candidate to the record it was derived from; the "best valid"
  pointer for the eventual (operator-approved) `final` is tracked in the decision
  field, not by mutating history.
- Keep changes minimal and reversible: if a candidate regresses, the next candidate
  branches from the last good parent, not from the regressed one.

---

## 4. Correctness checks (run mentally before every eval)

Reproduced from draft §4 as the pre-eval gate checklist. A candidate is not
evaluated until every item is argued true:

1. **RMSNorm fp32:** cast→fp32, `mean(x²)` over 128 in fp32, `rsqrt(var+1e-5)`,
   multiply by fp32-promoted bf16 weight, round to bf16 **before** RoPE.
2. **RoPE:** applied after RMSNorm; rotate-half split at 64 with signs
   `rot[:64]=-x[64:]`, `rot[64:]=x[:64]`; cos/sin indexed by (batch,pos), used
   verbatim across all 128 lanes; broadcast over heads. (fp32-internal RoPE + single
   round is allowed; it is strictly ≥ reference accuracy.)
3. **Scores/softmax:** scaling folded in; online max-subtracted softmax in fp32;
   PV matches "probs→bf16 then bf16 matmul, fp32 accum"; guard masked-row `0/0`.
4. **Causal + boundary:** query i sees keys j≤i; mask key cols j≥S and query rows
   i≥S; diagonal block does per-element compare; verify explicitly for S∈{293,373}.
5. **GQA mapping:** kv_head = q_head // 12 (contiguous group = fast-repeat axis).
6. **Bias:** present on q/k/v (added in fp32 epilogue, before RMSNorm); **absent**
   on o_proj.
7. **Accumulation:** all `tl.dot` fp32 accum (esp. O-GEMM over K=12288).
8. **Layout/output:** kernels agree on physical q/k/v layout; final output is
   contiguous bf16 `[B,S,4096]`; defensively `.contiguous()` inputs in plumbing.

"Runs-but-wrong" pitfalls to double-check every time: GQA head mapping, RoPE
order/split/duplication, bias placement, masked-row NaN, autotune picking a
shape-invalid block for S∈{293,373}.

**No local numeric diff is available** (Bash limited to the two launchers); the
evaluator tolerances (atol 0.0025–0.0051, rtol 0.05) are the only correctness
oracle. Hence the pre-eval proof is mandatory, not optional.

---

## 5. Performance hypotheses (falsifiable, tied to candidates)

| # | Hypothesis | Candidate | Expected signal | Falsified if |
|---|---|---|---|---|
| H1 | Fused flash attn removes multi-GB fp32 score traffic → big win on high-S | c001 | (2,2048),(1,2048),(4,1024) fastest speedups | high-S not improved |
| H2 | QKV-concat cuts launches + reuses LHS → helps small-M | c002 | (1,128),(4,128),(2,256) improve | small-M flat/worse |
| H3 | Epilogue-fused norm/rope removes 2 HBM passes → broad win | c003 | geomean up across all M | no geomean gain |
| H4 | GQA K/V reuse cuts attn K/V traffic 12× → high-S win | c004 | (2,2048),(1,2048) improve; no reg spill | occupancy/spill regresses |
| H5 | GEMM tuning lifts MMA util toward cuBLAS at large M | c005 | (32,256),(8,512),(4,1024) improve; ncu MMA% up | MMA% already saturated |
| H6 | Boundary/tiny shapes are launch-bound, need special path | c007 | (1,128),(293),(373) improve | already competitive |

Each hypothesis is confirmed/refuted by the eval's per-workload speedups plus, where
relevant, an ncu metric (MMA utilization, DRAM throughput, achieved occupancy).

---

## 6. Profiling protocol (ncu-report-skill)

- Invoke only via `./scripts/ncu_profile.sh <ncu args…>`; follow the
  `ncu-report-skill` workflow for metric selection and report reading.
- **Never overlap with an evaluation.** Sequence: finish eval → (optionally)
  profile → analyze → implement next candidate → eval. No background profiling.
- Representative 3-shape subset to bound profiling cost/time:
  - **(1,128)** — small-M, weight-streaming / launch-overhead bound.
  - **(4,512)** — balanced.
  - **(2,2048)** — long-S, attention-traffic bound.
- Metrics of interest: per-kernel duration split (which of the 4 stages dominates),
  GEMM MMA/tensor pipe utilization, DRAM throughput, achieved occupancy,
  launch/gap overhead on small-M. Use findings to pick between c005/c006/c007.

---

## 7. Evidence format (candidates.jsonl)

One JSON object appended per evaluated candidate (append-only). Fields:

```json
{
  "id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "Correct staged Triton baseline; passes all 16 tolerances.",
  "change_from_parent": "initial implementation",
  "validation": {
    "pre_eval_checklist": "all 8 correctness items argued true",
    "all_pass": true,
    "num_workloads": 16
  },
  "per_workload": [
    {"uuid": "c92bb0a5-...", "B": 1, "S": 512, "passed": true, "speedup": 0.0}
  ],
  "geomean_speedup": 0.0,
  "decision": "keep | discard | revert-to <id>",
  "decision_reason": "…",
  "cumulative_evals": 1,
  "skills_used": ["KernelWiki", "ncu-report-skill?"],
  "profiled": false,
  "notes": "…"
}
```

Rules:
- `speedup` = reference_time / candidate_time per workload (as reported by the
  evaluator); `geomean_speedup` = geometric mean over the 16 (all must pass to be
  a valid ranking).
- `cumulative_evals` increments by 1 per full-16-workload feedback eval; it must
  stay ≤ 100.
- If a workload fails correctness, record `passed:false` and the observed error, set
  `all_pass:false`, geomean invalid, `decision:"discard"` (or fix in next ID).
- `skills_used` records KernelWiki / ncu-report-skill consultation for that step.
- Never overwrite; each record is immutable once written.

---

## 8. Stopping criteria & completion

Stop the search and write `SEARCH_COMPLETE` (with reason) when any holds:
1. **Convergence:** best geomean improves by < ~2% across two consecutive
   accepted candidates and remaining backlog hypotheses are exhausted or
   ncu-refuted.
2. **Budget:** approaching the 100-eval cap or the token soft limit (9M) — leave
   margin; do not sail past 10M.
3. **Diminishing returns vs. risk:** further fusion adds correctness risk
   disproportionate to expected geomean gain (documented in the reason).

On completion: identify the best **valid** (all-16-pass) candidate as the `final`
nominee, record it, and **wait for explicit operator approval** before ever running
`./scripts/evaluate_candidate.sh final <id>`. Do not run `final` autonomously.

---

## 9. Progress log

- **c001 — DONE, KEPT (eval 1/100).** All 16 pass, geomean **2.1858x**, avg
  2.2675x. Current best-valid / `final` nominee. H1 confirmed: biggest wins on
  high-S small-B — (2,2048)=4.03x, (1,2048)=3.75x, (4,1024)=2.84x. Weakest on
  small-M/high-batch — (16,128)=1.52x, (16,256)=1.69x, (32,256)=1.86x,
  (4,128)=1.83x — i.e. projection/launch bound, exactly the regime c002/c003
  target.

- **c002 — DONE, KEPT (eval 2/100).** Single change: M-selected GEMM tile (M>=1024
  uses wide Hopper tile 128x256x64/warps8/stages3; M<1024 keeps c001 tile). All 16
  pass, geomean **2.1858x -> 2.6254x (+20%)**, avg 2.7520x. New best-valid / `final`
  nominee. H5 confirmed: large-M jumped ((2,2048) 4.03->5.07x, (1,2048)
  3.75->4.77x, (32,256) 1.86->2.65x, (16,256) 1.69->2.31x); small-M unchanged as
  designed. Weakest now: (4,128)=1.83x, (2,256)=1.93x, (16,128)=1.98x, (4,293)=2.03x,
  (1,512)=2.20x — all small-M (<1024), projection launch/traffic bound.

- **c003 — DONE, KEPT (eval 3/100).** Single change: added a mid-M GEMM tile bucket
  (256<=M<1024 -> 128x128 tile, warps8, stages3); other buckets unchanged. All 16
  pass, geomean **2.6254x -> 2.6743x (+1.9%)**, avg 2.7829x. New best-valid /
  `final` nominee. Targeted M=512 shapes improved: (1,512) 2.20->2.44x, (2,256)
  1.93->2.12x, (4,128) 1.83->2.01x, (2,512) 2.43->2.59x. Minor within-noise dips on
  some large-M shapes ((2,2048) 5.07->5.00x, (1,2048) 4.77->4.60x) — net positive.
  GEMM-tile lever is now plateauing (+1.9%); further tile tweaks unlikely to move
  geomean much.

- **c004 — DONE, DISCARDED / reverted to c003 (eval 4/100).** Single change:
  FlashAttention tile 64x64/warps4/stages2 -> 128x64/warps8/stages3. All 16 pass but
  geomean **2.6743x -> 2.6597x (-0.5%)** — small net regression, so reverted; c003
  stays best-valid / `final` nominee. Per-shape mixed: (1,1024) 2.94->3.02x up, but
  (1,512) 2.44->2.36x, (2,512) 2.59->2.48x, (4,512) 2.65->2.59x down. Lesson:
  BLOCK_M=128 halves the query-tile program count (grid=ceil(S/128)*B*H), hurting
  occupancy on the small/mid-S shapes that dominate the geomean; K/V-amortization
  doesn't outweigh it. **The flash-attn tile is not the bottleneck at these shapes** —
  attention is only dominant at very long S. Working source restored to c003 logic.

**Immediate next step (next turn): PROFILE, then a structural change (c005).** Two
consecutive tuning candidates now bracket the plateau: c003 (+1.9%) and c004
(-0.5%). Pure launch-config tuning is exhausted. Before the next candidate:
  1. **Profile** with `ncu_profile.sh` on (4,128) small-M, (4,512) balanced,
     (2,2048) long-S to get the real per-stage time split (QKV GEMM vs norm/rope vs
     flash-attn vs o-proj). NEVER overlap with an evaluation.
  2. **c005 (branch from c003)** = the highest-leverage *structural* change the
     profile points to. Leading candidates: (a) **GQA K/V reuse** — attention loads
     each kv-head's K/V 12x (once per q-head); making a program serve all 12 q-heads
     of a kv-group cuts K/V HBM traffic ~12x (best on long-S). (b) **Fuse
     RMSNorm+RoPE into the QKV GEMM epilogue** + **QKV-concat** to cut launches and
     q/k HBM round-trips (best on small-M). Gate: all 16 pass and geomean > 2.6743x;
     else revert to c003.
  3. If the profile shows no single stage dominates (time already spread thin and
     close to memory/compute rooflines across stages), that is evidence of
     convergence — consider `SEARCH_COMPLETE` rather than more low-yield candidates.

### Skills usage note
- **KernelWiki** already consulted (`lang-triton`: sm_90 ⇒ wgmma path, tcgen05/TMEM
  is sm_100-only; `kernel-flash-attention-4`: FA-4's tcgen05/2-CTA/software-exp are
  B200-only — use the standard Triton fused-attention pattern for H100 d=128).
- **ncu-report-skill** to be used per §6 during phases B/C.
