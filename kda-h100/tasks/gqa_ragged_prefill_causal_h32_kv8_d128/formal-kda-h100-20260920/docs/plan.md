# Plan — `gqa_ragged_prefill_causal_h32_kv8_d128`

Executable, sequential KDA optimization plan. Target: H100 (`sm_90`, Hopper).
Primary implementation: Triton. Metric: geometric-mean speedup over reference,
subject to every selected workload passing correctness.

This plan operationalizes `docs/draft.md`. It defines: the solution contract, the
candidate lineage, per-candidate procedure, correctness checks, performance
hypotheses, stopping criteria, and the evidence/record format.

---

## 0. Ground rules (from CLAUDE.md / TASK.md — binding)

- Submission is `solution/solution.py` exposing `run(q, k, v, qo_indptr, kv_indptr,
  sm_scale) -> (output, lse)`. Triton does all the math; PyTorch only for metadata /
  launch plumbing. **No** Torch/CPU/NumPy/CUDA-extension computational fallback.
- Evaluate **only** with `./scripts/evaluate_candidate.sh feedback <cNNN>`. The full
  feedback set = **one** evaluation. Budget = 100 evaluations. Token soft limit 9.0M,
  normal 10.0M, absolute 11.0M.
- Candidates are **immutable and sequential**: `c001`, `c002`, … Any meaningful
  source/config/launch change ⇒ new id. Never reuse an id for changed source. Never
  rewrite earlier `candidates.jsonl` records.
- Profiling only via `./scripts/ncu_profile.sh …` (ncu-report-skill workflow), and
  **never concurrently with an evaluation** (foreign process on the locked GPU ⇒
  controller discards measurement, return code 3, one eval burned). Serialize
  strictly: eval → (optional) profile → next eval.
- Do not touch evaluator/dataset/controller/launcher/config; do not change the fixed
  feedback workloads; do not run CUDA/nvidia-smi/the external evaluator directly.
- `final` is operator-only; never run it without explicit approval.

---

## 1. Solution contract (stable across all candidates)

`solution/solution.py` must:
1. Accept exactly `(q, k, v, qo_indptr, kv_indptr, sm_scale)` with the dtypes/shapes
   in `task/definition.json` (q `[Tq,32,128]` bf16, k/v `[Tkv,8,128]` bf16, indptrs
   int32 on device, `sm_scale` fp32 scalar).
2. Allocate `output` bf16 `[Tq,32,128]` (zero-init) and `lse` fp32 `[Tq,32]`
   (init `−inf`), matching reference initialization so skipped/zero-length sequences
   are correct with no kernel work.
3. Compute the launch grid using **at most one** device→host transfer (e.g. a single
   `int(qo_indptr[-1])`/`max_seqlen` read, or a batched read of the whole indptr in
   one `.tolist()`); **no per-sequence `.item()` loop** (that is the reference's main
   overhead sink — see draft §2).
4. Launch the Triton kernel(s) and return `(output, lse)`. No host sync between launch
   and return beyond what the harness does.

Invariant helper choices fixed for the whole search (constexpr): `HEAD_DIM=128`,
`NUM_QO_HEADS=32`, `NUM_KV_HEADS=8`, `GQA_RATIO=4`, `BLOCK_D=128`. `sm_scale` is always
a runtime scalar (two near-identical values appear in the set — never hardcode).

---

## 2. Candidate lineage strategy

Principle: **correctness first, then one variable at a time.** Each candidate changes
exactly one axis from its parent so the feedback delta is attributable. Escalate to
GPU-efficiency work only after the baseline is correct and only where profiling shows
it pays. The geomean is **tiny-case dominated** (17/21 workloads have `total_q ≤ 92`),
so guard every large-case optimization against tiny-case regressions.

Planned lineage (IDs are reserved; later branches may be pruned/re-pointed based on
evidence — the tree is a guide, not a commitment):

- **c001 — baseline correct fused flash kernel.** V1 dense grid
  `(num_seq, 32, cdiv(max_seqlen, BLOCK_M))`; `BLOCK_M=128, BLOCK_N=64, num_warps=8,
  num_stages=2`; bf16 `tl.dot` + fp32 accum; online base-2 softmax/LSE; causal
  upper-triangle skip; GQA via `kv_head = q_head // 4`. Purpose: pass all 21 +
  establish baseline geomean. **Gate: must be 100% correct before any perf branch.**

- **c002+ — correctness remediation (only if c001 fails any workload).** One fix per
  id, in this diagnostic order (most likely first): (a) LSE base-2 constant / `log2(e)`
  folding; (b) causal off-by-one (`j <= i+delta` vs `< i+1+delta`); (c) GQA head map;
  (d) masked-tile `row_max_fixed` guard; (e) bf16-dot precision — bump QKᵀ to
  `input_precision="tf32x3"` or `"ieee"` (see §5). If c001 is fully correct, skip to
  the perf branch and keep the id numbering contiguous.

- **Perf branch A — tile/launch sweep (small, cheap, high-confidence).** From the best
  correct baseline, sweep one axis per id: `num_stages ∈ {1,2,3}`, `num_warps ∈
  {4,8}`, `(BLOCK_M,BLOCK_N) ∈ {(128,64),(64,64),(128,128),(64,128)}`. Keep the winner.

- **Perf branch B — size-bucketed launch config.** One immutable kernel, host-side
  selection of `(BLOCK_M,BLOCK_N,num_warps,num_stages)` from a `total_q`/`max_seqlen`
  bucket (tiny → minimal-overhead config; large → pipelined config). This is a launch
  change ⇒ new id. Hypothesis: recovers tiny-case launch overhead without hurting
  large cases (draft §4.4).

- **Perf branch C — ragged scheduling V2 (flattened tile schedule).** Precompute
  (torch, one device→host transfer) a linear `tile_id → (seq, m_block)` map so ragged
  multi-seq batches (#5,#6,#20) don't waste blocks. New id. Gate on branch A/B winner.
  Only pursued if profiling shows load-imbalance/tail on the large cases (draft §4.3).

- **Perf branch D — GQA K/V reuse V3.** One program computes all 4 query heads sharing
  a kv head, loading each K/V tile once (4× less K/V global traffic). New id. Watch
  `sm_90` register pressure/occupancy; only pursued if branch A–C leave the large
  cases K/V-bandwidth bound (draft §4.3, §6 Q4).

Branch ordering rationale: A and B are cheap and help the dominant tiny cases; C and D
are more invasive and help only the 3 large cases, so they run later and must clear a
"no tiny-case regression" gate.

---

## 3. Per-candidate execution procedure (repeat for each `cNNN`)

1. **State hypothesis** (one sentence: what changes vs parent, expected effect, which
   regime it targets).
2. **Write source** as a new immutable version of `solution/solution.py` (one axis
   changed from parent). Do not edit a previously-evaluated candidate's source.
3. **Static self-check** against the reference behaviors in §4 (read-through, no
   execution) before spending an evaluation.
4. **Evaluate once**: `./scripts/evaluate_candidate.sh feedback cNNN`. This is the only
   correctness/timing oracle and counts as one of the 100 evaluations.
5. **Record** one complete JSON object appended to `candidates.jsonl` (schema §7).
   Never modify earlier records.
6. **Decide** keep / revert / branch per §6 stopping-and-decision rules; set the parent
   of the next candidate accordingly.
7. **(Optional) Profile** only if a perf decision needs it, via `./scripts/ncu_profile.sh`,
   strictly after the evaluation completes (never concurrent). Use `ncu-report-skill`
   to read the report; record findings in the next candidate's `notes`.

Rule: at most one in-flight change per id; if a fix requires touching two things, split
into two candidates so the feedback signal stays attributable (unless they are logically
inseparable, in which case note that in the record).

---

## 4. Correctness checks (static, applied before every evaluation)

Verify by inspection that the kernel reproduces the reference (`definition.json`):
1. **Causal boundary**: query row `i` (0-based in-sequence) attends to kv col `j` iff
   `j < i + 1 + delta`, `delta = kv_len − q_len`. All feedback data has `delta=0`
   ⇒ `j <= i`; implement general `delta` anyway.
2. **GQA map**: `kv_head = q_head // 4` (equiv. `q_head // GQA_RATIO`).
3. **Scale placement**: `sm_scale` multiplies raw QKᵀ *before* max/exp; fold
   `sm_scale·log2(e)` into one scale feeding `exp2`.
4. **Base-2 LSE**: `lse[q,h] = m2 + log2(l2)` where `m2`,`l2` are the base-2 running
   max/denominator of the scaled, masked logits (≡ `logsumexp(natural)/ln2`). fp32.
5. **Softmax/accum precision**: max, exp, rescale, denom, and the V-accumulator all in
   **fp32**; `p` cast to bf16 only as the `tl.dot(p, v)` operand.
6. **Output**: fp32 accumulator `acc / l`, cast to bf16 once at store.
7. **Masked-tile guard**: `row_max_fixed = where(row_max==−inf, −1e20, row_max)` so a
   fully-masked KV *tile* doesn't poison the running max (draft §3.3).
8. **Empty/zero-length sequences**: contribute no program work; `output=0`, `lse=−inf`
   preserved from init (draft §3.4). Guard final divide against `deno==0` defensively.
9. **No forbidden constructs**: no `repeat_interleave`/materialized expanded K/V, no
   full `logits` materialization, no computational fallback, no hardcoded `sm_scale`.

The evaluator's per-workload pass/fail is the authority; these static checks exist to
avoid wasting evaluations on avoidable mistakes.

---

## 5. Performance hypotheses (each tied to a candidate/branch and a check)

| # | Hypothesis | Target regime | Candidate | How confirmed |
|---|-----------|---------------|-----------|---------------|
| H1 | A single fused, host-sync-free Triton kernel beats the reference broadly, dominated by eliminating Python loop / `.item()` syncs / fp32 casts / `repeat_interleave` / logits materialization. | tiny (most workloads) | c001 | c001 geomean ≫ 1 |
| H2 | Correct bf16 `tl.dot` + fp32 accum is within evaluator tolerance on all 21. | all | c001 | c001 correctness 21/21 |
| H3 | If H2 fails on any workload, bumping QKᵀ precision (`tf32x3`/`ieee`) restores tolerance at modest large-case cost. | precision-sensitive | c002+ | correctness recovers, geomean cost bounded |
| H4 | Causal upper-triangle skip roughly halves QKᵀ/PV work on the 3 large cases. | large | c001 (built-in) | ncu FLOP/time on #5,#6,#20 |
| H5 | `num_stages`/`num_warps`/tile sweep improves large-case throughput; minimal-overhead config improves tiny-case latency. | large / tiny | branch A | per-workload speedup deltas |
| H6 | Size-bucketed launch config nets a geomean gain by cutting tiny-case launch overhead without large-case regression. | tiny (+neutral large) | branch B | per-regime speedup, no tiny regressions |
| H7 | Flattened ragged schedule (V2) removes load-imbalance/tail waste on multi-seq large cases. | large multi-seq (#5,#6) | branch C | ncu SM-util/tail before vs after |
| H8 | GQA K/V reuse (V3) cuts K/V bandwidth 4× on large cases, if register pressure permits occupancy. | large | branch D | ncu DRAM/L2 traffic + occupancy |

Evidence for H4/H7/H8 requires profiling — gathered via `ncu_profile.sh` only, and
only after the relevant evaluation, never concurrently.

---

## 6. Stopping and decision rules

Per-candidate decision:
- **keep** (new best) if geomean improves **and** correctness stays 21/21.
- **keep-neutral / note** if within noise (Δgeomean ≲ 1–2%) but strictly correct and
  useful as a base for a branch.
- **revert** if correctness regresses on any workload, or geomean drops materially; set
  next candidate's parent to the prior best.
- A large-case win that **regresses tiny cases** into a net geomean loss is a revert.

Search-level stopping (write `SEARCH_COMPLETE` with the reason when any holds):
1. **Convergence**: 3 consecutive kept candidates each improve geomean by < ~1%
   (diminishing returns) and remaining branches are lower-expected-value.
2. **Budget**: approaching the 100-evaluation cap or the token soft limit (9.0M) —
   stop with the best valid candidate recorded, leaving margin before 10.0M.
3. **Exhaustion**: all planned branches (A–D) evaluated and no further attributable
   gain hypothesis remains.

On stop: identify the single best valid candidate (highest geomean with 21/21
correctness), note it in `SEARCH_COMPLETE`, and **do not** run `final` — that requires
explicit operator approval.

---

## 7. Evidence / record format (`candidates.jsonl`, one JSON object per candidate)

Append exactly one line per evaluated candidate; never rewrite prior lines. Required
fields (superset of CLAUDE.md item 7):

```json
{
  "id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py as evaluated>",
  "hypothesis": "one-line change + expected effect + target regime",
  "change_from_parent": "single axis changed (e.g. num_stages 2->3)",
  "config": {"BLOCK_M":128,"BLOCK_N":64,"num_warps":8,"num_stages":2,"grid":"V1-dense",
             "qk_precision":"bf16","launch":"static"},
  "validation": {"correct": true, "workloads_passed": 21, "workloads_total": 21,
                 "failures": []},
  "per_workload": [
    {"uuid":"<uuid>","axes":{"total_q":13557,"len_indptr":26},"correct":true,"speedup":<x>}
  ],
  "geomean_speedup": <x>,
  "decision": "keep | keep-neutral | revert",
  "cumulative_evaluations": 1,
  "skills_used": ["KernelWiki", "ncu-report-skill?"],
  "notes": "profiling findings, precision escalations, next-parent choice"
}
```

Conventions:
- `source_sha256` records the exact evaluated source so an id is never reused for
  changed source.
- `per_workload` keyed by the workload `uuid` from `feedback_workloads.jsonl`, tagged
  with the regime (tiny/small/large) so tiny-vs-large effects are visible.
- `cumulative_evaluations` tracks budget consumption (out of 100).
- `skills_used` logs KernelWiki / ncu-report-skill usage per CLAUDE.md item 7.
- Speedups are the evaluator-reported per-workload numbers; `geomean_speedup` is the
  metric the evaluator reports / that ranks candidates.

---

## 8. Progress log / next action

- **c001 — DONE, KEPT (new best).** Evaluated 21/21 correct, geomean **6.99x**
  (arith 10.55x, min 3.95x, max 39.04x). H1/H2/H4 confirmed. Key finding: the 17
  tiny/micro cases (`total_q ≤ 92`) all sit at ~0.095–0.105ms **regardless of size**
  ⇒ dominated by a **fixed per-launch cost**, clustering at ~3.95–4.6x; large cases
  (#5/#6/#20) hit 28.7–32.1x, #17 (982 tok) 39x. Geomean is tiny-case dominated.

- **Next action — c002 (Perf branch B, size-bucketed launch config).** Highest-value
  move: cut the ~0.095ms fixed launch cost on the tiny/micro cases with a smaller,
  lower-overhead launch config selected host-side from a `total_q`/`max_seqlen` bucket
  (tiny → smaller `BLOCK_M`, fewer `num_warps`, `num_stages=1`; large → keep the
  current pipelined `BLOCK_M=128, num_warps=8, num_stages=2`). One immutable kernel,
  launch selection only ⇒ new id. Gate: no tiny-case regression, large cases neutral.
  Parent: c001. Static-check per §4, evaluate once, record per §7. Do not profile
  during evaluation; do not batch candidates.

- **c002 — DONE, KEPT (new best).** Instead of size-bucketing configs, the cheapest
  attributable branch-B move was tried first: **remove the `int(q_lens.max().item())`
  device→host sync** for `total_q ≤ BLOCK_M` (every seq fits one m-block ⇒
  `n_m_blocks=1`, no sync). Kernel body byte-identical to c001. Result: 21/21 correct,
  geomean **6.99 → 10.27x**. Tiny/micro sol time dropped ~0.095 → 0.060ms; tiny
  speedups ~4 → 6.3–7.3x; #21 (92 tok) 16.97 → 26.83x (now single-block sync-free).
  Large cases unchanged (still 1-sync path). Confirms the tiny cases were host-sync
  bound, not GPU bound.

- **Next action — c003.** Tiny cases still ~0.060ms ⇒ residual fixed cost = single
  kernel launch + the two `zeros`/`full` init kernels for output/lse. Since the packed
  ragged layout has every output/lse row owned and written by exactly one program
  (under `mask_m`; zero-length seqs contribute no rows), switch to `torch.empty` (no
  memset) to drop 2 kernel launches per call. Launch/alloc change only ⇒ new id,
  parent c002. Gate: 21/21 correct (verify uninitialized rows can never be read — they
  can't, since each row is fully written), no regression. If c003 shows diminishing
  returns, pivot to the large-case GPU-efficiency frontier (branch A tile/stages/warps
  sweep on #5/#6/#20).

- **c003 — DONE, KEPT (new best).** `torch.empty` for output/lse (no memset). 21/21
  correct, geomean **10.27 → 11.71x**. Tiny/micro sol ~0.060 → 0.049ms; #21 (92 tok)
  26.83 → 31.82x. Large cases within noise. Host/launch axis now near its floor:
  gains are decaying (c001→c002 +3.28, c002→c003 +1.44) and tiny cases (~0.049ms) are
  close to irreducible single-launch dispatch latency.

- **Next action — c004 (Perf branch A, num_stages sweep).** The remaining untapped
  frontier is the 3 large cases (#5/#6/#20, ~29–33x, sol ~1.05ms) which are
  GPU-efficiency bound. Try `num_stages=3` (deeper TMA/wgmma pipeline overlap on
  Hopper; KernelWiki `technique-pipeline-stages`) to raise large-case throughput.
  Config change only ⇒ new id, parent c003. Gate: no tiny-case regression from higher
  smem/register pressure; keep only if geomean improves. **Caveat for expectations:**
  geomean is tiny-dominated (17/21 near floor), so even a large 30→40x jump on the 3
  large cases moves the geomean only modestly — weigh remaining eval budget vs. the
  small expected geomean delta when deciding whether to continue past branch A.

- **c004 — DONE, KEPT (new best, marginal).** `num_stages=2→3`. 21/21 correct, geomean
  **11.71 → 11.93x** (+0.22). Confirmed the hypothesis: compute-bound cases improved
  (#5 31.54→34.36, #6 33.44→36.42, #20 29.48→31.55, #17 42.80→44.08, #4 14.57→15.15);
  tiny/micro cases ~neutral (8.2–8.55x, some cross-run timing noise but ratio held).
  Returns are clearly decaying: c002→c003 +1.44, c003→c004 +0.22.

- **Next action — c005 (Perf branch A, BLOCK_N sweep).** `BLOCK_N=64→128` to halve the
  KV-loop trip count on the large cases (more wgmma per iteration, better pipeline
  amortization). Smem check: K+V bf16 [128,128] tiles ≈ 64KB, fits Hopper's 228KB.
  Config change only ⇒ new id, parent c004. Gate: tiny cases neutral (single tile), no
  correctness regression, keep only if geomean improves. **Convergence watch:** with
  the geomean tiny-dominated and near its launch floor, per-step gains are now ≲0.2.
  If c005 (and any one further branch-A knob, e.g. num_warps) do not yield a
  meaningful gain, declare convergence and write SEARCH_COMPLETE rather than spending
  more of the budget for sub-noise deltas.

- **c005 — DONE, REVERTED.** `BLOCK_N=64→128`. 21/21 correct but geomean **11.93 →
  11.87x (−0.06)** — a net regression. Large cases were neutral-to-slightly-negative
  (#5 34.36→33.44, #20 31.55→31.12, #17 44.08→42.05; only #6 up 36.42→36.78). At
  BLOCK_N=64 the large-case KV loop was already latency-hidden by num_stages=3;
  doubling BLOCK_N adds per-iter register/smem pressure (bigger [128,128] qk/p tile)
  that offsets the fewer iterations. Confirms **c004 (BLOCK_M=128, BLOCK_N=64,
  num_warps=8, num_stages=3) is the branch-A optimum** among {stages, BLOCK_N}. Best
  remains **c004 @ 11.93x**.

- **Convergence assessment.** Gains have plateaued: c002→c003 +1.44, c003→c004 +0.22,
  c004→c005 −0.06. The geomean is tiny-dominated; the 17 launch-bound cases are pinned
  at their ~0.05ms single-launch floor (constant across c003/c004/c005 ⇒ no launch-side
  headroom), and the large cases are pipeline-saturated. One last low-risk probe is
  justified before declaring done:

- **Next action — c006 (Perf branch A, num_warps).** `num_warps=8→4` on parent **c004**
  (revert BLOCK_N to 64 first). Hypothesis: fewer warps may lower per-CTA
  scheduling/launch cost on the dominant tiny cases without materially hurting the
  large cases. Config change only ⇒ new id, parent c004. Gate: keep only if geomean
  beats 11.93x. **If c006 does not beat c004's 11.93x, the search has converged →
  write SEARCH_COMPLETE naming c004 as the best valid candidate.** Do NOT branch
  further from c005.

- **c006 — DONE, REVERTED.** `num_warps=8→4`. 21/21 correct but geomean **11.93 →
  11.79x (−0.14)**. Clear net-negative tradeoff: tiny cases improved slightly (#18
  8.38→10.14, #19 8.55→9.22) but the 4 compute-bound cases dropped sharply (4 warps
  under-feed wgmma: #5 34.4→28.5, #6 36.4→28.3, #20 31.6→25.3, #17 44.1→37.9). The
  large-case losses are large multiplicative geomean factors and outweigh the small
  tiny gains. Confirms `num_warps=8` is correct.

- **SEARCH COMPLETE.** Both remaining branch-A knobs regress (c005 BLOCK_N=128 −0.06,
  c006 num_warps=4 −0.14); host axis exhausted (tiny cases at launch floor), GPU axis
  saturated (num_stages=3 optimal). No positive-EV next candidate remains (branches C/D
  rejected on analysis — see SEARCH_COMPLETE). Best valid candidate = **c004 @ 11.93x,
  21/21 correct**. `solution/solution.py` restored to the c004 config. `SEARCH_COMPLETE`
  written. Final evaluation is operator-only and has NOT been run. 6/100 evaluations used.
