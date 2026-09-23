# Plan — L1/070 Mamba2 Fused Intra-Chunk Diagonal Computation

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`. This turn writes the
plan only — no candidate is implemented or evaluated here.

Entry point: `solution/solution.py` exposing `run(hidden_states, A_cumsum, B, C) -> Y_diag`.
Fixed axes: `L=128, H=32, P=128, Gp=8, N=128`; only `batch b` and `num_chunks nc` vary.
Per-tile math (fixed `(b,c,h)`, `g=h//4`): `G = C_h·B_hᵀ`; `S = incl.cumsum(a)`;
`Ldec[i,j]=exp(S[i]−S[j])` for `i≥j` else `0`; `M = G·Ldec`; `Y = M·X_h`; store bf16.

---

## 0. Operating rules (bind every step)

- **Triton primary**; Torch only for shape/stride/launch. No Torch/CPU/NumPy/CUDA-ext/alternate
  computational fallback. A failing Triton kernel is invalid — fix it or revert, never substitute.
- **Evaluate only** with `./scripts/evaluate_candidate.sh feedback cNNN`. `final` is
  operator-only; never run it without explicit approval.
- **One immutable candidate = one source version = one full-feedback-set evaluation.** Any
  meaningful source/config/launch change ⇒ new candidate ID. Never mutate or reuse an ID.
- After each eval, **append exactly one JSON record** to `candidates.jsonl` (§7 schema); never
  rewrite earlier records.
- **Profiling only** via `./scripts/ncu_profile.sh ...` (ncu-report-skill). Never run
  `ncu`/CUDA/`nvidia-smi`/the evaluator directly. **Never profile while an evaluation is running**
  (foreign process on the locked GPU ⇒ return code 3, discarded timing, one eval wasted).
- Correctness authority is the official evaluator; **do not** build a private GPU comparison
  harness (that is an alternate correctness harness).
- Budgets: 100 evaluations; tokens 9M soft / 10M normal / 11M absolute. Track cumulative eval
  count in every record.

---

## 1. Candidate lineage strategy

Single active lineage; each candidate is a minimal, attributable delta over a named parent so
speedups are causally isolated. **Change exactly one thing per candidate.** Promote a candidate
to "parent for next" only if it is correct on all 14 workloads and does not regress geomean;
otherwise revert to the last good parent and branch a different axis.

Planned sequence (IDs are reserved; later IDs may be re-purposed to the next untried axis if an
earlier hypothesis is settled — the *content*, not the number, defines the candidate):

- **c001 — Correct fused baseline.** Per-`(b,c,h)` grid, `BLOCK_M=128` (whole chunk), single
  `j`-pass, two `tl.dot`s (bf16 in, fp32 acc), exact decay formula. `num_warps=4/8`, `stages=2/3`
  chosen conservatively (no autotune yet). Goal: pass all 14 + capture the fusion win. This is the
  reference geomean baseline for the whole search.
- **c002 — Query-row split + causal skip.** `BLOCK_M∈{64,32}`, grid `(b,nc,h,i_block)`, loop only
  `j`-blocks `≤` i-block (skip strict-upper triangle). Targets occupancy on small `b·nc` and
  ~25–40% matmul/exp reduction. Parent = best of c001.
- **c003 — Warps/stages tuning** over the c002 winner (hand-picked grid of `num_warps∈{4,8}`,
  `num_stages∈{2,3,4}`), or Triton `autotune` keyed on `(b,nc)` shape class if a single static
  config can't serve both tiny and large workloads.
- **c004 — Group-level fusion.** Grid `(b,nc,Gp)`; compute `G=C_g·B_gᵀ` once, inner-loop the 4
  heads (each own `a`,`X`,`Y`). Cuts `C·Bᵀ` 4× and `B/C` HBM reads 4×. Expected win on large
  workloads, possible loss on tiny (max 512 programs). Compare against best per-head candidate.
- **c005 — Precision/layout refinements.** Only if needed: fp32 (or TF32-off) second matmul if
  bf16-cast `M` breaks rtol on high-dynamic-range workloads (§4); block-ptr vectorization tweaks;
  ensure decay `exp` computed once per tile (not per `d`-tile).
- **c006+ — Combine winners / fine autotune.** Merge the best occupancy + group-fusion + config
  choices; final autotune pass. Continue only while geomean improves (§6).

Branching rule: if an axis regresses or breaks correctness, record it, revert parent, and move to
the next axis. Do not stack unexplained changes.

---

## 2. Per-candidate execution loop (repeat for each ID)

1. **State hypothesis** (what changes vs parent, expected effect, why) in the record draft.
2. **Implement** the single delta in `solution/solution.py`. Keep `L,N,P,H,Gp` as `tl.constexpr`;
   derive `b,nc` from tensor shapes at runtime (never hardcode `b`/`nc`). Guard launch plumbing
   (contiguity/dtype/strides) in Python.
3. **Pre-eval self-check** (static, no GPU): re-verify index algebra and the decay formula against
   draft §1.2–§1.3 (matmul orientations `C·trans(B)` and `M·X`; group `g=h//4`; inclusive-diagonal
   `i≥j`; `−inf` before `exp`). Confirm the file imports cleanly and exposes `run`.
4. **Evaluate once**: `./scripts/evaluate_candidate.sh feedback cNNN`. Ensure no profiling job is
   running first.
5. **Record**: append one JSON object to `candidates.jsonl` (§7). Note parent, source hash,
   hypothesis, per-workload pass/speedup, geomean, decision, cumulative eval count, skills used.
6. **Decide**: promote (new parent) / reject (revert) / branch. Update lineage.
7. **Optionally profile** the promoted candidate via `ncu_profile.sh` (never concurrent with an
   eval) to steer the next hypothesis.

---

## 3. Correctness checks (in priority order)

- **C1 — Paper equivalence (pre-eval, mandatory).** The derivation in draft §1.2–§1.3 is the
  correctness contract each kernel is written against: (a) `G[i,j]=Σ_n C_h[i,n]B_h[j,n]`;
  (b) `S` inclusive prefix sum, `Ldec=exp(S[i]−S[j])`; (c) mask `i≥j` (diagonal **included**),
  `−inf` on `i<j` *before* `exp`; (d) `Y=M·X_h`; (e) group `g=h//4`; (f) bf16 inputs → fp32
  accumulate → bf16 store.
- **C2 — Official evaluator (authoritative).** All 14 workloads must pass
  (`|Δ| ≤ atol + rtol·|ref|`, `atol=1e-5`, `rtol=0.05`). The rtol=5% gate is the effective
  criterion (bf16 output makes 1e-5 atol unreachable). A candidate that fails any workload is
  **invalid regardless of speedup** and cannot be promoted or submitted as `final`.
- **C3 — No fallback.** If a Triton kernel fails correctness, diagnose and fix (or revert to last
  good parent); never insert a Torch/CPU path to "pass".
- **C4 — Numerical levers** if C2 fails on specific workloads:
  - bf16-cast `M` too lossy under large `exp` range → fp32 / TF32-off second matmul (new ID).
  - Suspected overflow (`exp(S[i]−S[j])`) → keep `−inf`-before-`exp` masking; inspect via ncu /
    static analysis, not by altering the reference math (no row-max/softmax reformulation).
  - cumsum grouping differences are negligible at rtol=5% (draft §1.3) — not a lever.

---

## 4. Performance hypotheses (each testable by one candidate)

- **H1 (fusion, c001):** eliminating the reference's `[b,nc,128,128,H,P]` HBM temp makes the op
  BW-bound at ~670 MB (big case) vs tens of GB. Predict large geomean win (≈5–20×) from c001
  alone. *Test:* c001 geomean vs 1.0; ncu HBM bytes ≈ X+Y+B+C traffic.
- **H2 (occupancy, c002):** small `b·nc` workloads (≤256 tiles) underfill 132 SMs at
  `BLOCK_M=128`; splitting query rows raises grid size and speeds the many small workloads.
  *Test:* per-workload speedup deltas on `(1,2),(1,3),(4,1),(16,1)` vs c001.
- **H3 (causal skip, c002):** skipping strict-upper `j`-blocks removes ~25–40% of matmul+exp
  work. *Test:* speedup on large workloads `(4,16),(1,32)`; ncu instruction/FLOP drop.
- **H4 (warps/stages, c003):** better pipelining/occupancy of the two 128³ dots. *Test:* config
  sweep geomean; ncu `sm__throughput`, achieved occupancy.
- **H5 (group fusion, c004):** sharing `G` across 4 heads cuts one matmul 4× and B/C reads 4×;
  net win on large workloads, monitored for regression on tiny ones. *Test:* c004 vs best
  per-head, per-workload and geomean.
- **H6 (precision, c005):** fp32 second matmul only if needed for correctness; expect a small
  perf cost — accept only to satisfy C2.

Profiling protocol: profile one large (`b=4,nc=16`) and one small (`b=1,nc=2`) representative per
promoted candidate to confirm the regime (BW- vs occupancy- vs compute-bound) before the next
hypothesis. Always sequential with evaluations, never concurrent.

---

## 5. Kernel-structure decisions (fixed unless evidence overturns)

- **Single fused kernel** (both matmuls + decay in one launch); do not materialize `G`/`M` in HBM
  — that is the entire fusion win. Split into two kernels only if register/SRAM pressure proves it
  necessary (two 128×128 fp32 accumulators at `num_warps=8` is within the flash-attention
  envelope, so a single kernel is expected to hold).
- **No online softmax / running max**: the reduction is a single `j`-pass over one 128-chunk, so
  standard masked matmul suffices; introducing flash-style normalization would change rounding for
  no benefit.
- **Access pattern:** `tl.make_block_ptr` for coalesced loads — `X`/`Y` inner-128 (`d`) contiguous
  with `j`-stride `H·P=4096`; `B`/`C` inner-128 (`n`) contiguous with `i/j`-stride `Gp·N=1024`.

---

## 6. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when **any** holds:
- Geomean improvement over the best candidate is `< ~2%` across two consecutive new candidates
  (genuine convergence), **and** the remaining untried axes (§1) are exhausted or predicted
  immaterial.
- Evaluation budget reached (100) or approaching (leave margin for a possible re-eval).
- Token budget: wind down and finalize records before the 9M soft limit; hard stop well under
  10M normal / 11M absolute.
- No correct candidate improves on the best while multiple axes have been tried → declare the best
  correct candidate the winner.

The `SEARCH_COMPLETE` note names the best candidate ID, its geomean, per-workload status, and why
further search is not worthwhile. `final` is run **only** after explicit operator approval, once,
on the best valid candidate.

---

## 7. Evidence format (one JSON object per line in `candidates.jsonl`)

Append-only; never edit prior lines. Required fields:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "Fused single-kernel baseline; expect large fusion win vs reference temp.",
  "change_vs_parent": "initial implementation",
  "validation": {
    "static_checks": "paper-equivalence C1 verified (matmul orientation, decay S[i]-S[j], i>=j incl. diagonal, g=h//4)",
    "correctness": "official evaluator feedback set"
  },
  "per_workload": [
    {"uuid": "46445623-...", "axes": {"batch_size": 4, "num_chunks": 7}, "passed": true, "speedup": 0.0}
  ],
  "all_passed": true,
  "geomean_speedup": 0.0,
  "decision": "promote|reject|branch",
  "decision_reason": "…",
  "cumulative_evaluations": 1,
  "skills_used": ["ncu-report-skill?", "KernelWiki?"],
  "notes": "profiling findings, next hypothesis"
}
```

Rules: `source_sha256` binds the record to the exact evaluated source (compute over
`solution/solution.py`). `per_workload` covers all 14 feedback UUIDs with pass/fail + speedup.
`geomean_speedup` is over passing workloads; if any fail, mark `all_passed=false` and treat as
non-promotable. `cumulative_evaluations` increments by exactly 1 per evaluated candidate.
`skills_used` lists KernelWiki/ncu-report-skill invocations for that candidate.

---

## 8. Decision log

- **c001 — DONE, PROMOTED (parent).** Single fused per-`(b,c,h)` kernel, `num_warps=4`,
  `num_stages=2`. 14/14 correct; geomean **362.24x**, average 374.57x (1 cumulative eval).
  Confirms H1 (fusion). Weakest: `(1,2)`=187.6x, `(1,3)`=248.8x, `(4,1)`=302.2x — the small
  `b·nc` (occupancy-starved) cases, exactly as H2 predicted. Best: `(4,16)`=512.8x, `(1,32)`=472.3x.

- **c002 — DONE, REJECTED.** Query-row split `BLOCK_M=64`, grid `(b*nc, H, num_m_blocks)` with a
  causal `j`-loop `range(m_block+1)`. 14/14 correct but geomean **235.68x** — a regression on
  **every** workload vs c001 (362.24x), including the small ones it targeted (`(1,2)` 187.6→148.5,
  `(1,3)` 248.8→167.4). H2/H3 disproven at this tile size: c001's monolithic 128³ dots beat
  64-row split dots, and the one-hot `S` selection + `j`-loop overhead outweigh any occupancy /
  causal-skip gain. Reverted; c001 remains parent (2 cumulative evals).
  **Lesson:** keep the whole-128 M-tile; the small-workload gap is not fixable by row-splitting.

- **c003 — DONE, REJECTED.** c001 kernel unchanged, only `num_warps` 4→8. 14/14 correct but
  geomean **292.37x** — regression vs c001 (362.24x) on nearly every workload (tiny `(1,2)`/`(1,3)`
  ~flat). c001 at 4 warps is not register-spill-bound; 8 warps hurts. Reverted; c001 remains
  parent (3 cumulative evals). **Lesson:** H4 settled — `num_warps=4` is best for this tile.

- **c004 — DONE, REJECTED.** Group-level `G` reuse: grid `(b,nc,Gp)`, compute `C_g@B_g^T` once,
  inner-loop the 4 heads. 14/14 correct but geomean **143.18x** — large regression vs c001
  (362.24x). The 4× fewer programs starve occupancy on small/medium `b·nc` (`(4,1)` 302→69,
  `(1,2)` 188→38, `(1,4)` 303→70); even the biggest cases dip slightly. H5 disproven — the kernel
  is parallelism/BW-bound, not bound by the shared `G` matmul/reads. Reverted; c001 remains parent
  (4 cumulative evals).
  **Lesson:** maximize program count (per-`(b,c,h)` grid) — do not coarsen the grid.

## 9. Status & next actions

Four distinct axes now disproven — row-split (c002), warp count (c003), group-fusion (c004),
pipeline stages (c005) — all regress or are flat vs c001. c001's per-`(b,c,h)` grid maximizes
parallelism and its 128³ tiles are already efficient; the op is parallelism/BW-bound with many
tiny workloads. No further justified candidate remains that changes the winning grid+tile.

**SEARCH CONVERGED.** Best valid candidate = **c001**, feedback geomean **362.24x**, 14/14
correct. `solution/solution.py` restored to the c001 source. `SEARCH_COMPLETE` written. `final`
awaits explicit operator approval.

- **c005 — DONE, REJECTED.** c001 kernel+grid unchanged, only `num_stages` 2→3. 14/14 correct but
  geomean **358.48x** — flat/marginally below c001 (362.24x); helps mid cases slightly but hurts
  the largest (`(4,16)` 512.8→463.9, `(1,32)` 472.3→454.3, `(16,1)` 424.1→397.9). Within noise;
  `num_stages=2` stays best. Reverted; c001 is the winner (5 cumulative evals).
