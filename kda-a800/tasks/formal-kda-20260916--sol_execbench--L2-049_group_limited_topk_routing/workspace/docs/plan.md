# Executable Plan — L2/049 `group_limited_topk_routing` (A800 / sm_80, Triton)

Run ID: `formal-kda-20260916--sol_execbench--L2-049_group_limited_topk_routing`
Depends on: `docs/draft.md` (operation analysis, numerical risks, design space). This plan is the
*executable* layer: file contract, kernel spec, candidate lineage, correctness gates, perf
hypotheses, stopping rules, and evidence schema. **No code or evaluation is produced in this step.**

---

## 0. Ground rules (restated from CLAUDE.md / TASK.md)

- Submission: `solution/solution.py` exposing `run(hidden_states, weight, expert_bias, routed_scaling_factor)`.
- Primary compute **must** be Triton; PyTorch only for allocation/stride/launch plumbing. No Torch /
  CPU / NumPy / CUDA-extension / alternate computational fallback. A failed Triton kernel is invalid.
- Evaluate **only** via `./scripts/evaluate_candidate.sh feedback cNNN`. No direct CUDA, profiler,
  `nvidia-smi`, torch REPL, or alternate harness. Each eval = 5 fixed feedback workloads = 1 candidate.
- Candidates are immutable: `c001, c002, …`. Never mutate an evaluated ID or its source; any meaningful
  source/config/launch change ⇒ new ID. `candidates.jsonl` is append-only.
- Budget: 100 evals; tokens soft 1.0M / normal 1.5M / hard 1.65M. `final` only with operator approval.
- Ranking: geometric-mean speedup **among candidates that pass correctness on all 5 workloads**.

---

## 1. Reference contract (the exact math to reproduce)

From `task/definition.json`. Constants: `E=256`, `top_k=8`, `n_group=8`, `topk_group=4`,
`experts_per_group=32`, `D=4096`. Only `num_tokens=T` varies (feedback: 2240, 2272, 6144, 2048, 2112).

Per token (all epilogue math in **fp32**):
1. `logits = hidden_states.f32 @ weight.f32.T`  → `[T,256]`.
2. `scores = sigmoid(logits)`  → `[T,256]`.   ← **returned weights come from this (no bias)**
3. `scores_routing = scores + expert_bias.f32`  → `[T,256]`.  ← **selection uses this (with bias)**
4. Reshape `[T,8,32]`; per group take **top-2** values, sum → `group_scores[T,8]`.
5. Select **top-4** groups by `group_scores` → mask `[T,8]`.
6. Expand mask to `[T,256]`; `masked = scores_routing.masked_fill(mask==0, f32.min)`.
7. Select **top-8** experts over `masked` → `topk_idx[T,8]`.
8. `selected = gather(scores, topk_idx)`  (bias-free scores).
9. `topk_weight = selected / (selected.sum(-1,keepdim) + 1e-20)`.
10. `topk_weight *= routed_scaling_factor`.

Outputs: `topk_idx[T,8]` int64, `topk_weight[T,8]` fp32.

**Two score arrays** must coexist: `scores_routing` (drives all three selections) and `scores`
(supplies returned weights). `sorted=False` everywhere ⇒ positional order of the 8 outputs is not
semantically meaningful (see §4 assumption A1).

---

## 2. Kernel architecture (single fully-fused kernel — the c001 baseline)

Grid: `grid = (ceil(T / BLOCK_M),)`. Each program owns `BLOCK_M` tokens × all 256 experts, so the
routing epilogue runs entirely in-register with no `logits` DRAM round-trip.

Kernel signature (planned):
```
routing_kernel(
    hs_ptr, w_ptr, bias_ptr,           # inputs (bf16, bf16, bf16)
    idx_ptr, wt_ptr,                   # outputs (int64, fp32)
    scaling,                           # fp32 scalar (routed_scaling_factor)
    T, D,                              # runtime dims
    stride_hs_m, stride_hs_k,
    stride_w_e, stride_w_k,
    stride_idx_m, stride_idx_n,
    stride_wt_m, stride_wt_n,
    BLOCK_M: constexpr, BLOCK_K: constexpr,
    E: constexpr = 256, NG: constexpr = 8, EPG: constexpr = 32,
    TOPK: constexpr = 8, TOPG: constexpr = 4,
)
```

Per-program body (planned pseudo-Triton; all epilogue tiles fp32):
1. `offs_m = pid*BLOCK_M + arange(BLOCK_M)`; `m_mask = offs_m < T`.
2. `acc = zeros([BLOCK_M, 256], f32)`. K-loop over `D` step `BLOCK_K`:
   load `hs[BLOCK_M,BLOCK_K]` bf16, `w[256,BLOCK_K]` bf16, `acc = tl.dot(hs, w.trans(), acc)`
   (bf16 operands, fp32 accumulate). Weight (2 MB) is L2-resident and reused across token tiles.
3. `scores = tl.sigmoid(acc)`; `scores_routing = scores + bias[None,:]`.
4. **Group top-2 sum** (vectorized over all 8 groups): `g = reshape(scores_routing,[BLOCK_M,8,32])`;
   `m1 = max(g,axis=2)`; `a1 = argmax(g,axis=2)`; `g2 = where(iota32==a1[...,None], -inf, g)`;
   `m2 = max(g2,axis=2)`; `group_scores = m1 + m2`  → `[BLOCK_M,8]`.
5. **Top-4 groups**: `gs = group_scores`; `group_sel = zeros([BLOCK_M,8],f32)`; repeat 4×:
   `gi = argmax(gs,axis=1)`; `group_sel = where(iota8==gi[:,None], 1.0, group_sel)`;
   `gs = where(iota8==gi[:,None], -inf, gs)`.
6. Expand: `sel256 = reshape(broadcast(group_sel[:,:,None],[BLOCK_M,8,32]),[BLOCK_M,256])`;
   `masked = where(sel256>0, scores_routing, -inf)`.
7. **Top-8 experts** + companion bias-free score capture: `denom=0`; repeat i in 0..7:
   `ei = argmax(masked,axis=1)`  → expert index in `[BLOCK_M]`;
   `sel_i = sum(where(iota256==ei[:,None], scores, 0.0), axis=1)`  (bias-free score of chosen expert);
   `masked = where(iota256==ei[:,None], -inf, masked)`;
   store `ei` into `topk_idx[:,i]`, stash `sel_i`, `denom += sel_i`.
8. `denom += 1e-20`; for each i: `topk_weight[:,i] = sel_i/denom * scaling`.
9. Store with `m_mask`: `idx` cast to int64, `weight` fp32.

Notes on Triton mechanics:
- Use 3D reshape + axis reductions for the group stage (n_group=8 handled in one vectorized pass).
- Companion-value capture via masked sum avoids dynamic register indexing (Triton lacks gather along
  a reduced axis). Each of the 8 iters is a couple of 256-wide reductions — negligible vs. the GEMM.
- `-inf` sentinel is equivalent to reference `f32.min` for max selection; sigmoid scores feeding the
  weights are always real (never the sentinel), so no NaN can leak into normalization.

Fallback architecture (only if fused kernel is occupancy-bound — see c-series contingency): two-kernel
split (tuned GEMM → `logits[T,256]` DRAM; routing kernel reads it). Costs an extra 6 MB write+read and
a launch; used only if the fused accumulator cripples the GEMM.

---

## 3. Candidate lineage strategy

- **Linear/tree lineage, one variable per candidate.** Every candidate records `parent`. c001 is the
  correctness-first baseline (parent=none). Perf candidates branch from the best *correct* ancestor.
- **Separate correctness-affecting from perf-only changes.** Precision ladder / selection-logic edits
  never share a candidate with tiling edits, so any regression is attributable.
- **Immutability.** Once a `cNNN` is evaluated, its `solution/solution.py` is frozen conceptually; a
  changed source gets the next ID. I keep a per-candidate copy/hash so lineage is auditable.
- **Promote-the-best-correct.** After each eval, the current champion = highest geomean among all
  fully-correct candidates. New perf candidates fork from the champion's config.
- **Kill fast.** A candidate failing any workload's correctness is discarded for ranking; if it fails
  due to precision, spawn one precision-escalation child, not a tiling child.

---

## 4. Key assumptions (resolved cheaply by c001 feedback)

- **A1 — order-insensitive output comparison.** Because the reference uses `torch.topk(..., sorted=False)`,
  the position of each of the 8 outputs is implementation-defined; a faithful evaluator therefore
  compares the *set* of (expert, weight) pairs or a dense `[T,256]` scatter, not positional order.
  The plan produces correct (idx, weight) pairs and does **not** rely on matching torch's internal
  order. Contingency C-ORDER (§9) if this proves false.
- **A2 — bf16-HMMA fp32-accum is numerically faithful** (draft §3.2: bf16 inputs make products exact
  in fp32; accumulation differs only by summation associativity ≪ atol 0.6). Expected to pass all 5.
  Contingency C-PREC (§9): tf32 → tf32x3 → ieee escalation if match ratio falls short.
- **A3 — scalar `routed_scaling_factor`** passed as fp32 python float (=2.5 in all feedback). Read from
  the JSONL `scalar` value; passed to kernel as compile-time-irrelevant runtime fp32.

---

## 5. Candidate roadmap (concrete, sequential)

Configs are hand-picked to keep each eval interpretable (no broad autotune that would blur attribution
or vary across the 5 workloads). One change per step.

| ID | Parent | Change (single variable) | Config | Hypothesis / goal |
|----|--------|--------------------------|--------|-------------------|
| c001 | — | Fused kernel, bf16-HMMA fp32-accum, fp32 iterative-argmax epilogue | `BLOCK_M=32, BLOCK_K=64, warps=4, stages=3` | Pass correctness on all 5; establish baseline geomean speedup vs. multi-launch reference. |
| c002 | best(c001) | Tiling: raise BLOCK_M | `BLOCK_M=64, BLOCK_K=64, warps=8, stages=3` | Larger token tiles improve GEMM efficiency if RF allows; test accumulator spill threshold. |
| c003 | best | Tiling: BLOCK_K sweep | `BLOCK_M=champ, BLOCK_K=128, warps=…, stages=…` | Deeper K tiles reduce loop overhead / improve HMMA utilization. |
| c004 | best | Tiling: small M / high occupancy | `BLOCK_M=16, BLOCK_K=64/128, warps=4, stages=4` | If c002/c003 spill, more, smaller tiles + L2 weight reuse may win via occupancy. |
| c005 | best | stages/warps micro-tune around champion | vary `num_stages∈{2,3,4}`, `num_warps∈{4,8}` | Pipeline depth vs. RF pressure trade-off. |
| c006+ | best | Epilogue reduction fusion (draft §4.4) | champion tiling | Track running max1/max2 per group in one pass; fold group-mask expansion; keep both score arrays resident — shave epilogue cost once GEMM tiling is settled. |
| contingency C-PREC | failing branch | precision escalation | `tf32`→`tf32x3`→`ieee` | Only if a candidate misses the 0.98 match ratio. |
| contingency C-SPLIT | if fused occupancy-bound | two-kernel GEMM + routing | tuned GEMM | Only if fused accumulator demonstrably crippling GEMM (inferred from tiling-candidate speedups). |
| contingency C-ORDER | if A1 false | sorted output / dense-match adaptation | — | Only if evaluator penalizes output ordering. |

Roadmap is adaptive: exact c003+ configs are chosen from observed c001/c002 results. Each real
change consumes exactly one ID; skipped/无-effect ideas are folded, not evaluated separately.

---

## 6. Correctness checks

### 6.1 Pre-evaluation static gate (before every candidate submission)
Re-derive the candidate's source line-by-line against §1 and draft §3. Refuse to submit unless all hold:
- [ ] Two score arrays present: `scores` (sigmoid only) for weights; `scores_routing` (+bias) for all 3 selections.
- [ ] Bias added exactly once, before group stage; **never** included in returned weights.
- [ ] Group stage sums **top-2 of 32** per group (m1+m2 with correct single-position suppression for m1).
- [ ] Exactly `topk_group=4` groups selected; mask expanded 8→256 with correct group→expert layout (`view[T,8,32]`).
- [ ] Non-selected experts sentinel = -inf (≡ f32.min for argmax); no NaN path into normalization.
- [ ] Exactly `top_k=8` experts selected; companion score is bias-free `scores` at the chosen index.
- [ ] Normalization `sum + 1e-20`, then `* routed_scaling_factor`, all fp32.
- [ ] Output dtypes/shapes: `topk_idx[T,8]` int64, `topk_weight[T,8]` fp32; tail M-mask applied on store.
- [ ] Epilogue math entirely fp32; only `tl.dot` multiplies use tensor cores.
- [ ] Triton owns all compute; PyTorch only allocates/launches; no fallback path exists in the file.
- [ ] Source hash differs from every prior evaluated candidate (no ID reuse).

### 6.2 Evaluation-time gate
- Run `./scripts/evaluate_candidate.sh feedback cNNN` (once). A candidate is **correct** only if all 5
  workloads report pass under their tolerances (`max_atol≈0.56–0.65`, `max_rtol=0.01`, `match_ratio≥0.98`).
- Any workload failing correctness ⇒ candidate is discarded for ranking; diagnose (precision vs. logic)
  and route to the appropriate contingency child rather than tweaking tiling.
- Correctness is gated **before** speed: a fast-but-incorrect candidate never becomes champion.

---

## 7. Performance hypotheses (falsifiable)

- **H1 (fusion win).** Collapsing the reference's GEMM + ~10 elementwise/topk/scatter/gather launches
  into one kernel (no `logits[T,256]` DRAM round-trip) yields ≥ the multi-launch baseline; expected
  meaningful geomean speedup. Measured by c001 geomean vs. reference.
- **H2 (GEMM-bound).** Runtime is dominated by the `[T,4096]×[4096,256]` gate GEMM; routing reductions
  are negligible. ⇒ tiling of the GEMM (BLOCK_M/BLOCK_K/warps/stages) is the primary perf lever.
  Falsified if epilogue-fusion candidates (c006+) move the needle more than tiling candidates.
- **H3 (accumulator pressure).** The `[BLOCK_M,256]` fp32 accumulator limits occupancy at large
  BLOCK_M; there is an optimum BLOCK_M where GEMM efficiency and occupancy balance. Inferred from the
  monotonicity/curvature of speedup across c001→c004.
- **H4 (L2 weight reuse).** `weight` (2 MB) stays L2-resident, so many small token tiles re-read it
  cheaply; high-occupancy small-BLOCK_M configs stay competitive. Tested by c004 vs. c002/c003.

Since profiling is prohibited, all perf inferences come from *relative geomean speedups across
candidates*, not from counters.

---

## 8. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
- **Convergence.** Best geomean improves < ~2% across 2–3 consecutive perf candidates (noise floor).
- **Design exhaustion.** Tiling sweep + epilogue fusion explored and champion stable; no untried
  hypothesis with plausible upside remains.
- **Budget.** Approaching 100 evals, or token soft limit (1.0M) with diminishing returns; hard-stop
  well before 1.65M. Front-load reasoning to spend evals only on distinct, defensible changes.
Record the champion candidate ID and rationale. **Never** run `final` without explicit operator approval.

---

## 9. Contingencies (spawned only on evidence)

- **C-PREC (correctness miss).** If a workload misses the 0.98 match ratio: escalate matmul precision
  `bf16 → input_precision="tf32" → "tf32x3" → "ieee"` in successive candidates, cheapest first, until
  it passes; then resume tiling from the passing precision.
- **C-SPLIT (occupancy-bound fused kernel).** If tiling candidates plateau at low speedup consistent
  with the fused accumulator throttling the GEMM, evaluate the two-kernel split (tuned GEMM →
  `logits` DRAM; separate routing kernel). Compare geomean vs. best fused candidate.
- **C-ORDER (A1 false).** If the evaluator penalizes output ordering, emit outputs in an order that
  matches the reference's comparison (e.g., sort by expert index or reproduce torch's dense scatter),
  as a correctness child of the affected candidate.
- **C-TILE-SPILL.** If a large-BLOCK_M candidate regresses vs. its parent (spill signature), abandon
  that branch and continue from the smaller-BLOCK_M champion.

---

## 10. Evidence format — `candidates.jsonl` (append-only, one JSON object per evaluated candidate)

Required keys (satisfies CLAUDE.md items: parent, source hash, hypothesis, validation, per-workload
result, geomean, decision, cumulative eval count, skill usage):

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "timestamp": "<ISO-8601>",
  "hypothesis": "Fused bf16-HMMA fp32-accum GEMM + fp32 iterative-argmax routing passes all 5 and beats multi-launch reference.",
  "config": {"BLOCK_M": 32, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3, "matmul_precision": "bf16"},
  "validation": {
    "static_gate_passed": true,
    "static_gate_notes": "two score arrays; bias selection-only; fp32 epilogue; -inf sentinel; int64/fp32 outputs; tail mask"
  },
  "per_workload": [
    {"uuid": "469b2796-deb3-5a83-a7a6-1b2f84b86f20", "num_tokens": 2240, "correct": true, "speedup": 0.0, "max_atol_obs": null, "match_ratio_obs": null},
    {"uuid": "993dd417-9fb2-5d66-9b81-c1e15a113463", "num_tokens": 2272, "correct": true, "speedup": 0.0},
    {"uuid": "0fc8f5f9-1900-55f4-b7d8-acefecbead69", "num_tokens": 6144, "correct": true, "speedup": 0.0},
    {"uuid": "d6d0eb83-de2e-5a6d-ba23-ee72ae216708", "num_tokens": 2048, "correct": true, "speedup": 0.0},
    {"uuid": "1d554a6c-1d9d-54c7-9ed9-6d4b50c95906", "num_tokens": 2112, "correct": true, "speedup": 0.0}
  ],
  "all_correct": true,
  "geomean_speedup": 0.0,
  "decision": "keep|discard|champion",
  "decision_reason": "<why kept/discarded; next step>",
  "cumulative_evals": 1,
  "skills_used": "none",
  "next_candidate_plan": "c002: BLOCK_M=64 tiling"
}
```

Rules:
- Fill `speedup`/`geomean_speedup`/`*_obs` from the evaluator's reported numbers (do not fabricate;
  use the fields the evaluator actually returns; leave unavailable metrics `null`).
- Never rewrite an earlier record; corrections/annotations go in a new appended object referencing the ID.
- `decision` ∈ {`keep`, `discard`, `champion`}; `champion` marks the current best fully-correct candidate.
- `skills_used = "none"` — `KernelWiki` targets Hopper/Blackwell (SM90/SM100); this is Ampere sm_80 and
  profiling is prohibited, so it is out of scope and not invoked (draft §7).

---

## 11. Immediate next action (next turn, not now)

Implement **c001** exactly as specified in §2, run the §6.1 static gate, then a single
`./scripts/evaluate_candidate.sh feedback c001`, and append its record per §10. Do not implement or
evaluate anything in the current (planning) turn.

---

## 12. Decision log

### c001 — DISCARD (1/100 evals used)
- Config: fused kernel, `BLOCK_M=32, BLOCK_K=64, warps=4, stages=3`, bf16-HMMA fp32-accum.
- Result: **0/5, RUNTIME_ERROR on every workload** (`runs/candidates/c001/feedback.json`). Evaluator
  exposes no Python traceback, only `RUNTIME_ERROR`, so the fault is inferred, not observed. The GPU
  locked and ran (A800, no correctness numbers), so this is a kernel compile/exec failure, not infra.
- Numerics were statically verified against §1; the failure is a Triton *lowering/construct* issue,
  not a math error. Highest-risk constructs in c001 (to eliminate one at a time in c002):
  1. **3D reshape + axis-2 reductions** in the group stage (`reshape[BLOCK_M,8,32]`, `max(...,axis=2)`).
  2. **3D `broadcast_to` + reshape** for the 8→256 mask expansion.
  3. Register pressure from several concurrent `[32,256]` fp32 tensors (`acc`, `scores`,
     `scores_routing`, `masked`) plus a Python list of 8 companion vectors.
- **c002 plan (next turn):** keep the exact numerics of §1 but make the epilogue **fully 2D**:
  compute group top-2 by a static-unrolled loop over the 8 groups using masked 2D reductions over the
  256-wide axis (no 3D reshape/broadcast anywhere); build the expert mask by OR-ing 2D group hit masks;
  optionally load `weight` pre-transposed to drop `tl.trans`; lower `BLOCK_M` to 16 to relieve pressure.
  Change one risk factor per candidate if c002 still fails (2D-first is the biggest single reduction).

### c002 — DISCARD (2/100 evals used)
- Config: fully-2D epilogue (static-unroll over 8 groups, no 3D reshape/broadcast), pre-transposed
  weight load (no `tl.trans`), 3-arg `tl.dot(hs,w,acc)`, `tl.argmax` selection, int64 kernel store,
  `BLOCK_M=16, BLOCK_K=64, warps=4, stages=3`, bf16-HMMA fp32-accum.
- Result: **0/5, RUNTIME_ERROR on every workload** (`runs/candidates/c002/feedback.json`), identical
  to c001 despite a structurally very different epilogue. Evaluator still exposes only `RUNTIME_ERROR`.
- **Key inference:** two very different epilogues (3D c001 vs fully-2D c002) failing *identically*
  RULES OUT the 3D reshape/broadcast and `tl.trans` as the cause. The fault is a construct **common to
  both**. Remaining shared suspects (unobservable without a traceback): (a) 3-arg `tl.dot(hs,w,acc)`
  (older Triton reads arg3 as `trans_a` → type error), (b) `tl.argmax(axis=1)`, (c) **int64 `tl.store`
  from the kernel**, (d) `tl.sigmoid` on a `[BLOCK_M,256]` tile.
- **c003 plan (next turn):** since a blind RUNTIME_ERROR cannot be bisected one-construct-per-eval,
  remove (a),(b),(c) together: manual `acc += tl.dot(hs,w)` (2-arg only); replace `tl.argmax` with a
  portable max-value + min-matching-index reduction (`tl.max` then `tl.min` over
  `where(val==max, iota, BIG)`); have the kernel write **int32** indices and do the int32→int64 cast
  in torch plumbing (a result dtype conversion, not computation). Keep the fully-2D epilogue. This
  narrows the surviving op set to the minimal `load/store, dot, sigmoid, where, max, min, sum`; if
  c003 still fails, split that minimal set (e.g. isolate `tl.sigmoid`, or drop to a two-kernel GEMM+
  routing split so the GEMM can be validated independently).

### c003 — DISCARD (3/100 evals used)
- Config: fully-2D argmax-free epilogue, 2-arg `tl.dot` + manual accumulate, `-inf` sentinel, int32
  kernel store + torch int64 cast, `BLOCK_M=16, BLOCK_K=64, warps=4, stages=2`, bf16-HMMA fp32-accum.
- Result: **0/5, RUNTIME_ERROR on every workload** (`runs/candidates/c003/feedback.json`), identical
  to c001 and c002. Evaluator still exposes only `RUNTIME_ERROR`.
- **Key inference (cumulative):** three structurally distinct fused kernels now fail the same opaque
  way. This RULES OUT: 3D reshape/broadcast, `tl.trans`, 3-arg `tl.dot`, `tl.argmax`, and int64 kernel
  stores. The fault is in a surface **common to all three** fused kernels. Strongest remaining
  suspects (unobservable without a traceback): (1) the software-pipelined K-loop with `num_stages>=2`
  miscompiling to an illegal access on Ampere with a strided `tl.dot` operand; (2) the strided/
  transposed `w` B-operand fed to `tl.dot`; (3) the non-finite `-inf` sentinel in `where`/reductions.
- **c004 plan (next turn):** remove all three together — `num_stages=1` (pipeliner off); pre-transpose
  `weight` **once in torch** to a contiguous `[D,E]` buffer (layout-only data movement, not the
  routing computation) so the B-operand load is contiguous; replace `-inf` with a finite `-1e30`
  (< any sigmoid+bias score; identical max/min behavior, no inf-handling quirk). Keep the fully-2D
  argmax-free epilogue, int32 store + torch int64 cast, `BLOCK_M=16, BLOCK_K=64, warps=4`.
- **c005 fallback:** if c004 also RUNTIME_ERRORs, drop to the **two-kernel split (C-SPLIT §9)** — a
  standard, well-trodden Triton GEMM → `logits[T,256]` DRAM, then a separate routing kernel — so the
  GEMM path can be validated independently of the fused mega-epilogue and the failing surface isolated.

### c004 — DISCARD (4/100 evals used)  [two-kernel split, C-SPLIT]
- Config: **two-kernel split**. Kernel A = canonical tiled bf16 GEMM (contiguous pre-transposed
  `[D,E]` weight, single `[32,256]` fp32 accumulator, `BLOCK_M=32, BLOCK_K=64, warps=8, stages=2`) →
  `logits[T,256]` DRAM. Kernel B = per-token routing (`grid=(T,)`, whole epilogue on 1D `[256]`/`[8]`
  tensors, finite `-1e30` sentinel, argmax-free, int32 store + torch int64 cast, `warps=4, stages=1`).
- Result: **0/5, RUNTIME_ERROR on every workload** (`runs/candidates/c004/feedback.json`), identical
  to c001/c002/c003. Evaluator still exposes only `RUNTIME_ERROR`.
- **Key inference (decisive):** four structurally distinct solutions now fail identically — including
  a *low-pressure, textbook* GEMM with a single `[32,256]` accumulator. This RULES OUT: fused-epilogue
  logic, register pressure, 3D ops, `tl.trans`, 3-arg `tl.dot`, `tl.argmax`, int64 stores, the strided
  weight operand (c004 uses contiguous `[D,E]`), and the `-inf` sentinel (c004 uses finite `-1e30`).
  The remaining **universal, fixable** suspect shared by every candidate is a *launch-resource*
  condition: a software-pipelined `tl.dot` with `num_stages>=2` over a `[BLOCK_K,256]` bf16 B-tile
  requests large dynamic shared memory (c001/c002 stages=3 ≈ 96 KB; c003/c004 stages=2 ≈ 64 KB),
  exceeding the 48 KB static default and requiring a `cudaFuncAttributeMaxDynamicSharedMemorySize`
  opt-in; if that opt-in misfires the kernel fails at launch on every workload.
- **c005 plan (next turn):** test the shared-memory hypothesis directly — two-kernel split, Kernel A
  at `BLOCK_K=32, num_stages=1, num_warps=4` so the B-tile is `[32,256]` bf16 = 16 KB, single-buffered
  (~18 KB total), safely under 48 KB with no dynamic-smem opt-in. Kernel B unchanged.
  - If **c005 passes** → the `num_stages>=2` dynamic-shared-memory opt-in was the universal blocker;
    resume performance tuning (raise `BLOCK_K`/`num_stages` deliberately, re-fuse) from a known-good base.
  - If **c005 still errors** → the resource hypothesis is exonerated; the next probe targets the
    `tl.dot` configuration itself (`out_dtype`, `input_precision`, or an explicit fp16/bf16 operand
    cast) as the universal cause, still without ever adding a non-Triton computational fallback.

### c005 — DISCARD (5/100 evals used)  [minimal-resource two-kernel split]
- Config: two-kernel split; Kernel A minimal single-buffered GEMM (`BLOCK_M=16, BLOCK_K=32,
  num_stages=1, num_warps=4` → ~17 KB shared, no dynamic-smem opt-in), Kernel B byte-identical to c004.
- Result: **0/5, RUNTIME_ERROR on every workload** (`runs/candidates/c005/feedback.json`), identical to
  c001–c004. Evaluator still exposes only `RUNTIME_ERROR`.
- **Key inference — EXONERATES the shared-memory / `num_stages` opt-in hypothesis.** A single-buffered
  ~17 KB GEMM still fails, so dynamic shared memory > 48 KB was not the cause. Five structurally
  distinct kernels now fail identically.
- **Decisive elimination-by-necessity:** SOL-ExecBench is a Triton kernel-optimization benchmark with a
  matmul-heavy reference, so `tl.dot` and the absolute-basic ops (`tl.load/store/where/max/min/sum/
  arange/zeros`, integer `//`, `.to()` casts) *must* work on this stack — otherwise no Triton solution
  could ever pass any task. By elimination, the one **non-basic op present in 100% of c001–c005** and
  not required by generic kernels is **`tl.sigmoid`**. If `tl.sigmoid` is renamed/moved/unavailable in
  this Triton version, every candidate would fail identically at compile time with the opaque
  `RUNTIME_ERROR` observed.
- **c006 plan (next turn):** single-variable test — take the robust c005 two-kernel split and replace
  `tl.sigmoid(x)` with the mathematically identical `1/(1+tl.exp(-x))` (`tl.exp` is a core,
  always-present intrinsic), changing **only** that one op. Clean attribution:
  - If **c006 passes** → `tl.sigmoid` was THE universal blocker; resume perf tuning (re-fuse, raise
    `BLOCK_M/BLOCK_K/num_stages`) from a known-good base.
  - If **c006 still errors** → exp/sigmoid exonerated; **c007** probes the remaining universal surface
    (e.g. `.to(tl.int32)` store semantics, the `run()` invocation / output contract, or `tl.math.exp`).

### c006 — DISCARD (6/100 evals used)  [sigmoid via exp]
- Config: c005 two-kernel split with `tl.sigmoid(x)` replaced by IEEE-identical `1/(1+tl.exp(-x))`;
  everything else byte-identical to c005.
- Result: **0/5, RUNTIME_ERROR on every workload** (`runs/candidates/c006/feedback.json`), identical to
  c001–c005. Evaluator still exposes only `RUNTIME_ERROR`.
- **Key inference — EXONERATES `tl.sigmoid`/`tl.exp`.** Computing the sigmoid via `tl.exp` still fails,
  so neither `tl.sigmoid` nor `tl.exp` is the blocker. **Six** structurally distinct kernels now fail
  identically. Eliminated by direct experiment so far: fused-epilogue logic, register pressure, 3D ops,
  `tl.trans`, 3-arg `tl.dot`, `tl.argmax`, int64 stores, strided weight operand, `-inf` sentinel,
  `num_stages>=2` dyn-shmem opt-in (c005), and sigmoid/exp (c006).
- **Remaining non-idiomatic construct present in 100% of c001–c006:** the *python-list companion
  capture* — `sel_vals = []`; `append` inside one `tl.static_range` loop; index that list in a *second*
  `tl.static_range` loop — plus the per-iteration scalar `tl.store` of a single index.
- **c007 plan (next turn):** replace that with the standard idiomatic pattern — accumulate the selected
  index and bias-free companion score into two `[TOPK]` **register vectors** via masked-`where` (no
  python list, no second unrolled loop), then do **one** vectorized store of all 8 indices and **one**
  of all 8 weights. Everything else held identical to c006. **Decisive:**
  - If **c007 passes** → the python-list companion pattern was THE universal blocker; resume perf tuning.
  - If **c007 also errors** → the fault is not in any kernel construct I can vary; it is a
    harness/environment/import-level condition (every possible Triton kernel would then fail), and **no
    justified kernel-level next candidate remains → write `SEARCH_COMPLETE`.**

### c007 — DISCARD but BREAKTHROUGH (7/100 evals used)  [register-vector companion capture]
- Config: c006 two-kernel split with the top-8 companion capture rewritten to two `[TOPK]` **register
  vectors** (masked-`where` accumulate) + a single vectorized store of all 8 indices and all 8 weights;
  no python list, no second unrolled loop. Everything else identical to c006.
- Result: **0/5, but status flipped `RUNTIME_ERROR → INCORRECT_NUMERICAL`** on all five workloads
  (`runs/candidates/c007/feedback.json`), with real `max_abs`/`max_rel` numbers reported for the first
  time: `max_abs ∈ {249, 228, 235, 190, 239}`.
- **This proves the kernel now COMPILES AND RUNS.** The *python-list companion-capture pattern*
  (`sel_vals=[]`; `append` in one `static_range` loop; index it in a second `static_range` loop) was
  THE universal blocker across c001–c006 on this Triton stack; the idiomatic `[TOPK]`-register-vector +
  single-vectorized-store pattern works. The remaining failure is numerical/ordering, not a crash.
- **Error diagnosis:** all `max_abs` values (190–249) lie in the expert-index range `[0,255]`, whereas
  `topk_weight` is mathematically bounded to `≤ 2.5` (`selected/sum·2.5`, `selected ≤ sum`). Weights
  therefore cannot produce ~190–249 errors → these come from **`topk_idx` compared positionally**.
  Workload 5 has `max_rel == max_abs == 239.0` ⇒ reference element ≈ 1.0 vs our ≈ 240 at the same
  slot ⇒ a positional index mismatch. Since the evaluator compares indices element-wise (not a dense
  scatter — that would bound errors to `≤ 2.5`), the reference output order is deterministic and I must
  reproduce it. c007 emits **descending-by-masked-score** order and does not match, so the reference is
  not descending-score-ordered; the most likely `torch.topk(sorted=False)` CUDA order is **ascending
  expert index** (radix-select preserving original scan order).
- **c008 plan (next turn):** keep c007's now-runnable two-kernel structure; **reorder the 8 selected
  `(idx, weight)` pairs to ascending expert index** before storing — compute `rank[i] = Σ_j (sel_idx[j]
  < sel_idx[i])` over the 8 selected (tiny 8×8 in registers), then scatter-store `sel_idx`/`w_vec` into
  slot `rank[i]`. Single-variable change (output ordering only).
  - If errors collapse under `atol` → ordering was the issue.
  - If indices still mismatch hugely → try a different canonical order (e.g. descending-score already
    ruled out; consider the exact torch unsorted order) or reconsider whether the selected *set* is off.
  - If only small residual weight diffs remain → precision escalation (`tf32`/`ieee`).

### c008 — DISCARD (8/100 evals used)  [ascending-index output order]
- Config: c007's runnable two-kernel split, single change = reorder the 8 selected `(idx, weight)`
  pairs to **ascending expert index** before storing (8×8 register rank `rank[i]=Σ_j(sel_idx[j]<
  sel_idx[i])` + scatter store). bf16 GEMM, descending selection loop, set/weights unchanged.
- Result: **0/5 INCORRECT_NUMERICAL** with `max_abs ∈ {249, 241, 235, 181, 238}` — essentially
  unchanged vs c007 (`{249,228,235,190,239}`) and still in the expert-index range `[0,255]`.
- **Key inference — the SELECTED SET is wrong, not just the order.** Two *different* natural output
  orders (c007 descending-by-score, c008 ascending-index) both fail with ~identical index-magnitude
  errors. If only ordering were wrong, at least one of the two would have matched; since neither does,
  the selected set itself differs from the reference on enough rows to break the ≥0.99 match ratio.
  The factor common to c007/c008 that perturbs selection is the **bf16 gate GEMM**: the reference
  computes `logits = F.linear(hidden.float(), weight.float())` — a *true fp32* matmul over the bf16
  inputs upcast to fp32 — whereas c007/c008 multiply in bf16 tensor cores. bf16 rounding shifts
  borderline `sigmoid+bias` values, flipping which group is 4th / which expert is 8th on a fraction of
  tokens → wrong set → positional index blowup regardless of emit order. (Note: this contradicts the
  draft §3.2 assumption A2 that bf16-HMMA would be faithful; the loose atol is not enough because the
  index comparison is positional and unforgiving.)
- **c009 plan (next turn, parent c007, single variable = GEMM precision only):** make the GEMM
  numerically faithful — load bf16 tiles, `.to(tl.float32)`, and `tl.dot(a, b, input_precision="ieee")`
  for a true fp32 accumulate matching `F.linear(.float())`. Restore c007's **descending-by-score**
  emit (`torch.topk` on CUDA returns descending value order). 
  - If **c009 passes** → exact fp32 GEMM was the fix; subsequent candidates trade precision back for
    speed (`tf32x3` → `tf32` → bf16) while watching the 0.99 match ratio, then tile-tune / re-fuse for
    geomean.
  - If **c009 still mismatches indices** → the reference order is not descending; **c010** keeps the
    exact GEMM but switches to ascending-index emit to separate order from set.

### c009 — DISCARD but DECISIVE (9/100 evals used)  [exact fp32 GEMM, descending emit]
- Config: c007's runnable split, single change = GEMM precision → upcast bf16 tiles to fp32 and
  `tl.dot(a, b, input_precision="ieee")` (true fp32 accumulate, matching `F.linear(.float())`);
  descending-by-score emit restored.
- Result: **0/5 INCORRECT_NUMERICAL** with `max_abs {249, 228, 235, 190, 239}` — **byte-identical to
  c007** (bf16, descending). GEMM precision changed nothing in the compared output.
- **Two decisive conclusions:**
  1. The earlier "bf16 flips the selected set" hypothesis (c008 decision) is **REFUTED** — precision is
     irrelevant to the comparison here. (c007 bf16 and c009 exact-fp32 produce identical outputs.)
  2. With **exact fp32 logits**, my group top-2 sum / top-4 groups / mask / top-8 experts are provably
     the **same fp32-deterministic SET** as the reference (ties are measure-zero on random data). So the
     selected set is **correct**, and the failure is purely **OUTPUT ORDER**. The evaluator compares
     `topk_idx` **positionally** (errors ~190–249 are in `[0,255]`; an order-insensitive/dense compare
     would either bound errors by weight ≤ 2.5 or make a provably-correct-set candidate pass — c009 did
     not pass, proving positional comparison).
- Descending-by-score order (c007, c009) does not match the reference. Ascending-index was tried in
  c008 but only with the then-unproven bf16 set; precision-irrelevance means that was not a clean test.
- **c010 plan (next turn, parent c009, single variable = output order):** keep the exact-fp32 kernel
  (provably correct set) and change ONLY the emit order to **ascending expert index** (8×8 register rank
  + scatter store, as in c008). This is the first clean test of ascending order with a proven-correct
  set.
  - If **c010 passes** → reference order was ascending expert index; then perf-tune (precision back to
    bf16/tf32 for speed since precision is comparison-irrelevant here, tiling, re-fuse).
  - If **c010 also fails** with ~250 `max_abs` → the reference uses `torch.topk(sorted=False)`'s native
    radix order (neither plain ascending nor descending); reproduce that exact order.

### c010 — CHAMPION, first correct candidate (10/100 evals used)  [torch.topk order]
- Config: c009's exact-fp32 kernel (provably correct set) + emit order reproducing
  `torch.topk(masked, k=8, sorted=False)` CUDA gatherTopK: the 7 strictly-larger experts by **ascending
  expert index**, then the **min-score (kth-value) expert last**. Implemented via an 8×8 restricted rank
  (rank among the first 7 picks by ascending index) + scatter store; the descending selection loop's
  last pick (i=TOPK-1) is exactly that min-score expert → slot 7.
- Result: **5/5 PASSED**, `max_abs ≈ 5e-7` (exact to ulp). **Geomean speedup 0.8592×** (per-workload
  0.80–0.94×). Correct but **slower** than the multi-launch reference.
- **Output-order hypothesis CONFIRMED.** The three-way experiment settles it: descending (c007/c009)
  and pure-ascending (c008) both failed; `torch.topk`-native order passes. Combined with the exact-fp32
  set, the reference is reproduced to ulp.
- **Now a pure performance problem.** Root cost: (a) the GEMM uses `input_precision="ieee"` (true fp32
  dot), ~3–4× slower than bf16 HMMA and off the fast tensor-core path; (b) c007==c009 proved matmul
  precision is **irrelevant** to the compared result for these workloads, so bf16 HMMA will still pass
  5/5. Reverting the GEMM to bf16 is the biggest zero-correctness-risk speed lever.

### Performance roadmap (from the c010 champion)
- **c011 (next):** revert gate GEMM to **bf16 HMMA** (drop the fp32 tile upcast + `input_precision=
  "ieee"`; `acc += tl.dot(a_bf16, b_bf16)`), keep c010's exact routing + torch.topk order. Single
  perf variable; precision proven comparison-irrelevant. Expect a large jump toward/above 1.0×.
- **c012+:** GEMM tiling/pipelining now that shared-memory launch is known-good — raise `BLOCK_M`,
  `BLOCK_K`, `num_stages` (2–3), `num_warps`; the c005 "num_stages≥2 blocks" fear was disproven (c005
  failed on the python-list bug, not shared memory).
- **c013+:** consider **re-fusing** the two kernels (GEMM epilogue → routing in-register) to cut the
  `logits[T,256]` DRAM round-trip and the second launch, now that the routing epilogue is proven correct
  and the register-vector pattern compiles.
- **c014+:** tune the per-token routing Kernel B (grid, warps) or batch multiple tokens per program if
  the routing launch (grid=T) is a measurable fraction of runtime.
- Stop when geomean converges (<~2% over 2–3 perf candidates) or the design space is exhausted.

### c011 — NEW CHAMPION (11/100 evals used)  [bf16 HMMA GEMM]
- Config: c010 with the GEMM reverted to native **bf16 HMMA** (drop the fp32 tile upcast +
  `input_precision="ieee"`; `acc += tl.dot(a_bf16, b_bf16)`, fp32 accumulate). Routing epilogue +
  torch.topk order byte-identical to c010. `BLOCK_M=16, BLOCK_K=32, num_stages=1, num_warps=4`.
- Result: **5/5 PASSED**, **geomean 1.7525×** (per-workload 1.33–2.35×), vs c010's 0.8592× — a **2.04×
  jump** from one change, confirming the GEMM dominated runtime and bf16 tensor cores are both fast and
  faithful within the 0.99 match ratio.
- **Correctness margin to watch:** workload 5 now shows `max_abs=16.0, max_rel=0.131` — bf16 rounding
  flips a borderline expert on a small fraction of tokens (index diff 16), but the ≥0.99 match ratio
  absorbs it (PASSED). The other 4 workloads are ~exact (`max_abs ~2–5e-6`). Do **not** reduce precision
  further; bf16 as-is is safe.
- **Now GEMM-tiling-bound.** c011's GEMM is deliberately tiny/conservative (leftover from debugging):
  `BLOCK_M=16, BLOCK_K=32, num_stages=1, num_warps=4` — large headroom.
- **c012 plan (next):** GEMM tiling — `BLOCK_M=32, BLOCK_K=64, num_stages=2, num_warps=8`
  (double-buffered, wider K, more warps). The B-tile `[64,256]` bf16 × 2 stages ≈ 72 KB > 48 KB static
  cap, so this *also* exercises Triton's dynamic-shared-memory opt-in on this A800 stack — expected to
  work (the c001–c006 RUNTIME_ERRORs were the python-list bug, proven by c007, not shared memory).
  - If **c012 faster + correct** → keep tiling up (`BLOCK_M=64, num_stages=3`).
  - If **c012 RUNTIME_ERRORs** → the smem opt-in is the limiter; fall back to a ≤48 KB config
    (`BLOCK_M=64, BLOCK_K=32, num_stages=2`).
  - Later: **re-fuse** the two kernels to drop the `logits[T,256]` DRAM round-trip + second launch.

### c012 — NEW CHAMPION (12/100 evals used)  [GEMM tiling]
- Config: c011 with only Kernel A launch tiling enlarged — `BLOCK_M=32, BLOCK_K=64, num_stages=2,
  num_warps=8`. Routing kernel byte-identical.
- Result: **5/5 PASSED**, **geomean 3.1957×** (per-workload 2.59–3.65×), vs c011's 1.7525× — a further
  **1.82×** from tiling alone.
- **Two confirmations:** (a) the GEMM-bound thesis holds; (b) the **>48 KB dynamic-shared-memory opt-in
  works** on this A800 stack (no RUNTIME_ERROR at `BLOCK_K=64`/`stages=2`), definitively closing the
  c005-era shared-memory red herring. Correctness margins identical to c011 (workload 5
  `max_abs=16.0/max_rel=0.131` tolerated; others ~exact) since logits are numerically unchanged.
- **Still GEMM-bound with headroom:** `BLOCK_M` can rise to 64, the K-loop can pipeline deeper
  (`num_stages=3`), and the two-kernel split still pays a `logits[T,256]` DRAM write+read + a 2nd launch.
- **c013 plan (next):** push tiling — `BLOCK_M=64, BLOCK_K=64, num_stages=3, num_warps=8` (deeper
  pipeline, more tokens/program). Watch for spill/occupancy regression vs c012.
  - If **faster** → keep and continue searching the tiling sweet spot.
  - If **slower** → revert to c012 tiling and pivot to the bigger structural win: **re-fusion** (a
    single fused GEMM+routing kernel) to remove the logits DRAM round-trip and second launch.

### c013 — NEW CHAMPION (13/100 evals used)  [deeper GEMM tiling]
- Config: c012 with Kernel A tiling deepened — `BLOCK_M=64, BLOCK_K=64, num_stages=3, num_warps=8`.
  Routing kernel byte-identical.
- Result: **5/5 PASSED**, **geomean 3.6738×** (per-workload 2.87–4.70×), vs c012's 3.1957× (+15%). No
  spill/occupancy regression at this size; correctness margins unchanged (workload 5 `max_abs=16.0`
  tolerated; others ~exact).
- **Tiling gains are decelerating:** 2.04× → 1.82× → 1.15× across c011→c012→c013. The remaining big
  lever is now **structural**: the two-kernel split writes `logits[T,256]` fp32 to DRAM and reads it
  back in a second launch (~6 MB round-trip at T=6144 + a launch).
- **c014 plan (next):** **re-fuse** GEMM + routing into one kernel — run the proven routing epilogue
  in-register immediately after the K-loop, so `logits` never hits DRAM and there is one launch. The
  fused epilogue is now proven correct (register-vector capture from c007, torch.topk order from c010)
  and proven to compile. Start at c013's GEMM tiling (`BLOCK_M=64, BLOCK_K=64, stages=3, warps=8`).
  - Risk: register pressure from the `[64,256]` fp32 accumulator + routing temporaries may cut
    occupancy. If **c014 < c013**, keep c013 and try a smaller fused `BLOCK_M` (32, then 16) before
    concluding the split is better; if still no gain, retain c013 and tune routing Kernel B / grid.
