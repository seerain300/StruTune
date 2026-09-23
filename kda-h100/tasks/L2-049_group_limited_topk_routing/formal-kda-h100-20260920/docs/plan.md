# Plan — L2/049 `group_limited_topk_routing` (H100 / sm_90)

Executable, sequential KDA plan. Builds on `docs/draft.md` (semantics, numerical analysis,
design space). This turn: plan only — no candidate implemented or evaluated.

Governing constants (fixed): `E=256`, `n_group=8`, `experts_per_group=32`, `topk_group=4`,
`top_k=8`, `K=4096`; inputs bf16 (`hidden_states [T,K]`, `weight [E,K]`, `expert_bias [E]`)
+ fp32 scalar `routed_scaling_factor`; outputs `topk_idx [T,8] int64`, `topk_weight [T,8]
float32`. Feedback `T ∈ {2048,2080,2112,2144,2176,2208,2240,2272,2851,3169,3557,4093,6144,
8192,12288,16384}`, all with `routed_scaling_factor = 2.5`.

---

## 0. Budgets, guardrails, invariants

- **Eval budget:** 100 candidate evaluations (one full 16-workload feedback run = 1 eval).
  Target: converge in well under 20 evals. Each eval is precious — no eval is spent until
  the source is reasoned-correct and the diff vs. its parent is meaningful.
- **Token budget:** soft 9.0M / normal 10.0M / absolute 11.0M. Keep per-turn reads scoped.
- **Hard rules (from CLAUDE.md / TASK.md):**
  - Triton is the primary compute; PyTorch only for metadata/launch plumbing. No Torch/
    CPU/NumPy/CUDA-extension/alternate fallback. A failing Triton kernel is *invalid* — fix
    or abandon the candidate, never paper over with a fallback.
  - Only `./scripts/evaluate_candidate.sh feedback cNNN` runs/evaluates code.
  - Profiling only via `./scripts/ncu_profile.sh` under the `ncu-report-skill` workflow.
  - **Never** profile and evaluate at the same time (foreign proc on locked GPU → rc 3,
    wasted eval). Strictly sequence them.
  - Do not touch evaluator/dataset/controller/launcher/shared config; do not change the
    feedback workloads; never run `final` without explicit operator approval.
  - `candidates.jsonl` is append-only; candidate IDs are immutable; any meaningful source/
    config/launch change ⇒ new ID; never reuse an ID for changed source.
- **Convergence / stop:** write `SEARCH_COMPLETE` when improvement genuinely converges
  (see §6) or a budget limit is reached.

---

## 1. Solution skeleton (stable across candidates)

`solution/solution.py` exposes `run(hidden_states, weight, expert_bias,
routed_scaling_factor)` and returns `(topk_idx, topk_weight)`. Structure kept constant so
diffs between candidates are minimal and attributable:

- Thin host wrapper (allowed torch): read `T` from `hidden_states.shape[0]`; assert dtypes/
  shapes; allocate outputs (`topk_idx` int64, `topk_weight` float32) on the input device;
  compute grid; launch Triton kernel(s); return. No torch compute on the routing math.
- All compute (projection GEMM, sigmoid, bias add, group top-2, group top-4, expert top-8,
  normalize, scale) lives in Triton. `weight` passed so the kernel reads `[E,K]` directly
  (present `tl.dot` operands via strides; no host transpose of data values).
- The kernel/config identity (block sizes, `input_precision`, num_warps/stages, fusion
  topology) is what a candidate ID pins.

---

## 2. Correctness checks (applied to every candidate before eval)

Because local execution is unavailable, correctness is established by *static reasoning*
against the reference, then confirmed by the single eval. Each candidate must satisfy this
checklist before spending an eval:

**C1 — Semantic parity with reference (`docs/draft.md §1`, §4.5):**
- Two score tensors kept distinct: `scores_for_routing = sigmoid(logits)+bias` drives *all*
  selection; bias-free `scores = sigmoid(logits)` drives the returned weights.
- `sigmoid` computed in fp32; `expert_bias` upcast bf16→fp32 before add.
- Group score = `max1 + max2` over the 32 members (top-2, order-independent).
- top-4 groups → build an 8-wide group mask (set membership, not order).
- Masked-out experts set to fp32-min sentinel (`-3.4028235e38`) on `scores_for_routing`
  before the expert top-8.
- top-8 experts by masked `scores_for_routing`; weights gather bias-free `scores` at those
  indices; `w = sel/(sum(sel)+1e-20)*routed_scaling_factor`.
- Output dtypes exactly int64 / float32.

**C2 — Tie-break parity (`draft §4.3`):** iterative argmax must break ties by **lowest
index** (matches CUDA topk/argmax). Every mask step uses a strict `>` update seeded so the
first (lowest) index wins on equality. This governs both group top-4 and expert top-8.

**C3 — Boundary/masking:** M masked for `T` not divisible by `BLOCK_M` (out-of-range rows
must not read/write OOB); K-loop tail (4096 is clean for BK∈{32,64,128}); grid covers all
tokens. Verify with the two odd shapes explicitly in reasoning (e.g. 2851, 4093).

**C4 — Precision budget (`draft §4.1–4.2`):** confirm the chosen `input_precision` keeps
the token-flip rate under the 2% mismatch budget (`required_match_ratio=0.98`,
`max_rtol=0.01`, `max_atol` per-workload 0.44–0.84). Start safe, step down only with a
recorded hypothesis.

**C5 — Order-of-output assumption:** until eval evidence says otherwise, do **not** rely on
a particular within-row order of the 8 indices matching the reference (reference uses
`sorted=False`). c001 both (a) confirms pass and (b) probes how strict the comparison is on
index order — see §3.

If a candidate fails any Cx in reasoning, fix within the *same not-yet-evaluated* draft
(same ID only if never evaluated); once evaluated, corrections require a new ID.

---

## 3. Sequential candidate roadmap

Each step = one immutable candidate. IDs advance on any meaningful change. Later steps are
contingent on earlier evidence; only the next 1–2 are firmly specified, the rest are
decision points.

### c001 — Correctness anchor (safe precision)
- **Design:** simplest defensible fused kernel. Grid over M-blocks (`BLOCK_M=64`), N-tile =
  full 256, K-loop `tl.dot` with **safe precision** `input_precision="tf32x3"` (≈fp32
  fidelity, far faster than IEEE fp32), fp32 accumulate. Full routing epilogue in-kernel
  (group top-2 → group top-4 → mask → expert top-8 → normalize → scale) with lowest-index
  tie-break. If a single fused kernel raises register/occupancy concerns, fall back to the
  two-kernel split (Option B, draft §5.1) — decided before writing, recorded in hypothesis.
- **Hypothesis:** passes all 16 workloads; establishes baseline speedup and reveals
  evaluator strictness on index order + precision. Expected ≥1× (likely >2× since reference
  matmul is fp32-path + ~7 launches).
- **Success:** all workloads pass; record geomean.
- **If it fails:** diagnose from evaluator output — correctness (semantic/tie-break/order)
  vs. runtime (compile/launch/OOB). Correctness → new ID with the fix; never add a fallback.

### c002 — Precision step-down (if c001 passes)
- Change only `input_precision`: `tf32x3 → tf32` (or directly `bf16` if the flip-rate
  argument in draft §4.2 looks safe). Everything else identical to the best passing anchor.
- **Hypothesis:** lower precision materially raises GEMM throughput while staying within the
  2% flip budget. This isolates the precision↔speed↔correctness frontier in one variable.
- **Decision:** if it passes and is faster → new baseline; if it fails correctness → keep
  the previous precision as the floor and stop stepping down.

### c003 — Fusion topology / launch reduction (if not already single-kernel)
- If the anchor was two-kernel, collapse to the single fused kernel (draft §5.1 Option A) at
  the best passing precision. If already fused, skip.
- **Hypothesis:** removing the `[T,256]×2` gmem round-trip and one launch improves latency,
  especially at smaller T.

### c004+ — GEMM tiling / pipeline tuning
- Sweep, one variable per candidate: `BLOCK_M ∈ {32,64,128}`, `BLOCK_K ∈ {32,64,128}`,
  `num_warps ∈ {4,8}`, `num_stages ∈ {2,3,4}`. Prefer an **M-agnostic** fixed config over
  `T`-keyed autotune (16 distinct M → recompile/registration churn; draft §5.4).
- Guided by profiling evidence (§4): if GEMM is DRAM/pipeline-bound, adjust stages/warps and
  BLOCK_K; if the routing epilogue is the register-pressure limiter, adjust BLOCK_M or split
  the epilogue.
- One change per ID; keep the best; abandon regressions.

### c00x — Routing-epilogue micro-opts (only if it shows up in profile)
- E.g. reduce argmax passes, fuse group top-2 into the accumulator write, better sentinel
  handling. Only pursued if profiling shows the epilogue is non-negligible (expected small
  vs. the K-loop, draft §5.3).

**Lineage rule:** each candidate names its `parent`. The default parent is the current best
*passing* candidate. A/B branches (e.g. bf16 vs tf32 at same tiling) are allowed but must be
recorded with distinct IDs and a shared parent.

---

## 4. Performance hypotheses & profiling protocol

**Primary hypotheses (ranked by expected impact):**
1. Replacing the fp32-path reference matmul with bf16/TF32 tensor-core `tl.dot` is the
   dominant win (draft §3). Ceiling depends on whether the reference is true fp32 or TF32 —
   to be measured.
2. Fusing the ~7-launch epilogue into 1 kernel removes launch overhead + `[T,256]` gmem
   round-trips; larger relative effect at smaller T (2048–3557).
3. Lower GEMM precision trades correctness margin for throughput; there is a safe floor
   (§C4) — likely tf32 or bf16 given sigmoid saturation robustness (draft §4.1).

**Profiling protocol (`ncu-report-skill` via `./scripts/ncu_profile.sh`):**
- Run **only when no evaluation is in flight**, and never launch an eval while a profile is
  running (draft §7; rc-3 risk).
- Use to answer: (a) achieved tensor-core throughput vs. peak & whether the kernel is DRAM-
  bound on `hidden_states`; (b) occupancy / registers / spills of the fused kernel; (c)
  whether the routing epilogue is a measurable fraction. Profile representative shapes only
  (e.g. one mid `T=4093` and the max `T=16384`), not all 16.
- Convert findings into a single-variable candidate change with a written hypothesis.
- Profiling does **not** consume the eval budget, but its results are advisory — the eval is
  the source of truth for correctness and ranking.

---

## 5. Evidence format (`candidates.jsonl`, append-only)

One complete JSON object appended per evaluated candidate; earlier records never rewritten.
Required fields:

```json
{
  "id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "faithful anchor at tf32x3; establish correctness + baseline speedup",
  "design": {
    "fusion": "single-kernel|two-kernel",
    "block_m": 64, "block_k": 64, "num_warps": 8, "num_stages": 3,
    "input_precision": "tf32x3"
  },
  "validation": {
    "checklist": {"C1": true, "C2": true, "C3": true, "C4": "tf32x3", "C5": "probe"},
    "correctness": "pass|fail",
    "notes": "any semantic/tie-break/order observations from evaluator output"
  },
  "per_workload": [
    {"uuid": "d6d0eb83-...", "num_tokens": 2048, "pass": true, "speedup": 0.0,
     "match_ratio": 1.0}
  ],
  "geomean_speedup": 0.0,
  "decision": "keep|reject|new-baseline",
  "cumulative_evals": 1,
  "skills_used": ["KernelWiki", "ncu-report-skill"],
  "profiling_ref": "profile/rNNN or null"
}
```

Rules: `geomean_speedup` = geometric mean over all 16 workloads (only meaningful if all
pass; if any workload fails correctness the candidate is `reject` regardless of speed).
Record the actual per-workload numbers emitted by the evaluator verbatim; do not synthesize.
`decision` explains why we keep/branch/abandon and names the next intended candidate.

---

## 6. Stopping criteria (convergence → `SEARCH_COMPLETE`)

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
- **Converged:** the best passing candidate's geomean has not improved by > ~2% over the
  last 3 evaluated candidates, and the profile shows the GEMM near its achievable
  tensor-core / DRAM ceiling (no clear remaining lever).
- **Precision floor reached:** further precision step-down fails correctness and tiling/
  fusion sweeps are exhausted.
- **Budget:** approaching the eval budget (100) or the token soft limit (9.0M) — leave
  margin for the operator-approved `final` run.
- **Diminishing returns vs. risk:** remaining ideas are speculative micro-opts with small
  expected gain and non-trivial correctness risk.

`SEARCH_COMPLETE` records: best candidate ID, its geomean, evals consumed, the converged
bottleneck, and the reason. `final` is run only after explicit operator approval, on the
best valid candidate.

---

## 7. Immediate next actions (subsequent turns)
1. Implement `solution/solution.py` as **c001** per §3 (anchor, full checklist §2).
2. Re-verify C1–C5 by reading the written kernel against reference semantics.
3. Evaluate: `./scripts/evaluate_candidate.sh feedback c001`.
4. Append the c001 record to `candidates.jsonl` (§5); set decision + next candidate.
5. Optionally profile the best passing candidate (§4) before the next tuning change.

## 8. Decision log

### c001 — REJECT (eval 1/100)
- **Result:** 0/16 pass, all `RUNTIME_ERROR`. `feedback.json` has `valid=false`,
  `ref_ms/sol_ms=null`, `max_abs/max_rel=0.0`; controller shows `return_code=1` and
  `foreign_process_detected=false` → a genuine consumed evaluation, **not** a discarded
  (rc-3) measurement. No Python traceback surfaced in the controller log/json.
- **Diagnosis (static):** design was semantically faithful, but the routing epilogue leaned
  on version-sensitive Triton constructs — 3D `tl.reshape([BM,256]→[BM,8,32])`,
  `tl.max(axis=2, keep_dims=True)` over a 3D tile, and `tl.broadcast_to` for the group→expert
  mask expansion. The uniform compile-time failure across every shape (independent of `T`)
  points to a **kernel compilation/lowering error**, not a data or numerical issue. The
  first-launched `T=2048` failing identically to all others is consistent with a compile
  error rather than a runtime OOB.
- **Correction for c002:** rewrite the epilogue entirely in **2D ops** on the `[BM,256]`
  score row (no 3D reshape, no `axis=2` reductions, no `tl.broadcast_to`):
  - group id per expert: `gid = ecol // 32` (`ecol = arange(0,256)`).
  - group top-2: for each group `g in 0..7`, `mg = where(gid==g, sr, _NEG)`; `m1 =
    max(mg,axis=1)`; mask the lowest-index argmax; `m2 = max(...)`; accumulate
    `group_scores[:,g]` by writing into an `[BM,8]` tile via `where(col8==g, m1+m2, ...)`.
    (Or keep group scores as 8 separate `[BM]` scalars combined at the end.)
  - top-4 groups: iterative argmax over the 8 group scores (2D), building a per-expert
    `keep` mask via `keep |= (gid==gsel)`.
  - expert top-8 and weights: unchanged 2D logic already used in c001 (that part is
    2D-only and is not suspected).
  - Keep bf16 `tl.dot` + fp32 accumulate; keep lowest-index tie-break; keep M-masking.
- **Note:** because c001 did not run, the open questions from draft §4.4/§8 (evaluator
  order-strictness, safe precision floor, bf16 flip-rate) remain **unmeasured**; c002 is
  still a correctness anchor, not yet a tuning step.

### c002 — REJECT (eval 2/100)
- **Result:** 0/16 pass, all `RUNTIME_ERROR` — identical pattern to c001 (`valid=false`,
  `ref_ms/sol_ms=null`, `max_abs/max_rel=0.0`, `return_code=1`,
  `foreign_process_detected=false`). Genuine consumed eval, not rc-3.
- **What changed vs c001:** routing epilogue rewritten to pure 2D ops (removed 3D
  `tl.reshape`, `axis=2` reductions, `tl.broadcast_to`); projection switched to fp32
  `tl.dot(input_precision="tf32")`. Removing the 3D constructs **did not** fix the failure.
- **Revised diagnosis:** the c001 hypothesis (3D constructs) was wrong/incomplete — the
  fault is **common to both versions**. The evaluator surfaces no traceback and I cannot run
  a private harness (CLAUDE.md forbids direct CUDA / alternate harness), so the cause must
  be narrowed **empirically, one variable per candidate**. Ranked shared suspects:
  1. **Resource exhaustion (most likely).** Both versions hold ~4–5 live `[BLOCK_M=32,256]`
     fp32 tiles (`acc`, `scores`, `sr`, `keep_expert`, `masked/ms`) simultaneously with only
     `num_warps=4` (128 threads). That is ~256 fp32 slots/thread/tile → well over the
     255-register limit → "out of resource" at launch. Config-dependent ⇒ uniform across all
     `T`, exactly what we see.
  2. Unsupported `tl.dot` precision/shape on the installed Triton.
  3. An epilogue idiom (Python list-of-tensors across unrolled loops, or the
     `tl.min(tl.where(...), axis=1)` argmax pattern).
- **Correction for c003:** test suspect (1) with the cheapest high-value change —
  `num_warps=8` (halves fp32 registers/thread) and `BLOCK_K=128`, keeping `BLOCK_M=32` and
  identical 2D semantics. If c003 still errors, pivot to a **two-kernel split** (K1:
  GEMM+sigmoid+bias→gmem; K2: routing) to localize the failing kernel and cut per-kernel
  live-tile pressure.
- **Open questions** (order-strictness, precision floor, bf16 flip-rate) remain unmeasured;
  no candidate has yet run to completion.
