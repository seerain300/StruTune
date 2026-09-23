# Executable Optimization Plan — L1/058 MoE Expert Token Radix Sort + Prefix Sum

Target: NVIDIA A800 (`sm_80`, Ampere). Primary implementation = **Triton**; PyTorch only for
tensor allocation/metadata/launch plumbing. **No** Torch/CPU/NumPy/CUDA-extension computational
fallback. Submission entry point: `solution/solution.py::run(topk_idx) -> (sorted_token_indices,
expert_offsets)`.

This plan operationalizes `docs/draft.md`. It defines: the fixed pipeline architecture, a sequential
candidate lineage with per-candidate hypotheses and accept/reject rules, correctness checks, the
stopping criteria, and the evidence schema. **No candidate is implemented or evaluated in this
turn.**

---

## 0. Problem recap (authoritative constraints)

- `run(topk_idx: int32[B, S, 8]) -> (sorted_token_indices: int32[N], expert_offsets: int32[257])`
  with `num_experts = 256`, `num_experts_per_tok = 8`, `N = B*S*8`.
- Semantics to reproduce **exactly** (integer, exact-match comparison; atol=1e-5/rtol=0.01 on ints):
  1. `flat = topk_idx.reshape(-1)` (C-order), length `N`, values in `[0,255]`.
  2. `sorted_token_indices = argsort(flat, stable=True)` — **stable** counting sort permutation:
     within each expert bucket the original flat indices appear in **strictly ascending** order.
  3. `expert_offsets[0] = 0`; `expert_offsets[e+1] = sum_{j<=e} count[j]` (inclusive prefix of the
     256-bin histogram). `expert_offsets[e]:expert_offsets[e+1]` = expert `e`'s output slice.
- Feedback workloads (5 fixed): `N ∈ {17920, 18432, 17408, 16384, 16384}` — all ~16–18k, keys
  uniform over 256 experts (mean bucket ≈ 64–72). Flat array ≈ 64–72 KB → fully L2-resident on A800.

---

## 1. Fixed pipeline architecture

Two logical stages with a hard cross-stage dependency (offsets must be complete before scatter can
compute bucket bases), so ≥2 kernel launches are required.

### Stage A — histogram + prefix sum → `expert_offsets[257]`
Compute the 256-bin histogram of `flat`, then an inclusive prefix sum, writing
`expert_offsets[0]=0` and `expert_offsets[1:257]=inclusive_cumsum`. Purely order-independent
(commutative counting), so any histogram method is stability-safe.

### Stage B — stable scatter → `sorted_token_indices[N]`
Given `expert_offsets`, place each flat index into its expert's slice in ascending-index order.
**Scheme S1 (expert-parallel sequential scan)** — decisive stability property:
- Grid = 256 programs, program `e` owns expert `e`.
- `base = expert_offsets[e]`; `running = 0`.
- Loop over `flat` in tiles of `BLOCK`, in **ascending index order**:
  - `v = load(flat + off, mask)`; `m = (v == e) & mask` (masked/tail lanes never match).
  - `excl = tl.cumsum(m.to(int32)) - m.to(int32)`  (exclusive within-tile prefix).
  - `pos = base + running + excl`; `store(out + pos, off, mask=m)`  (`off` = ascending flat index).
  - `running += tl.sum(m.to(int32))`.
- Correctness: tiles ascending × lanes-within-tile ascending ⇒ matched indices written in strictly
  ascending order ⇒ **stable by construction**. Each program writes only `[base, base+count)` —
  disjoint ranges ⇒ **no output atomics, no write races**, fully deterministic.
- Cost: O(E·N) compares (~4.7M), but `flat` is L2-resident and reused by all 256 programs ⇒
  L2-bandwidth-bound; 256 blocks over 108 SMs ≈ 2–3 blocks/SM.

Contingency **Scheme S2 (tiled counting sort, O(N) reads)** is held in reserve (see §3, c00x) only
if evidence shows Stage B dominates and the E·N reads are the bottleneck.

**Rejected:** atomic-counter scatter (`pos = atomicAdd(write_pos[e],1)`) — non-stable ⇒ exact-match
fail. Never used.

---

## 2. Candidate lineage strategy

Rules (from CLAUDE.md / TASK.md):
- Immutable IDs `c001, c002, …`; **one source version per ID**; never reuse an ID for changed source.
- Each candidate = one immutable `solution/solution.py`, evaluated once via
  `./scripts/evaluate_candidate.sh feedback cNNN` (5 workloads = 1 evaluation).
- Record every evaluated candidate as one JSON line appended to `candidates.jsonl` (§6); never
  rewrite prior lines.
- Budget: 100 evaluations; token soft/normal/hard = 1.0M / 1.5M / 1.65M. Expect to converge in far
  fewer than 100 evals.

Lineage shape: a **correctness-first trunk** (c001), then a **breadth phase** that isolates which
stage dominates (one variable changed per candidate), then a **depth/tuning phase** on the winning
branch. Parent = the best correct-and-faster ancestor. Do not branch off a broken/rejected node
except to fix the specific defect.

```
c001 (correct baseline: robust histogram + cumsum + S1 scatter)
  ├─ c002  scatter tuning (BLOCK/num_warps)         [depends c001 pass]
  ├─ c003  Stage-A launch reduction / fusion
  ├─ c004  histogram method swap (atomic ↔ tl.histogram ↔ grid-1)
  ├─ c005  S2 tiled counting sort (only if Stage B dominates)
  └─ c006+ depth tuning on best branch (block/warps/stages, expert-range packing)
```

The concrete next steps after c001 are **evidence-driven** (§3 decision tree), not pre-committed.

---

## 3. Sequential candidate roadmap (executable)

Each step lists: parent, hypothesis, the single change, what to check, accept/reject rule.

### c001 — Correctness baseline (robustness over speed)
- Parent: none.
- Change: implement the full pipeline choosing the **most universally-available** Triton primitives
  so the first eval cannot fail on a version/feature gap:
  - **K1 histogram (grid = ceil(N/BLOCK))**: `tl.atomic_add(counts_ptr + v, 1, mask)` into a
    pre-zeroed global `counts int32[256]` (`torch.zeros` = allowed plumbing). Atomics on counts are
    commutative ⇒ stability-safe.
  - **K2 prefix (grid = 1)**: load `counts[256]`, `inc = tl.cumsum(counts)`; store
    `expert_offsets[0]=0`, `expert_offsets[1:257]=inc`.
  - **K3 scatter (grid = 256)**: Scheme S1 as in §1, `BLOCK=1024`, `num_warps=4`.
  - Conservative launch params everywhere; no `tl.histogram` yet (kept for c004 once availability is
    confirmed via a passing baseline).
- Hypothesis: correct on all 5 workloads; likely already ≥ baseline because it replaces a generic
  `torch.sort` + `bincount` + `cumsum` (3+ generic launches) with range-specialized counting sort.
- Check: §4 pre-eval invariants; then eval — require **5/5 correct**.
- Accept/reject: if 5/5 correct → c001 becomes trunk parent, record geomean. If any workload fails
  correctness → **do not tune**; diagnose (most likely prefix off-by-one, tail mask, or flatten
  order) and issue a corrected new ID (c002) before any performance work.

### c002 — First performance lever (chosen by c001 evidence)
- Parent: c001 (if correct).
- Decision input: from c001 per-workload speedups + reasoning about where time goes at N~18k.
  Two likely branches:
  - **(B-scatter branch)** if scatter is suspected dominant → sweep `BLOCK ∈ {512,1024,2048}` and
    `num_warps ∈ {2,4,8}` for K3 (pick one concrete combo per candidate; one change per ID).
  - **(A-launch branch)** if launch overhead of 3 small kernels dominates (very plausible at 18k) →
    **fuse K1+K2** into a single grid-1 histogram+cumsum kernel (loop-accumulate a 256-bin
    histogram, then cumsum in-program), cutting the pipeline to **2 launches**.
- Hypothesis: fewer launches and/or better-shaped scatter reduce wall time; N is tiny so launch
  overhead is expected to be the largest single term ⇒ the A-launch branch is the strong prior.
- Accept/reject: keep only if **5/5 correct AND geomean ≥ parent** (within noise, treat <1% as tie
  and prefer fewer kernels / simpler code).

### c003 — Second lever (the branch not taken in c002)
- Parent: best of {c001, c002}.
- Change: apply the other lever from c002 (if c002 fused launches, now tune scatter BLOCK/warps; if
  c002 tuned scatter, now fuse launches). One change vs parent.
- Accept/reject: as c002.

### c004 — Histogram method / Stage-A utilization
- Parent: current best.
- Change: swap Stage-A histogram implementation to test utilization vs launch trade-off, e.g.
  `tl.histogram` per tile (now that baseline confirmed the toolchain), or parallel atomic histogram
  with `num_warps`/`BLOCK` tuning, or reduce to a single fused grid-1 count+cumsum. One method per ID.
- Accept/reject: as above.

### c005 — Scatter algorithm change (only if warranted)
- Parent: current best.
- Precondition: only pursue if evidence indicates Stage B (E·N scan) is the bottleneck (e.g. scatter
  BLOCK/warps tuning in c002/c003 moved geomean materially, implying scan-bound).
- Change: **Scheme S2** tiled counting sort — per-tile histograms `H[tiles][256]`; cross-tile scan to
  `start[t][e] = expert_offsets[e] + Σ_{t'<t} H[t'][e]`; scatter each tile using intra-tile keyed
  rank (T×T comparison or segmented scan). Reduces reads to ~2N. Higher complexity + a
  `tiles×256` intermediate.
- Accept/reject: keep only if **5/5 correct AND geomean ≥ best by a clear margin (>2%)**; otherwise
  revert to S1 branch (S2's added complexity must earn its place).

### c006+ — Depth tuning on the winning branch
- Parent: current best.
- Changes (one per ID): `num_stages` sweep; `num_warps` fine-tune; expert-range packing (each scatter
  program owns a small contiguous range of experts to trade grid size vs redundant reads); optional
  fusion of leading-zero write; `BLOCK` fine steps around the c002/c003 optimum.
- Stop per §5 when geomean plateaus.

**Invariant across all candidates:** exactly one meaningful source/config/launch change per new ID;
never edit an already-evaluated ID's source.

---

## 4. Correctness checks

### 4.1 Pre-eval (static reasoning before every candidate)
Confirm in the source/diff before spending an evaluation:
1. **Flatten order** is C-contiguous: `flat = topk_idx.reshape(-1)` (add `.contiguous()` defensively);
   scatter uses the flat index `i = ((b*S)+s)*8+k` as the stored value.
2. **Stability**: Stage-B writes ascending flat indices within each bucket (tile order ascending +
   in-tile lane order ascending; exclusive within-tile prefix; monotonically increasing `running`).
3. **Prefix bookkeeping**: `expert_offsets[0]==0`; inclusive prefix for `[1:257]`; per-element base =
   **exclusive** prefix `expert_offsets[e]` + exclusive running count. No inclusive/exclusive mixups.
4. **Output shapes/dtypes**: `sorted_token_indices` length exactly `N`, int32; `expert_offsets` length
   exactly `257`, int32.
5. **Tail masking**: masked/OOB lanes use a sentinel outside `[0,255]` (or are excluded by `mask`) so
   they are never counted in the histogram and never match any expert in scatter.
6. **Empty experts**: expert with 0 tokens writes nothing and keeps
   `expert_offsets[e]==expert_offsets[e+1]` (S1 handles this automatically: `count==0` ⇒ no stores).
7. **Zero-init**: any global histogram buffer is zeroed (`torch.zeros`) before K1.
8. **No computational fallback**: no `torch.sort/bincount/cumsum/argsort` in the compute path; only
   allocation/reshape/view/contiguous/grid-math in Python.

### 4.2 Eval-time (via the trusted evaluator only)
- Run `./scripts/evaluate_candidate.sh feedback cNNN` — the only permitted execution path. It reports
  per-workload correctness (exact-match vs reference) and speedup, plus geomean.
- **Correctness gate**: a candidate is *valid* only if **all 5 workloads pass**. A candidate that
  fails any workload is invalid for ranking regardless of speed.
- Never run CUDA/profiler/`nvidia-smi`/python/`final`/any alternate harness directly.

### 4.3 Invariants to eye-check from the evaluator output / reasoning
`expert_offsets[0]==0`, `expert_offsets[256]==N`, offsets non-decreasing; `sorted_token_indices` is a
permutation of `0..N-1`; each bucket strictly ascending. (These follow from the exact-match pass but
are the debugging lens if a workload fails.)

---

## 5. Performance hypotheses & how each is tested

| ID | Hypothesis | Lever | Test / expected signal |
|----|-----------|-------|------------------------|
| H1 | Specialized counting sort beats generic `torch.sort`+`bincount`+`cumsum` mainly by cutting launch/generic-sort overhead at N~18k | whole pipeline (c001) | c001 geomean > 1.0 |
| H2 | Launch overhead of small kernels is the dominant term at N~18k | fuse to 2 launches (c002/c003) | fusing K1+K2 raises geomean |
| H3 | Stage-B scatter is L2-bandwidth/scan bound, not compute bound | scatter BLOCK/warps sweep (c002/c003) | little/large sensitivity to BLOCK reveals bound |
| H4 | `tl.histogram`/parallel atomic improves Stage-A utilization negligibly (Stage A is tiny) | histogram swap (c004) | flat geomean ⇒ Stage A not on critical path |
| H5 | S2 O(N)-read scatter only helps if E·N reads dominate (unlikely at 18k) | S2 (c005) | geomean gain >2% ⇒ keep, else revert |
| H6 | Marginal gains from `num_stages`/`num_warps`/expert-range packing | depth tuning (c006+) | small monotone geomean creep to plateau |

Ranking metric = **geometric mean speedup across the 5 workloads**, gated on 5/5 correctness. Treat
geomean deltas <1% as noise/tie (prefer the simpler candidate on ties).

---

## 6. Evidence format (`candidates.jsonl`)

Append exactly one JSON object per evaluated candidate (never rewrite earlier lines). Schema:

```json
{
  "id": "c001",
  "parent": null,
  "source_sha256": "<sha256 of solution/solution.py at eval time>",
  "timestamp": "<ISO-8601>",
  "hypothesis": "correctness baseline: atomic histogram + cumsum + S1 expert-parallel stable scatter",
  "change_from_parent": "initial implementation",
  "config": {"hist": "atomic", "scatter": "S1", "BLOCK": 1024, "num_warps": 4, "kernels": 3},
  "validation": {
    "preeval_invariants": "pass (flatten/stability/prefix/shape/tail/empty checked)",
    "correct_workloads": "5/5",
    "per_workload": [
      {"uuid": "13149551-...", "B": 2, "S": 1120, "N": 17920, "correct": true, "speedup": 0.00},
      {"uuid": "2eb27620-...", "B": 8, "S": 288,  "N": 18432, "correct": true, "speedup": 0.00},
      {"uuid": "c84061fe-...", "B": 4, "S": 544,  "N": 17408, "correct": true, "speedup": 0.00},
      {"uuid": "e259b448-...", "B": 8, "S": 256,  "N": 16384, "correct": true, "speedup": 0.00},
      {"uuid": "f761ce94-...", "B": 4, "S": 512,  "N": 16384, "correct": true, "speedup": 0.00}
    ]
  },
  "geomean_speedup": 0.00,
  "decision": "accept|reject",
  "decision_reason": "5/5 correct, geomean ... vs parent ...",
  "cumulative_evaluations": 1,
  "skill_usage": "none (Ampere sm_80; KernelWiki is Blackwell/Hopper-only, not applicable)"
}
```

Notes:
- `source_sha256`: compute with `sha256sum solution/solution.py` (plumbing only) and record it so the
  ID↔source binding is auditable and immutability is verifiable.
- Fill `speedup`/`geomean_speedup` from the evaluator's actual report; the `0.00`s above are
  placeholders for the schema, not results.
- `cumulative_evaluations` is monotone (each candidate = +1). `decision` ∈ {accept, reject}; `parent`
  points at the ancestor the change was made against.

---

## 7. Stopping / convergence criteria

Stop the search and write `SEARCH_COMPLETE` (with the reason) when **any** holds:
1. **Convergence**: best geomean improves by <1% across two consecutive accepted candidates AND both
   remaining structural levers (histogram method, scatter algorithm) have been tried and rejected.
2. **Exhausted levers**: c001–c006 branch fully explored, depth tuning plateaued, no untried
   hypothesis with a credible >1% upside remains.
3. **Budget**: approaching the token soft limit (1.0M) with no active improving branch, or nearing
   the 100-evaluation cap.
4. **Regression floor**: repeated attempts fail to beat the current best and further changes only add
   complexity/risk.

At stop: the best **valid** (5/5 correct) candidate is the selected result. `solution/solution.py`
must hold that candidate's exact source. **Never run `final` without explicit operator approval.**

---

## 8. Skill usage

`KernelWiki` covers NVIDIA **Blackwell (SM100)** and **Hopper (SM90)** kernels; this task targets
**Ampere A800 (sm_80)** and is an integer counting-sort/prefix-sum problem — outside that skill's
scope. No skill invoked. If a Triton primitive/version question blocks c001, the first evaluation's
feedback is the empirical probe (baseline deliberately uses only broadly-available primitives).

---

## 9. Risk register & rollback

| Risk | Mitigation | Rollback |
|------|-----------|----------|
| Stability mismatch | S1 ascending-order scatter (design-guaranteed); pre-eval check #2 | revert to last correct ID; fix scatter ordering under new ID |
| Prefix off-by-one | pre-eval check #3 (exclusive base + inclusive offsets, leading 0) | new ID with corrected indexing |
| `tl.histogram` unavailable | c001 avoids it (atomic add) | keep atomic histogram branch |
| Tail-lane contamination | sentinel + mask (check #5) | new ID fixing mask |
| A change regresses geomean | one change per ID + accept gate (5/5 & ≥parent) | keep parent as best; abandon branch |
| Over-engineering (S2) for tiny N | S2 gated behind >2% margin (c005) | stay on S1 |

---

## 10. Immediate next action (next turn, not now)

Implement **c001** exactly per §3 (atomic histogram K1 + cumsum K2 + S1 scatter K3, `BLOCK=1024`,
`num_warps=4`), run pre-eval checks §4.1, then `./scripts/evaluate_candidate.sh feedback c001`,
and append the c001 record per §6. Proceed down §3 based on the observed evidence.

---

## 11. Decision log / learnings

### c001 — REJECTED (STATIC_CHECK_FAILED, no GPU run)
- Error: `computational torch operators are forbidden in the submitted path`.
- Root cause: `run()` used `flat = topk_idx.reshape(-1).contiguous()` and
  `flat = flat.to(torch.int32)`. The evaluator's **static allowlist** rejects `.contiguous()` and
  `.to()`/dtype-cast as "computational" torch ops — even though they are plumbing in intent. The
  kernels themselves were never reached; this is a pure host-side plumbing rejection, not a kernel
  bug and not a stability/prefix defect.
- **New constraint discovered (applies to all future candidates):** the submitted `run()` path may
  use only the narrowest tensor plumbing. Confirmed-forbidden so far: `.contiguous()`, `.to(dtype)`.
  Safe-so-far plumbing to rely on: `torch.empty`, `torch.zeros` (allocation), `.reshape(-1)` /
  `.view(-1)` (shape view), `.numel()`, `.device`, Triton launch/grid math. Do **not** cast dtype or
  call `.contiguous()` in `run()`.
- **c002 fix (next candidate):** identical kernels to c001, but strip the two offending calls:
  - Flatten with a pure view: `flat = topk_idx.reshape(-1)` (input from `torch.randint(...,
    dtype=int32)` is already contiguous int32, so no cast/contiguous needed). Match reference
    exactly, which also just does `topk_idx.reshape(-1)`.
  - Keep everything else (atomic hist K1 + cumsum K2 + S1 scatter K3, `BLOCK=1024`, `num_warps=4`).
  - This is the first candidate expected to actually reach the GPU and report per-workload
    correctness + speedup; it is the true correctness baseline for the trunk.
- Note: if a later workload ever delivers a non-int32 / non-contiguous input and a cast is truly
  required, it must be done **inside a Triton kernel** (e.g. load + cast in-kernel), never via a
  host-side torch computational op. Not expected here (spec fixes int32 contiguous input).

### c002 — REJECTED (STATIC_CHECK_FAILED, no GPU run)
- Same error. c002 removed `.contiguous()` and `.to()`, flattening with a pure `reshape(-1)`.
- Still failed → the offender was elsewhere. Remaining prime suspect at the time: `torch.zeros`
  (value-filling / computational). This **falsified** the c001-era assumption that `torch.zeros` is
  "safe allocation" — the static allowlist appears to distinguish `torch.empty` (pure alloc) from
  `torch.zeros` (fills values).
- c003 fix: replace all `torch.zeros` with `torch.empty` + zero-init inside Triton.

### c003 — REJECTED (STATIC_CHECK_FAILED, no GPU run)
- Same error, third time. Removing `torch.zeros` did **not** help → `torch.zeros` was not the (sole)
  offender either.
- **Elimination narrows the culprit.** The torch ops present across ALL three failing versions and
  still present in c003 are: `topk_idx.reshape(-1)`, `torch.empty`, `flat.numel()`, `flat.device`,
  `@torch.no_grad()`. Outputs must be allocated, so `torch.empty` (pure allocation) essentially has
  to be permitted. The leading remaining suspect is **`.reshape(-1)`** — a tensor-transforming op the
  static checker likely flags as computational.
- **Updated allowlist model (working hypothesis, to confirm empirically):** even shape-view ops like
  `.reshape`/`.view` may be forbidden in the submitted path. Only raw allocation (`torch.empty`) +
  attribute reads (`.shape`, `.device`) + Triton launches are assumed safe until proven otherwise.
- **c004 fix (this-was-next):** remove `.reshape` entirely. Pass the contiguous 3-D `topk_idx`
  directly to the Triton kernels; because it is contiguous, `data_ptr + linear_offset` walks memory
  in exactly C-flattened order, so kernels are unchanged in behavior. Compute `N` from
  `topk_idx.shape` via pure Python int multiply (`b*s*k`), not `.numel()`. Keep `torch.empty` for the
  three output/scratch buffers; keep in-kernel zero-init. If c004 *still* fails the static check, the
  next elimination step is to probe whether `torch.empty` itself or `@torch.no_grad()` is flagged
  (e.g. by minimizing run() to the smallest possible torch surface).
