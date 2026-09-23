# Plan — L1/002 VAE Conv3x3 → GroupNorm → SiLU → (×2) → Residual Add

Executable, sequential KDA plan. Builds on `docs/draft.md`. This turn produces the plan
only — no candidate is implemented or evaluated here.

Target: H100 `sm_90`, FP32 I/O. Primary metric: geomean speedup vs the PyTorch reference
over the 20 feedback workloads, subject to per-workload correctness (atol ≈ 0.0027–0.0034,
rtol 1e-5).

---

## 0. Operating rules (binding on every step)

- One immutable candidate = one source version. Any meaningful source/config/launch change
  ⇒ new candidate ID (`c001`, `c002`, …), sequential, never reused, never rewritten.
- Compute must be Triton. PyTorch only for metadata / allocation / launch plumbing. No
  Torch/CPU/NumPy/CUDA-ext compute fallback. A failing Triton kernel is invalid and is
  fixed in a *new* candidate, never replaced by a Torch path.
- Evaluate only via `./scripts/evaluate_candidate.sh feedback <id>`. Full 20-workload set =
  1 evaluation. Budget: 100 evaluations. Token budget: soft 9M / normal 10M / absolute 11M.
- Profiling only via `./scripts/ncu_profile.sh` (ncu-report-skill). **Never** profile while
  an evaluation runs (foreign process ⇒ RC 3, wasted eval). Serialize the two.
- `final` only after explicit operator approval.
- After each eval, append exactly one JSON record to `candidates.jsonl` (schema in §7).
- Change **one axis per candidate** so each eval result attributes cleanly.

---

## 1. Strategy overview

Two-phase search:

- **Phase A — Establish a correct, fully-Triton baseline (c001–c00x).** Get all 20
  workloads passing tolerance with a straightforward, clearly-correct kernel. Do not chase
  speed yet. This de-risks the numerics (TF32 acceptability, GroupNorm variance, padding,
  residual) and gives a per-shape latency/speedup baseline to optimize against.
- **Phase B — Optimize measured bottlenecks (c00x onward).** Guided by ncu, change one
  lever per candidate: conv formulation, data layout, GroupNorm reduction strategy, fusion
  boundaries, autotune tiles/warps/stages. Keep only Pareto-improving candidates as parents.

Guiding cost model (from draft §3): convs are the FLOP floor and hard to beat cuDNN on;
the *reliable* win is eliminating the reference's ~9–10 memory passes/launches by fusing
the norm+SiLU+residual epilogues. So Phase A must already fuse epilogues; Phase B mostly
tunes the conv and the GroupNorm reduction.

Lineage is a tree, not a chain: each candidate names a `parent`. A candidate becomes the
new baseline (parent for the next) only if it is correct **and** improves geomean (or is
geomean-neutral but strictly simpler/faster on the worst shape). Losing experiments are
recorded and abandoned; the parent pointer stays on the last winner.

---

## 2. Candidate roadmap (planned; IDs assigned at implementation time)

Each entry: hypothesis → concrete change → success criterion → decision rule. Only c001 is
fully specified; later entries are conditional and may be reordered/dropped based on
evidence. At most a handful of the "if-needed" branches will actually be spent.

### c001 — Correctness-first fully-Triton baseline  (parent: none)
- **Hypothesis:** A fused Triton implementation with TF32 tensor-core convs passes all 20
  tolerances and, via epilogue fusion + fewer launches, is already ≥ ~0.8–1.0× geomean.
- **Design (simplest clearly-correct version):**
  - Conv as **9× shifted 1×1 GEMM** accumulation (formulation B from draft §6.1): loop the
    9 (kh,kw) taps, each a `[M×C_in]·[C_in×C_out]` `tl.dot`, accumulated in FP32, with
    edge/zero-pad masking. Weight reshaped `(256,256,3,3)`→per-tap `(256,256)` via metadata.
    Chosen over full im2col first because per-tap masking and contiguous channel access are
    easier to get provably correct.
  - `tl.dot(..., input_precision="tf32")`, FP32 accumulators, FP32 output buffers.
  - GroupNorm via a **two-kernel split reduction**: (1) per (batch, group, spatial-tile)
    partial (count, mean, M2) using Welford; (2) combine partials → mean/rstd; then a
    normalize+affine+SiLU kernel. This scales to both 32-group and 2048-group regimes.
    (If a single-pass per-group Welford is simpler to land first and occupancy is
    acceptable, that is an allowed c001 simplification — but the split path is preferred
    because the 1024²/768² 32-group cases otherwise idle ~100 SMs.)
  - Stage structure: conv1→buf1; gn1 stats; gn1 apply+SiLU→act1; conv2→buf2; gn2 stats;
    gn2 apply+SiLU+**residual add of original x**→output. `x` never overwritten.
  - Biased variance (÷N), `rstd = rsqrt(var+eps)`, affine before SiLU, `tl.sigmoid`.
  - NCHW-direct index math initially (no transpose pass) to minimize moving parts; layout
    is a Phase-B lever.
- **Success:** all 20 workloads correct.
- **Decision:** if correct → becomes baseline, proceed to profiling then Phase B. If any
  workload fails tolerance → diagnose (likely TF32 or variance) and spawn c002 as the fix
  (one change). If it fails to compile/run → fix in c002; never fall back to Torch.

### c002 — Numerical fix *or* first optimization (parent: c001)
- **If c001 failed correctness:** apply the single most-likely fix:
  - tolerance failure on large-reduction shapes (1024², 768², 293²) ⇒ verify Welford merge;
    if still failing, switch the failing matmul to `input_precision="ieee"` (true FP32) —
    measure the speed cost.
  - failure concentrated near edges/odd dims ⇒ fix padding mask; re-run.
- **If c001 passed:** begin Phase B with the highest-leverage lever indicated by ncu
  (expected: conv formulation or layout). See c003+ menu; pick one change only.

### Phase-B optimization menu (each = one candidate, one axis, parent = current best)
Ordered by expected leverage; actual order set by ncu evidence.

1. **Conv formulation A vs B.** Try full implicit im2col GEMM (K=2304, single big `tl.dot`)
   against the 9-shift baseline. Hypothesis: fewer, larger GEMMs raise tensor-core
   utilization on the compute-bound shapes (1024², 768², B=64). Success: geomean ↑ with all
   correct.
2. **Layout NCHW-direct vs explicit NHWC transpose.** Add a transpose-in / transpose-out
   plumbing pass so the GEMM contraction (C_in) is contiguous. Hypothesis: coalesced K
   loads outweigh the transpose passes on large shapes. Success: geomean ↑; watch that
   small shapes don't regress from the extra passes.
3. **GroupNorm reduction tuning.** Tune spatial-tile size / programs-per-group split for the
   bimodal regime (32 huge groups vs 2048 small). Possibly regime-select at launch from
   `B*32` vs SM count. Success: better occupancy (ncu) and geomean ↑, esp. 1024²/768².
4. **Autotune GEMM tiles/warps/stages.** `BLOCK_M/N/K`, `num_warps`, `num_stages` configs
   keyed on regime (large-batch-small-spatial vs small-batch-large-spatial). Use Triton
   autotune with a config set informed by KernelWiki SM90 GEMM guidance. Success: geomean ↑.
5. **Deeper fusion / fewer temporaries.** Fold conv1's output write into stage-1 GroupNorm
   partial accumulation, and/or fuse gn-apply+SiLU directly after conv where the reduction
   allows. Hypothesis: removes 1–2 full tensor passes. Success: geomean ↑, correctness held.
6. **Conv weight precision / caching.** Ensure weights (2.36 MB, L2-resident) are loaded
   efficiently; consider pre-reshaping once. Minor; only if ncu shows weight-load stalls.

Each menu item may spawn follow-ups only if it wins; otherwise it is abandoned and the
parent pointer stays put.

---

## 3. Correctness checks (per candidate, before spending an eval)

Paper/desk review checklist — must pass before each eval (we have no independent harness):

1. **Shapes/strides:** output `(B,256,H,W)` NCHW contiguous; weight reshape matches
   `(C_out, C_in, kh, kw)` semantics for the chosen formulation.
2. **Padding:** all masked (out-of-range) input taps contribute exactly `0.0`; check 4
   edges + 4 corners + odd dims (131, 293).
3. **GroupNorm math:** reduction over the correct contiguous `8·H·W` block per (b,group);
   biased variance (÷N); `rstd=rsqrt(var+eps)`; partial-merge uses Welford/parallel formula
   (not naive E[x²]−E[x]²); affine per output channel; order = normalize→affine→SiLU.
4. **SiLU:** `x*tl.sigmoid(x)` (stable form).
5. **Residual:** added after the *second* SiLU, from the *original* `x`; `x` untouched.
6. **Dtype:** FP32 accumulators and FP32 output; TF32 only inside `tl.dot`.
7. **eps:** inside the sqrt.

Then the evaluator's built-in tolerance check across all 20 workloads is the authoritative
correctness verdict. Record pass/fail per workload from its output.

---

## 4. Performance hypotheses (falsifiable, tied to evidence)

- **H1 (fusion win):** collapsing the ~9–10 reference passes into ~4–5 fused passes yields
  a speedup that grows with tensor size; the largest single-image shapes (1024², 768²)
  benefit most in absolute ms. Test: per-shape speedup vs size trend in eval output.
- **H2 (conv is the ceiling):** on compute-bound shapes the Triton conv, not the epilogue,
  bounds speedup; if a shape is < 1.0× it is because the conv GEMM trails cuDNN. Test: ncu
  tensor-core utilization + roofline on 1024²/B=64 shapes.
- **H3 (TF32 sufficient):** TF32 convs pass all tolerances. Test: c001 eval; if it fails,
  H3 is falsified for specific shapes → selective IEEE matmul.
- **H4 (layout):** NHWC contiguous-K beats NCHW-strided-K on large shapes by enough to pay
  for the transpose passes. Test: c00x layout candidate geomean + ncu memory throughput.
- **H5 (GroupNorm occupancy):** split reduction keeps the 32-group cases from idling SMs.
  Test: ncu achieved occupancy on B=1,1024² before/after split tuning.

Each hypothesis is confirmed or rejected by a specific eval or ncu run and logged in the
candidate record's `notes`.

---

## 5. Profiling protocol (ncu-report-skill)

- Invoke only via `./scripts/ncu_profile.sh <ncu args…>`. Never while an eval is running.
- Use for: (a) confirming compute- vs memory-bound per representative shape, (b) tensor-core
  utilization of conv GEMMs, (c) achieved occupancy for the bimodal GroupNorm, (d) DRAM
  throughput to validate the fusion traffic argument.
- Representative shapes to profile (cover the regimes): `B=1,1024²` (few huge groups,
  compute-heavy), `B=64,64²` (many groups, large batch), `B=4,128²` (mid), `B=1,131²`
  (odd/boundary). Do not profile all 20.
- Consult **KernelWiki** before conv/GEMM tiling and warp-specialization decisions on SM90;
  cite the specific guidance/PR in the candidate `notes`.
- Order of operations each cycle: implement → desk-check (§3) → eval (records result) →
  (later, separately) profile the current best → decide next lever. Never overlap eval and
  profile.

---

## 6. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any of:

1. **Convergence:** 3 consecutive new candidates fail to improve geomean by > ~1% over the
   current best (and no untried high-leverage lever remains).
2. **Budget:** approaching the evaluation budget (reserve ≥ 2 evals margin) or the token
   soft limit (9M) — wind down, record best, stop before hard limits.
3. **Ceiling reached:** ncu shows conv GEMMs near tensor-core roofline and epilogues fully
   fused (memory-bound at HBM roofline) — no structural headroom left.

At stop: identify the best correct candidate by geomean, ensure its record is complete,
write `SEARCH_COMPLETE` with the rationale and the winning candidate id. Do **not** run
`final` — that requires explicit operator approval.

---

## 7. Evidence format (`candidates.jsonl`)

Append exactly one JSON object per evaluated candidate, in evaluation order; never edit
prior lines. Schema:

```json
{
  "candidate": "c001",
  "parent": null,
  "timestamp": "2026-09-21T00:00:00Z",
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "TF32 fused Triton baseline passes all tolerances and beats reference via epilogue fusion",
  "change_from_parent": "initial implementation",
  "validation": {
    "desk_check_passed": true,
    "compiled": true,
    "all_workloads_correct": true
  },
  "results": {
    "per_workload": [
      {"uuid": "7b00c2c8-...", "B": 16, "H": 64, "W": 64, "correct": true, "speedup": 0.00, "latency_ms": 0.0}
    ],
    "geomean_speedup": 0.00,
    "worst_workload": {"uuid": "...", "speedup": 0.00},
    "num_correct": 20
  },
  "decision": "keep-as-baseline | reject | fix-in-next",
  "cumulative_evals": 1,
  "skills_used": ["KernelWiki", "ncu-report-skill"],
  "notes": "hypotheses confirmed/rejected; ncu findings; next lever"
}
```

Rules:
- `source_sha256` recomputed from the actual `solution/solution.py` used for that eval.
- `per_workload` includes every workload with correctness and speedup; capture from the
  evaluator's own output (do not fabricate).
- `cumulative_evals` monotonic; equals the number of eval invocations spent so far.
- `decision` states whether this candidate becomes the new parent.
- `skills_used` lists KernelWiki / ncu-report-skill usage for that candidate.
- If an eval is discarded (RC 3 / foreign-process interference), record it honestly as a
  spent eval with `decision: "discarded-rc3"` and no results.

---

## 8. Execution checklist (per candidate cycle)

1. Pick the single next lever (§2 menu) with a written hypothesis (§4).
2. Implement in `solution/solution.py` as a new immutable candidate id.
3. Desk-check correctness (§3).
4. Ensure no profiling job is running; then `./scripts/evaluate_candidate.sh feedback cNNN`.
5. Parse per-workload correctness + speedup + geomean from output.
6. Append the record to `candidates.jsonl` (§7).
7. If kept as baseline: (separately, no eval running) profile with ncu to choose the next
   lever. If rejected: keep parent pointer, choose an alternative lever.
8. Check stopping criteria (§6). If met, write `SEARCH_COMPLETE`.

Next turn: implement and evaluate **c001** per §2.
