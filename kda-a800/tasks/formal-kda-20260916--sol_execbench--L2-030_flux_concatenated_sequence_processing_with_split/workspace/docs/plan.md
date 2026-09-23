# Executable Optimization Plan — L2/030 Flux Concatenated Sequence Processing with Split

Run: `formal-kda-20260916--sol_execbench--L2-030_flux_concatenated_sequence_processing_with_split`
Target: NVIDIA A800 (`sm_80`, Ampere). Triton primary; PyTorch only for metadata/launch plumbing.
No Torch/CPU/NumPy/CUDA-extension computational fallback (a failed Triton kernel is invalid, not
fallback-able).

This plan operationalizes `docs/draft.md`. It fixes the algorithm, the file/API contract, the exact
candidate lineage, correctness gates, performance hypotheses (each falsifiable), stopping criteria,
and the append-only evidence schema. **No candidate is implemented or evaluated in this turn.**

---

## 0. Fixed decisions carried from the draft (do not re-litigate mid-search)

1. **Algebra:** replace `cat → matmul → split` with **two independent per-stream GEMMs**
   `Y = X @ Wᵀ`, `K = N = H = 3072`. This is row-for-row identical to the reference (see draft §2)
   and deletes the `cat` allocation/copy and the `split` slice-copy.
2. **Precision:** primary `input_precision="tf32x3"` (near-fp32, tensor-core speed). Accuracy floor
   `input_precision="ieee"` (true fp32 CUDA-core). **Never** plain `"tf32"` in a submitted candidate.
3. **Launch:** start with **two launches** (one per stream), weight `W` reused hot across both.
4. **Accumulator/outputs:** always fp32. Two output tensors of exact reference shapes.

These are the search's fixed frame; candidates vary only precision, tiling/occupancy, and launch
strategy, one axis at a time.

---

## 1. Solution contract (`solution/solution.py`)

The submission must expose:

```python
def run(hidden_states, encoder_hidden_states, process_weight):
    # returns (processed_encoder, processed_hidden)
```

matching `task/definition.json`:
- `hidden_states`   : `[B, I, H]` fp32  (image latents)
- `encoder_hidden_states` : `[B, T, H]` fp32  (text conditioning)
- `process_weight`  : `[H, H]` fp32  (`nn.Linear` weight; output = `X @ Wᵀ`)
- returns `(processed_encoder [B, T, H] fp32, processed_hidden [B, I, H] fp32)`, in that order.

Host-side responsibilities (PyTorch allowed here only):
- Read `B, T, I` from tensor shapes; `H = 3072` (const, but read from tensors, never hard-code as a
  correctness dependency).
- Flatten each stream `[B, S, H] → [B*S, H]` **without a copy when contiguous** (use `.reshape`;
  pass strides to the kernel rather than forcing `.contiguous()`). Only copy a stream if it is
  genuinely non-contiguous (not expected for the feedback set — verify, do not assume).
- Allocate the two fp32 output tensors with `torch.empty` in the exact reference shapes.
- Compute launch grids, launch the Triton kernel(s), reshape outputs back to `[B, S, H]`, return.
- Guard degenerate `T==0` or `I==0` (skip that stream's launch; return an empty tensor of the right
  shape). Not in the feedback set but cheap and safe.

Kernel responsibilities (Triton, the only compute path):
- Tiled `Y = X @ Wᵀ`: grid over `(M, N)` tiles, reduce over `K` in `BLOCK_K` chunks, fp32 accumulate,
  `tl.dot(x_tile, w_tile, acc, input_precision=PREC)`.
- `W` indexed as `W[n, k]` to realize the transpose (`Y = X·Wᵀ`); pick the stride pattern that keeps
  both `X` and `W` loads coalesced (`W` is `[H,H]` row-major contiguous).
- Mask the `M` dimension (streams like `M=77`, `M=2846` are not block multiples). `N=K=3072` is a
  multiple of 64/128, so N/K masking can be omitted for speed **only after** confirming every block
  size chosen divides 3072 (64,128,256 all do; 96,192 do too; avoid non-divisors).

---

## 2. Candidate lineage strategy (sequential, immutable IDs)

One candidate = one immutable `solution/solution.py` source, evaluated once over all five fixed
feedback workloads (= one evaluation). IDs are never reused for changed source. Each phase changes
**one** axis so the evaluation attributes cause cleanly. Lineage is a tree rooted at c001; `parent`
is recorded per row.

### Phase A — Correctness + assumption validation (root)
- **c001** — Fused two-launch GEMM, `tf32x3`, fixed conservative tiling
  `BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=3`, grouped-M schedule
  `GROUP_M=8`. Parent: none.
  - **Primary purpose:** (a) prove correctness on all 5 workloads; (b) empirically confirm the
    reference is true fp32 (if `tf32x3` passes the tight atol, the reference is fp32 — see draft §5);
    (c) establish the baseline speedup number.
  - **Branch rules after c001:**
    - c001 passes correctness → proceed to Phase B tuning (parent = c001).
    - c001 fails correctness on ≥1 workload → run **c002 = ieee** variant (same tiling) to separate
      *accuracy* from *bug*:
      - ieee passes everywhere → the failure was `tf32x3` accuracy (i.e. reference tighter than
        assumed / reference is not the fp32 we modeled); adopt `ieee` as the precision floor and tune
        from there.
      - ieee also fails → the failure is a **shape/stride/indexing bug**, not precision; fix the bug
        under a new id (c003…) and re-run. Do **not** touch precision to mask a bug.
    - c001 fails to compile/import (e.g. `tf32x3` unsupported in installed Triton) → **c002 = ieee**
      variant becomes the working root; note the Triton limitation in the row.

### Phase B — Precision confirmation / accuracy floor (only if not already forced)
- If c001 (`tf32x3`) passed, still keep an **`ieee` sibling** available in the plan as a safety net,
  but only spend an evaluation on it if a later tuned `tf32x3` candidate shows any accuracy drift.
  Rationale: don't burn evals confirming what c001 already established; reserve `ieee` as a fallback.

### Phase C — Tiling / occupancy sweep (around the best correct precision)
Change exactly one axis per candidate; parent = current best correct+fastest candidate. Candidate
ordering (skip a step if a prior result makes it clearly dominated):
- Vary `BLOCK_M ∈ {64, 128, 256}` (holding others).
- Vary `BLOCK_N ∈ {64, 128, 256}`.
- Vary `BLOCK_K ∈ {32, 64, 128}` (note `tf32x3` ≈ triples effective K-work; watch smem).
- Vary `num_warps ∈ {4, 8}`.
- Vary `num_stages ∈ {2, 3, 4, 5}` (fp32 tiles are smem-heavy on `sm_80`, 164 KB/SM configurable —
  a too-large stages×block may fail to compile; treat a compile failure as a recorded negative
  result, not a crash to hide).
- Vary `GROUP_M ∈ {1, 4, 8, 16}` for L2 reuse of `W` (37.7 MB ≈ A800 L2).

**Efficiency shortcut (token-budget aware):** after 2–3 manual fixed-config probes bound the good
region, introduce **one autotuned candidate** whose source carries a *curated* `triton.autotune`
config list (the survivors from the manual probes). This lets the evaluator's own timing pick the
best config per workload shape within a single immutable source = one evaluation, covering the small
(`M=77`) and large (`M=4096`) extremes without a separate candidate each. This is the preferred way
to converge once the region is bounded; it trades per-config attribution for eval/token efficiency.

### Phase D — Stream specialization (only if evidence demands)
- If the thin encoder streams (`M_enc = 77` in WL3/WL5) measurably drag geomean (their per-workload
  speedup is the weakest), add a candidate that uses a **thin-M config for the encoder launch** and
  a **large-M config for the image launch** (different tiling per stream in the same source). One
  candidate, one eval.

### Phase E — Launch-strategy alternative (only if small-WL overhead shows)
- If small workloads (WL2/WL4) show launch-overhead-dominated timing (poor speedup despite good
  large-WL speedup), try **single virtual-concat launch** (draft §6.2 Option B): one grid over
  `M_total` row-tiles selecting encoder/image source & destination by global row index. Compare
  against the two-launch best. Keep whichever wins geomean.

At every step, if a change regresses geomean or breaks correctness, **discard it** (do not carry it
forward) and continue from the last best. The best-correct candidate id is always tracked.

---

## 3. Correctness checks (gates before and after each evaluation)

**Static pre-eval checklist (before writing/locking each candidate source):**
1. `run` signature and return order/shapes/dtype match §1 exactly (two fp32 tensors, encoder first).
2. Algebraic equivalence preserved: encoder rows → encoder output, image rows → image output, never
   crossed; C-contiguous flatten `[B,S,H]→[B*S,H]` and reshape back preserve row order.
3. `Y = X @ Wᵀ` transpose semantics correct (`W[n,k]`), verified against the reference `matmul(X, W.t())`.
4. All chosen block sizes divide 3072 if N/K masking is omitted; otherwise masking present.
5. `M`-dimension masking present for non-multiple `M` (77, 256, 512, 768, 1024, 2846, 2978, 4096…).
6. Accumulator dtype fp32; no down-cast of inputs, accumulator, or outputs.
7. Precision knob set to the intended `tf32x3` / `ieee` (never plain `tf32` in a submission).
8. No `torch.matmul`/`F.linear`/cuBLAS/CPU/NumPy on the compute path; no fallback branch.
9. Strides passed for non-unit-stride safety; `.contiguous()` only on a proven non-contiguous stream.

**Evaluation-driven correctness gate (the only runtime oracle):**
- Correctness is judged solely by `./scripts/evaluate_candidate.sh feedback cNNN` over the 5 fixed
  workloads. A candidate is **valid** only if **all 5 workloads pass** correctness
  (`|out-ref| ≤ atol + rtol·|ref|` for ≥ 98% elements, per workload tolerances in
  `task/feedback_workloads.jsonl`).
- Numerical margin expectation (draft §5): `tf32x3` error ≈ `5e-5` ≪ budget ≈ `0.00255` at output
  magnitude ≈ 55 → expect ~100% element match, comfortable pass. Any failure is treated as a real
  signal (precision assumption wrong, or a bug) and triaged via the c001 branch rules (§2 Phase A).
- A candidate that fails any workload is **invalid** and cannot be selected as best, regardless of
  its speed.

---

## 4. Performance hypotheses (each falsifiable, tied to a decision)

- **H1 (main lever).** Removing `cat`/`split` + `tf32x3` tensor path beats the reference (true-fp32
  SIMT matmul + cat traffic). Ceiling ≈ `52/19.5 ≈ 2.7×` on matmul plus deleted cat traffic.
  - *Test:* c001 geomean > 1.0×.
  - *Falsifier:* if c001 is **slower** than reference, the reference matmul is likely already
    tensor-accelerated (TF32) — which would also endanger accuracy. Re-examine assumptions
    (inspect per-workload speedups + whether accuracy is marginal) rather than chase tiling.
- **H2 (precision cost).** `tf32x3` (~52 TFLOPS eff) is faster than `ieee` (~19.5 TFLOPS) while
  still passing tolerance.
  - *Test:* if both are run, `tf32x3` geomean > `ieee` geomean and both pass.
  - *Falsifier:* `tf32x3` fails accuracy on any workload → fall back to `ieee` as the floor.
- **H3 (L2 reuse of W).** `W` (37.7 MB ≈ L2) reuse across `M` tiles matters; grouped-M scheduling
  and larger `BLOCK_M` improve large-M workloads (WL1, WL5).
  - *Test:* increasing `GROUP_M` / `BLOCK_M` improves WL1/WL5 speedup without hurting small WL.
  - *Falsifier:* no change or regression → `W` reuse is not the bottleneck; stop tuning that axis.
- **H4 (small-M efficiency).** Tiny encoder streams (`M=77`, WL3/WL5) underutilize a large-`BLOCK_M`
  config and drag geomean.
  - *Test:* WL3/WL5 encoder-side speedup is the weakest; a thin-M / per-stream config lifts it.
  - *Falsifier:* per-stream specialization yields < ~2% geomean gain → not worth the complexity.
- **H5 (launch overhead).** For small workloads (WL2/WL4), two launches add measurable overhead.
  - *Test:* single virtual-concat launch improves WL2/WL4 speedup.
  - *Falsifier:* no improvement → keep the simpler two-launch design.
- **H6 (pipelining).** `num_stages`/`BLOCK_K` tuning improves K-loop overlap, more so given
  `tf32x3`'s 3× K-cost, up to the `sm_80` smem limit.
  - *Test:* a stages/BK setting improves geomean without compile failure.
  - *Falsifier:* best is the conservative baseline → stop.

Metric of record: **geometric mean speedup over the 5 feedback workloads**, plus per-workload
speedups (to attribute large vs small-M behavior). A candidate is only comparable if valid (all 5
pass).

---

## 5. Stopping criteria

Stop the search and write `SEARCH_COMPLETE` (with the reason) when any of:
1. **Convergence:** the best valid geomean improves by **< ~2%** across **3 consecutive** new
   candidates spanning distinct tuning axes (i.e. tiling, launch, and specialization have each been
   probed and stopped paying off).
2. **Evaluation budget:** cumulative evaluations approach **100** (hold a small reserve; do not
   exceed).
3. **Token budget:** approaching the soft limit **1,000,000** tokens — wind down and consolidate;
   never exceed the normal-completion limit **1,500,000** (absolute **1,650,000**). Prefer stopping
   at soft limit with a clean best candidate over squeezing marginal gains.
4. **No valid candidate path:** if neither `tf32x3` nor `ieee` can pass all 5 workloads after bug
   triage, stop and report the blocker (do **not** introduce any non-Triton fallback).

On stop: the recorded best is the **valid candidate with the highest geomean** (correctness is a
hard gate; a faster invalid candidate never wins). `SEARCH_COMPLETE` states the reason (converged /
budget / blocker) and names the best candidate id + its geomean. **Never run `final` without
explicit operator approval** (final = one 16-workload eval, operator-only).

---

## 6. Evidence format (append-only `candidates.jsonl`)

Append exactly **one JSON object per evaluated candidate**, in evaluation order. Never rewrite or
delete a prior line. One object per line (JSONL). Schema:

```json
{
  "id": "c001",
  "parent": null,
  "timestamp": "2026-09-17T00:00:00Z",
  "source_sha256": "<sha256 of solution/solution.py at eval time>",
  "hypothesis": "Fused two-launch tf32x3 GEMM (no cat/split) beats fp32 reference; validates that the reference is true fp32.",
  "change_axis": "root: precision=tf32x3, launch=two, tiling BM128/BN128/BK32/w4/s3, GROUP_M8",
  "config": {
    "precision": "tf32x3",
    "launch": "two",
    "BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32,
    "num_warps": 4, "num_stages": 3, "GROUP_M": 8
  },
  "validation": {
    "static_checklist_passed": true,
    "shape_dtype_ok": true,
    "triton_only_compute": true,
    "notes": "algebraic equivalence per draft §2; block sizes divide 3072; M masked"
  },
  "per_workload": [
    {"uuid": "7ae93ff7", "B": 2, "T": 1423, "I": 1489, "passed": true, "match_ratio": 1.0, "speedup": 0.0},
    {"uuid": "7a91d659", "B": 4, "T": 128,  "I": 256,  "passed": true, "match_ratio": 1.0, "speedup": 0.0},
    {"uuid": "a9d58452", "B": 1, "T": 77,   "I": 1024, "passed": true, "match_ratio": 1.0, "speedup": 0.0},
    {"uuid": "9de8abcb", "B": 2, "T": 128,  "I": 256,  "passed": true, "match_ratio": 1.0, "speedup": 0.0},
    {"uuid": "e92f1fac", "B": 1, "T": 77,   "I": 4096, "passed": true, "match_ratio": 1.0, "speedup": 0.0}
  ],
  "all_passed": true,
  "geomean_speedup": 0.0,
  "decision": "keep|discard|new-best|root|invalid",
  "reason": "one-line why this decision (e.g. new best geomean; regressed; failed WL3 accuracy)",
  "cumulative_evaluations": 1,
  "skill_usage": "none (A800/sm_80 Ampere; KernelWiki scoped to Hopper/Blackwell, N/A)"
}
```

Rules:
- Numeric fields (`speedup`, `match_ratio`, `geomean_speedup`) are filled **from the evaluator
  output** for that candidate; the example zeros above are placeholders for the schema only.
- `source_sha256` is computed on the exact `solution/solution.py` that was evaluated (immutability
  anchor). If source changes, it is a new id.
- `decision` ∈ {`root`, `new-best`, `keep`, `discard`, `invalid`}; `invalid` for any candidate that
  fails ≥1 workload. Only valid candidates can be `new-best`.
- `cumulative_evaluations` is monot, incremented by 1 per appended row (5 workloads = 1 eval).
- `skill_usage` records `KernelWiki`/other skill invocation, or "none" with the reason (Ampere
  target outside the skill's Hopper/Blackwell scope).

A running human-readable log of decisions/lineage may additionally be kept in `docs/` (e.g.
`docs/progress.md`), but `candidates.jsonl` is the canonical append-only evidence.

---

## 7. Execution order (next turns, one action per step)

1. Implement **c001** source per §1/§2 (Phase A root). Run static checklist §3.
2. Evaluate: `./scripts/evaluate_candidate.sh feedback c001`. Append evidence row (§6).
3. Apply c001 branch rules (§2 Phase A) → either tune (Phase C) or run the `ieee` triage sibling.
4. Continue Phase C tiling sweep, one axis per candidate, appending one row each; consolidate with a
   curated-autotune candidate once the good region is bounded.
5. Phase D/E only if their hypotheses (H4/H5) are supported by observed per-workload speedups.
6. Monitor stopping criteria (§5) continuously. On convergence/budget, write `SEARCH_COMPLETE` with
   the reason and the best valid candidate id + geomean.
7. Do **not** run `final` unless the operator explicitly approves.

---

## 8. Guardrails (from CLAUDE.md / TASK.md — always in force)

- Work only inside this workspace; no parent dirs, other tasks, evaluator internals, or full
  workloads. Do not modify evaluator/dataset/controller/launcher/config/eval script.
- Only evaluate via `./scripts/evaluate_candidate.sh feedback cNNN`. No direct CUDA, profiler,
  `nvidia-smi`, external evaluator, or alternate correctness harness.
- Triton is the only compute path. No Torch/CPU/NumPy/CUDA-extension computational fallback; a
  failed Triton kernel is invalid and is fixed, not replaced by a fallback.
- Candidate IDs are immutable; never reuse an id for changed source; never rewrite earlier evidence.
- Do not change the five fixed feedback workloads.
- The only permitted external knowledge source is the `KernelWiki` skill (not applicable to this
  Ampere target; recorded as such).
```
