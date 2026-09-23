# Plan — L2/036 ConvNeXtV2 Backward (Triton, H100 sm_90)

This is the executable, sequential KDA optimization plan derived from `docs/draft.md`. It defines
the candidate lineage, per-candidate hypotheses and acceptance gates, correctness checks,
performance-diagnosis workflow, stopping criteria, and the evidence record format. No candidate is
implemented or evaluated in this turn.

---

## 0. Ground rules (recap of the binding constraints)

- Entry point: `solution/solution.py::run(...)` with the exact signature in `task/definition.json`,
  returning the 11-tuple in the specified order.
- **Triton-only compute.** Torch only for allocation / view / permute / launch plumbing. No
  Torch/cuDNN/cuBLAS/NumPy/CUDA-extension computational fallback. A failed Triton kernel = invalid
  candidate (discard; do not fall back).
- One immutable source version per candidate ID (`c001`, `c002`, …). Never reuse an ID for changed
  source. Any meaningful source/config/launch change ⇒ new ID.
- Evaluation only via `./scripts/evaluate_candidate.sh feedback cNNN` (all 14 workloads = 1 eval).
  Profiling only via `./scripts/ncu_profile.sh …`. **Never** run profiling and evaluation
  concurrently (foreign process on the locked GPU ⇒ return code 3 ⇒ wasted eval). `final` only on
  explicit operator approval.
- Budget: 100 evaluations; token soft 9M / normal 10M / hard 11M. Correctness authority = the
  feedback evaluator only (no local/alternate correctness harness).
- Append exactly one JSON record per evaluated candidate to `candidates.jsonl`; never rewrite
  earlier records.

---

## 1. Strategy overview

Two phases:

- **Phase A — Correctness (c001).** Land a full-Triton reimplementation that reproduces the
  reference arithmetic verbatim (including the §5.1 `grad_gf_mean` non-sum quirk and the drop-path-
  on-projected-branch-only detail from the draft). Prioritize passing all 14 workloads over speed.
  Conservative choices: fp32/`ieee` GEMMs, simple (possibly many) kernels, no risky fusion.
- **Phase B — Performance.** Iteratively optimize, guided by `ncu` between evals: replace the
  reference's per-channel weight-grad loop (expected dominant cost), then GEMM tiling/precision,
  fusion of elementwise+reduction chains, split-K/split-M for tall-skinny reductions, SMEM tiling
  for depthwise, and M-bucketed launch configs. Each meaningful change is a new candidate.

Lineage is a mostly-linear chain: each new candidate's parent is the current best *valid* candidate.
Branch (keep an earlier parent) only when an optimization is speculative and might regress; then the
losing branch is abandoned and we continue from the retained best.

Design invariants carried through all candidates (from draft §6):
- Work in NHWC `(M, {C,C4})` with `M=B·H·W`, `C=128`, `C4=512`.
- Transpose-in kernel: NCHW `grad_output` → NHWC `grad_x_projected`, folding `drop_mask/keep`.
- Transpose-out: NHWC `grad_x_nhwc` → NCHW, adding depthwise-input-grad and `grad_output` residual.
- Four GEMMs: two activation (K=128 / K=512), two tall-skinny weight (K=M).
- Reduction/elementwise kernels for GRN, GELU, LayerNorm; two depthwise kernels (input-grad,
  weight+bias-grad).

---

## 2. Candidate lineage (planned; IDs are reserved, not yet implemented)

Each entry: parent · goal · concrete change · hypothesis · risk · accept/reject gate. Later IDs are
provisional and will be re-prioritized using evidence; only c001 is fully specified up front.

### c001 — Correctness baseline (parent: none)
- **Goal:** all 14 workloads pass; establish the geomean baseline for the Triton path.
- **Change:** full-Triton decomposition, maximal simplicity (draft §6.5): transpose-in;
  4 GEMMs (fp32/ieee); GRN reduction kernel + GRN/GELU elementwise kernel; LayerNorm-bwd kernel;
  bias/affine column-reductions; depthwise input-grad kernel; depthwise weight+bias-grad kernel;
  transpose-out. Verbatim reference arithmetic incl. §5.1 quirk and two-branch drop-path.
- **Hypothesis:** even unfused, replacing the reference's 128-iter Python weight-grad loop and its
  temporaries yields geomean speedup > 1.0 (likely ≫ 1 on large-M workloads 12/13).
- **Risk:** depthwise flip/pad indices; GEMM transpose conventions; reduction axes.
- **Gate:** must pass all 14. If it fails, the next ID fixes the specific failing region (no perf
  work until green).

### c002 — Depthwise weight-grad optimization (parent: best valid)
- **Goal:** ensure the (expected) hottest kernel is efficient.
- **Change:** restructure `grad_dwconv_weight`/`grad_dwconv_bias` reduction — split-M partials +
  combine (or atomics), preload the 49 taps, coalesced NCHW reads. Only pursue if ncu confirms it is
  hot after c001.
- **Hypothesis:** largest single speedup on high-M workloads.
- **Gate:** pass 14 AND geomean ≥ parent × 1.03 (else reject, keep parent).

### c003 — Depthwise input-grad + transpose-out fusion (parent: best valid)
- **Change:** SMEM-haloed tiling of `grad_x_dwconv` for the 49-tap gather; fuse `+ grad_output`
  residual and NHWC→NCHW write into one kernel.
- **Hypothesis:** removes a full HBM round-trip for `grad_x` on 56×56 shapes.
- **Gate:** pass 14 AND geomean ≥ parent × 1.02.

### c004 — Activation-GEMM tiling / autotune (parent: best valid)
- **Change:** tuned `BLOCK_M/N/K`, warps, stages for the two activation GEMMs; pre-declared config
  set (avoid huge autotune search under warmup=2). fp32/ieee kept.
- **Gate:** pass 14 AND geomean ≥ parent × 1.02.

### c005 — Tall-skinny weight-GEMM split-K (parent: best valid)
- **Change:** split-K (two-pass or atomic) for `grad_pwconv1_weight`/`grad_pwconv2_weight` at large M.
- **Hypothesis:** exposes parallelism at M≈100k (workloads 12/13).
- **Gate:** pass 14 AND geomean ≥ parent × 1.02.

### c006 — Epilogue fusion of GRN reductions (parent: best valid)
- **Change:** fold `grad_grn_weight/bias`, `grad_pwconv2_bias`, and `grad_norm_features` spatial
  partials into the pwconv2 activation-GEMM epilogue / a single GRN pass.
- **Gate:** pass 14 AND geomean ≥ parent × 1.02.

### c007 — GELU/GRN pass-2 + pwconv1 fusion (parent: best valid)
- **Change:** compute `grad_x_gelu` (three contributions) and `grad_x_expanded = grad_x_gelu *
  gelu_grad` fused, feeding the pwconv1 GEMM with minimal temporaries.
- **Gate:** pass 14 AND geomean ≥ parent × 1.02.

### c008 — LayerNorm-bwd fusion + column-reduction consolidation (parent: best valid)
- **Change:** single LN-bwd kernel emitting `grad_x_nhwc`, `grad_layernorm_weight/bias`
  (split-M+combine), minimizing passes over `(M,C)`.
- **Gate:** pass 14 AND geomean ≥ parent × 1.02.

### c009 — TF32 / tf32x3 GEMM precision (parent: best valid) — *speculative branch*
- **Change:** switch the four GEMMs to `input_precision="tf32"` (and/or `tf32x3`).
- **Hypothesis:** speed win; must confirm it stays within `rtol=1e-3` / `match_ratio=0.98` on the
  largest-magnitude outputs (workloads 12/13).
- **Gate:** pass 14 AND geomean ≥ parent × 1.02. If any workload fails ⇒ reject, keep parent
  (do not weaken accuracy for speed).

### c010 — M-bucketed launch configs (parent: best valid)
- **Change:** select kernel configs by M-bucket (small/medium/large) instead of one cache; special
  small-M path (fuse more) for latency-bound workloads 4/9.
- **Gate:** pass 14 AND geomean ≥ parent × 1.02.

### c011+ — Evidence-driven (parent: best valid)
- Further candidates decided from ncu evidence: mega-kernel for tiny shapes, reducing temporaries in
  place, tap-vectorization in depthwise, reduction-tree tuning. Each: pass 14 AND geomean ≥ parent ×
  1.02, else reject.

**Acceptance rule for the chain:** a candidate becomes the new best only if it passes all 14 and
improves geomean over the current best by the stated threshold. Rejected candidates are recorded but
do not change the parent pointer.

---

## 3. Correctness checks (per candidate, before spending an evaluation)

Since the evaluator is the only correctness authority, do a **static risk-first review** of the diff
before each eval to avoid burning evaluations on avoidable bugs:

1. **Return contract:** 11 tensors, exact order, exact shapes (esp. `grad_dwconv_weight (C,1,7,7)`,
   `grad_grn_weight/bias (1,1,1,C4)`), dtype fp32, correct device, contiguous as the evaluator
   expects.
2. **Drop-path two branches:** `grad_x_projected` uses `grad_output*drop_mask/keep`;
   `grad_residual`/final `grad_x` add un-scaled `grad_output`. Confirm not conflated.
3. **grad_gf_mean quirk (draft §5.1):** replicate the reference's per-channel (non-summed) form
   exactly; do NOT substitute the textbook backward.
4. **GEMM transpose conventions:** `grad_x_grn = grad_x_projected @ pwconv2_weight` (not `.T`);
   `grad_x_ln = grad_x_expanded @ pwconv1_weight`; weight-grads are `A.T @ B` with correct operand.
5. **Reduction axes:** every `grad_*` reduces over the axes in draft §1 table (M vs spatial-only vs
   channel). `grad_norm_features` reduces spatial per-batch; GRN weight/bias over all M.
6. **GELU constants:** `a=0.7978845608028654`, `0.044715`; derivative uses `1 - t^2`; fp32 tanh.
7. **Depthwise indices (top hazard):** input-grad is flipped/transposed correlation with boundary
   mask; weight-grad uses `residual[b,c,y+i-3,x+j-3]` zero-padded. Hand-check a corner index.
8. **eps placement:** `gf_mean+eps`, `(gf_mean+eps)^2`, `global_features+eps`, `var+eps` — distinct.
9. **Grid/mask coverage:** M not divisible by BLOCK_M, C4 tiling, and B=1 edge cases masked; no OOB.

Only after this review passes do we run `evaluate_candidate.sh feedback cNNN`.

If a candidate fails correctness: read the evaluator's per-workload diagnostics, localize to a
region (usually depthwise or a reduction axis), and create the *next* ID with the fix. Never mutate
an already-evaluated candidate's source.

---

## 4. Performance hypotheses (ranked, to be confirmed by ncu)

1. **H1 (highest):** the reference's `for g in range(C)` weight-grad loop dominates; a proper
   reduction kernel is the biggest win, scaling with M (workloads 12,13,6,8,10,5).
2. **H2:** many fp32 temporaries in GRN/GELU/LN cause excess HBM traffic; fusion cuts it several-fold.
3. **H3:** tall-skinny weight GEMMs (K=M up to 100k) serialize without split-K; split-K helps big M.
4. **H4:** activation GEMMs are FLOP-real but small-N; tiling/warps/stages matter modestly.
5. **H5:** depthwise input-grad on 56×56 benefits from SMEM haloed tiling.
6. **H6:** tiny shapes (4,9) are launch-overhead-bound; fewer/fused kernels help there.
7. **H7:** TF32 GEMMs speed up big-M but risk rtol on largest-magnitude weight grads.

Diagnosis loop (between evals only): run `./scripts/ncu_profile.sh --set basic -o profile/rN python
harness.py` on a small self-authored perf harness (perf only — NOT a correctness oracle) that calls
`run` on representative shapes (e.g. one large: B=32,H=56; one medium: B=8,H=28; one tiny: B=1,H=14).
Read the report via the ncu-report-skill workflow, identify the dominant kernel and limiter (memory
vs compute vs latency/occupancy), and let it choose the next candidate. **Never** profile while an
evaluation is running.

---

## 5. Stopping / convergence criteria

- **Hard stops:** reach 100 evaluations, or token soft-limit (9M) approached → wind down; hard limit
  (11M) is absolute.
- **Convergence:** stop optimizing when the best-of-last-3 accepted candidates improve geomean by
  < 1.5% cumulatively, or when ncu shows the top kernels are within a small factor of a roofline
  bound (memory-bandwidth or FLOP limited) with no clear remaining lever.
- On convergence: write `SEARCH_COMPLETE` with the reason, the best candidate ID, its geomean, and
  the evidence pointer.
- **Do NOT** run `final` without explicit operator approval; when approved, run `final` once on the
  best valid candidate.

---

## 6. Evidence format (one JSON object appended per evaluation to `candidates.jsonl`)

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "Full-Triton reimplementation removes the per-channel weight-grad loop and temporaries.",
  "change_summary": "Initial correct decomposition: transpose-in, 4 fp32 GEMMs, GRN/GELU/LN kernels, 2 depthwise kernels, transpose-out.",
  "validation": {
    "passed": true,
    "per_workload": [
      {"uuid": "817ddfe6-...", "B":16,"H":14,"W":14, "pass": true, "speedup": 0.0, "max_abs_err": 0.0}
    ],
    "num_passed": 14,
    "num_total": 14
  },
  "geomean_speedup": 0.0,
  "decision": "accept|reject|baseline",
  "decision_reason": "new best / regressed vs parent / failed workload N",
  "cumulative_evaluations": 1,
  "skills_used": ["ncu-report-skill", "KernelWiki"],
  "notes": "ncu: dominant kernel = <name>, limiter = <mem/compute/latency>",
  "return_code": 0
}
```

Rules:
- `source_sha256` recorded at eval time so source↔record is verifiable and IDs stay immutable.
- `per_workload` mirrors the 14 feedback UUIDs with pass flag and speedup vs reference.
- `decision`: `baseline` for c001, else `accept` (new best) or `reject` (keep parent).
- `cumulative_evaluations` is monotonic; never rewrite prior lines.
- `skills_used` lists KernelWiki / ncu-report-skill usage for that iteration.
- On a controller error (e.g. return code 3 from interference), record it with `decision:"invalid"`
  and note that the eval was consumed without a valid measurement.

---

## 7. Immediate next actions (subsequent turns, not this one)

1. Implement `solution/solution.py` for **c001** per §2 (correctness-first, verbatim arithmetic).
2. Run the §3 static review; then `./scripts/evaluate_candidate.sh feedback c001`.
3. Append the c001 record (§6). If green, profile with `ncu_profile.sh` to confirm H1, then proceed
   to the highest-value candidate; if red, create c002 as a targeted fix.
4. Continue the lineage (§2), honoring the accept/reject gates (§2, §5) until convergence, then write
   `SEARCH_COMPLETE`. `final` only on operator approval.
