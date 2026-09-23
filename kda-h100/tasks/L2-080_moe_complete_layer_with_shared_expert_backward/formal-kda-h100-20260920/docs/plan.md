# Plan — L2/080 MoE Complete Layer with Shared Expert (Backward)

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`. Target H100
(`sm_90`), Triton-only compute, submission `solution/solution.py::run(...)`. Metric: geomean
speedup over reference, every selected workload must pass correctness (`rtol=0.05`, per-workload
`atol` 0.11–0.39, 98% match).

Budget: 100 candidate evaluations; tokens soft/normal/hard 9M/10M/11M. One immutable source
version per candidate ID; full 16-workload feedback run = 1 evaluation. `final` only on operator
approval.

---

## 1. Strategy overview

Two independent levers, developed one candidate at a time:

- **Lever A — drop the routing branch (H0).** Analysis in draft §2 shows `grad_router_weight`
  and the router term of `grad_hidden` are ~10¹⁵× below every `atol`. Replacing them with zeros
  removes 2 GEMMs + a full-`H` reduction + scatter + sigmoid-derivative. Must be empirically
  confirmed, with a faithful fallback retained.
- **Lever B — reduce launches / fuse.** Collapse the reference's ~20 kernels into a small number of
  Triton kernels with fp32-accumulate GEMMs and fused SwiGLU-backward epilogues, plus per-regime
  tile tuning across B=192…8192.

Sequencing principle: **establish a correct Triton baseline first**, then apply the highest-margin,
lowest-risk change (Lever A), then incremental fusions/tuning (Lever B), validating each in isolation
so a regression can be attributed to exactly one change. Never stack an unvalidated change on another.

---

## 2. Candidate lineage

Each row is one immutable candidate. "Parent" is the best-known-good ancestor whose source it edits.
Later IDs are contingent — actual choices depend on measured evidence; this is the intended tree, not
a promise to run all of them.

| ID | Parent | Change (single hypothesis) | Risk |
|---|---|---|---|
| c001 | — | Faithful Triton port. All 8 GEMMs as separate `tl.dot` kernels (fp32 accum, typed downcast); SwiGLU-backward as one fused pointwise kernel; routing computed faithfully (fused reduction+scatter+sigmoid-deriv where convenient). Establish correctness + baseline speedup. | Correctness (masking, dtypes, transpose strides) |
| c002 | c001 | **Lever A / H0**: drop routing branch — `grad_router_weight ← zeros[E,H] f32`, no router term in `grad_hidden`; delete R1/R2, grad_norm_sq reduction, scatter, sigmoid-deriv. | H0 could fail a workload (very unlikely; ~10¹⁵ margin) |
| c003 | c002 | **F-A**: fuse G1 (down-backprop) + SwiGLU-backward epilogue → emit `grad_shared_up_output` & `grad_shared_gate_output` directly; remove `grad_shared_activated` materialization + 3 elementwise kernels. | Register pressure in epilogue |
| c004 | best(c002,c003) | **F-B**: dual grad_hidden GEMM — `grad_hidden = grad_up@up_w + grad_gate@gate_w` in one kernel over shared K=I, single `[B,H]` write. | Occupancy vs cuBLAS |
| c005 | best-so-far | **F-C**: dual weight-grad GEMM — G5+G6 share `hidden[B,H]` K-tile, two `tl.dot`s, two writes. | Small-B memory-bound tuning |
| c006 | best-so-far | **F-D**: standalone down weight-grad G2 kernel tuned; confirm it is not the bottleneck. | — |
| c007+ | best-so-far | Autotune / shape-keyed tile configs (small-B vs large-B regime), `num_warps`/`num_stages`/`GROUP_M` raster; guided by `ncu-report-skill`. | Overfit to feedback shapes |

Rules: any source/config/launch change ⇒ new ID; never reuse an ID for changed source; never rewrite
earlier `candidates.jsonl` records. If c002 fails any workload, freeze on c001 and pursue Lever B from
c001 instead (keep faithful routing).

### Decision log (updated as candidates run)

- **c001** (faithful port): 16/16 pass, geomean **3.2568×**. Kept as parent.
- **c002** (H0, drop routing): **0/16 pass — H0 REFUTED.** Reference routing gradients are NOT below
  `atol`; the draft §2 `eps=1e-20` softmax-normalization cancellation estimate does not hold in
  practice. Lever A is abandoned. **Faithful routing (c001) is mandatory.** All further candidates
  branch from **c001** and keep the routing branch intact; work proceeds on **Lever B only**
  (fusion + tuning). Do not re-attempt routing removal.

Revised lineage from here: c003 = F-A (fuse G1 down-backprop + SwiGLU-backward epilogue) on top of
c001; then F-B / F-C / F-D and per-regime tuning as originally scoped, each a new immutable ID
branching from the current best (starting c001).

---

## 3. Per-candidate execution loop

For each candidate, in order:

1. **Implement** `solution/solution.py` as the single immutable source for that ID (edit from parent).
2. **Static pre-eval checks** (§4) — no GPU compute of our own; pure code review.
3. **Evaluate**: `./scripts/evaluate_candidate.sh feedback cNNN` (exactly once per ID). Ensure no
   profiling is running concurrently (foreign process ⇒ rc 3, wasted eval).
4. **Record** one JSON object appended to `candidates.jsonl` (§6).
5. **Decide** keep / discard / iterate (§5 stopping criteria), set next parent.
6. **Profile only if needed** to choose the next change: `./scripts/ncu_profile.sh --set basic -o
   profile/rN python harness.py` — strictly never concurrent with an evaluation.

---

## 4. Correctness checks (pre-evaluation, static)

Gate every candidate on these before spending an evaluation:

- **Output contract**: returns a 5-tuple in order `(grad_hidden_states, grad_router_weight,
  grad_shared_expert_gate_weight, grad_shared_expert_up_weight, grad_shared_expert_down_weight)`;
  shapes `[B,H],[E,H],[I,H],[I,H],[H,I]`; dtypes `bf16, f32, bf16, bf16, bf16`. `grad_router_weight`
  is the only f32.
- **Accumulation**: every `tl.dot` uses fp32 accumulator (`out_dtype=tl.float32`); only the *result*
  is downcast to the target dtype. No f32 materialization of large bf16 inputs beyond in-register cast.
- **Ragged B**: all load/store use `mask=` for M/K tiles; OOB reduction lanes contribute 0; grid math
  never launches 0 programs (all B≥192).
- **Transpose GEMMs** (G2/G5/G6, R2): `Aᵀ@X` realized via strides, not a physical transpose; contraction
  along the correct (row/B) dimension.
- **SwiGLU-back rounding lattice** (draft §4.2): round `grad_shared_activated` to bf16 first, do
  silu + sigmoid-derivative in f32, round outputs to bf16.
- **Triton-only**: no `torch.matmul`/`F.*`/cuBLAS/NumPy/CPU compute in the hot path; PyTorch only for
  metadata, allocation, launch grid.
- **No fallback**: if a Triton kernel fails, the candidate is invalid — do not substitute Torch.

Post-evaluation correctness signal = the evaluator's per-workload pass/fail (the only oracle).

---

## 5. Performance hypotheses & stopping criteria

**Hypotheses (each tied to a candidate, validated by geomean delta):**

- H-c001: A straightforward fp32-accumulate Triton port reaches at least parity with the reference
  (which itself is cuBLAS-backed); small-B shapes may already gain from fewer launches.
- H-c002 (H0): Dropping the routing branch gives a measurable geomean gain, largest at small B
  (launch/memory-bound), with **zero** correctness regression.
- H-c003..c006 (fusion): Each fusion reduces launches/HBM traffic and improves small–mid B; must not
  regress large-B GEMM efficiency. Accept a fusion only if geomean improves and no workload regresses.
- H-c007+ (tuning): Per-regime tile configs close the remaining gap to cuBLAS at large B.

**Decision rule per candidate:** keep as new parent iff (a) all 16 workloads pass, and (b) geomean
≥ current best (ties broken toward simpler source). Otherwise discard and revert to prior parent;
a fusion that helps small B but regresses large B is rejected unless net geomean improves.

**Stopping criteria (any one triggers stop):**

- Geomean improvement < ~2% across two consecutive kept candidates (convergence).
- Evaluation budget (100) approached, or token soft limit (9M) reached.
- Profiling shows large-B GEMMs are within a few % of cuBLAS and small-B is launch-bound with no
  remaining fusion opportunity (diminishing returns).

On genuine convergence, write `SEARCH_COMPLETE` with the reason and the best candidate ID. Never run
`final` without explicit operator approval.

---

## 6. Evidence format (`candidates.jsonl`, one JSON object per evaluated candidate)

Append-only; never rewrite. Each record contains at minimum:

```json
{
  "id": "cNNN",
  "parent": "cMMM | null",
  "source_sha256": "<hash of solution/solution.py as evaluated>",
  "hypothesis": "one-line change + expected effect",
  "static_checks": "pass | notes",
  "results": [
    {"uuid": "<workload uuid>", "batch_seq_len": <B>, "passed": true,
     "speedup": <ref_time/cand_time>, "max_atol_used": <tol>}
  ],
  "geomean_speedup": <float>,
  "all_passed": true,
  "decision": "keep-as-parent | discard | fallback",
  "cumulative_evaluations": <int>,
  "skills_used": ["KernelWiki", "ncu-report-skill?"],
  "notes": "profiling findings / next step"
}
```

Also record, in `notes` or dedicated fields, any evaluator return code ≠ 0 (e.g. rc 3 = interference —
does not count as a valid measurement but still consumes an eval; log and retry cleanly).

---

## 7. Immediate next action (next turn, not this one)

Implement **c001** (faithful Triton port) as `solution/solution.py`, run static checks §4, then a
single `./scripts/evaluate_candidate.sh feedback c001`, and append its record to `candidates.jsonl`.
No implementation or evaluation in this planning turn.
