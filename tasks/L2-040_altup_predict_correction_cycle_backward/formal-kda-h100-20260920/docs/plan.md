# Plan — L2/040 `altup_predict_correction_cycle_backward`

Executable, sequential KDA optimization plan. Builds on `docs/draft.md`. Target H100 (`sm_90`),
Triton compute only, submission at `solution/solution.py::run(...)`. This turn writes the plan
only — no candidate is implemented or evaluated here.

---

## 0. Ground rules recap (binding on every step)

- One immutable kernel version over the full 16-workload feedback set = **one** evaluation.
  Budget: 100 evals. Token limits: soft 9M / normal 10M / hard 11M.
- Any meaningful source/config/launch change ⇒ **new** `cNNN`; never reuse an id for changed
  source; never rewrite earlier `candidates.jsonl` records.
- Evaluate only with `./scripts/evaluate_candidate.sh feedback cNNN`. Profile only with
  `./scripts/ncu_profile.sh …` under the `ncu-report-skill` workflow. **Never** run profiling
  and evaluation at the same time (foreign process ⇒ return code 3, discarded, budget burned).
- No Torch/CPU/NumPy/CUDA-extension compute fallback. A failing Triton kernel stays Triton;
  fix it, do not substitute a torch path.
- `final` only after explicit operator approval.

---

## 1. Solution skeleton (fixed across candidates)

`solution/solution.py` structure that stays stable while kernels evolve:

1. `run(...)` — the required entry point. Responsibilities (Torch allowed here, plumbing only):
   - Read shapes: `N=3`, `H=2304` (assert against constants), `B`, `S`; flatten `M = B*S`.
   - Make inputs contiguous as needed; keep bf16 (kernel upcasts internally).
   - Allocate outputs with **exact dtypes**: `grad_hidden_states` bf16 `[N,B,S,H]`,
     `grad_activated` bf16 `[B,S,H]`; and **`torch.zeros`** (not `empty`) fp32 for the four
     weight grads (`grad_prediction_coef_weight [9,3]`, `grad_correction_coef_weight [3,3]`,
     `grad_router_weight [3,H]`, `grad_norm_weight [H]`) because they are accumulated.
   - Compute strides, build the launch grid, launch the Triton kernel(s).
   - Return the 6-tuple in the documented order.
2. `@triton.jit` kernel(s) — all math in fp32; the design in §3.
3. Optional `@triton.autotune` wrapper (introduced only when tuning starts, as its own `cNNN`).

Guard rails inside `run`: handle `altup_active_idx` as a Python int (all workloads = 0) passed
to the kernel; keep the row-selection/residual-injection logic parametric on it so a non-zero
index is still correct. Treat `rms_norm_eps` as fp32 scalar arg.

---

## 2. Candidate lineage strategy

Sequential, one source version at a time. Each `cNNN` is immutable once evaluated. Lineage is a
mostly-linear chain; branch only if a perf idea regresses and we need to return to a known-good
parent. Planned arc (later ids are contingent on earlier evidence):

- **c001 — correctness baseline (single fused kernel, no autotune).**
  Parent: none. The simplest fused token-parallel kernel that compiles and is correct.
  Design 5.1 from the draft with a fixed, safe config (`BLOCK_M=1`, H as `256×9` internal loop
  or a single masked `H` block, `num_warps=4`). Weight grads via block-local fp32 accumulate +
  one `tl.atomic_add` per program per buffer. Goal: **pass all 16 workloads**. Perf is
  secondary; we just need a green, trustworthy anchor.

- **c002 — H-tiling / block shape correctness+perf variant.**
  Parent: c001. Only if c001 passes. Switch to whichever `H` handling c001 revealed as best
  (single masked `2304` block vs `256×9` loop), pick a better default `BLOCK_M`. Establishes
  the real baseline speed before autotuning.

- **c003 — weight-grad reduction strategy.**
  Parent: best of c001/c002. Compare block-local-accumulate-then-atomic (default) vs
  partial-buffer + small reduction kernel, if profiling shows atomic contention (likely only on
  large-M 13/16 where many programs hit the same `[H]`/`[3,H]` buffers).

- **c004 — autotune sweep.**
  Parent: best so far. Add `@triton.autotune` over `BLOCK_M ∈ {1,2,4,8}`,
  H-block ∈ {256,512,768,2304}, `num_warps ∈ {2,4,8}`, `num_stages ∈ {1,2,3}`. Because small-M
  and large-M want opposite configs, key the autotune cache on `M` (and pick configs that serve
  both regimes). This is where most of the geomean gain is expected.

- **c005+ — targeted micro-opts** driven by ncu evidence, e.g.:
  - fuse/reduce H passes (5.1A whole-row-in-SRAM vs 5.1B streaming) to cut L2 re-reads;
  - vectorized/coalesced loads of the 3 `grad_corrected`/`hidden_states` rows (contiguous over
    H; the leading `N` stride is `B*S*H`), consider a single strided load pattern;
  - specialize `altup_active_idx=0` as `constexpr` to drop a branch;
  - `tl.multiple_of`/`max_contiguous` hints on H pointers;
  - split-M grid for the tiny-M cases to raise occupancy.

Each bullet above becomes its own `cNNN` only when it is a concrete, single, testable change.
Do not bundle unrelated changes into one id (keeps attribution of speedups clean).

Naming: strictly increasing `c001, c002, …`. A reverted idea does not free its id; the next
new source gets the next number, with `parent` pointing at the good ancestor.

---

## 3. c001 implementation spec (the concrete first build)

Single `@triton.jit` kernel, grid = `ceil(M / BLOCK_M)` programs, `BLOCK_M=1` for c001
(one token per program — simplest correct mapping; raise later).

Per program (token row `t`, all fp32 internally):
1. **Loads** (upcast bf16→fp32): `h0,h1,h2 = hidden_states[0..2, t, :]`, `xa = activated[t,:]`,
   `gc0,gc1,gc2 = grad_corrected[0..2, t, :]`, plus shared `nw = norm_weight[:]`,
   `rw0,rw1,rw2 = router_weight[0..2, :]`. Small coef weights `pcw[9,3]`, `ccw[3,3]` as scalars.
   `x_p = h_idx` (idx=0 → h0).
2. **Forward recompute** exactly per draft §2: `RN` on `x_p` and on `xa`; `routed_*` via fp32
   `tl.sum` over H (no `tl.dot`); `mod_* = tanh(routed_*)`; `coefs_flat[0..8]`,
   `coefs_correct[0..2]`; `predictions_idx`, `innovation`.
3. **Correct backward** per draft §2: `grad_innovation`, `grad_coefs_correct[3]`,
   `grad_mod_c`, `grad_routed_c`, `grad_scaled_c`, `grad_normed_c`, `mean_c`, `grad_act_router`,
   `grad_activated` (write bf16), `grad_predictions[k]` (with `−grad_innovation` on idx row).
4. **Predict backward** per draft §2: `grad_h_permuted[:,k]`, `grad_all_coefs_flat[9]`,
   `grad_mod_p`, `grad_routed_p`, `grad_scaled_p`, `grad_normed_p`, `mean_p`,
   `grad_active_input`, `grad_hidden_states[k]` (write bf16; idx row adds `grad_active_input`).
5. **Weight-grad accumulation**: build per-program fp32 partials for
   `grad_prediction_coef_weight[9,3]`, `grad_correction_coef_weight[3,3]`,
   `grad_router_weight[3,H]`, `grad_norm_weight[H]`; with `BLOCK_M=1` these are the row's own
   contributions. Commit via `tl.atomic_add` into the zero-initialized fp32 outputs.
   (`[3,H]` and `[H]` atomics are the H-wide ones; `[9,3]`/`[3,3]` are tiny.)

H tiling: c001 uses an internal loop of 9 tiles of 256 over H (exact factor, no H mask needed),
carrying `rstd`, the three `routed` accumulators, etc. across the loop; means divide by the
true `H=2304`. If the row fits, a single masked `H` block is an acceptable alternative to try
first — decide by what compiles cleanly and passes.

M tail: guard the token index with `t < M`; masked programs do nothing (no partial writes, no
atomics).

Explicitly **not** in c001: autotune, whole-row-SRAM caching tricks, partial-buffer reductions,
constexpr idx specialization. Those are later candidates so their effect is measurable.

---

## 4. Correctness checks (per candidate, before it counts as "accepted")

The only sanctioned oracle is `./scripts/evaluate_candidate.sh feedback cNNN` (full 16 set).
Process for each candidate:

1. **Static self-review before evaluating** (no GPU, no tokens on the eval budget):
   - Signature, return order, and the 6 output dtypes match `definition.json` exactly.
   - Weight-grad outputs allocated with `torch.zeros` (accumulation correctness).
   - Every index/sign in the kernel cross-checked line-by-line against the `reference` string in
     `definition.json` — especially draft risks §4.6 (mean divisor = true H) and §4.7 (residual
     & `−grad_innovation` ordering; `grad_router_weight`/`grad_norm_weight` receive **both**
     predict and correct contributions).
   - fp32 everywhere on the hot path; only the two activation grads cast to bf16 at the end.
   - No `tl.dot` on the width-3 H-contractions.
2. **Evaluate once**: run the feedback script. Record the per-workload pass/fail, atol/rtol
   achieved (if the evaluator reports it), match ratio, and geomean speedup.
3. **Correctness gate**: a candidate is *valid* only if **all 16** workloads pass correctness.
   Any failure ⇒ candidate is invalid regardless of speed; diagnose and fix in a **new** `cNNN`.
4. **Canary focus** when reading results:
   - Precision canaries: workload 4 (atol 0.0014, M=256), 6 (0.0026), 2/7/15 (0.0029).
   - Tail-mask canaries: 3 (S=293), 5 (373), 13 (613), 14 (449).
   - Throughput/atomic-contention canaries: 13 (M=39232), 1 (16384), 16 (8192, B=64).
   If a precision canary fails but large-M passes, suspect a bf16-accumulation leak or the mean
   divisor; if a tail canary fails, suspect M-masking; if only large-M fails, suspect atomic
   accumulation or an overflow.
5. **Never** substitute an alternate correctness harness or run CUDA directly to "pre-check."
   Reason about correctness statically, then spend exactly one eval.

---

## 5. Performance hypotheses (each maps to a candidate + a measurable prediction)

| ID | Hypothesis | Change | Predicted effect | How measured |
|----|-----------|--------|------------------|--------------|
| H1 | The op is DRAM-bandwidth-bound; a single fused kernel beats the multi-op reference by removing all `[M,H]`/`[3,M,H]` temporaries and redundant passes. | c001 fused kernel | Large geomean speedup vs reference (reference does ~dozens of passes; fused ≈ 11·M·H traffic). | feedback geomean; ncu DRAM throughput ≈ roofline. |
| H2 | `BLOCK_M=1` under-utilizes; batching a few tokens per program improves coalescing/occupancy on large-M. | c002 raise `BLOCK_M`, tune H-block | Faster on 1/13/16; neutral on tiny-M. | per-workload times c001 vs c002; ncu occupancy/achieved-BW. |
| H3 | On large-M, many programs atomic-add into the same `[3,H]`/`[H]` buffers → contention. Block-local accumulate over `BLOCK_M` (or partial buffers) cuts atomic traffic. | c003 reduction strategy | Faster large-M (13/1/16); neutral small-M. | ncu L2/atomic stalls; per-workload times. |
| H4 | One config cannot serve M∈[256,39232]; autotuning per-M unlocks both regimes. | c004 autotune keyed on M | Best overall geomean; small-M gets high occupancy config, large-M gets throughput config. | feedback geomean vs c003; per-workload. |
| H5 | Redundant L2 re-reads of the row across H passes cost bandwidth; caching the row in SRAM (5.1A) or fusing passes cuts it. | c005 pass fusion | Marginal BW improvement if L2-bound. | ncu L2 hit-rate & DRAM bytes. |
| H6 | Misc micro-opts (constexpr idx, `multiple_of`/`max_contiguous` hints, vectorized N-row loads). | c006+ | Small single-digit % each. | per-workload deltas; keep only if net-positive on geomean. |

Rule: adopt a change only if it does not regress the correctness gate and improves (or is
neutral to) the geomean without hurting any single workload materially. If a hypothesis is
falsified by evidence, record it and revert to the parent for the next branch.

Profiling discipline: profile with `./scripts/ncu_profile.sh` **only between** evaluations,
never concurrently. Prefer profiling the two extreme workloads (a tiny-M and workload 13) to
characterize both regimes with minimal profiling runs.

---

## 6. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with the reason) when **any** of:
1. **Converged**: two consecutive accepted candidates improve geomean by < ~2% and ncu shows the
   kernel is within ~10–15% of the DRAM roofline (little headroom left).
2. **Budget**: approaching the evaluation budget (reserve ≥1 eval for a final re-check of the
   chosen best) or the token soft limit (9M) — wind down and finalize the best valid candidate.
3. **Diminishing returns**: several successive micro-opt candidates fail to move the geomean.

Do not run `final` on convergence; `final` requires explicit operator approval. When stopping,
identify the single best valid candidate (all-16-pass, highest geomean) as the finalization
target and note it in `SEARCH_COMPLETE`.

---

## 7. Evidence format (`candidates.jsonl`, one JSON object per evaluated candidate)

Append exactly one line per evaluated candidate; never rewrite prior lines. Schema:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "Fused single-kernel bandwidth-bound baseline; pass all 16 workloads.",
  "change_summary": "Initial fused Triton kernel, BLOCK_M=1, H tiled 256x9, atomic weight grads.",
  "validation": {
    "stage": "feedback",
    "all_pass": true,
    "num_workloads": 16,
    "per_workload": [
      {"uuid": "e9c4303d-...", "B": 64, "S": 256, "pass": true, "speedup": 0.0}
    ],
    "geomean_speedup": 0.0
  },
  "decision": "accept|reject|revert",
  "reason": "why accepted/rejected; which hypothesis (H1..) confirmed/falsified",
  "cumulative_evaluations": 1,
  "skills_used": ["KernelWiki", "ncu-report-skill"]
}
```

Notes:
- `source_sha256`: hash of the exact submitted `solution/solution.py` so a record is tied to
  immutable source. Compute at eval time.
- `per_workload`: capture whatever the evaluator reports (pass, speedup, and atol/rtol/match if
  available) for all 16 uuids; at minimum pass + speedup.
- `geomean_speedup`: primary ranking metric (geometric mean over the 16 workloads).
- `cumulative_evaluations`: running count of feedback+final evaluations consumed (budget = 100).
- `skills_used`: list skills actually used for that candidate (`KernelWiki` for design,
  `ncu-report-skill` for profiling); `[]` if none.
- `decision`: `accept` (new best/kept baseline), `reject` (worse/regresses, parent stays best),
  or `revert` (invalid correctness → fix in next id).

Companion notes (human-readable) go in `docs/` per candidate if analysis is long; the JSONL
stays the machine record.

---

## 8. Immediate next actions (after this plan)

1. Consult **KernelWiki** for SM90 fused RMSNorm-backward patterns, non-pow2 `H` tiling, and
   `tl.atomic_add` reduction idioms.
2. Implement `c001` per §3 (correctness-first, no autotune).
3. Static self-review per §4.1; then evaluate once with the feedback script.
4. Record the result in `candidates.jsonl` per §7; branch to c002+ per §2 driven by §5.
