# Plan — L2/080 MoE complete-layer + shared-expert (backward)

Run ID: `formal-kda-20260916--sol_execbench--L2-080_moe_complete_layer_with_shared_expert_backward`
Target: **NVIDIA A800, `sm_80` (Ampere)**. Compute path is **Triton-only** (torch for
allocation / metadata / launch only; no Torch/CPU/NumPy/CUDA-extension computational fallback).

This plan operationalizes `docs/draft.md`. It defines the executable candidate sequence, the
lineage rules, the correctness gate, the per-candidate performance hypotheses, the stopping
criteria, and the exact evidence schema. **No code is written and no candidate is evaluated in
this step.**

---

## 0. Ground truth recap (frozen constants)

| symbol | meaning | value |
|--------|---------|-------|
| `H` | hidden_size | 4096 |
| `I` | moe_intermediate_size | 1408 |
| `E` | n_routed_experts | 128 |
| `K` | num_experts_per_tok | 8 |
| `T` | batch_seq_len (variable) | feedback set: 211, 293, 384, 512, 997 |

Baked scalars: `routed_scaling_factor = 1.0`, `norm_topk_prob = True`, `score_mask` all ones,
`eps = 1e-20`. `router_logits` input is unused by the reference and by us.

Output tuple (order + dtype are part of the contract):
1. `grad_hidden_states` bf16 `[T,H]`
2. `grad_router_weight` **f32** `[E,H]`
3. `grad_shared_expert_gate_weight` bf16 `[I,H]`
4. `grad_shared_expert_up_weight` bf16 `[I,H]`
5. `grad_shared_expert_down_weight` bf16 `[H,I]`

Feedback tolerances (per workload): `max_rtol = 0.05`, `required_match_ratio = 0.98`,
`max_atol ∈ {211→0.13, 293→0.12, 384→0.18, 512→0.20, 997→0.36}`.

### The eight ops we must reproduce op-for-op

| id | formula | shape | K (contraction) | dtype in→out |
|----|---------|-------|-----------------|--------------|
| G1 | `grad_shared_activated = go @ down_w` | `[T,I]` | `H=4096` | bf16·bf16 → bf16 |
| SW | SwiGLU-bwd elementwise (`silu`, `silu'` in f32) on `[T,I]` | `[T,I]` | — | bf16/f32 → bf16 |
| G2 | `g_down_w = go.T @ shared_activated` (f32 mm) | `[H,I]` | `T` | bf16→f32 mm → bf16 |
| G3 | `g_h_up = grad_up_output @ up_w` | `[T,H]` | `I=1408` | bf16·bf16 → bf16 |
| G4 | `g_h_gate = grad_gate_output @ gate_w` | `[T,H]` | `I=1408` | bf16·bf16 → bf16 |
| G5 | `g_up_w = grad_up_output.T @ hidden` (f32 mm) | `[I,H]` | `T` | bf16→f32 mm → bf16 |
| G6 | `g_gate_w = grad_gate_output.T @ hidden` (f32 mm) | `[I,H]` | `T` | bf16→f32 mm → bf16 |
| R  | router prologue: `grad_norm_sq` rowsum, `/K`, norm-Jacobian, scatter→`[T,E]`, `*score_mask`, sigmoid-deriv | `[T,E]` | — | f32 |
| G7 | `g_h_router = grad_router_logits.bf16 @ router_w` | `[T,H]` | `E=128` | bf16·bf16 → bf16 |
| G8 | `g_router_w = grad_router_logits.T @ hidden` | `[E,H]` | `T` | f32 mm → **f32** |

`grad_hidden_states = g_h_up + g_h_gate + g_h_router` (outputs #3/#4/#5 = `g_gate_w`/`g_up_w`/`g_down_w`).

**Router-path cancellation (from draft §4.3):** `grad_topk_weights` is isotropic across the `K`
slots and the normalization Jacobian removes the common mode, so `grad_router_logits ≈ 0`.
Therefore outputs #2 (`grad_router_weight`) and the router contribution to #1 are ~0 and pass on
`atol` alone. **All real error budget lives in the shared-expert path.** We still reproduce the
router formula faithfully (robust to a hypothetical non-zero `e_score_correction_bias`).

---

## 1. Deliverable & candidate mechanics

- Single source of truth: `solution/solution.py`, exposing
  `run(grad_output, hidden_states, router_weight, e_score_correction_bias, router_logits, scores,
  topk_indices, topk_weights, score_mask, shared_expert_gate_weight, shared_expert_up_weight,
  shared_expert_down_weight, shared_gate_output, shared_up_output, shared_activated)` and returning
  the 5-tuple in the order above.
- Each candidate `cNNN` is an **immutable snapshot** of that source. Any meaningful change to
  source, Triton config (block sizes / warps / stages / precision knob), or launch structure ⇒ a
  **new** candidate ID. Never reuse an ID for changed source; never rewrite a prior
  `candidates.jsonl` record.
- Evaluate **only** with `./scripts/evaluate_candidate.sh feedback cNNN`. The five fixed workloads
  together are one evaluation. Never run `final` without explicit operator approval.
- Do not touch evaluator/dataset/controller/launcher/config; do not run CUDA/profiler/`nvidia-smi`
  or any alternate correctness harness directly.

---

## 2. Sequential candidate roadmap (executable)

Each step is implemented as one immutable source version, then evaluated once (5 workloads). Do
**not** start `cN+1` before `cN` has been evaluated and logged. Advance the "best" pointer only to
a candidate that (a) passes all five workloads and (b) does not regress geomean.

### c001 — D0 correctness anchor (pure per-op Triton)
- **Goal:** establish a fully-Triton path that passes all five workloads and gives the baseline
  speed number. Prioritize correctness clarity over fusion.
- **Kernels (one Triton kernel per op; no torch mm):**
  - `gemm_g1`: `go[T,H] @ down_w[H,I] → grad_shared_activated[T,I]`, bf16 in / fp32 acc / bf16 out.
  - `swiglu_bwd`: elementwise over `[T,I]`; loads `grad_shared_activated`, `shared_up_output`,
    `shared_gate_output`; computes in f32 `sig=sigmoid(gate)`, `silu=gate*sig`,
    `silu'=sig*(1+gate*(1-sig))`; writes `grad_up_output = gsa*up` (bf16) and
    `grad_gate_output = (gsa*up... )` — specifically `grad_gate_silu = gsa*up_output`,
    `grad_up_output = gsa*silu(gate)`, `grad_gate_output = grad_gate_silu*silu'(gate)`. **Match the
    reference exactly:** `grad_shared_gate_silu = grad_shared_activated * shared_up_output`,
    `grad_shared_up_output = grad_shared_activated * silu(gate)`,
    `grad_shared_gate_output = grad_shared_gate_silu * silu'(gate)`.
  - `gemm_g3`, `gemm_g4`: `[T,I] @ [I,H] → [T,H]`, bf16/fp32acc/bf16.
  - `gemm_g7`: `[T,E] @ [E,H] → [T,H]` (left operand `grad_router_logits` cast to bf16), bf16/fp32acc/bf16.
  - `add_hidden`: `grad_hidden_states = g_h_up + g_h_gate + g_h_router` (may be an epilogue add in
    the last GEMM, or a tiny elementwise kernel; keep explicit & correct in c001).
  - `gemm_g2`: `go.T @ shared_activated → [H,I]`, contraction `K=T`, fp32 acc → bf16 out. Handle the
    logical transpose via strides (no physical transpose).
  - `gemm_g5`, `gemm_g6`: `grad_*_output.T @ hidden → [I,H]`, `K=T`, fp32 acc → bf16 out.
  - `router_prologue`: computes `grad_norm_sq = rowsum(go.f32^2)`; `g = grad_norm_sq / K`;
    `S = rowsum(topk_weights)+1e-20`; `sum_grad = rowsum(g_slots*topk_weights)/S`;
    `grad_before = (g/1 - sum_grad)/S` (routed_scaling_factor=1); scatter-add into `grad_scores[T,E]`
    via `topk_indices`; multiply by `score_mask`; `grad_router_logits = grad_scores*scores*(1-scores)`.
    Since the isotropic `g` is identical across slots, this can be a per-row `[E]`-wide kernel; the
    scatter over `K=8` uses `topk_indices` (int64, values <128 — safe to use directly or cast i32).
  - `gemm_g8`: `grad_router_logits.T @ hidden → [E,H]`, `K=T`, **f32 out** (TF32 acc acceptable —
    numerically ~0).
- **Tiling (fixed, conservative):** short-M GEMMs (G1/G3/G4/G7) `BM=64, BN=128, BK=32/64`,
  `num_warps=4`, `num_stages=3`, mask on M (`T` not a multiple of 64) and on N/K tails. Outer-product
  GEMMs (G2/G5/G6/G8) tile the large output `[BM=64,BN=64]`, single/short K-loop over `T` with mask.
- **Correctness self-checks before evaluating:** run the §4 checklist mentally/statically.
- **Decision rule:** must pass all 5 workloads. Record baseline geomean; this is the lineage root.

### c002 — D1a: fuse SwiGLU-bwd into G1 epilogue
- **Parent:** c001. **Change:** `gemm_g1` writes, from its `[T,I]` accumulator, directly the two
  SwiGLU-bwd products `grad_up_output` and `grad_gate_output` (bf16), consuming `shared_up_output` /
  `shared_gate_output` tiles in the epilogue. Removes the `grad_shared_activated` materialization and
  the separate `swiglu_bwd` pass over `[T,I]`.
- **Hypothesis:** −1 kernel launch and −1 full `[T,I]` write/read round-trip → small but real win,
  larger at big `T` (997). Correctness unchanged (same arithmetic, still f32 intermediates).
- **Guard:** if match-ratio drops (unexpected), revert epilogue f32 handling and re-attempt as c003.

### c003 — D1b: concatenated grad_hidden GEMM (G3+G4+G7 → one)
- **Parent:** best of {c001,c002}. **Change:** single GEMM producing `grad_hidden_states[T,H]` with
  concatenated contraction `K = I + I + E = 2944`, reading three left operands (`grad_up_output`,
  `grad_gate_output`, `grad_router_logits.bf16`) and three right operands (`up_w`, `gate_w`,
  `router_w`) via separate base pointers in one K-loop (no physical `torch.cat`). One accumulator
  replaces three GEMMs + two adds.
- **Hypothesis:** biggest single fusion win — 3 launches + 2 elementwise adds → 1 launch, one
  `[T,H]` write instead of three. Expect the largest geomean gain of the D1 steps.
- **Risk:** ordering of the three partial sums differs from the reference's separate add order; this
  is fp associativity on bf16-input/fp32-acc — well inside `atol`. Verify match-ratio holds.

### c004 — D1c: fuse shared weight-grad GEMMs (G5+G6 → one; keep G2, G8 separate)
- **Parent:** best so far. **Change:** one GEMM `[grad_up_output | grad_gate_output].T @ hidden →
  [2I,H]`, `K=T`, split the `[2I,H]` output into outputs #4 (`g_up_w`) and #3 (`g_gate_w`). G2
  (`[H,I]`, different operands) and G8 (f32) stay separate.
- **Hypothesis:** −1 launch, shared `hidden` reads across the two outputs → modest win, mostly at
  small `T` where launch overhead dominates.

### c005+ — D2 tuning (one knob per candidate; measured, reversible)
Explore in priority order, each as its own candidate; keep only correctness-passing, non-regressing
steps. Stop early per §5 if converged.
- **c005:** `T`-bucketed fixed configs for the short-M GEMMs (block/warps/stages) — e.g. small `T`
  (211/293) vs large `T` (512/997) buckets. Deterministic table, no open autotune sweep.
- **c006:** split-K (or 2-pass reduce) for the `K=T` outer-product GEMMs (G2/G5+G6/G8) to raise SM
  occupancy at small `T`; guard the reduction dtype (fp32 accumulate, cast on final store).
- **c007:** router prologue fusion — fold `grad_norm_sq` reduction + normalization + scatter +
  sigmoid-deriv into one launch; optionally fuse G8 as an epilogue. Router is ~0 so precision is
  free; the win is launch reduction only (small).
- **c008+:** precision knobs (`tl.dot(input_precision=...)`, bf16 vs tf32) on the shared path *only
  if* a match-ratio regression appears; and any remaining launch/occupancy micro-tuning suggested by
  the per-workload speed breakdown. Each change = new ID.

Advance the lineage only through candidates that pass all five workloads.

---

## 3. Lineage strategy

- **Root:** c001 (correctness anchor). Every later candidate names its `parent` = the current best
  passing candidate it was derived from (not necessarily its numeric predecessor).
- **Linear-with-backtrack:** normally c(N) branches from c(N−1). If a step regresses or fails, its
  successor branches from the last good candidate instead, and the plan records why.
- **One variable per step:** each candidate changes exactly one fusion/tiling/precision dimension so
  any geomean or match-ratio delta is attributable.
- **Immutability:** source is frozen at evaluation; the controller locks the hash. A changed config
  is a new ID even if the `.py` diff is one line.
- **Best pointer:** maintained explicitly in each `candidates.jsonl` record (`is_new_best`), so the
  final submission choice is unambiguous.

---

## 4. Correctness checks (gate — must hold before every evaluation)

Static/local self-review (not the evaluator) applied to each candidate's source:

1. **Contract shape/dtype:** returns exactly 5 tensors in order; dtypes = bf16, **f32**, bf16, bf16,
   bf16; shapes `[T,H],[E,H],[I,H],[I,H],[H,I]`; contiguous, on the input device.
2. **Op-for-op fidelity:** each of G1..G8 + SW + R matches the reference arithmetic, including the
   *intended* router isotropic approximation (do **not** "fix" it) and the exact SwiGLU-bwd
   expressions and f32-intermediate/bf16-store casts.
3. **`T`-tail masking:** `T ∈ {211,293,997}` are not multiples of typical blocks. Every M-tile
   (short-M GEMMs) and every K-tile over `T` (outer-product GEMMs) is masked; masked lanes
   contribute exactly 0 to accumulators / stores.
4. **Transpose via strides:** G2/G5/G6/G8 realize logical transposes through pointer/stride math,
   never a physical transpose that would change results or add copies. Verify `down_w`=[H,I],
   `up_w`/`gate_w`=[I,H], `router_w`=[E,H] layouts are read with correct strides.
5. **Additive `grad_hidden_states`:** built either by one fused accumulator (c003+) or a single
   zero-init + non-racing adds; no cross-program read-modify-write race into the same output element.
6. **Scatter semantics:** `grad_scores[T,E]` zero-initialized; `scatter_add_` over `K=8` slots using
   `topk_indices` (int64; values <128, safe as index); add is commutative so ordering is immaterial.
   `* score_mask` applied (no-op here but kept general).
7. **f32 where required:** `grad_router_weight` and the router prologue math stay f32; `silu`/`silu'`
   computed in f32 before bf16 store, exactly as the reference.
8. **No fallback:** grep the final source to confirm no `torch.matmul`/`@`/`F.*`/NumPy compute path
   remains in `run` (torch only for `empty`/`zeros`/`stride`/`view`/launch). A failing Triton path is
   reported as failure, never swapped for torch.
9. **Determinism:** fixed config tables only; no wall-clock/random-seeded autotune that could vary
   between evaluations of the same ID.

**Evaluation gate:** a candidate is *valid* only if it passes all five feedback workloads
(≥98% element match at that workload's `atol`/`rtol`). A candidate failing any workload is invalid,
is not promoted, and its failure mode is logged before the next candidate.

---

## 5. Performance hypotheses & stopping criteria

### Hypotheses (ranked expected impact)
- **H1 (dominant):** the reference issues ~8 cuBLAS GEMMs + ~6 elementwise/reduction kernels + a
  scatter; with small `T` these are launch/round-trip bound. Fusion → fewer launches + fewer
  materialized intermediates is the primary lever (c002–c004), **not** out-GEMMing cuBLAS.
- **H2:** the concatenated grad_hidden GEMM (c003) yields the single largest step (3→1 launches, 3→1
  `[T,H]` writes, drops 2 adds).
- **H3:** launch-overhead-bound wins are largest at small `T` (211/293); FLOP-bound behavior only
  starts to matter at `T=997`. Track per-workload speedup to catch a config that wins big-`T` but
  regresses small-`T`.
- **H4:** split-K on the `K=T` outer-product GEMMs (c006) helps occupancy at small `T`; may be neutral
  or slightly negative at `T=997`.
- **H5:** router-path optimization (c007) is low-impact (it is ~11× cheaper and numerically ~0);
  pursue only after shared-path fusion is exhausted.

### Stopping criteria (any one triggers stop → write `SEARCH_COMPLETE`)
- **Convergence:** best geomean improves by **< 2%** across **2 consecutive** newly-evaluated
  candidates, and the remaining roadmap ideas are exhausted or judged low-yield.
- **Budget:** approaching the evaluation cap (100) or token limits (soft 1.0M / normal 1.5M /
  absolute 1.65M) — stop with margin; do not start a candidate that cannot be completed+logged.
- **Regression plateau:** several successive tuning attempts fail or regress with no new hypothesis.
- On stop, `SEARCH_COMPLETE` records: best candidate ID, its geomean + per-workload speedups,
  cumulative evaluations used, and the convergence reason. `final` is run **only** on explicit
  operator approval, on the best valid candidate.

---

## 6. Evidence format (append one JSON object per evaluated candidate to `candidates.jsonl`)

Append-only; never rewrite a prior record. Schema:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "timestamp": "<ISO-8601>",
  "hypothesis": "D0 per-op Triton correctness anchor + baseline speed",
  "design": "D0|D1a|D1b|D1c|D2-...",
  "change_from_parent": "one-sentence single-variable diff",
  "validation": {
    "static_checks_passed": true,
    "notes": "shape/dtype/mask/transpose/scatter/no-fallback confirmed"
  },
  "results": [
    {"uuid": "53b1fc00-0392-5132-9121-a8dcca9d1b51", "batch_seq_len": 997, "passed": true,
     "match_ratio": 0.0, "max_atol_used": 0.36, "speedup": 0.0},
    {"uuid": "229c89b2-ade9-5bee-a787-ad8617fbb601", "batch_seq_len": 293, "passed": true,
     "match_ratio": 0.0, "max_atol_used": 0.12, "speedup": 0.0},
    {"uuid": "bdc774f5-1c4f-5f57-ba9b-1049c6966e28", "batch_seq_len": 211, "passed": true,
     "match_ratio": 0.0, "max_atol_used": 0.13, "speedup": 0.0},
    {"uuid": "8686a737-9875-569c-a926-68c9787c7c67", "batch_seq_len": 384, "passed": true,
     "match_ratio": 0.0, "max_atol_used": 0.18, "speedup": 0.0},
    {"uuid": "43f3b0f2-344c-5778-ad7a-79c347ce2166", "batch_seq_len": 512, "passed": true,
     "match_ratio": 0.0, "max_atol_used": 0.20, "speedup": 0.0}
  ],
  "all_passed": true,
  "geomean_speedup": 0.0,
  "decision": "keep|revert|new_best|invalid",
  "is_new_best": false,
  "cumulative_evaluations": 1,
  "skill_usage": "KernelWiki not applicable (Ampere sm_80; skill covers SM90/SM100 only)",
  "next_action": "what the next candidate will change and why"
}
```

Notes:
- Fill `match_ratio`, `speedup`, `max_atol_used`, and `passed` from the evaluator output verbatim;
  do not fabricate numbers (the `0.0` placeholders above are schema illustration only).
- `geomean_speedup` = geometric mean of the five per-workload `speedup` values.
- `decision`/`is_new_best`/`next_action` capture the lineage move and keep the best pointer explicit.

---

## 7. Skill usage & isolation

- **KernelWiki:** target is A800 / `sm_80` (Ampere). The skill covers Blackwell (SM100) / Hopper
  (SM90) techniques (tcgen05/TMEM/CLC/NVFP4/2-SM/FA4/warp-spec) that do not apply here. Recorded as
  **"not applicable (Ampere)"**; not invoked.
- No other external agents/tools/web/subagents/MCP, per isolation rules. Work stays inside this
  workspace. Evaluator/dataset/controller/launcher/config are never modified or run directly.

---

## 8. First concrete action (done)

Implemented **c001** (D0 per-op Triton) in `solution/solution.py` per §2, ran the §4 static checklist,
evaluated once via `./scripts/evaluate_candidate.sh feedback c001`, appended its record to
`candidates.jsonl` per §6.

## 9. Decision log

- **c001 (D0, root)** — evaluated on A800 (g0056/gpu1). **5/5 PASS**, geomean **3.880x**
  (per-workload: T=997 4.77x, T=384 4.17x, T=512 3.68x, T=211 3.49x, T=293 3.44x). Correctness
  clean at these tolerances despite pure per-op decomposition. Decision: **new_best**, lineage root.
  Observation: the win is already large from fusing nothing but running Triton with fp32-acc bf16
  GEMMs + a cheap in-kernel router prologue; smallest speedups are at small `T` (211/293) as
  predicted by H3 (launch-overhead bound). Fusion (c002–c004) should lift the small-`T` cases most.
- **Next:** c002 — fuse SwiGLU backward into G1's epilogue (remove `grad_shared_activated`
  materialization + one `[T,I]` elementwise pass). Parent = c001.
