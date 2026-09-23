# Draft — L2/049 `group_limited_topk_routing` (A800 / sm_80, Triton)

Run ID: `formal-kda-20260916--sol_execbench--L2-049_group_limited_topk_routing`
Target: NVIDIA A800 (Ampere, sm_80). Primary implementation: **Triton** (PyTorch only for
metadata/launch plumbing). No Torch/CPU/NumPy/CUDA-extension computational fallback.

This document is analysis-only. No `plan.md`, no solution code is produced in this step.

---

## 1. Operation analysis

### 1.1 Signature (from `task/definition.json`)

Entry point `run(hidden_states, weight, expert_bias, routed_scaling_factor)`:

| tensor | shape | dtype | notes |
|---|---|---|---|
| `hidden_states` | `[num_tokens, 4096]` | bf16 | token reps |
| `weight` | `[256, 4096]` | bf16 | gating/routing matrix |
| `expert_bias` | `[256]` | bf16 | per-expert routing bias |
| `routed_scaling_factor` | scalar | f32 | =2.5 in all feedback workloads |

Outputs:

| tensor | shape | dtype |
|---|---|---|
| `topk_idx` | `[num_tokens, 8]` | int64 |
| `topk_weight` | `[num_tokens, 8]` | float32 |

Fixed constants: `num_experts=256`, `top_k=8`, `n_group=8`, `topk_group=4`,
`experts_per_group=32`, `gating_dim=4096`. Only `num_tokens` varies.

This is the DeepSeek-V3 / Ring-flash "noaux_tc" group-limited routing (sigmoid gate + bias +
2-stage group-limited top-k).

### 1.2 Reference math (the exact contract to reproduce)

1. `logits = (hidden_states.float()) @ (weight.float()).T` → `[T, 256]` (fp32 matmul).
2. `scores = sigmoid(logits)` → `[T, 256]`.
3. `scores_routing = scores + expert_bias.float()` → `[T, 256]`. **Bias used for selection only.**
4. View `[T, 8, 32]`; within each group take **top-2** values and sum → `group_scores [T,8]`.
   (top-2 uses `sorted=False`; only the *sum* is used, so tie-order is irrelevant.)
5. Select **top-4** groups by `group_scores` (`sorted=False`) → `group_idx [T,4]`.
6. Build `group_mask [T,8]` (1.0 for selected groups), expand to expert level `[T,256]`.
7. `masked_scores = scores_routing.masked_fill(mask==0, float32.min)`.
8. Select **top-8** experts over the 256 masked scores (`sorted=False`) → `topk_idx [T,8]`.
9. `selected = gather(scores, topk_idx)` — **original sigmoid score, WITHOUT bias**.
10. `topk_weight = selected / (selected.sum(-1, keepdim) + 1e-20)`.
11. `topk_weight *= routed_scaling_factor`.

Two distinct score arrays must be tracked per token: `scores_routing` (sigmoid+bias) drives all
three selections; `scores` (sigmoid only) supplies the returned weights.

Because the 4 selected groups contribute 4×32 = 128 candidate experts and we pick 8, there are
always ≥ top_k valid candidates → no degenerate/empty-selection case.

### 1.3 Cost / roofline sketch

Dominant work is the gate GEMM `[T,4096] × [4096,256]`:
- FLOPs ≈ `T·256·4096·2`. For `T=6144`: ≈ 12.9 GFLOP.
- Memory: `hidden_states` = `T·4096·2` B (50 MB @ T=6144), `weight` = 2 MB (fits A800's ~40 MB
  L2 → reused across all token tiles), `logits` intermediate = `T·256·4` (6 MB @ T=6144).

Per-token routing is 256-wide reductions → negligible vs. the GEMM.

The reference path launches the GEMM **plus** ~10 elementwise/topk/scatter/gather kernels over
`[T,256]` tensors, each a separate launch and each re-reading/writing `[T,256]`. The optimization
thesis is: **fuse the GEMM epilogue with the entire routing pipeline** so `logits` is never
written to DRAM and the routing reductions happen in-register right after the K-loop, collapsing
~11 launches into 1 and removing several `[T,256]` DRAM round-trips.

---

## 2. Constraints & environment

- Triton must own the computation; PyTorch only for shapes/dtype/stride/launch. No fallback of any
  kind (a failed Triton kernel is invalid, not something to paper over).
- Hardware A800 sm_80: bf16 tensor cores (HMMA) and TF32 tensor cores available; no fp8, no
  tcgen05/TMEM (those are Hopper/Blackwell). `KernelWiki` (SM90/SM100) and `ncu-report-skill`
  (B200/sm_100) are **out of scope** for this Ampere target, and profiling is prohibited, so they
  will not be used.
- Evaluation only through `./scripts/evaluate_candidate.sh feedback cNNN`. I may **not** run CUDA,
  a profiler, `nvidia-smi`, torch directly, or any alternate correctness harness. Therefore every
  correctness/perf signal costs exactly one of the 100 candidate evaluations — reasoning must
  front-load correctness before each eval.
- Five fixed feedback workloads = one candidate evaluation. `num_tokens ∈ {2240, 2272, 6144, 2048,
  2112}`, all with `routed_scaling_factor=2.5`, random inputs.
- Budget: 100 evals; tokens soft 1.0M / hard 1.2M. Final 16-workload eval is operator-approved only.
- Immutable candidates: `c001, c002, …`; never mutate an evaluated ID; append-only `candidates.jsonl`.

### 2.1 Tolerance interpretation

Per-workload tolerances: `max_atol ≈ 0.56–0.65`, `max_rtol = 0.01`, `required_match_ratio = 0.98`.
`topk_weight` lies in roughly `[0, 2.5]` (normalized-to-1 weights × 2.5; mean selected weight
≈ 2.5/8 ≈ 0.31). An `atol ≈ 0.6` is ~2× a typical weight — deliberately loose. Interpretation:
near-tie experts/groups can be selected differently by a faithful-but-not-bit-identical
implementation, changing a few weights substantially; the 0.98 match ratio tolerates ≤ 2% such
elements. The unknown exact comparison (likely a set/order-insensitive or dense-scatter comparison,
since `sorted=False` makes positional index order meaningless) means: **reproduce the reference
*set* of selected experts and their weights**; do not rely on any particular ordering of the 8
outputs. This is the single most important correctness design constraint.

---

## 3. Numerical risks

1. **Selection sensitivity near boundaries.** `sigmoid(logit)+bias` decides which group is 4th and
   which expert is 8th. Tiny logit perturbations flip borderline picks. The loose tolerance exists
   precisely because of this; still, minimizing GEMM error maximizes match ratio.

2. **GEMM precision — and why bf16 tensor cores are actually faithful here.** Reference upcasts
   bf16→fp32 then does an fp32 `F.linear` over the *exact bf16 values*. Key observation: the inputs
   already carry only 8 mantissa bits (bf16). A bf16 tensor-core matmul multiplies these exact bf16
   operands and **accumulates in fp32**; each product of two 8-bit-mantissa numbers is exact in
   fp32, and the fp32 accumulator matches the reference's fp32 accumulation up to summation order.
   Hence bf16-HMMA-with-fp32-accum ≈ reference fp32 matmul (differences only from associativity of
   the 4096-term reduction, far below the 0.6 atol). TF32 (10-bit) is likewise lossless for the
   inputs. So the fast path (native bf16 tensor cores, fp32 accumulate) is *also* the numerically
   faithful path. Escalation ladder if a candidate shows correctness pressure:
   `tl.dot(bf16, acc=fp32)` → `input_precision="tf32"` → `"tf32x3"` → `"ieee"` (true fp32, slowest).
   Whether the evaluator's reference ran with `allow_tf32` on/off is unknown; the bf16-input
   argument above makes all these variants agree, so I expect bf16-HMMA to pass.

3. **Epilogue must be fp32.** sigmoid, bias add, top-2 sums, group scores, normalization, and the
   `1e-20` epsilon all in fp32 to match reference. Only the matmul multiplies use tensor cores;
   accumulation and everything downstream stays fp32.

4. **Masking sentinel.** Reference uses `float32.min` (`-3.4e38`). For our in-register iterative
   argmax, using `-inf` (or a large negative) for both group-masked experts and
   already-selected-slot suppression is equivalent for max selection. Must ensure no NaN leaks into
   the normalization (selected values come from real sigmoid scores, never from the sentinel).

5. **Tie-breaking vs. torch.topk.** For exact ties torch picks a specific index; iterative argmax
   picks by scan order. Exact fp ties on random data are measure-zero; within-group top-2 only needs
   the *sum* (tie-order irrelevant); group/expert selection sets (not orders) matter. Residual
   near-tie flips are absorbed by the 0.98 match ratio.

6. **Summation order of the weight normalization.** `selected.sum` over 8 fp32 values — reproduce in
   fp32; ordering effect is ≪ atol.

7. **Output dtypes.** `topk_idx` int64 (compute int32, store as int64), `topk_weight` fp32. Ensure
   correct casts.

---

## 4. Triton design space

### 4.1 Primary architecture — single fully-fused kernel

Grid over token tiles: `program handles BLOCK_M tokens × all N=256 experts`. Because `N=256` is
small, one block can hold the entire expert dimension, so after the K-loop the block owns the full
`logits[BLOCK_M, 256]` and can run the *complete* routing epilogue in-register — no second kernel,
no `logits` DRAM round-trip.

Per program:
1. fp32 accumulator `acc[BLOCK_M, 256]`.
2. K-loop over 4096 in steps `BLOCK_K`: load `hs[BLOCK_M,BLOCK_K]` (bf16), `w[256,BLOCK_K]` (bf16),
   `acc += tl.dot(hs, w.T, acc)` (bf16 inputs, fp32 accum). Weight tiles hit L2 (2 MB, reused).
3. `scores = sigmoid(acc)`; `scores_routing = scores + bias[None,:]` (fp32, bias broadcast over 256).
4. **Group top-2 sum** (8 groups × 32): per group, `m1 = max over 32`; mask out the argmax column;
   `m2 = max over 32`; `group_score = m1 + m2`. Implement over the `[BLOCK_M, 8, 32]` view via
   reshaped reductions (reduce over the size-32 axis). → `group_scores[BLOCK_M,8]`.
5. **Top-4 groups**: iterative argmax over the 8 group scores, 4 iterations, building
   `group_mask[BLOCK_M,8]` (set chosen to selected, suppress with `-inf` between iters).
6. Expand `group_mask` to expert level `[BLOCK_M,256]`; `masked = where(group selected,
   scores_routing, -inf)`.
7. **Top-8 experts**: iterative argmax over 256, 8 iterations. Each iter: `mval,midx = max/argmax`;
   record `topk_idx[:,i]=midx`; capture the companion **original** score via masked reduction
   `sel_i = sum(where(col==midx, scores, 0))` (avoids dynamic register indexing); suppress
   `masked[col==midx] = -inf`.
8. `denom = sum_i sel_i + 1e-20`; `topk_weight[:,i] = sel_i/denom * routed_scaling_factor`.
9. Store `topk_idx` (→int64) and `topk_weight` (fp32) with M-masking for the tail tile.

Companion-value trick (step 7) avoids a post-hoc gather into a register tile (Triton has no dynamic
indexing along a reduced axis); each of the 8 iterations costs a couple of 256-wide reductions —
negligible vs. the GEMM.

### 4.2 Register-pressure / tiling knobs

The `[BLOCK_M,256]` fp32 accumulator is the pressure point (256 KB RF / SM, 255 regs/thread).
- `BLOCK_M=16` → 4096 fp32 accum (÷128 threads ≈ 32/thread) — safe, high occupancy, many tiles
  (weight re-read from L2, cheap).
- `BLOCK_M=32/64` → larger tiles, better GEMM efficiency, more RF/possible spills.
Sweep `BLOCK_M ∈ {16,32,64}`, `BLOCK_K ∈ {32,64,128}`, `num_warps ∈ {4,8}`, `num_stages ∈ {2,3,4}`.
Autotune is possible but adds compile cost/nondeterminism across the 5 workloads; prefer a small
hand-picked config set chosen per-candidate to keep evaluations interpretable.

### 4.3 Alternative — two-kernel split (fallback)

Kernel A: tuned GEMM → `logits[T,256]` in DRAM (6 MB). Kernel B: per-token routing reading `logits`.
Pros: each kernel independently tunable, lower register pressure (no fused mega-accumulator), can
reuse a well-shaped GEMM config. Cons: extra 6 MB write + 6 MB read (~a few µs) and a second launch;
loses the core fusion win. Keep as fallback if the fused kernel is occupancy-limited by the
`[BLOCK_M,256]` accumulator or if the epilogue inflates register usage enough to cripple the GEMM.

### 4.4 Epilogue micro-optimizations (later candidates)

- Fuse the group top-2 with the group-mask expansion to avoid re-materializing.
- Since `topk_group=4` of `n_group=8`, the group selection is tiny — an explicit
  4-iteration argmax over 8 is cheaper than a general sort.
- Represent group selection as a per-expert boolean without a full `[T,8]→[T,256]` broadcast where
  the compiler can fold it.
- Consider computing `group_scores` directly from a single pass that tracks running max1/max2 per
  group to halve the group reductions.
- Keep `scores` and `scores_routing` both resident (they differ only by the bias add) rather than
  recomputing sigmoid.

### 4.5 Things to avoid

- No `tl`-level general sort / full topk over 256 (expensive); iterative argmax with small k is
  cheaper and matches semantics.
- No fp16/bf16 in the epilogue reductions (precision + selection flips).
- No autotune configs that vary results across the 5 workloads in a way that muddies attribution.

---

## 5. Candidate roadmap (high level; concrete plan deferred to `plan.md`)

- **c001** — Correctness-first fused kernel: bf16-HMMA fp32-accum GEMM + fp32 iterative-argmax
  routing epilogue, conservative `BLOCK_M=16/32`. Goal: pass all 5 workloads' correctness and
  establish baseline speedup. This is the highest-value, lowest-risk first shot.
- **c002+** — Performance tuning of `BLOCK_M/BLOCK_K/num_warps/num_stages` on the confirmed-correct
  kernel; one knob-set change per candidate for clean attribution.
- **cNNN (contingency)** — precision escalation (tf32/tf32x3/ieee) *only if* a candidate shows a
  match-ratio miss; two-kernel split if fused occupancy is the bottleneck; epilogue reduction
  fusions once GEMM tiling is settled.
- Stop when speedup converges or budget nears; then write `SEARCH_COMPLETE`. Never run `final`
  without operator approval.

---

## 6. Validation strategy

Because no local torch/CUDA/profiler execution is permitted, validation is **reason-then-measure**:

1. **Static faithfulness proof before each eval.** Re-derive each kernel step against §1.2 and §3
   (two score arrays; bias-in-selection-only; fp32 epilogue; sentinel handling; dtype casts; tail
   masking). Only submit a candidate whose math I can defend line-by-line against the reference.
2. **bf16-HMMA-faithfulness argument (§3.2)** justifies expecting correctness on the fast path;
   keep the precision-escalation ladder ready if the evaluator reports a match-ratio shortfall.
3. **One evaluation per immutable candidate** via `./scripts/evaluate_candidate.sh feedback cNNN`.
   Read per-workload pass/fail + speedup; record parent, source hash, hypothesis, per-workload
   result, geomean, decision, cumulative eval count, skill usage in `candidates.jsonl` (append-only).
4. **Isolate variables.** Correctness-affecting changes (precision, selection logic) and
   perf-only changes (tiling) go in separate candidates so a regression is attributable.
5. **Correctness gate before speed.** A candidate that fails any workload's correctness is discarded
   regardless of speed; ranking is geomean speedup *among fully-correct* candidates.
6. **Budget discipline.** Front-load reasoning to minimize evals; prefer a curated handful of tiling
   configs over broad autotune sweeps; converge and stop rather than chase noise.

### Open questions to resolve empirically (cheaply, via c001 feedback)
- Does bf16-HMMA meet the 0.98 match ratio on all 5 workloads? (Expected yes per §3.2.)
- Actual speedup magnitude of fusion vs. the multi-launch reference.
- Whether the `[BLOCK_M,256]` fp32 accumulator spills at `BLOCK_M=32/64` (inferred from relative
  speedups across tiling candidates, since direct profiling is disallowed).

---

## 7. Skill usage note

`KernelWiki` and `ncu-report-skill` target Hopper/Blackwell (SM90/SM100) and B200 profiling; this
task is Ampere (sm_80) and profiling is prohibited. Neither is applicable, so neither is invoked.
This will be recorded as `skills: none` in the candidate records.
