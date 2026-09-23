# Draft — L2/049 `group_limited_topk_routing` (H100 / sm_90)

Status: analysis only. No `plan.md`, no solution code yet. This draft fixes the exact
semantics, enumerates numerical hazards, maps the Triton design space, and defines how
candidates will be validated against the sanctioned evaluator.

---

## 1. Operation specification (from `task/definition.json`)

DeepSeek-V3 / Ring-flash style **group-limited top-k expert routing**. All constants are
fixed by the definition:

| Symbol | Value | Meaning |
|---|---|---|
| `num_experts` | 256 | total experts |
| `n_group` | 8 | expert groups |
| `experts_per_group` | 32 | `256 // 8` |
| `topk_group` | 4 | groups kept |
| `top_k` | 8 | experts selected per token |
| `gating_dim` | 4096 | hidden dim of the gating projection |
| `num_tokens` (`T`) | variable | 2048 … 16384 in the feedback set |

### Inputs
- `hidden_states` : `[T, 4096]` **bf16**
- `weight`        : `[256, 4096]` **bf16**
- `expert_bias`   : `[256]` **bf16**
- `routed_scaling_factor` : python `float` (fp32 scalar), always `2.5` in the feedback set.

### Outputs
- `topk_idx`    : `[T, 8]` **int64** — selected expert indices.
- `topk_weight` : `[T, 8]` **float32** — normalized, scaled routing weights.

### Reference algorithm (exact, step by step)
1. `logits = (hidden_states.float()) @ (weight.float()).T` → `[T, 256]`, an fp32 GEMM.
   Inputs are bf16, so the `.float()` cast is lossless; only the *products/accumulation*
   run in fp32.
2. `scores = sigmoid(logits)` → `[T, 256]` (fp32).
3. `scores_for_routing = scores + expert_bias.float()` → `[T, 256]`. **Bias is added only
   for selection**, not for the final weights.
4. View as `[T, 8, 32]`; take **top-2 per group** (`largest=True, sorted=False`) and **sum
   the two** → `group_scores` `[T, 8]`.
5. **top-4 groups** by `group_scores` (`sorted=False`) → the *set* `group_idx` `[T, 4]`.
6. `group_mask` `[T, 8]` = scatter 1.0 at `group_idx`.
7. Expand mask to expert level `[T, 256]` (each group covers 32 contiguous experts).
8. `masked_scores = scores_for_routing.masked_fill(mask == 0, finfo(f32).min)`
   (`neg_inf = -3.4028235e38`).
9. **top-8 experts** from `masked_scores` (`sorted=False`) → `topk_idx` `[T, 8]`.
10. `selected = gather(scores, 1, topk_idx)` — weights use **sigmoid scores WITHOUT bias**.
11. `topk_weight = selected / (selected.sum(-1, keepdim) + 1e-20)`.
12. `topk_weight *= routed_scaling_factor`.
13. Cast `topk_idx→int64`, `topk_weight→float32`; return.

Two distinct score tensors are needed downstream: `scores_for_routing` (bias-added, drives
group top-2 / group top-4 / expert top-8 selection) and `scores` (bias-free, drives the
returned weights). A correct kernel must keep both.

---

## 2. Workload characterization

16 feedback workloads, `num_tokens ∈ {2048, 2080, 2112, 2144, 2176, 2208, 2240, 2272,
2851, 3169, 3557, 4093, 6144, 8192, 12288, 16384}`. `routed_scaling_factor = 2.5` for all;
all inputs are `random`. Tolerances per workload: `max_rtol = 0.01`,
`required_match_ratio = 0.98`, `max_atol ∈ [0.44, 0.84]`.

Observations that shape the design:
- **Always large-batch.** Smallest `T = 2048`. This is squarely the *throughput /
  compute-bound* regime — no decode-latency corner cases, no tiny-M tail. A single
  compute-efficient GEMM config should serve all shapes.
- **Odd M values** (2080, 2851, 3169, 3557, 4093, …) are not multiples of typical block
  sizes → the M dimension needs boundary masking; K=4096 and N=256 are clean.
- **N=256 is tiny and fixed** = exactly one N-tile. This enables fusing the entire routing
  epilogue into the GEMM (a full row of 256 logits is produced by one M-block program).
- Data volume is dominated by `hidden_states` (up to 16384×4096×2 = 134 MB). `weight` is
  2 MB (fits/streams cheaply, reusable across M-blocks). Outputs are tiny (`T×8`).

---

## 3. Cost model & the speedup lever

FLOPs are dominated by the projection: `2·T·4096·256`. For `T=16384` ≈ **34 GFLOP**;
routing work is on `[T,256]` (≤ 16 MB traffic) — cheap by comparison.

The central lever is **precision of the projection GEMM**:
- The reference runs the matmul in **fp32**. On H100 there are no fast fp32 tensor cores;
  depending on `torch.backends.cuda.matmul.allow_tf32`, the reference is either true IEEE
  fp32 (~67 TFLOPS → ~0.5 ms at T=16384) or TF32 (~few-hundred TFLOPS). Either way it is
  far below bf16 tensor-core throughput (~1000 TFLOPS peak).
- A Triton bf16-input / fp32-accumulate `tl.dot` should run the projection ~5–10× faster
  than a true-fp32 reference, and TF32x3 sits in between at higher fidelity.
- On top of raw GEMM, the reference is a **~7-launch** pipeline (linear, sigmoid, add,
  topk×3, scatter, gather, …) that round-trips `[T,256]` and mask tensors through global
  memory. Fusing to 1–2 launches removes those launches and the intermediate traffic.

So the win is: **(a) bf16/TF32 tensor-core projection + (b) fused routing epilogue**. The
realized multiplier hinges on the reference's actual precision path, which we will measure
via profiling and the first candidate.

---

## 4. Numerical analysis & risks

### 4.1 Sigmoid saturation dominates the statistics (important)
Each logit is a dot product of two ~N(0,1) bf16 vectors of length 4096 ⇒ logit ~ N(0, 4096),
**std ≈ 64**. `sigmoid` saturates to exactly `1.0f` for logit ≳ 16 and to `0.0f` for
logit ≲ −16. With std 64, the large majority of experts have scores that are *exactly* 0.0
or 1.0 in fp32, essentially independent of small GEMM errors. Consequences:
- Selection among the saturated "1.0" experts is decided **entirely by `expert_bias`**
  (`scores_for_routing = 1.0 + bias`). Group top-2/top-4 and expert top-8 among these are
  bias-ordered.
- The precision-sensitive experts are only those with `|logit| ≲ 16`, i.e. `P(|N|<16/64) ≈
  P(|N|<0.25) ≈ 20%` of experts sit in the sigmoid transition band. Selection flips can
  only originate there.
- **Net:** saturation makes most of the routing robust to GEMM precision, but ~20% of
  experts remain genuinely precision-sensitive, so bf16 GEMM error can still flip a
  boundary group (4th vs 5th) or boundary expert (8th vs 9th).

### 4.2 GEMM precision vs. selection flips
bf16×bf16→fp32 accumulate introduces per-term rounding ~2⁻⁸; over 4096 random-sign terms
the logit error std ≈ √4096·2⁻⁸ ≈ 0.25 (absolute). That is negligible for saturated experts
but can reorder transition-band experts whose scores differ by < ~0.06. A flipped **group**
is the expensive failure: it swaps the 32-expert candidate pool and can change several of
the 8 selected experts for that token at once.

Match budget: outputs have `T·8` elements per field; `required_match_ratio = 0.98` ⇒ up to
2% of elements may mismatch. If a fraction `f` of tokens flip and each flipped token spoils
up to 8/8 index elements, we need `f ≲ 2%`. Group flips are rarer than single-expert flips,
so the practical question is whether bf16 keeps token-flip rate under ~2%. **Mitigation
ladder** (cheap→safe): bf16 `tl.dot` → `input_precision="tf32"` → `"tf32x3"` → `"ieee"`.
We start conservative-but-fast and only escalate if correctness fails.

### 4.3 Tie-breaking in top-k (subtle)
Because saturated scores collapse to `1.0` and `expert_bias` is **bf16** (only 8 mantissa
bits), *exact* ties in `scores_for_routing` and in `group_scores` are plausible (bf16 bias
collisions). `torch.topk`/`argmax` resolve ties deterministically (CUDA topk tends to favor
the lower index). An iterative-argmax Triton implementation must reproduce the same
tie-break (lowest index wins) to keep indices aligned with the reference. This is a real,
if low-frequency, mismatch source and is worth a dedicated check.

### 4.4 Output-comparison semantics (open question, drives c001 design)
`torch.topk(..., sorted=False)` leaves the *order* of the 8 indices unspecified, so any
sane evaluator must compare in an order-independent way (e.g. sort both rows, or compare a
densified `[T,256]` weight vector, or set-compare indices with aligned weights). We must not
assume position-wise comparison. **c001 will be a faithful implementation** whose primary
job is to (i) confirm correctness passes and (ii) reveal how strict the comparison is on
index order and on precision, before we optimize aggressively.

### 4.5 Other exactness details to preserve
- Compute `sigmoid` in **fp32** (`tl.sigmoid` on fp32 logits) — matches reference saturation.
- Bias added in fp32 (`expert_bias.float()`); load bf16 bias, upcast in-kernel.
- Mask sentinel = `-3.4028235e38` (fp32 min), applied to `scores_for_routing` before the
  expert top-8. Using `-inf` instead is acceptable only if it never changes the argmax.
- Denominator `+ 1e-20` before dividing; final `* 2.5`. Weight sum uses **bias-free** scores.
- Output dtypes exactly `int64` / `float32`.
- Group top-2 is `max1 + max2` of the 32 group members of `scores_for_routing`.
- top-4 groups needs the **set** (for masking), not an order.

---

## 5. Triton design space

Every program handles a block of tokens; **tokens are independent**, so the kernel is
embarrassingly parallel over M. N=256 fits in one tile, which unlocks fusion.

### 5.1 Fusion topology (primary axis)
- **Option A — single fused kernel (preferred target).** Grid over M-blocks only. Each
  program computes the full `[BLOCK_M, 256]` logits via a K-loop `tl.dot`, applies
  `sigmoid`, adds bias, runs the two-stage top-k, and writes `topk_idx`/`topk_weight`.
  Logits/scores/mask never touch global memory → maximal traffic saving, 1 launch.
  Cost: `[BLOCK_M,256]` fp32 accumulator (BLOCK_M=64 → 64 KB) plus routing temporaries;
  register/occupancy pressure to watch.
- **Option B — two kernels.** K1: fused GEMM+sigmoid+bias → `scores`,`scores_for_routing`
  in gmem (`2·T·256·4` bytes). K2: routing on those. Simpler to get correct; still ≪ 7
  launches. Good fallback / correctness anchor if Option A has occupancy or register issues.

Plan: prototype the correctness anchor first, then push toward the fused single kernel.

### 5.2 GEMM tiling
- N-tile = 256 (whole row). K-loop over 4096 with `BLOCK_K ∈ {32,64,128}`.
- `BLOCK_M ∈ {32,64,128}`; M masked for odd `T`. `weight` tile is `[256, BK]`; feed
  `tl.dot(hidden[BLOCK_M,BK], weightᵀ[BK,256])` (load/transpose weight appropriately, or
  choose strides to present `[BK,256]`).
- `num_warps`, `num_stages` tuned for the K-loop pipeline. Weight (2 MB) is reused across
  all M-blocks → benefits from L2 residency.
- Precision knob = `input_precision` on `tl.dot` (see 4.2 ladder).

### 5.3 Routing epilogue (small, register-resident on `[BLOCK_M,256]`)
- **Group top-2:** reshape logical `[BLOCK_M,8,32]`; `max1 = max` over 32; mask the argmax;
  `max2 = max`; `group_scores = max1 + max2` → `[BLOCK_M,8]`.
- **top-4 groups:** iterative argmax ×4 over 8 group scores, masking each pick with the
  fp32-min sentinel and **lowest-index tie-break**, to build `group_mask [BLOCK_M,8]`.
  (Thresholding on the 4th-largest is faster but mishandles ties — prefer iterative argmax
  to mirror torch.)
- **Expert mask:** broadcast `group_mask` to `[BLOCK_M,256]`; where 0, set
  `scores_for_routing = fp32_min`.
- **Expert top-8:** iterative argmax ×8 over 256, each pick records the index and the
  **bias-free** `scores` value at that index, then masks the position. Produces indices in
  descending masked-score order.
- **Weights:** `w = sel / (sum(sel) + 1e-20) * 2.5`; store `[BLOCK_M,8]`.
- All argmax reductions are over ≤256 elements and fit in registers; 2+4+8 = 14 reduction
  passes total, negligible vs the K-loop.

### 5.4 Autotuning / config
All feedback M are large and similar in character; a single well-chosen config likely
covers the set. Autotune (within a single candidate) is permissible but adds compile cost;
initial candidates will pin explicit configs and iterate via new candidate IDs. Keying
autotune on `T` risks recompiles across 16 distinct M — prefer a config that is M-agnostic.

---

## 6. Candidate roadmap (high level; concrete plan deferred to `plan.md`)
- **c001 — correctness anchor.** Simplest defensible design (likely Option B or a modest
  Option A), GEMM at a *safe* precision (`tf32x3`, possibly `ieee` if needed), faithful
  tie-breaking. Goal: pass all 16 workloads and learn the comparison strictness + baseline
  speedup. This is the reference point for everything after.
- **Then** step down precision (tf32x3 → tf32 → bf16) to find the fastest that still passes,
  fuse to a single kernel (Option A), and tune `BLOCK_M/BLOCK_K/num_warps/num_stages`.
- Each meaningful source/config/launch change ⇒ a **new immutable** candidate ID; record
  parent, source hash, hypothesis, per-workload result, geomean, decision, cumulative eval
  count, and skill usage in `candidates.jsonl` (append-only).

---

## 7. Validation strategy
- **Only** `./scripts/evaluate_candidate.sh feedback cNNN` runs code (one full 16-workload
  run = one candidate eval; budget = 100). No direct CUDA / torch / numpy / nvidia-smi /
  external harness; no Torch/CPU fallback in the solution (a failing Triton kernel is
  invalid, not to be papered over).
- Because ad-hoc local execution is not available, correctness must be reasoned through
  before spending an eval: keep c001 conservative so the first eval is informative rather
  than a wasted debug cycle.
- **Profiling** only via `./scripts/ncu_profile.sh` under the `ncu-report-skill` workflow,
  and **never concurrently with an evaluation** (a foreign process on the locked GPU during
  timing ⇒ return code 3, discarded measurement, wasted eval). Sequence profiling and
  evaluation strictly.
- Use profiling to confirm the GEMM is the bottleneck, measure achieved vs. peak
  tensor-core throughput, and check occupancy / register pressure of the fused epilogue
  before committing tuning changes.
- Track the geomean-speedup ranking metric; every selected workload must pass correctness.
  Stop and write `SEARCH_COMPLETE` when improvement genuinely converges or the budget is hit.
  Never run `final` without explicit operator approval.

## 8. Open questions to resolve empirically
1. Does the evaluator compare `topk_idx` order-independently? (Governs whether we must
   replicate torch's `sorted=False` order or just the set.)
2. What is the reference's real matmul precision (true fp32 vs TF32)? Sets the achievable
   speedup ceiling and the safe precision floor.
3. Does bf16-input GEMM keep the token-flip rate under the 2% mismatch budget, or must we
   hold at tf32x3/tf32?
4. Fused single-kernel occupancy at `BLOCK_M=64` with a `[BLOCK_M,256]` fp32 accumulator —
   is register pressure acceptable, or is the two-kernel split faster in practice?
