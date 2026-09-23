# Draft — L2/080 MoE complete layer with shared expert (backward)

Run ID: `formal-kda-20260916--sol_execbench--L2-080_moe_complete_layer_with_shared_expert_backward`
Target GPU: **NVIDIA A800, `sm_80` (Ampere)**. Triton is the required compute path.

> Status: analysis only. No `plan.md` and no solution code are written in this step.

---

## 1. Task contract and entry point

We must expose `solution/solution.py::run(...)` with the exact signature and output
ordering of the reference in `task/definition.json`. All *computation* must be Triton;
PyTorch may only be used for tensor metadata, allocation, dtype/stride bookkeeping, and
kernel launch plumbing. **No** Torch/CPU/NumPy/CUDA-extension computational fallback is
permitted — a failing Triton path is invalid, not something to substitute.

Constant axes (fixed across all workloads):

| symbol | meaning | value |
|--------|---------|-------|
| `H`  | `hidden_size`            | 4096 |
| `I`  | `moe_intermediate_size`  | 1408 |
| `E`  | `n_routed_experts`       | 128  |
| `K`  | `num_experts_per_tok`    | 8    |
| `T`  | `batch_seq_len` (variable) | 211, 293, 384, 512, 997 (feedback set) |

Scalars baked into the reference: `routed_scaling_factor = 1.0`, `norm_topk_prob = True`,
`n_group = topk_group = 1` (so `score_mask` is all ones), `eps = 1e-20`.

### Inputs (dtype / shape)
- `grad_output`  bf16 `[T,H]`
- `hidden_states`  bf16 `[T,H]`
- `router_weight`  bf16 `[E,H]`
- `e_score_correction_bias`  f32 `[E]`  (all zeros in the generator, but not assumed)
- `router_logits`  f32 `[T,E]`  (unused by the reference `run`)
- `scores`  f32 `[T,E]`  (sigmoid outputs)
- `topk_indices`  int64 `[T,K]`
- `topk_weights`  f32 `[T,K]`  (already normalized to sum≈1 per row in the generator)
- `score_mask`  f32 `[T,E]`  (all ones here)
- `shared_expert_gate_weight`  bf16 `[I,H]`
- `shared_expert_up_weight`  bf16 `[I,H]`
- `shared_expert_down_weight`  bf16 `[H,I]`
- `shared_gate_output`  bf16 `[T,I]`
- `shared_up_output`  bf16 `[T,I]`
- `shared_activated`  bf16 `[T,I]`

### Outputs (ordered tuple)
1. `grad_hidden_states`  bf16 `[T,H]`
2. `grad_router_weight`  **f32** `[E,H]`
3. `grad_shared_expert_gate_weight`  bf16 `[I,H]`
4. `grad_shared_expert_up_weight`  bf16 `[I,H]`
5. `grad_shared_expert_down_weight`  bf16 `[H,I]`

Note the reference `run` is decorated `@torch.no_grad()` — it is a hand-written backward,
not autograd. We reproduce its arithmetic exactly (op-for-op semantics), not "the correct
math"; the reference contains an intentional *approximation* for the routed-expert path
(see §4.3) that we must match, not fix.

---

## 2. Exact decomposition of the reference computation

Two independent paths that both accumulate into `grad_hidden_states`.

### 2.1 Shared-expert path (numerically dominant)

Let `go = grad_output` (`grad_shared_output` is just a clone of it).

- **G1** `grad_shared_activated = go @ down_w`  → `[T,I]`.
  `down_w` is `[H,I]`; bf16·bf16, fp32 accumulate, output bf16. Contraction `K=H=4096`.
- **G2** `grad_shared_expert_down_weight = (go.T.f32) @ (shared_activated.f32)` → `[H,I]`,
  cast to bf16. Contraction `K=T` (small). **Output #5.**
- **SwiGLU backward** (elementwise, `[T,I]`):
  - `grad_gate_silu = grad_shared_activated * shared_up_output`
  - `grad_up_output = grad_shared_activated * silu(shared_gate_output)`   ← `grad_shared_up_output`
  - `sig = sigmoid(gate.f32)`; `silu'(x) = sig·(1 + x·(1−sig))`
  - `grad_gate_output = (grad_gate_silu.f32) · silu'(gate)` → bf16   ← `grad_shared_gate_output`
- **G3** `grad_hidden_from_shared_up = grad_up_output @ up_w`  → `[T,H]` bf16. `up_w=[I,H]`, `K=I`.
- **G4** `grad_hidden_from_shared_gate = grad_gate_output @ gate_w` → `[T,H]` bf16. `K=I`.
- **G5** `grad_shared_expert_up_weight = (grad_up_output.T.f32) @ (hidden.f32)` → `[I,H]`→bf16.
  `K=T`. **Output #4.**
- **G6** `grad_shared_expert_gate_weight = (grad_gate_output.T.f32) @ (hidden.f32)` → `[I,H]`→bf16.
  `K=T`. **Output #3.**
- `grad_hidden_states = grad_hidden_from_shared_up + grad_hidden_from_shared_gate`.

### 2.2 Router path (numerically ~zero, but must be produced)

- `grad_norm_sq = rowsum(go.f32²)` → `[T,1]`; `grad_topk_weights = grad_norm_sq / K` broadcast to `[T,K]`.
  (This is the *isotropic straight-through* approximation; see §4.3.)
- Normalization-Jacobian (quotient rule), with `routed_scaling_factor=1`:
  - `S = rowsum(topk_weights) + 1e-20`
  - `sum_grad = rowsum(grad_topk_weights · topk_weights) / S`
  - `grad_topk_before_norm = (grad_topk_weights − sum_grad) / S`   → `[T,K]`
- **Scatter** `grad_scores[T,E] = 0; scatter_add_(dim=1, topk_indices, grad_topk_before_norm)`.
- `grad_scores *= score_mask` (all ones ⇒ no-op, but keep general).
- `grad_router_logits = grad_scores · scores · (1 − scores)` → `[T,E]` f32 (sigmoid derivative).
- **G7** `grad_hidden_from_router = grad_router_logits.to(bf16) @ router_w` → `[T,H]` bf16. `K=E=128`.
- **G8** `grad_router_weight = grad_router_logits.T @ hidden.f32` → `[E,H]` **f32**. `K=T`. **Output #2.**
- `grad_hidden_states += grad_hidden_from_router`.

---

## 3. Compute / cost analysis

Eight GEMMs plus a handful of elementwise/reduction/scatter kernels. Dominant FLOPs are the
six shared-path GEMMs (each ≈ `2·T·H·I`, with `H·I ≈ 5.77M`); router GEMMs use `E=128`
instead of `I=1408` (≈11× cheaper) and the scatter is trivial.

Two structural families of GEMM:
- **"Tall/skinny output" GEMMs** with contraction `K=H` or `K=I` (G1, G3, G4, G7): output has
  `T` rows (211–997) → short-M GEMMs.
- **"Outer-product" weight-grad GEMMs** with contraction `K=T` (small): G2 `[H,I]`, G5/G6 `[I,H]`,
  G8 `[E,H]`. Small contraction, large outputs (`[H,I]` bf16 ≈ 11.5 MB each). These are
  latency/memory sensitive, not FLOP-bound.

Because `T` is small, every GEMM here is a *small* GEMM where kernel-launch overhead and
memory round-trips dominate. That is precisely the regime where a **fused Triton**
implementation can beat the reference's ~8 cuBLAS calls + ~6 elementwise kernels + a
scatter, even if a single Triton GEMM is not faster than cuBLAS in isolation. The win comes
from *fewer launches* and *fewer materialized intermediates*, not from out-GEMMing cuBLAS.

### 3.1 Fusion opportunities (the core of the speedup thesis)

- **Epilogue-fuse SwiGLU backward into G1**: G1 produces `grad_shared_activated[T,I]`; the
  SwiGLU-backward elementwise turns it (with `shared_up_output`, `shared_gate_output`) into
  `grad_up_output` and `grad_gate_output`. Fusing this into G1's epilogue avoids writing
  `grad_shared_activated` and a separate elementwise pass over `[T,I]`.
- **Concatenate the grad_hidden GEMMs G3+G4+G7** into one GEMM:
  `grad_hidden = [grad_up_output | grad_gate_output | grad_router_logits.bf16] @ [up_w; gate_w; router_w]`
  with contraction `K = I + I + E = 2944`. One `[T,H]` output, one accumulator, replaces three
  GEMMs and two adds. (All three left operands are bf16, all three right operands bf16 ⇒ clean.)
- **Concatenate the shared weight-grad GEMMs G5+G6** into one GEMM
  `[grad_up_output | grad_gate_output].T @ hidden → [2I, H]` (contraction `K=T`), splitting the
  `[2I,H]` output into the two `[I,H]` bf16 results. G2 is a separate `[H,I]` output (different
  left operand `go`, different N-operand `shared_activated`) but shares `K=T`.
- **Router path** (G8 f32, scatter, sigmoid-deriv, normalization) is cheap and numerically
  delicate; keep it as small dedicated kernels rather than forcing it into the fused bf16 GEMMs.

These are *plan-stage* choices; the draft only enumerates the design space.

---

## 4. Numerical analysis and risks

### 4.1 Magnitudes (from the generator's distributions)
- `grad_output ~ N(0,1)`, `H=4096` ⇒ per-row `grad_norm_sq ≈ 4096`; `grad_topk_weights ≈ 512`.
- Weights scaled by `0.02`; `shared_gate_output/up ~ std≈1.3`; `silu(gate) ~ O(0.5)`.
- `grad_shared_activated ~ std≈1.3`; `grad_up_output/grad_gate_output ~ O(0.6)`.
- `grad_hidden_states ~ O(0.4–0.6)` (dominated by the shared up+gate contributions).
- Shared weight-grads `[I,H]/[H,I]`: contraction `K=T` of `O(0.6)·O(1)` terms ⇒ `O(√T · 0.6)`,
  i.e. tens; cast to bf16.

### 4.2 dtype-matching (must reproduce reference casts)
- G1, G3, G4, G7 are bf16·bf16 → bf16 in the reference. Triton `tl.dot` on bf16 inputs with an
  fp32 accumulator reproduces this (tensor-core computes exact bf16 products, fp32 accumulate),
  then we round to bf16 on store. **Match risk: low.**
- G2, G5, G6 upcast bf16 operands to f32 then matmul, then cast the *output* to bf16. Since
  bf16→f32 is exact, an fp32-accumulate tensor-core GEMM of the bf16 operands yields the same
  products; the only question is the accumulator/rounding. Given the bf16 output cast, TF32 or
  bf16-input fp32-accumulate is more than sufficient. **Match risk: low.**
- G8 keeps f32 output (`grad_router_weight`). Its left operand `grad_router_logits` is genuine
  f32 (not bf16). But it is numerically ~0 (see §4.3), so precision of this GEMM is immaterial
  to passing tolerance; still, prefer an fp32 (TF32-accumulate acceptable) path here.
- `silu`, `sigmoid`, and the SiLU derivative are computed in f32 in the reference before the
  bf16 cast; do the same in-kernel (compute in fp32, store bf16).

### 4.3 The router-path cancellation (key insight)
Because `grad_topk_weights` is *isotropic* (identical value `g` across the `K` slots) and the
normalization Jacobian removes the common-mode component:
`grad_before[j] = (g − g·S/S)/S ≈ 0`. With `topk_weights` already summing to ≈1 and `eps=1e-20`,
`grad_topk_before_norm ≈ 0` up to fp rounding. Consequently `grad_router_logits ≈ 0`,
`grad_router_weight ≈ 0`, and `grad_hidden_from_router ≈ 0`.

Implications:
- `grad_router_weight` (output #2) is a near-zero tensor; comparisons are effectively `atol`-only,
  which is very forgiving. We must still *produce* it via the same formula (a tiny residual), but
  precision is a non-issue.
- The **entire measured error budget lives in the shared-expert path**. Optimization aggressiveness
  (TF32 vs bf16 accumulate, tiling, split-K) is bounded by the shared-path outputs, not the router.
- Risk: if we naively "simplify" the router path to exactly 0, the residual might differ from the
  reference by ~1e-4, still far inside `atol`. We will nonetheless reproduce the formula faithfully
  to avoid surprises and to be robust if a non-zero `e_score_correction_bias` ever appears.

### 4.4 Tolerance interpretation
Feedback tolerances: `max_atol ∈ {0.12, 0.13, 0.18, 0.2, 0.36}`, `max_rtol = 0.05`,
`required_match_ratio = 0.98`. Assumed check: element passes if `|a−b| ≤ atol + rtol·|b|`, and
≥98% of elements per tensor must pass. Given output magnitudes `O(0.5)` and these atols, bf16
rounding (~2⁻⁸ relative ≈ 0.4%) is comfortably inside budget. This confirms we may use bf16
tensor cores / TF32 freely on the shared path. The 2% "free" fraction also absorbs occasional
large-relative-error elements near zero.

### 4.5 Other correctness hazards
- **`int64` topk_indices**: scatter target indices; must gather/scatter with int64 or safely cast
  to int32 (values < 128, safe). Triton scatter via `tl.atomic_add` or a per-token serial loop over
  `K=8`; must guard bounds and match `scatter_add_` accumulation order semantics (add is commutative
  so order does not affect result, only fp associativity — negligible here since values ~0).
- **Transposes / strides**: G2/G5/G6/G8 use logical transposes of the left operand. Handle via
  stride arithmetic in the kernel (do not physically transpose). Weight tensors have specific
  row/col-major layouts (`down_w`=[H,I], `up_w`/`gate_w`=[I,H]); tile pointers must respect these.
- **Non-multiple `T`**: `T ∈ {211,293,384,512,997}` — 211, 293, 997 are not multiples of typical
  block sizes; masking on the `T` dimension (both as M-dim and as K-dim in the outer-product GEMMs)
  is mandatory. Masked K-loop tails must contribute zero.
- **Accumulator init / additive outputs**: `grad_hidden_states` is a sum of contributions; if built
  from a single fused concatenated GEMM the accumulator handles it, otherwise use one zero-init plus
  in-kernel adds (avoid read-modify-write races across programs).
- **Autotune / immutability**: Triton autotuning must be deterministic and self-contained within a
  candidate; any config change is a new candidate ID. Prefer a small fixed config set keyed on `T`
  buckets to avoid nondeterministic tuning across evaluations.

---

## 5. Constraints checklist
- Triton-only compute; torch for allocation/metadata/launch only. No fallback of any kind.
- Immutable sequential candidates `c001, c002, …`; new ID for any meaningful source/config/launch change.
- Evaluate exclusively via `./scripts/evaluate_candidate.sh feedback cNNN`; five fixed workloads = one evaluation.
- Never run `final` without operator approval. Do not touch evaluator/dataset/controller/launcher/config.
- Budget: ≤100 candidate evaluations; token soft/normal/absolute = 1.0M / 1.5M / 1.65M.
- Do not run CUDA/profiler/`nvidia-smi`/the raw evaluator directly.

---

## 6. Triton design space

### 6.1 Kernel decomposition options (from conservative → aggressive)
- **D0 (baseline correctness)**: one Triton kernel per reference op — 8 GEMM kernels + separate
  elementwise (SwiGLU-bwd, sigmoid-deriv), a row-reduction for `grad_norm_sq`, a scatter kernel,
  and the normalization. Maximizes correctness confidence; minimal fusion. Establishes that a pure
  Triton path passes and gives a speed reference vs. torch.
- **D1 (epilogue fusion)**: fuse SwiGLU backward into G1's epilogue; fuse the sigmoid-derivative and
  normalization into small elementwise kernels; fuse the three grad_hidden GEMMs into one
  concatenated-K GEMM (G3+G4+G7); fuse G5+G6 into one concatenated-output GEMM.
- **D2 (aggressive)**: additionally fuse the `grad_norm_sq` row-reduction + normalization + scatter +
  sigmoid-deriv into a single router prologue; consider split-K for the `K=T` outer-product GEMMs
  (G2/G5/G6) when `T` is small to keep SMs busy; persistent-kernel / grouped launch to cut overhead.

### 6.2 GEMM tiling and precision
- Short-M GEMMs (G1/G3/G4/G7): block `[BM, BN]` with `BM` covering the small `T` (e.g. 64/128), `BN`
  along `H` or `I`; K-loop over `H`/`I`/`2944`. bf16 inputs, fp32 accumulate.
- Outer-product GEMMs (G2/G5/G6/G8): contraction `K=T` (small, 211–997). Large outputs; tile
  `[BM,BN]` over the output; single (or few) K-tiles. Consider split-K + atomic/second-pass reduce for
  small `T` to improve occupancy. bf16 in / fp32 accum for G2/G5/G6; f32/TF32 for G8.
- Precision knobs: `tl.dot(..., input_precision="tf32"|"ieee")` and bf16 operands. Default plan: bf16
  tensor cores with fp32 accumulate on the shared path (matches reference, fast on Ampere); reserve
  IEEE/TF32 only where a candidate shows a match-ratio regression.

### 6.3 Autotuning strategy
- Small, fixed candidate-local config lists (block sizes, `num_warps`, `num_stages`) selected by a `T`
  bucket, chosen to be deterministic. Ampere `num_stages` typically 3–4; `num_warps` 4–8. Avoid broad
  autotune sweeps that could vary across evaluations or blow the token/eval budget.

### 6.4 Launch/plumbing
- Preallocate all five outputs with `torch.empty`/`zeros` (zeros only where an additive accumulation
  requires it). Pass strides explicitly. Concatenated-K GEMMs read multiple source tensors via
  separate base pointers within one kernel rather than a physical `torch.cat` (avoid extra copies),
  unless a cheap `cat` is measured to be faster.

---

## 7. Candidate roadmap (sketch, to be formalized in plan.md)
1. **c001** — D0 straightforward per-op Triton (correctness anchor + baseline speed).
2. **c002** — D1 epilogue-fuse SwiGLU backward into G1; keep other GEMMs separate.
3. **c003** — D1 concatenated grad_hidden GEMM (G3+G4+G7) and fused G5+G6.
4. **c004+** — D2 tuning: split-K for `K=T` GEMMs, block/warp/stage tuning per `T` bucket, router
   prologue fusion. Each measured; keep only non-regressing, correctness-passing steps.
Advance only when the previous candidate passes all five workloads.

---

## 8. Validation strategy
- **Correctness gate**: every candidate must pass all five feedback workloads (≥98% element match at
  the per-workload tolerance) before its speed is considered. A candidate that fails any workload is
  invalid and is not promoted.
- **Numerical self-checks before evaluating** (local reasoning / offline mental checks, not the
  evaluator): confirm output dtypes/shapes match the tuple contract; confirm bf16 outputs are stored
  bf16 and `grad_router_weight` is f32; confirm masking on `T` tails and `K`/`N` tails; confirm the
  scatter accumulates over `K=8` slots per token.
- **Precision ablation discipline**: change one precision knob per candidate so a match-ratio change
  is attributable. Watch specifically the shared-path outputs (`grad_hidden_states`,
  `grad_shared_expert_{gate,up,down}_weight`) since the router outputs are ~0 and near-trivially pass.
- **Speed accounting**: geometric mean speedup across the five workloads is the ranking metric; record
  per-workload speedup to catch a config that wins on large `T` but regresses on small `T` (211/293).
- **Evidence logging**: for each evaluated candidate append one JSON record to `candidates.jsonl`
  (parent, source hash, hypothesis, validation, per-workload result, geomean, decision, cumulative
  eval count, skill usage). Never rewrite prior records.
- **Convergence / stop**: stop and write `SEARCH_COMPLETE` when successive candidates no longer improve
  geomean meaningfully, or on hitting the eval/token budget. `final` only with operator approval.

---

## 9. Skill usage
- `KernelWiki` covers **Blackwell (SM100) / Hopper (SM90)** only. This target is **A800 / `sm_80`
  (Ampere)**, and the techniques (tcgen05/TMEM/CLC/NVFP4/2-SM/FA4/tcgen05 warp-spec) do not apply.
  Therefore KernelWiki is **not invoked** for this task; recorded as "not applicable (Ampere)".
- No other external agents/tools/web are used, per isolation rules.

## 10. Open questions / assumptions
- Assumed comparison rule `|a−b| ≤ atol + rtol·|b|` with per-tensor 98% match; if the evaluator uses a
  stricter/looser rule, the generous atols still leave large headroom on the shared path.
- Assumed the reference torch implementation is the speed baseline; our job is a faster *Triton* path,
  primarily via fusion/launch reduction rather than beating cuBLAS on isolated GEMMs.
- Assumed `torch.backends.cuda.matmul.allow_tf32` state does not materially change the reference's f32
  weight-grad GEMMs relative to our Triton path — irrelevant for router (≈0) and absorbed by tolerance
  for shared weight-grads (bf16 output).
- `router_logits` input is unused by the reference `run`; we ignore it too.
- To be verified on first evaluation: installed Triton/torch versions and available `tl.dot`
  precision/`input_precision` API surface on this Ampere build (probed indirectly via a passing c001).
