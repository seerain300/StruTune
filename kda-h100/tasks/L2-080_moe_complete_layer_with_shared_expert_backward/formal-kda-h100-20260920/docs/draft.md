# Draft — L2/080 MoE Complete Layer with Shared Expert (Backward)

Target: NVIDIA H100 (`sm_90`). Submission: `solution/solution.py::run(...)`. Primary
compute in Triton; PyTorch only for tensor metadata / launch plumbing. No Torch/CPU/NumPy/
CUDA-extension computational fallback. Ranking metric: geometric-mean speedup over the
reference, with every selected workload required to pass correctness.

Skills consulted this turn: `KernelWiki` (pages `kernel-fused-moe`, `kernel-gated-dual-gemm`,
`technique-kernel-fusion`, `technique-epilogue-fusion`, `technique-pipeline-stages`,
`technique-warp-specialization`, `technique-tile-scheduling`, `lang-triton`/`vs-triton-3.6-blackwell-tcgen05`).
`ncu-report-skill` will be used later for profiling (never concurrently with an evaluation).

---

## 1. Operation analysis

### 1.1 Fixed dimensions

| Symbol | Meaning | Value |
|---|---|---|
| `B` | `batch_seq_len` (variable axis) | 192 … 8192 |
| `H` | `hidden_size` | 4096 |
| `I` | `moe_intermediate_size` | 1408 |
| `E` | `n_routed_experts` | 128 |
| `K` | `num_experts_per_tok` | 8 |

Feedback workloads (16, all evaluated together = 1 candidate evaluation), sorted by `B`:
`192, 211, 293, 384, 512, 997, 1321, 1571, 1879, 2053, 3072, 3089, 3719, 4093, 6144, 8192`.
Per-workload tolerance: `rtol = 0.05`, `required_match_ratio = 0.98`, `atol` grows with `B`
(0.11 at B=192 → 0.39 at B=8192), consistent with bf16 accumulation error scaling with the
contraction length.

### 1.2 Inputs / outputs

Inputs (dtypes as in `definition.json`): `grad_output[B,H] bf16`, `hidden_states[B,H] bf16`,
`router_weight[E,H] bf16`, `e_score_correction_bias[E] f32`, `router_logits[B,E] f32`,
`scores[B,E] f32`, `topk_indices[B,K] int64`, `topk_weights[B,K] f32`, `score_mask[B,E] f32`
(all ones for n_group=1), `shared_expert_gate_weight[I,H] bf16`, `shared_expert_up_weight[I,H] bf16`,
`shared_expert_down_weight[H,I] bf16`, `shared_gate_output[B,I] bf16`, `shared_up_output[B,I] bf16`,
`shared_activated[B,I] bf16`.

Outputs (required, in order):
1. `grad_hidden_states[B,H] bf16`
2. `grad_router_weight[E,H] f32`
3. `grad_shared_expert_gate_weight[I,H] bf16`
4. `grad_shared_expert_up_weight[I,H] bf16`
5. `grad_shared_expert_down_weight[H,I] bf16`

Note routed-expert weight grads are intentionally omitted by the reference.

### 1.3 Reference dataflow, decomposed

**Shared-expert branch** (`grad_shared_output = grad_output.clone()`):

- G1 `grad_shared_activated = grad_output @ down_weight`  → `[B,H]·[H,I] = [B,I]` (bf16 in, bf16 out, fp32 accum via cuBLAS).
- G2 `grad_shared_expert_down_weight = grad_outputᵀ(f32) @ shared_activated(f32)` → `[H,B]·[B,I] = [H,I]`, f32 accum → bf16. **Weight-grad, contraction = B.**
- SwiGLU backward (pointwise on `[B,I]`):
  - `grad_shared_gate_silu = grad_shared_activated · shared_up_output`
  - `grad_shared_up_output = grad_shared_activated · silu(shared_gate_output)`
  - `sig = sigmoid(gate_f32)`; `grad_shared_gate_output = grad_shared_gate_silu_f32 · (sig·(1 + gate_f32·(1−sig)))` → bf16
    (derivative of SiLU: `d/dx silu = sig·(1 + x·(1−sig))`).
- G3 `grad_hidden_from_shared_up = grad_shared_up_output @ up_weight` → `[B,I]·[I,H] = [B,H]` (bf16).
- G4 `grad_hidden_from_shared_gate = grad_shared_gate_output @ gate_weight` → `[B,H]` (bf16).
- G5 `grad_shared_expert_up_weight = grad_shared_up_outputᵀ(f32) @ hidden(f32)` → `[I,B]·[B,H] = [I,H]`, f32→bf16. **Weight-grad.**
- G6 `grad_shared_expert_gate_weight = grad_shared_gate_outputᵀ(f32) @ hidden(f32)` → `[I,H]`, f32→bf16. **Weight-grad.**
- `grad_hidden_states = grad_hidden_from_shared_up + grad_hidden_from_shared_gate` (+ router term below).

**Routing branch** (straight-through approximation):

- `grad_norm_sq = Σ_H grad_output_f32²` → `[B,1]`; `grad_topk_weights = grad_norm_sq/K` broadcast to `[B,K]` (uniform across K).
- With `norm_topk_prob=True`, `routed_scaling_factor=1.0`:
  `denom = Σ_K topk_weights + 1e-20`; `sum_grad = (grad_topk_weights·topk_weights).sum_K / denom`;
  `grad_topk_weights_before_norm = (grad_topk_weights − sum_grad)/denom`.
- `grad_scores_for_choice = scatter_add(zeros[B,E], topk_indices, grad_before_norm) · score_mask`.
- `grad_router_logits = grad_scores · scores · (1−scores)`  `[B,E]`.
- R1 `grad_hidden_from_router = grad_router_logits(bf16) @ router_weight` → `[B,E]·[E,H] = [B,H]`.
- R2 `grad_router_weight = grad_router_logitsᵀ @ hidden(f32)` → `[E,B]·[B,H] = [E,H]` f32.
- `grad_hidden_states += grad_hidden_from_router`.

### 1.4 Cost model

Shared branch = 6 GEMMs, each ≈ `2·B·I·H` FLOPs → `12·B·I·H`. `I·H = 5.77M`.
Router branch = 2 GEMMs, each ≈ `2·B·E·H`, `E·H = 0.52M` → ~11× smaller than one shared GEMM.

- **Large B (e.g. 8192):** shared branch ≈ `1.1e15` FLOP → ~1.1 ms ideal at ~990 bf16 TFLOPS ⇒ **compute-bound**; GEMM efficiency dominates.
- **Small B (192–512):** weights (34 MB) + activations dominate; ~10–20 µs of compute but many kernel launches ⇒ **launch/memory-bound**; fusion and launch reduction dominate.

Weight-grad GEMMs (G2/G5/G6) contract over `B` (small K, large `M×N` output `H×I` or `I×H`),
i.e. tall/thin reduction GEMMs — memory-heavy at small B.

---

## 2. Key numerical insight — the routing branch is negligible

`grad_topk_weights` is **uniform across the K columns** (all equal to `c[b] = grad_norm_sq[b]/K`).
Feeding a uniform vector `c·1` through the softmax-style normalization gradient:

```
grad_before_norm = (c − c·S/(S+ε)) / (S+ε) = c·ε/(S+ε)²,   S = Σ_K topk_weights, ε = 1e-20
```

The inputs are generated with normalized `topk_weights` (`Σ_K topk_weights ≈ 1`), so `S ≈ 1` and
`grad_before_norm ≈ c·1e-20`. With `c ≈ ‖grad_output_row‖²/K ≈ 4096/8 ≈ 512`, that is ≈ `5e-18`.
Propagating: `grad_router_logits ≈ 5e-18·scores·(1−scores) ~ 1e-18`; then

- `grad_router_weight[e,h] = Σ_b grad_router_logits·hidden ~ (B·K/E)·1e-18 ≲ 5e-16` for all B.
- router contribution to `grad_hidden ~ E·1e-18 ~ 1e-16`.

Both are **~15 orders of magnitude below** every workload's `atol` (≥ 0.11). The `isclose`
criterion `|a−b| ≤ atol + rtol·|b|` passes on the `atol` term alone even if we output exact
zeros for `grad_router_weight` and omit the router term from `grad_hidden`.

**Hypothesis H0 (headline):** the entire routing branch — the full-`H` `grad_norm_sq`
reduction, the scatter, the sigmoid-derivative elementwise, and GEMMs R1/R2 — can be replaced
by `grad_router_weight ← zeros[E,H] (f32)` and no router term in `grad_hidden`, saving 2 GEMMs +
a `B·H` reduction + scatter, with zero correctness impact within tolerance.

Robustness: the cancellation depends only on `Σ_K topk_weights ≈ 1` (true by construction of
`get_inputs`) and the `1e-20` epsilon; the margin is ~10¹⁵, so it is not fragile to bf16 noise.
**Still, H0 must be validated by a real feedback evaluation, not assumed** — a faithful (but
fused) routing path will be kept available as a fallback candidate if any workload regresses.

---

## 3. Constraints

- **Triton-only compute.** Every GEMM and pointwise op must be a Triton kernel; PyTorch limited
  to shape/stride/dtype queries, empty/zeros allocation, and launch grid math. No `torch.matmul`/
  `F.*`/cuBLAS compute in the hot path, no CPU/NumPy path, no CUDA-extension fallback. A failed
  Triton kernel is an invalid candidate — never silently substitute Torch.
- **Immutable candidates.** `c001, c002, …` one source version each; append one JSON record per
  evaluated candidate to `candidates.jsonl`; never rewrite. Any source/config/launch change → new ID.
- **Evaluation discipline.** Only `./scripts/evaluate_candidate.sh feedback <id>`; full 16-workload
  set = 1 evaluation. Budget 100 evals. Token soft/normal/hard limits 9M/10M/11M. `final` only with
  operator approval. Never run profiling and evaluation concurrently (foreign-process on the locked
  GPU → return code 3, wasted eval). Profiling only via `./scripts/ncu_profile.sh` per `ncu-report-skill`.
- **Output contract.** Exactly 5 tensors, in order, with the specified shapes/dtypes; `grad_router_weight`
  is **f32**, the other four are **bf16**. Must return a tuple.
- **Variable B, non-power-of-two.** B includes 211, 293, 997, 1321, 1571, 1879, 2053, 3089, 3719,
  4093 — kernels must mask ragged M/K tiles (no assumption `B % BLOCK == 0`).

---

## 4. Numerical risks & mitigations

1. **Accumulation dtype.** Reference weight-grads (G2/G5/G6, R2) cast to f32 and accumulate in f32;
   grad_hidden GEMMs (G1/G3/G4) keep bf16 operands with cuBLAS fp32 accumulation. Mitigation: use
   `tl.dot(..., out_dtype=tl.float32)` (fp32 accumulator) everywhere, cast the *result* to the
   reference output dtype (bf16 for 1/3/4/5, f32 for 2). Matches cuBLAS behavior.
2. **bf16 rounding order in the SwiGLU-back epilogue.** Reference rounds `grad_shared_activated`
   to bf16 (matmul output) *before* the elementwise chain, computes `grad_shared_gate_output` fully
   in f32 then rounds to bf16. Mitigation: in the fused epilogue, round `grad_shared_activated`
   to bf16 first, do `silu` and the sigmoid-derivative in f32, then round outputs to bf16 — matches
   the reference rounding lattice. (Even without exact replication, atol ≥ 0.11 leaves ample room.)
3. **`silu(shared_gate_output)` recompute.** Reference recomputes `F.silu(gate)` (bf16). We recompute
   `x·sigmoid(x)` in f32 from the saved bf16 `shared_gate_output`, then cast — negligible drift.
4. **Ragged-tile masking.** Non-multiple B needs correct `mask=` on loads/stores and zeroed OOB
   contributions in reductions; otherwise NaNs/garbage leak into the 98%-match check.
5. **Numerically-dropped routing (H0).** Only risk is if a workload's true `grad_router_weight`
   magnitude approached `atol`; analysis shows ~10¹⁵ margin. Guarded by validation.
6. **`grad_router_weight` all-zeros & rtol.** For elements where the reference is exactly 0 (or ~1e-16),
   `rtol·|ref| ≈ 0`, but `|0 − ref| ≤ atol` holds ⇒ passes. Safe.
7. **Empty/degenerate handling.** All B ≥ 192, so no zero-row case; still guard grid math for the
   smallest tiles so no kernel launches with 0 programs.

---

## 5. Triton design space

Reference issues ~20 separate CUDA kernels (clone, 8 matmuls, several elementwise, casts, reduction,
scatter). The win is **fewer launches + fused pointwise/epilogue + dropped routing**, while keeping
GEMM throughput competitive with cuBLAS at large B.

### 5.1 Kernel decomposition options

Starting from the 6 shared GEMMs (routing dropped per H0), candidate fusions:

- **F-A: Fused down-backprop + SwiGLU-backward epilogue.** One kernel computes G1
  (`grad_output @ down_weight → [B,I]`) and, in the epilogue with `shared_up_output`,
  `shared_gate_output` tiles resident, directly emits `grad_shared_up_output[B,I]` and
  `grad_shared_gate_output[B,I]` (bf16). Removes the separate `grad_shared_activated`
  materialization and 3 elementwise kernels. (Ref: `technique-epilogue-fusion`,
  `technique-kernel-fusion`.)
- **F-B: Dual grad_hidden GEMM with shared output accumulation.** One kernel computes
  `grad_hidden = grad_shared_up_output @ up_weight + grad_shared_gate_output @ gate_weight`
  by iterating the shared K=I once and issuing two `tl.dot` accumulations into one `[B,H]` tile
  → single write of `grad_hidden` (bf16). Merges G3+G4+the add. (Analogous to the gate-up dual-GEMM
  pattern in `kernel-gated-dual-gemm`, here on the backward/output-accumulation side.)
- **F-C: Dual weight-grad GEMM sharing `hidden`.** G5 and G6 both contract over B against the same
  `hidden[B,H]`; a single kernel over output tiles `(I×H)` reads the `hidden` K-tile once and issues
  two `tl.dot`s (`grad_upᵀ`, `grad_gateᵀ`) → writes `grad_shared_expert_up_weight` and
  `grad_shared_expert_gate_weight`. Halves `hidden` traffic.
- **F-D: Weight-grad down GEMM (G2)** stands alone: `grad_outputᵀ @ shared_activated → [H,I]` f32→bf16.

Net: **~20 reference kernels → ~4 Triton kernels** (F-A, F-B, F-C, F-D) plus trivial `zeros` for
`grad_router_weight`. A more conservative first candidate may keep GEMMs unfused (6–8 kernels) to
de-risk correctness, then fuse incrementally.

### 5.2 GEMM implementation notes (`lang-triton`, SM90)

- Standard tiled `tl.dot` with fp32 accumulator; autotune `BLOCK_M/N/K`, `num_warps`, `num_stages`,
  and grouped-M raster (`GROUP_M`) for L2 locality (`technique-tile-scheduling`,
  `technique-pipeline-stages`). Triton 3.6+ on SM90 still lowers to `wgmma`; tcgen05/TMEM are SM100-only,
  so no Blackwell-specific paths apply here — this is Hopper `wgmma` + async-copy pipelining territory.
- **Weight-grad GEMMs (G2/G5/G6)** contract over `B` (K small at small B), output large `H×I`/`I×H`.
  These are memory-bound at small B → prioritize wide vectorized loads and enough stages to hide HBM;
  at large B they become compute-bound. Autotune per regime; a shape-keyed config table (small vs
  large B) is likely needed because a single tile config won't be optimal across B=192…8192.
- **grad_hidden GEMMs (G1/G3/G4)** are `[B,·]@[·,H]`, well-shaped; M=B ragged → mask.
- Transpose handling for `Aᵀ@X` done via strides in the pointer arithmetic (no physical transpose):
  contract along the leading (row) dimension of A.
- Consider fp32-accumulate + bf16-store throughout; avoid needless f32 materialization of inputs
  (load bf16, cast in-register) to cut memory traffic vs the reference's explicit `.to(f32)` copies.

### 5.3 Progression plan (high level, for plan.md later)

1. **c001 — faithful, unfused Triton port**, routing computed faithfully (fused pointwise but
   GEMMs separate). Purpose: establish correctness + a Triton baseline speedup number, de-risk
   masking and dtype handling. (No H0 yet.)
2. **c002 — apply H0**: drop routing branch (zeros `grad_router_weight`, no router grad_hidden term).
   Confirms the numerical claim on real hardware and quantifies the saving.
3. **c003+ — fusions** F-A, F-B, F-C, F-D introduced one at a time (each a new candidate ID),
   measuring each; then autotune / shape-keyed configs; then any epilogue/pipeline tuning guided by
   `ncu-report-skill`.

Each step is one immutable candidate; stop when geomean converges or budget/token limits hit.

---

## 6. Validation strategy

- **Primary oracle:** `./scripts/evaluate_candidate.sh feedback <id>` runs the full 16-workload set
  and checks correctness (`rtol=0.05`, per-workload `atol`, 98% match) *and* timing. This is the only
  correctness/perf signal; use it deliberately (100-eval budget).
- **Pre-eval self-checks (no GPU compute of our own reference):** static review that (a) all 5 outputs
  have exact required shapes/dtypes and tuple order, (b) every ragged B path is masked, (c) accumulators
  are fp32 and only results are downcast, (d) no Torch compute leaked into the hot path.
- **Sequencing vs H0:** c001 faithful first so that if c002 (H0) ever fails a workload we have a passing
  reference candidate to fall back to; H0's ~10¹⁵ tolerance margin makes failure very unlikely, but it is
  verified empirically before building fusions on top of it.
- **Profiling:** only via `./scripts/ncu_profile.sh` (per `ncu-report-skill`), and **never** while an
  evaluation is running (foreign process on the locked GPU → discarded measurement, wasted eval).
  Use profiling to pick between fusion/config variants and to confirm launch-count / memory-traffic
  reductions at small B and GEMM efficiency at large B.
- **Record-keeping:** append one complete JSON object per evaluated candidate to `candidates.jsonl`
  (parent, source hash, hypothesis, validation, per-workload result, geomean, decision, cumulative
  eval count, skill usage); never rewrite earlier records. Create `SEARCH_COMPLETE` when converged.
  Never run `final` without operator approval.

---

## 7. Open questions to resolve during search

1. Real speedup of c001 (faithful Triton port) vs reference — is a straightforward Triton port already
   ahead, or do we need fusion just to reach parity at large B?
2. Magnitude of the H0 saving across the B range (expected largest at small B).
3. Best tile/stage configs per B regime; whether one autotuned space covers all 16 shapes or a
   shape-keyed table is needed.
4. Whether F-A/F-B/F-C fusions actually help at large B (they may add register pressure / hurt GEMM
   occupancy) — validate each independently before stacking.
