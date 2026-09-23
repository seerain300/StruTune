# Draft — L2/012 MoE Expert Batched Execution with Capacity Factor

Run ID: `formal-kda-20260916--sol_execbench--L2-012_moe_expert_batched_execution_with_capacity_factor`
Target: NVIDIA A800 (`sm_80`, Ampere). Primary implementation: **Triton**. PyTorch allowed only for
tensor metadata / launch plumbing. No Torch/CPU/NumPy/CUDA-extension computational fallback.

This document is analysis only. No plan, no solution code is written yet (per workflow steps 2–4).

---

## 1. Operation semantics (from `task/definition.json`)

Capacity-based, drop-if-overflow MoE feed-forward with SwiGLU experts.

Constants (all fixed by the task axes):

| symbol | meaning | value |
|---|---|---|
| `H` | hidden_size | 6144 |
| `I` | moe_intermediate_size | 2560 |
| `E` | num_experts | 160 |
| `K` | num_experts_per_tok | 8 |
| `cf` | capacity factor | 1.25 (fixed) |
| `dtype` | all tensors | bfloat16 (indices int64) |

Inputs:
- `hidden_states` `[N, H]` bf16
- `selected_experts` `[N, K]` int64 — each row is `K` **unique** expert ids (reference builds them via `randperm`, so no duplicates within a token)
- `routing_weights` `[N, K]` bf16 — softmax over the K logits, sums to 1 per token
- `expert_gate_weights` `[E, H, I]` bf16
- `expert_up_weights`   `[E, H, I]` bf16
- `expert_down_weights` `[E, I, H]` bf16

Output: `output` `[N, H]` bf16.

### Reference algorithm (exact semantics we MUST reproduce)

1. `capacity = max(int((N*K/E) * 1.25), 1)`  (Python `int()` truncates toward zero).
2. Flatten the `N*K` (token, expert) assignments. `flat_token_ids = arange(N).repeat_interleave(K)`, so the
   flat index of assignment `(token t, slot k)` is `t*K + k`.
3. **Stable** sort of the flat assignments by expert id. Stability means within one expert the assignments
   keep increasing `(t*K + k)` order — i.e. lower token id first, and for the same token lower slot first.
4. Within-expert position `within_pos = global_sorted_idx - group_start[expert]`.
5. **Capacity admission:** keep assignments with `within_pos < capacity`. The rest are **dropped** (their
   contribution to that token's output is simply omitted; there is **no renormalization** of routing weights).
6. Scatter admitted token rows into a padded batch `expert_inputs[E, capacity, H]` (unfilled rows are zero).
7. Three batched GEMMs per expert with SwiGLU:
   - `gate = X_e @ Wg_e`  (`[cap,H]·[H,I]`)
   - `up   = X_e @ Wu_e`  (`[cap,H]·[H,I]`)
   - `act  = silu(gate) * up`  — **note:** `gate` and `up` are bf16 GEMM outputs; silu and the multiply are
     performed in **bf16** (torch rounds each GEMM result to bf16 before the elementwise ops).
   - `y    = act @ Wd_e`  (`[cap,I]·[I,H]`)
8. Gather admitted rows back, weight by `routing_weight`, and `index_add_` into `result[N,H]` (bf16
   accumulation, since `result` is bf16). Dropped tokens contribute nothing; a token that had all 8 experts
   admitted sums 8 weighted contributions.

**Correctness-critical invariants** we cannot deviate from:
- Identical **admitted set** (same drop decisions). This depends on the stable-sort order and the exact
  `capacity` integer. Easiest safe route: reuse the reference's cheap metadata ops in torch (sort / bincount /
  cumsum / mask) verbatim — these are "tensor metadata", explicitly allowed.
- **No renormalization** of routing weights after dropping.
- SwiGLU is `silu(gate) * up` with `silu(x)=x*sigmoid(x)`; gate is the first projection, up the second.
- Weight layouts: gate/up are `[H, I]` (contract over dim-1 = `H`); down is `[I, H]` (contract over `I`).

---

## 2. Per-workload characterization (feedback set)

`capacity = int(N*K/E * 1.25)`, `N*K` = total assignments, padded rows processed by the reference = `E*capacity`.
Arithmetic intensity of the three-GEMM core, in FLOP per weight-byte, works out to exactly `≈ capacity`
(derivation in §3), so the capacity value alone predicts the compute/memory regime.

| uuid (short) | N | capacity | mean load `N*K/E` | assigns `N*K` | padded rows `E*cap` | pad/useful | core FLOP | AI≈cap | regime | atol |
|---|---|---|---|---|---|---|---|---|---|---|
| b983758c | 1536 | 96  | 76.8  | 12288 | 15360 | 1.25 | ~1.45 TF | 96  | **memory** | 1.9 |
| d36f3c8b | 1568 | 98  | 78.4  | 12544 | 15680 | 1.25 | ~1.48 TF | 98  | **memory** | 1.2 |
| c2a09e88 | 1344 | 84  | 67.2  | 10752 | 13440 | 1.25 | ~1.27 TF | 84  | **memory** | 1.6 |
| f850f2b7 | 4096 | 256 | 204.8 | 32768 | 40960 | 1.25 | ~3.87 TF | 256 | **compute** | **0.0076** |
| 2fe16676 | 1571 | 98  | 78.55 | 12568 | 15680 | 1.248| ~1.48 TF | 98  | **memory** | 1.7 |

All five share `rtol=0.05`, `required_match_ratio=0.98`.

Observations that steer the whole design:

- **Four of five workloads are memory-bound** on weight traffic (capacity ≈ 84–98 < Ampere ridge point ≈ 150,
  see §3). Only `N=4096` (capacity 256) is compute-bound.
- **All 160 experts are active in every workload** (mean load 67–205 ≫ 0), so no expert can be skipped and the
  full weight tensor must be streamed. **Weight traffic is a hard floor** (§3).
- Padding overhead is a uniform 1.25× because capacity = 1.25× mean load; useful rows ≈ `N*K` minus a small
  number of dropped tokens on over-subscribed experts.
- `N=4096` has an unusually tight **`atol=0.0076`** (≈ one bf16 ULP near 1.0). Because `rtol=0.05` also applies
  and only 98% of elements must pass, large-magnitude outputs are governed by the loose rtol; the tight atol
  only bites for near-zero outputs. Still, this workload is the one most sensitive to accumulation ordering and
  drop-set fidelity — treat it as the numerical canary.

---

## 3. Roofline / cost model (A800 sm_80)

A800 ≈ A100 die: bf16 tensor-core peak ≈ **312 TFLOP/s**, HBM2e bandwidth ≈ **~2.0 TB/s**, 108 SMs, 164 KB
smem/SM. Ridge point ≈ `312e12 / 2.0e12 ≈ 156` FLOP/byte.

**Weight bytes (streamed once, minimum):**
`3 tensors × E·H·I × 2 B = 3 × 160 × 6144 × 2560 × 2 ≈ 15.1 GB`. At ~2.0 TB/s the pure weight-streaming floor
is **≈ 7.5 ms** for *every* workload, independent of N. This is the dominant term and cannot be reduced
(all experts active, bf16 fixed).

**Core FLOP** = `E·cap · 6·H·I` (gate + up + down = 3 matmuls, 2 FLOP/MAC). `6·H·I ≈ 9.44e7` per row.

**Arithmetic intensity** = `E·cap·6HI / (3·H·I·2) = cap` → **AI (FLOP/byte) ≈ capacity**. Hence:
- cap ≤ ~156 ⇒ memory-bound ⇒ wall-clock ≈ weight floor ≈ 7.5 ms (workloads 1,2,3,5).
- cap = 256 (N=4096) ⇒ compute-bound ⇒ core compute ≈ `3.87e12 / 312e12 ≈ 12.4 ms` > 7.5 ms weight floor.

**Where the reference loses time above these floors (the actual optimization budget):**
1. **Intermediate HBM traffic.** The reference materializes and re-reads padded tensors:
   `expert_inputs[E,cap,H]` (scatter-write + read twice for gate & up), `gate_out`, `up_out`, `activated`
   (`[E,cap,I]` each), `expert_outputs[E,cap,H]` (write + gather-read). Rough totals ≈ 1 GB (cap96) to ≈ 3.7 GB
   (cap256) on top of the 15 GB weights — i.e. ~7% (memory cases) to ~20% (compute case) extra traffic that a
   fused kernel can largely eliminate.
2. **Padded-zero compute.** ~20% of the core FLOP is spent on zero-padded rows. Irrelevant when memory-bound;
   worth ~1.1–1.25× on the compute-bound `N=4096`.
3. **Launch / small-op overhead.** sort, bincount, cumsum, arange, compare, scatter, 3× bmm, silu, mul, gather,
   index_add — ~a dozen kernels. On a big GPU with small per-workload data, fixed launch overhead is a
   non-trivial fraction for the memory-bound cases.

**Honest speedup ceiling.** Weight traffic is fixed, and cuBLAS batched GEMM is already near BW-peak, so the
reachable win is bounded by items (1)–(3): roughly **~1.1–1.3× geomean**, with the largest headroom on the
compute-bound `N=4096` (de-padding + fusion) and thin margins on the memory-bound four (intermediate-traffic +
launch-overhead removal only). This realism should temper candidate ambitions and stopping criteria.

---

## 4. Numerical risks

1. **Capacity integer & drop set.** `capacity = int(N*K/E * 1.25)`. Any off-by-one (e.g. computing in fp32 vs
   float, `round` vs `trunc`) changes which tokens are admitted → wrong rows entirely → can blow past the 2%
   mismatch budget. Mitigation: compute capacity exactly as the reference and reuse its stable-sort admission
   metadata in torch.
2. **Stable-sort order.** Admission keeps the first `capacity` per expert in `(t*K+k)` order. A non-stable sort
   would admit a different subset on over-subscribed experts. Mitigation: reuse `sort(stable=True)` metadata.
3. **No routing-weight renormalization.** Dropped assignments are just omitted; remaining weights are NOT
   rescaled. Must replicate.
4. **SwiGLU precision ordering.** Reference rounds `gate` and `up` to bf16 *before* silu/multiply. A fused
   Triton kernel naturally keeps fp32 accumulators and would compute `silu(gate_fp32)*up_fp32` — *more* accurate
   than the reference. Tolerances (rtol 0.05, 98% match) are generous, so higher precision is expected to be
   fine, but this is a deviation to watch, especially on the tight-atol `N=4096`. If a candidate fails the
   canary, an option is to round gate/up to bf16 before the elementwise step to mimic the reference exactly.
5. **Accumulation dtype in the down-projection & aggregation.** Reference's `index_add_` accumulates in bf16
   (result is bf16). A Triton scatter that accumulates in fp32 and casts once at the end is more accurate; again
   generally safe under the tolerances, but a source of small per-element deltas. Atomic-add should target an
   **fp32** result buffer (bf16 atomics are unreliable on sm_80), cast to bf16 at the very end.
6. **tl.dot accumulation.** Must use fp32 accumulators for all three GEMMs (Triton default). bf16-accumulate
   would lose precision on the `K=H=6144` reduction and likely fail the tight-atol canary.
7. **Empty / min-capacity edge cases.** `capacity` floored at 1. Not triggered by the feedback set (min cap 84),
   but the kernel must not divide-by-zero or emit NaN for zero-token blocks; masked loads/stores must yield 0.
8. **bf16 overflow / NaN.** Inputs are unit-scale (randn, Xavier-scaled weights). `H=6144` reductions in fp32
   are safe; no overflow expected. silu is bounded below by ~-0.278, fine.
9. **Determinism vs the evaluator's reference.** The evaluator presumably regenerates inputs from `get_inputs`
   with a fixed seed and compares to the reference `run`. Both our kernel and the reference see identical inputs,
   so only algorithmic/precision deltas matter, not RNG.

---

## 5. Triton design space

All options keep weight traffic at the 15 GB floor (unavoidable). They differ in intermediate traffic, padded
compute, and complexity. Ampere-specific tooling available: `tl.dot` (mma.sync m16n8k16 bf16→fp32), `cp.async`
software pipelining via `num_stages`, masked loads for ragged/padded tiles. Blackwell/Hopper features
(tcgen05/TMEM/TMA/WGMMA/NVFP4) are **out of scope on sm_80** — the KernelWiki corpus is Blackwell/Hopper-first,
so only its *algorithmic* grouped-MoE patterns (token sorting, block-aligned grouping, fused SwiGLU, scatter-add
aggregation — the vLLM/SGLang `fused_moe` family) transfer; none of its hardware-intrinsic techniques do.

### Option A — Padded-batch grouped GEMM, two kernels (closest to reference layout)
Keep the reference's `expert_inputs[E,cap,H]` layout; replace the 3 bmm + silu + mul with:
- Kernel 1: per `(expert, M-block, N-block)` compute gate & up, fuse `silu(gate)*up`, write `h[E,cap,I]`.
- Kernel 2: per `(expert, M-block, N-block)` compute `h@Wd`, scale by routing weight, scatter-add to result.

Pros: simplest; drops `gate_out/up_out/activated` intermediates (fuses silu). Cons: still materializes
`expert_inputs` and computes all `E*cap` padded rows. Modest win, mostly on memory cases via intermediate
removal. Good **first, low-risk candidate** to establish a correct baseline.

### Option B — Sorted-token grouped GEMM with on-the-fly gather (vLLM `fused_moe` style)
Reuse admission metadata to get admitted `(token_id, expert, weight)` already grouped by expert. Build a
block-aligned index list (`moe_align`-style: per expert pad the last tile to `BLOCK_M` with a sentinel row).
- Kernel 1: each program = one `BLOCK_M` tile of one expert. Gather `X` rows via `token_id` (no
  `expert_inputs` materialization), compute gate & up over full `K=H`, fuse SwiGLU, write `h` to a compact
  intermediate `[total_aligned_rows, I]`.
- Kernel 2: load `h` tile, `h@Wd`, multiply by per-row routing weight, `atomic_add` into fp32 result via
  `token_id`; cast to bf16 at the end.

Pros: eliminates `expert_inputs` scatter + reads; compute only ~`sum ceil(count_e/BLOCK_M)*BLOCK_M` rows.
Cons: per-expert counts (67–205) are comparable to `BLOCK_M`, so block rounding reclaims padding only with
small `BLOCK_M` (e.g. 32 → ~cap; 16 → ~15% fewer rows) at some tensor-core-efficiency cost. Atomic-add
contention on the `[N,H]` result. Most upside on the compute-bound `N=4096`. **Primary target design.**

### Option C — Single fused megakernel (gate+up+silu+down in one launch)
Infeasible: the down projection needs the full `I=2560` intermediate per token block; holding `BLOCK_M×I`
fp32 in smem (e.g. 64×2560×4 = 640 KB) exceeds the 164 KB/SM budget. Would require tiny `BLOCK_M`, killing
GEMM efficiency. **Rejected.**

### Option D — Split-K / stream-K on the `K=H=6144` reduction
With `E=160` experts × several N-tiles the grid already exceeds 108 SMs, so occupancy is fine without split-K;
split-K adds a reduction pass and extra traffic. Keep in reserve only if profiling shows tail/idle SMs on the
smallest workload. **Deferred.**

### Cross-cutting knobs (autotune space)
- `BLOCK_M ∈ {16, 32, 64}`, `BLOCK_N ∈ {64, 128, 256}`, `BLOCK_K ∈ {32, 64}`.
- `num_warps ∈ {4, 8}`, `num_stages ∈ {2, 3, 4}` (cp.async pipelining to hide the weight-streaming latency —
  the single most important knob for the memory-bound majority).
- Grid ordering / L2 grouping of tiles by expert to maximize weight-tile L2 reuse across the N-tiles of one
  expert (weights are the reused operand; group programs of the same expert together).
- fp32 accumulators everywhere; optional bf16-rounding of gate/up before SwiGLU as a numeric-match fallback.

**Recommended progression:** A (correct baseline, low risk) → B with a modest `BLOCK_M` (fusion + de-padding)
→ autotune `num_stages`/block shapes → consider bf16-round fallback only if the `N=4096` canary fails.

---

## 6. Validation strategy

Constraints: I may **not** run CUDA, a profiler, `nvidia-smi`, the external evaluator directly, or any alternate
correctness harness. The **only** correctness signal is `./scripts/evaluate_candidate.sh feedback <cid>`, which
runs all five fixed workloads and counts as **one** evaluation against the 100-eval budget. So local "unit
testing" is not available; correctness must be engineered up-front and confirmed via the sanctioned evaluator.

Approach:
1. **Reuse reference metadata verbatim** (capacity, stable sort, within-pos, capacity mask, admitted
   token/expert/weight lists) in torch — this makes the drop set provably identical and removes the biggest
   correctness risk. Only the three GEMMs + SwiGLU + weighted scatter-add move into Triton.
2. **Static self-checks inside `run`** (asserts on shapes/dtypes/contiguity, expert-id bounds) so a malformed
   launch fails loudly rather than silently mismatching.
3. **Reason about tolerances** per §4: fp32 accumulation should meet the loose rtol on large outputs; the tight
   `atol=0.0076` only affects near-zero outputs on `N=4096`, cushioned by the 98% match ratio.
4. **Candidate discipline / budget economy.** Because each eval costs budget and there's no cheap local check:
   - `c001` = the lowest-risk correct design (Option A style, minimal fusion) to lock in correctness and get a
     real baseline speedup number and per-workload pass/fail.
   - Only after `c001` passes all five, iterate toward Option B and autotuning, changing one thing per candidate
     so each result is attributable.
   - Treat `N=4096` (tight atol) as the canary; if a change breaks only it, suspect SwiGLU/accumulation ordering
     and apply the bf16-round fallback rather than reverting the whole design.
5. **Per-candidate record** (in `candidates.jsonl`): parent, source hash, hypothesis, validation, five
   per-workload results (pass + speedup), geomean, decision, cumulative eval count, skill usage.
6. **Stopping.** Given the ~1.1–1.3× ceiling in §3, stop when the geomean improvement across a couple of
   candidates is within noise of the weight-streaming floor, and write `SEARCH_COMPLETE` with the reason.
   Never run `final` without explicit operator approval.

---

## 7. Skill usage & environment notes

- **KernelWiki** (invoked): confirmed Blackwell/Hopper-first scope. Its hardware-intrinsic techniques
  (tcgen05, TMEM, TMA, WGMMA, NVFP4) do **not** apply to A800/sm_80; only its algorithmic grouped-MoE patterns
  (token sort → block-aligned grouping → fused SwiGLU → scatter-add, i.e. the `fused_moe` family) transfer. The
  wiki's query scripts require shell execution, which the current sandbox denies, so I relied on the loaded
  skill description plus first-principles analysis rather than page-level citations.
- **ncu-report-skill:** profiling is disallowed for this task (no direct profiler runs), so it is not used;
  bottleneck reasoning is done analytically via the roofline in §3.
- **Sandbox note:** arbitrary Bash (including a Python calculator and the KernelWiki query scripts) is denied in
  this environment. All numbers in §2–§3 are hand-derived from the fixed constants and stated for the plan phase
  to build against; they involve no execution of the operation.

---

## 8. Summary of decisions carried into the plan

- Move only the 3 GEMMs + SwiGLU + weighted scatter-add into Triton; keep admission metadata in torch to
  guarantee an identical drop set.
- Two-kernel grouped GEMM (gate+up+SwiGLU → down+weighted scatter-add), fp32 accumulators throughout, fp32
  result buffer with a final bf16 cast.
- First candidate: correctness-first (Option A, padded layout, fused SwiGLU). Then evolve to sorted-token
  Option B and autotune `num_stages`/block shapes for the memory-bound majority.
- Numeric canary = `N=4096` (atol 0.0076); bf16-round-before-SwiGLU is the reserved fallback if it fails.
- Expectation: realistic geomean ceiling ~1.1–1.3×; weight streaming (~15 GB, ~7.5 ms floor) dominates and is
  irreducible.
