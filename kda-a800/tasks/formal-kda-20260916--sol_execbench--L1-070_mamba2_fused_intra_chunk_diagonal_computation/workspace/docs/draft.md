# Draft — L1/070 Mamba2 Fused Intra-Chunk Diagonal Computation

Task: optimize the official SOL-ExecBench task
`L1/070_mamba2_fused_intra_chunk_diagonal_computation` on **NVIDIA A800 (`sm_80`, Ampere)**.
Submission is `solution/solution.py` exposing `run(hidden_states, A_cumsum, B, C) -> Y_diag`.
Primary implementation must be **Triton** (PyTorch only for metadata/launch plumbing; no
Torch/CPU/NumPy/CUDA-extension computational fallback).

---

## 1. What the operation actually computes

### 1.1 Shapes and constants (from `task/definition.json`)

Constants (fixed across all workloads):

| name        | value | role                      |
|-------------|-------|---------------------------|
| CHUNK_SIZE  | 128   | intra-chunk sequence len  |
| NUM_HEADS   | 32    | attention heads           |
| HEAD_DIM    | 128   | per-head feature dim (V/out) |
| N_GROUPS    | 8     | groups for B/C            |
| STATE_SIZE  | 128   | SSM state dim (contraction dim of G) |

Derived: `heads_per_group = NUM_HEADS // N_GROUPS = 4`, i.e. head `h` belongs to group
`g = h // 4`.

Variable axes: `batch_size (B)` and `num_chunks (C)`.

Inputs (all `bfloat16`):
- `hidden_states` (X): `[B, C, 128(s), 32(h), 128(d)]`
- `A_cumsum`      (a): `[B, 32(h), C, 128(s)]`
- `B` (Bmat):          `[B, C, 128(s), 8(g), 128(n)]`
- `C` (Cmat):          `[B, C, 128(s), 8(g), 128(n)]`

Output (`bfloat16`):
- `Y_diag` (Y): `[B, C, 128(s), 32(h), 128(d)]`  (same layout as `hidden_states`)

### 1.2 Per-(batch b, chunk c, head h) math

The reference expands B/C over heads via `repeat_interleave(4, dim=groups)`, so group
`g = h // 4`. For each independent `(b, c, h)` triple define, over local indices
`i, j ∈ [0,128)` (query row `i`, key/value row `j`) and state/feature index `n, d`:

- `Cmat = C[b, c, :, g, :]`  → `[128(i), 128(n)]`
- `Bmat = B[b, c, :, g, :]`  → `[128(j), 128(n)]`
- `a    = A_cumsum[b, h, c, :]` → `[128]`
- `X    = hidden_states[b, c, :, h, :]` → `[128(j), 128(d)]`

**Step 1 — decay mask `L` (segment_sum):**
The reference *re-cumsums* the input `A_cumsum` inside the chunk. Working it through the
`tril(diagonal=-1)` masked cumsum + `tril(diagonal=0)` `-inf` fill:

```
segsum[i, j] = sum_{k=j+1..i} a[k]           for i >= j     (diagonal i==j → empty sum → 0)
segsum[i, j] = -inf                          for i <  j
L[i, j]      = exp(segsum[i, j])             (i<j → 0, i==j → 1)
```

Equivalently, with prefix sum `cA[i] = sum_{k=0..i} a[k]`:
```
L[i, j] = exp(cA[i] - cA[j]) for i >= j, else 0.
```
`L` is **lower-triangular (incl. diagonal)** in `(i, j)`, and is **per-head** (because `a`
depends on `h`). NOTE: the input is *named* `A_cumsum` but the reference applies a further
cumulative sum to it — we must reproduce that extra cumsum, not treat the input as already
being `cA`.

**Step 2 — scores `G` (state contraction):**
```
G[i, j] = sum_n Cmat[i, n] * Bmat[j, n]   →   G = Cmat @ Bmat^T   ([128,128])
```
`G` depends only on `(b, c, g)` — it is **shared by the 4 heads of a group**.

**Step 3 — masked weights `M`:**
```
M[i, j] = G[i, j] * L[i, j]     (lower-triangular, per-head)
```

**Step 4 — output:**
```
Y[i, d] = sum_j M[i, j] * X[j, d]   →   Y = M @ X   ([128,128])
```

### 1.3 One-line characterization

This is **masked "attention" per `(b, c, h)`** with:
`Q≡Cmat`, `K≡Bmat` (contraction over `state_size`), scores `G = Q·Kᵀ`, an
exponential **causal decay mask** `L` (no softmax / no normalization), `V≡X`, output
`Y = (G ∘ L) · X`. Every reduction dimension (state, kv-seq, head_dim) is exactly **128**.
There are `B·C·32` independent `[128×128]` output tiles; each costs two
`128×128×128` matmuls (`G`, then `Y`).

### 1.4 Feedback workload sizes

`(B, C)` ∈ {(2,4), (1,3), (4,4), (4,7), (1,4)} → `B·C·H` programs (per-head layout) ∈
{256, 96, 512, 896, 128}. Largest is `B·C=28` → 896 head-tiles. With 108 SMs on A800 that
is ~2–8 waves; parallelism is adequate even before any grouping.

---

## 2. Constraints

- **Triton-only compute**; PyTorch allowed for shapes/strides/empty-alloc/launch only.
- **No fallback** of any kind (Torch/CPU/NumPy/CUDA-ext/alternate). A failing Triton kernel
  is simply invalid.
- Output must be `bfloat16`, exact shape `[B, C, 128, 32, 128]`, same layout as input `X`.
- **Cannot run** CUDA/profiler/`nvidia-smi`/external evaluator/alternate correctness harness
  directly. The *only* correctness+perf signal is `./scripts/evaluate_candidate.sh feedback cNNN`.
  → Validation must lean heavily on **static analysis and faithful math reproduction**, with
  each candidate confirmed through the official feedback evaluator.
- Immutable, sequential candidates `c001, c002, …`; one kernel version over all 5 feedback
  workloads = one evaluation. Budget: 100 evals; token soft/hard 1.0M / 1.2M.
- Metric: **geometric-mean speedup**, and **every** selected workload must pass correctness.
- Skills: `KernelWiki` (Blackwell/Hopper) and `ncu-report-skill` (B200/sm_100 profiling) are
  **out of scope** for an Ampere `sm_80` target and are additionally blocked by the
  "no profiler" rule; I will not invoke them. All other external agents/tools are forbidden.

---

## 3. Numerical risks and how to match the reference

Tolerance (all 5 feedback workloads): `max_atol = 1e-5`, `max_rtol = 0.05` (assumed combined,
`|out-ref| <= atol + rtol·|ref|`). `rtol = 5%` is loose; `atol = 1e-5` only bites near zero.

Reference numeric path: everything is promoted to **fp32** for the arithmetic (A, B, C, X all
cast to fp32), `L = exp(...)` in fp32, and the final `Y` is cast to bf16 on return.

1. **Exponential dynamic range (main risk).** Inputs are `random` (≈ N(0,1) bf16). `a` is not
   guaranteed negative, so `cA` behaves like a random walk (std ~`sqrt(128)≈11`), and
   `cA[i]-cA[j]` for `i>=j` can reach ±(30–60). `exp(60)≈1e26` is finite in fp32 (max ≈3.4e38)
   but **huge**; extreme tails could overflow to `inf`. Mitigations / rules:
   - Compute `exp` of the **difference** `segsum[i,j] = cA[i]-cA[j]` directly (as the reference
     does), **never** as `exp(cA[i])·exp(-cA[j])` — the factored form overflows/underflows even
     when the difference is bounded (`cA[i]` alone can be ±40).
   - **Do NOT** use the FlashAttention max-subtraction / online-rescale trick. There is **no
     softmax normalization here**, so subtracting a row max changes the result (it would not
     cancel). We must accumulate the true weighted sum. (We may still *tile* the kv loop, just
     without renormalization.)
   - Because both our kernel and the reference form the same `exp(difference)` in fp32, `inf`
     outputs (if any) should coincide; but `inf`/`nan` are a correctness landmine — watch the
     first feedback run for NaN/Inf failures and, if needed, keep the exp argument computation
     bit-faithful to the reference ordering.

2. **Matmul-1 (`G = Cmat @ Bmatᵀ`).** Reference: bf16→fp32 (lossless) then fp32 MAC. Triton
   `tl.dot(bf16, bf16, out=fp32)` uses bf16 tensor cores with fp32 accumulate; bf16×bf16 exact
   products + fp32 accumulate ≈ the reference to well within `rtol`. Acceptable.

3. **Matmul-2 (`Y = M @ X`).** `M = G∘L` is fp32 with large dynamic range; `X` is bf16 (the
   reference casts it to fp32 losslessly). Two options:
   - **bf16 tensor-core** (`M→bf16`, `X→bf16`, fp32 accum): fast; per-element relative error of
     `M` ≈ 2⁻⁸ ≈ 0.4% — comfortably inside `rtol=0.05`. Since each output element is dominated
     by the largest-`L` terms (same as the reference fp32 sum), small terms vanish in both.
   - **fp32/TF32** (`M` fp32, `X`→fp32, `tl.dot` TF32 ~10-bit): closer to the reference but
     slower. Keep as a correctness fallback if bf16 proves marginal on any workload.
   Default plan: bf16 tensor core; escalate to TF32 only if a workload fails.

4. **Near-zero atol.** Rows with few unmasked terms (e.g. row `i=0` has only `j=0`,
   `Y[0,d]=G[0,0]·X[0,d]`) are single products computed almost identically in both paths →
   safe. The bf16 output quantization the reference itself applies (~value·2⁻⁸) dominates and
   is inside tolerance.

5. **Group sharing correctness.** If a candidate computes `G` once per group and reuses it for
   4 heads, the reuse is exact (`G` truly is head-independent); only `L` and `X` differ per
   head. Must index `A_cumsum[b,h,c,:]` and `X[...,h,:]` per head, but `Bmat/Cmat` per group.

---

## 4. Triton design space

Because `chunk = state = head_dim = 128`, every tile is a clean multiple of 128; no ragged
remainders. The op is **memory-heavy**: `X` and `Y` are the big tensors
(`B·C·32·128·128·2 B` ≈ 59 MB each at `B·C=28`), while `B`/`C` are 4× smaller (8 groups) and
`A_cumsum` is tiny. So minimizing redundant `X`/`Y` traffic and keeping matmuls on tensor
cores are the levers; compute is small (~15 GFLOP at the largest workload).

### 4.1 Baseline candidate (c001) — robust, per-`(b,c,h)`
- **Grid:** `B·C·32` programs; program `pid` → `(b, c, h)`, `g = h//4`.
- Load `Cmat[128,128]`, `Bmat[128,128]` (group `g`); `G = tl.dot(Cmat, Bmatᵀ)` → fp32 `[128,128]`.
- Load `a[128]`; `cA = tl.cumsum(a_fp32)`; `segsum = cA[:,None]-cA[None,:]`;
  `L = tl.where(i>=j, exp(segsum), 0)`.
- `M = G * L` (fp32).
- Load `X[128,128]`; `Y = tl.dot(M.to(bf16), X)` fp32 accum; store `Y→Y_diag` as bf16.
- **Config:** `num_warps=8` to spread the `128×128` fp32 accumulators (16384 elts) to ~64
  regs/thread and avoid heavy spills; `num_stages=2–3`. Everything is 128-wide so no masking
  on loads/stores (all divisible by 128).
- Rationale: simplest faithful mapping; establishes a correct, passing reference before any
  perf tuning.

### 4.2 Optimization axes (later candidates, one change per ID)
1. **KV tiling / causal skip (flash-style, no renorm).** `BLOCK_M=64` query rows, loop kv in
   blocks of 64 with causal pruning (skip `j`-blocks strictly above the diagonal; apply the
   `i>=j` mask only on the diagonal block). Lowers register pressure ([64,128] accumulator →
   `num_warps=4`), improves occupancy, and skips ~half the score work. Compare vs full-128.
2. **Group fusion of `G`.** One program per `(b, c, g)` computes `Cmat@Bmatᵀ` **once**, then
   loops the 4 heads of the group: `M_h = G∘L_h`, `Y_h = M_h@X_h`. Cuts matmul-1 count 4× and
   `B`/`C` loads 4×. Grid shrinks to `B·C·8` (fewer waves — watch occupancy on small `B·C`).
3. **`num_warps` / `num_stages` / block-size autotune** across {4,8} warps, {2,3,4} stages,
   `BLOCK_M∈{64,128}`, `BLOCK_N∈{64,128}`. Pick per-workload-robust config (must be one
   immutable kernel across all 5).
4. **Load layout / block pointers.** `X`/`Y` tiles: row stride `= 32*128 = 4096`, contiguous
   in `d` (good coalescing on the 128-wide inner dim). `B`/`C` tiles: row stride
   `= 8*128 = 1024`, contiguous in `n`. `A_cumsum` contiguous in `s` (stride 1). Use
   `make_block_ptr` / precomputed strides; keep the contiguous 128-dim innermost for
   coalesced 128-element vector loads.
5. **Matmul-2 precision toggle** (bf16 tensor core ↔ fp32/TF32) as a correctness/perf knob.
6. **Epilogue fusion:** compute `L`, `M` in-register and write only `Y` (avoid materializing
   any `[128,128]` intermediate to global memory — the reference's `G`, `L`, `M` blowups are
   exactly the "memory-intensive" cost this kernel must avoid).

### 4.3 Things to avoid
- Materializing `L`/`G`/`M` (`[B,C,128,128,32]`) in HBM — that is the naive PyTorch cost we
  are beating; keep them on-chip.
- Softmax-style max subtraction (changes the un-normalized result).
- Factored `exp(cA[i])·exp(-cA[j])` (overflow/underflow).
- Any non-Triton compute path.

---

## 5. Validation strategy

Because direct CUDA/torch/profiler execution and any alternate correctness harness are
forbidden, validation is **analysis-first, evaluator-confirmed**:

1. **Static/index audit (pre-eval).** Re-derive every stride and index against §1.2 before
   each candidate: group mapping `g=h//4`; `A_cumsum[b,h,c,s]` layout (note `h` and `c` axes
   are swapped vs `X`); `L` lower-triangular incl. diagonal; `G = C@Bᵀ` (i indexes C, j
   indexes B); `Y = M@X` reduces over `j`; output layout identical to `X`.
2. **Numeric-faithfulness checklist.** fp32 accumulation everywhere; exp-of-difference (not
   factored); no max-subtraction; bf16 cast only at the final store (and at matmul-2 operands).
3. **First candidate = correctness anchor.** c001 is the simplest faithful mapping; its job is
   to *pass all 5 feedback workloads* (watch specifically for NaN/Inf from the exp range and
   for the `A_cumsum` axis-order pitfall). Only after a clean pass do we chase speed.
4. **One-variable-per-candidate.** Each subsequent candidate changes exactly one axis from §4.2
   so a correctness regression or speedup is attributable. Record per-workload pass/fail +
   time, geomean, parent, source hash, hypothesis, decision, cumulative eval count, skill usage
   in `candidates.jsonl` (append-only).
5. **Convergence / stop.** Stop when speedup plateaus across ~2–3 consecutive candidates or at
   the budget; then write `SEARCH_COMPLETE`. `final` only on explicit operator approval.

### Open questions to resolve empirically via feedback
- Does the largest `exp` range on the `random` inputs stay finite (no Inf failures)?
- Is bf16 matmul-2 within tolerance on all 5 workloads, or is TF32/fp32 needed?
- Full-128 tile (`num_warps=8`) vs `BLOCK_M=64` flash tiling — which wins on A800 given the
  memory-bound profile?
- Does per-group `G` fusion actually help, or does the reduced grid hurt occupancy on small
  `B·C` (e.g. (1,3) → only 24 group-programs)?
