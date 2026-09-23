# Draft Analysis — `gemm_n4096_k4096`

Run ID: `formal-kda-20260916--flashinfer--gemm_n4096_k4096`
Benchmark: `flashinfer` · Op type: `gemm` · Target HW: **NVIDIA A800 (`sm_80`, Ampere / GA100)**

This document is the pre-implementation analysis. No code is written yet. It defines the
operation, the hard constraints, the numerical risks, the Triton design space, and the
validation plan. `docs/plan.md` and any `solution/` code come in later turns.

---

## 1. Operation definition

From `task/definition.json`:

- Reference:
  ```python
  import torch
  def run(A, B):
      C = torch.matmul(A, B.T)
      return C
  ```
- Semantics: `C = A @ B.T`, i.e. `C[m, n] = sum_k A[m, k] * B[n, k]`.
- Shapes / dtypes:
  - `A`: `[M, K]`, `float16`
  - `B`: `[N, K]`, `float16`  (note: `B` is stored as `[N, K]`, NOT `[K, N]`)
  - `C`: `[M, N]`, `float16`
- Axes: `N = 4096` (const), `K = 4096` (const), `M` = variable (`var`).
- Provenance: Llama 3.1 8B `attn.o_proj`. This is the attention output projection:
  hidden→hidden matmul with the projection weight `B` and per-token activations `A`.
  `M` therefore corresponds to the number of tokens in a batch/decode step, which is
  why `M` is small and variable.

### Feedback workloads (fixed, 5 total)

| workload uuid (prefix) | M   |
|------------------------|-----|
| 67d4c8f3               | 64  |
| 59ca23f5               | 48  |
| e7c939ae               | 240 |
| 29ebd771               | 128 |
| f439da26               | 4   |

All share `N = K = 4096`. `M ∈ {4, 48, 64, 128, 240}` — all **skinny / tall-thin GEMM**
(a.k.a. "GEMV-adjacent" for M=4). The final (operator-only) evaluation uses 43 workloads,
so the kernel must generalize across a range of small-to-moderate `M`, not overfit these 5.

### Memory layout details (critical for the Triton kernel)

- `A[M,K]` is row-major: element `A[m,k]` at offset `m*K + k`. The contraction axis `K` is
  the **contiguous/fast** axis of A.
- `B[N,K]` is row-major: element `B[n,k]` at offset `n*K + k`. The contraction axis `K` is
  also the **contiguous/fast** axis of B.
- This is the "NT" GEMM layout (`A` normal, `B` transposed relative to a K-major operand).
  It is favorable: for both operands the reduction dimension is contiguous, so K-strips load
  coalesced. For `tl.dot(a, b)` we need `a=[BM,BK]` and `b=[BK,BN]`; the natural coalesced
  load of B is a `[BN,BK]` tile (n outer with stride K, k inner with stride 1), which is then
  transposed in-register (`tl.trans`) or fed to a dot that consumes the transposed operand.

---

## 2. Performance characterization (roofline)

Byte and FLOP counts per workload (fp16 = 2 bytes):

- `B` bytes = `N*K*2 = 4096*4096*2 = 33.55 MB` — **dominant, read at least once.**
- `A` bytes = `M*K*2` — 32 KB (M=4) … 1.97 MB (M=240). Small.
- `C` bytes = `M*N*2` — 32 KB (M=4) … 1.97 MB (M=240). Small.
- FLOPs = `2*M*N*K`.

Arithmetic intensity `AI = 2*M*N*K / total_bytes` (bytes ≈ B + A + C, single pass):

| M   | FLOPs    | ~bytes   | AI (FLOP/B) | regime            |
|-----|----------|----------|-------------|-------------------|
| 4   | 0.134 G  | 33.6 MB  | ~4.0        | strongly mem-bound|
| 48  | 1.61 G   | 34.3 MB  | ~47         | mem-bound         |
| 64  | 2.15 G   | 34.6 MB  | ~62         | mem-bound         |
| 128 | 4.29 G   | 35.6 MB  | ~121        | mem-bound         |
| 240 | 8.05 G   | 37.5 MB  | ~215        | near ridge        |

A800/A100 reference numbers: HBM2e bandwidth ≈ **~2.0 TB/s** (~2039 GB/s), fp16 tensor-core
peak ≈ **312 TFLOP/s**, L2 = 40 MB, 108 SMs, up to 164 KB shared mem/SM. Roofline ridge
point ≈ `312e12 / 2.0e12 ≈ 156 FLOP/B`.

**Conclusion:** For all feedback M values except possibly M=240 the kernel is **HBM-bandwidth
bound on reading B once**. The theoretical lower bound is roughly `33.55 MB / 2.0 TB/s ≈ 16.8 µs`
plus small A/C traffic. Tensor-core compute is NOT the bottleneck; the entire game is:

1. Read all of B exactly once with maximally coalesced, high-throughput loads.
2. Launch enough concurrent thread blocks to **saturate HBM bandwidth** despite tiny M.
3. Avoid redundant B traffic (don't re-read B across unnecessary M-tiles).

### Why cuBLAS/`torch.matmul` can be beaten here

For small `M`, a standard GEMM tiling produces very few output tiles:
`num_tiles = ceil(M/BLOCK_M) * ceil(N/BLOCK_N)`. With one M-tile (M≤BLOCK_M) and
`BLOCK_N=128`, that is only `1 * 32 = 32` blocks — far below the ~108–216 blocks needed to
keep A800's 108 SMs busy and to hide HBM latency. cuBLAS heuristics for skinny GEMM often
leave SMs idle and HBM under-fed. The main lever to beat it is **Split-K** (partition the
K-reduction across blocks) to multiply the block count and saturate bandwidth. This is the
core hypothesis of the search.

### Grid-size / occupancy table (no split-K, 1 M-tile)

| BLOCK_N | N-tiles | blocks (1 M-tile) | vs 108 SMs |
|---------|---------|-------------------|------------|
| 256     | 16      | 16                | badly under|
| 128     | 32      | 32                | under      |
| 64      | 64      | 64                | under      |
| 32      | 128     | 128               | ok, but small tiles hurt B load width |

To reach ≥108–216 blocks for M=4/48/64 we need Split-K (e.g. splitK=2–8) and/or moderate
BLOCK_N. The split factor should scale with how few M-tiles exist.

---

## 3. Constraints

From `CLAUDE.md`, `TASK.md`, `README.md`:

- **Triton is the primary implementation.** PyTorch allowed only for tensor metadata / launch
  plumbing / output allocation. No Torch compute fallback, no CPU/NumPy fallback, no
  CUDA-extension fallback, no alternate-implementation fallback. A failed Triton kernel is
  invalid — I must fix Triton, not substitute `torch.matmul`.
- Submission entry point: `solution/solution.py` exposing `run(...)`. It must accept `A, B`
  and return `C = A @ B.T` (match reference signature/behavior).
- Hardware: A800 `sm_80` — **no** Hopper/Blackwell features (no TMA, no `wgmma`, no tcgen05,
  no fp8/nvfp4 MMA). Ampere `mma.sync` (m16n8k16 fp16) via Triton `tl.dot`, `cp.async`
  pipelining (Triton `num_stages`), 164 KB smem. The KernelWiki skill is explicitly
  scoped to Blackwell/Hopper and does NOT cover sm_80, so it will not be used here.
- Evaluation: only via `./scripts/evaluate_candidate.sh feedback <cNNN>`. Five fixed
  workloads = one candidate evaluation. Do not run CUDA/profiler/nvidia-smi/evaluator
  directly. Cannot execute kernels locally for iteration — must reason carefully and spend
  evaluations deliberately.
- Immutable candidates: `c001, c002, …` sequential; any meaningful source/config/launch
  change ⇒ new ID; never reuse an ID for changed source; never rewrite prior
  `candidates.jsonl` records.
- Budget: **100 candidate evaluations**; token soft limit 1,000,000 / hard 1,200,000.
- Ranking: **geometric mean speedup** over reference; every selected workload must pass
  correctness first.
- Final 43-workload run is operator-only; never run `final` without explicit approval.

---

## 4. Numerical analysis and risks

### Reference numerics
`torch.matmul(A, B.T)` with fp16 inputs on GPU uses tensor cores with **fp32 accumulation**
internally, then rounds the result to fp16. So the "ground truth" is: exact products in
fp16→fp32, fp32 accumulation over `K=4096`, final round-to-fp16.

### Tolerance
`task/definition.json` declares no explicit `rtol`/`atol`; the trusted evaluator applies its
default GEMM tolerance. For fp16 GEMM these are typically loose (order `1e-2` relative, plus
an absolute floor). I will design to be **as close to reference as possible** so the exact
tolerance does not matter:

- Accumulate in **fp32** in the Triton kernel (`tl.dot(..., acc, out_dtype=tl.float32)` /
  `allow_tf32=False`). This matches the reference's fp32 accumulation.
- **Do NOT enable TF32** for the fp16 inputs. `allow_tf32=True` would truncate fp16 mantissas
  to TF32 (10-bit) — but inputs are already fp16 (10-bit mantissa), so TF32 vs fp16 MMA is
  essentially equivalent in precision; still, fp16 `mma.sync` with fp32 accum is the exact
  hardware path cuBLAS uses. Keep operands fp16, accumulate fp32.
- Final cast to fp16 at store, matching the reference's output dtype.

### Split-K reduction risk (the main numerical concern)
Split-K partitions the `K=4096` reduction into `S` chunks computed by different blocks, then
sums the `S` partials. Options and their risk:

1. **fp32 partials + fp32 reduction, then cast to fp16.** Each partial is a fp32 sum over
   `K/S` terms; summing `S` fp32 partials in fp32 is numerically *very close* to a single
   fp32 sum over K (only reassociation differences at fp32 ULP level). This is the **safest**
   and well within any fp16 tolerance. Preferred.
2. **`tl.atomic_add` into an fp32 scratch buffer**, then a cheap cast kernel to fp16.
   Deterministic in value (fp32), but atomic ordering makes the *bit pattern*
   non-deterministic across runs (fp add non-associative). Value differences are ~fp32 ULP —
   negligible vs fp16 tolerance, but run-to-run non-bit-identical. Acceptable if the evaluator
   checks tolerance (not bit-exactness), which is standard for GEMM.
3. **`tl.atomic_add` directly into fp16 C.** Accumulates in fp16 → catastrophic precision loss
   and heavy contention. **Rejected.**
4. **Two-pass deterministic**: write partials to `[S, M, N]` fp32 buffer, then a reduction
   kernel sums along S. Fully deterministic and precise. Extra memory: `S*M*N*4` bytes
   (e.g. S=8, M=240 → ~31 MB) — fine. Slightly more traffic on the (small) C side.

Plan will start with non-split-K (exact-match, zero risk), then add split-K using option (1)
or (4) for determinism, or (2) if the evaluator tolerates non-bit-exact.

### Masking / padding risk
- MMA requires operand tiles with `BLOCK_M ≥ 16`. For `M=4` (and any `M < BLOCK_M`), the
  M-tile is padded; loads of A must be **masked** (`mask = offs_m < M`, `other=0.0`) and
  stores of C masked so we never read/write out of bounds. Padded rows contribute zeros to
  the dot and are simply not stored. Wasted compute on padded rows is irrelevant (mem-bound).
- `N=4096` and `K=4096` are both multiples of 128/64/32/16, so N/K tiles divide evenly for
  typical block sizes — **no N/K boundary masking needed** for power-of-two blocks that divide
  4096 (simplifies and speeds up the inner loop). If split-K uses a chunk size that does not
  divide K, the last K-chunk needs masking; choosing S as a power of two dividing 4096 avoids
  this.
- fp16 accumulation of zeros is exact, so padding does not perturb results.

### Determinism
The reference is deterministic. Non-split-K Triton dot is deterministic. Split-K via atomics
is not bit-deterministic. If the evaluator requires reproducibility across its repeated timing
runs at the tolerance level (not bit level), atomics are fine; if it demands bit-identical
outputs, use the two-pass deterministic reduction. Draft assumes tolerance-based checking
(standard for GEMM KDA tasks) but keeps the deterministic path as a fallback.

---

## 5. Triton design space

### 5.1 Kernel structure candidates
1. **Baseline autotuned NT GEMM** (no split-K). Standard `pid → (pid_m, pid_n)` mapping with
   L2-friendly grouping along M (`GROUP_SIZE_M`). fp32 accumulator, fp16 store. Establishes a
   correct, exact-match reference point and a speed baseline. Likely already competitive for
   M=128/240; weak for M=4/48/64 due to too few blocks.
2. **Split-K GEMM.** Add a `pid_k` dimension: grid `(num_pid_m*num_pid_n, split_k)` (or 3D).
   Each block reduces a `K/S` slice; partials combined via fp32 atomic-add scratch (+cast) or
   two-pass reduction. This is the primary expected win for small M. `S` chosen so total
   blocks ≳ 2×108. Power-of-two `S` dividing 4096.
3. **M-adaptive split-K via autotune `key=['M']`.** One immutable kernel whose autotuner picks
   different `(BLOCK_M, BLOCK_N, BLOCK_K, S, num_warps, num_stages)` per `M`. This handles the
   5 different M values (and the 43-workload generalization) without changing source. Trade-off:
   first-call autotune cost; must ensure the evaluator's timing excludes warmup or that
   autotune space is small. Alternative: a **heuristic** that computes `S` from `M` and `N`
   in Python at launch (no autotune search) — cheaper, predictable, and easy to reason about.
4. **Dedicated skinny/GEMV-style kernel for very small M (e.g. M≤8).** Since M=4 is essentially
   4 independent GEMVs, a bandwidth-streaming kernel (each block owns a strip of N rows,
   streams K with wide vectorized fp16 loads, fp32 FMA accumulate, no tensor cores) can hit
   near-peak HBM bandwidth without MMA padding waste. Optional; only if tl.dot path
   underperforms on M=4. Must still be Triton.
5. **Persistent kernel.** Launch exactly `#SMs * k` blocks that loop over output tiles. Reduces
   launch overhead and improves scheduling, but for a single-pass mem-bound op the benefit is
   marginal vs. plain split-K. Lower priority.

### 5.2 Tiling / config parameters to explore
- `BLOCK_M ∈ {16, 32, 64}` (≥16 for MMA; 16 ideal for M=4..48 to minimize padding waste;
  larger only when M is large).
- `BLOCK_N ∈ {32, 64, 128, 256}` — larger N-tiles = wider coalesced B loads and better MMA
  efficiency but fewer blocks; balance against needing many blocks. For mem-bound, mid values
  (64–128) with split-K tend to win.
- `BLOCK_K ∈ {32, 64, 128}` — larger BLOCK_K = better B load width and fewer loop iters;
  bounded by smem and register pressure.
- `split_k / S ∈ {1, 2, 4, 8, 16}` — dominant lever for small M. Target ≥ ~216 blocks.
- `num_warps ∈ {2, 4, 8}`, `num_stages ∈ {2, 3, 4, 5}` — `cp.async` software pipelining depth
  to overlap HBM loads with MMA; deeper stages help hide HBM latency (key for mem-bound) but
  cost smem/registers. On A800, 3–4 stages typical.
- `GROUP_SIZE_M` — L2 locality reordering; low impact here since B is read once (limited reuse),
  but cheap to include.

### 5.3 Load/coalescing strategy
- Load A tile `[BLOCK_M, BLOCK_K]` coalesced along K (A's fast axis), mask rows `< M`.
- Load B tile as `[BLOCK_N, BLOCK_K]` coalesced along K (B's fast axis) — this is the natural,
  fully-coalesced layout since K is contiguous in `B[N,K]`. Feed to `tl.dot` with the operand
  transposed (`tl.dot(a, b_nt)` where the b operand is treated as `[BK, BN]` via `tl.trans` or
  Triton's built-in operand transpose). Verify the generated PTX uses `ldmatrix`/`cp.async`
  efficiently; the NT layout is the tensor-core-friendly case on Ampere.
- Since K=4096 and N=4096 are multiples of the block sizes, the inner K loop can drop masking
  for the non-split-K case (faster).

### 5.4 Expected best configuration (hypothesis to test)
Small M (4–64): `BLOCK_M=16`, `BLOCK_N=64–128`, `BLOCK_K=64`, `split_k=4–8`, `num_stages=3–4`,
`num_warps=4`, fp32 accum, deterministic-or-atomic fp32 reduction. This should push block
count to 200+ and approach the ~17–25 µs HBM-bound floor, beating cuBLAS's under-occupied
skinny GEMM. Moderate M (128–240): smaller/zero split-K, `BLOCK_M=32–64`, may already be near
optimal with plain tiling.

---

## 6. Validation strategy

Because I cannot run CUDA/profilers locally and each feedback run costs one of 100 evaluations,
validation is a mix of static reasoning and disciplined use of the evaluator.

1. **Static correctness review before every evaluation:** re-derive index math
   (`A[m,k]=m*K+k`, `B[n,k]=n*K+k`, `C[m,n]=m*N+n`), masks (`offs_m<M`, plus K-mask only if
   split chunk doesn't divide K), accumulator dtype (fp32), and output cast (fp16). Confirm the
   `run(A,B)` wrapper returns a contiguous fp16 `[M,N]` tensor on the same device.
2. **Correctness is gate #1:** the evaluator checks each workload against `torch.matmul(A,B.T)`
   within tolerance. A candidate is only ranked if all 5 (feedback) workloads pass. I treat any
   correctness failure as a hard stop to debug before chasing speed.
3. **Numerical safety margin:** fp32 accumulation + split-K option (1)/(4) keeps error at fp32
   ULP scale, far inside fp16 tolerance. If a split-K/atomic candidate ever shows a correctness
   miss, fall back to the deterministic two-pass reduction.
4. **Candidate discipline:** start with `c001` = plain autotuned NT GEMM (exact-match, low risk)
   to (a) confirm the harness/wrapper works and (b) get a real per-M speedup baseline. Then
   introduce split-K (`c002+`), then M-adaptive split/heuristics, changing **one structural
   idea per candidate** so each evaluation yields a clean signal. Record for every candidate:
   parent, source hash, hypothesis, validation, per-workload result, geomean, decision,
   cumulative eval count, skill usage — one JSON object appended to `candidates.jsonl`.
5. **Generalization guard:** since the final run is 43 workloads, avoid overfitting the 5
   feedback M values. Prefer heuristics/autotune keyed on `M` that behave sensibly for arbitrary
   small-to-moderate M, and sanity-check the block-count logic across a range of M, not just
   {4,48,64,128,240}.
6. **Convergence:** stop when geomean improvement flattens across successive structural ideas or
   when approaching the HBM-bound floor (little headroom left), then write `SEARCH_COMPLETE`
   with the reason. Never trigger `final` without operator approval.

---

## 7. Open questions to resolve during implementation

- Exact evaluator tolerance and whether it requires bit-reproducibility (drives atomic vs.
  two-pass split-K choice). Will infer from first split-K candidate's pass/fail behavior.
- Installed Triton/PyTorch versions and available `tl.dot` transpose ergonomics on this box
  (affects how B is fed to the dot). Will confirm empirically via the first candidate.
- Whether autotune warmup is counted in the evaluator's timing (drives autotune-vs-heuristic
  choice for the split factor).
- Real achieved HBM bandwidth fraction vs. the ~2 TB/s peak, to know how much headroom remains
  after the first split-K candidate (informs when to stop).
