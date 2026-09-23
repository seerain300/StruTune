# Draft — L1/020 Vision Patch Merger (Spatial Shuffle + LayerNorm + 2-layer GELU MLP)

Target device: **NVIDIA A800 (sm_80, Ampere)**. Framework: Triton primary compute,
PyTorch only for metadata/launch plumbing. No Torch/CPU/NumPy/CUDA-ext computational fallback.

## 1. Operation semantics (from `task/definition.json` reference)

Signature:
```
run(hidden[num_patches,1536] bf16,
    grid_thw[num_grids,3] int64,
    ln_weight[1536] bf16, ln_bias[1536] bf16,
    fc1_weight[6144,6144] bf16, fc1_bias[6144] bf16,
    fc2_weight[3584,6144] bf16, fc2_bias[3584] bf16,
    eps: float32) -> output[num_merged,3584] bf16
```
Constants: `hidden_size C = 1536`, `merge_size = 2`, `hidden_size_expanded E = 6144 = 4*C`,
`out_hidden_size O = 3584`, `num_merged = num_patches / 4`.

Reference pipeline (semantically):

1. **LayerNorm (pre-shuffle), computed in fp32, per input patch over the C=1536 axis**
   - `x = hidden.float()`
   - `mean = x.mean(-1, keepdim)`, `var = x.var(-1, keepdim, unbiased=False)` (÷N, N=1536)
   - `xhat = (x - mean) / sqrt(var + eps)`
   - `y = xhat * ln_weight.float() + ln_bias.float()`  → cast to **bf16** → `hidden_norm[num_patches,1536]`
   - Note: generator fills `ln_weight=ones`, `ln_bias=zeros` (identity affine), but we implement
     the **general** affine using the passed tensors (do not hardcode).

2. **Spatial shuffle (2×2 patch merge), per grid `(t,h,w)`** (h,w even by construction):
   - `view(t, h/2, 2, w/2, 2, C)` → `permute(0,1,3,2,4,5)` → `reshape(t*(h/2)*(w/2), 4*C=E)`.
   - Grids concatenated in order → `hidden_shuffled[num_merged, 6144]` bf16.
   - Decoding: for a grid with row-major patch index `p = ((tt*h + hh)*w + ww)`,
     with `hh = hm*2 + a`, `ww = wm*2 + b`:
     - output row (within grid) = `(tt*(h/2) + hm)*(w/2) + wm`
     - within-row **block** `β = a*2 + b ∈ {0,1,2,3}` occupies columns `[β*1536 : (β+1)*1536]`.
     - i.e. the 6144-vector = `concat( patch(a=0,b=0), patch(a=0,b=1), patch(a=1,b=0), patch(a=1,b=1) )`.
   - This is a **pure row-gather / permutation** of already-LayerNormed patches; LayerNorm is
     per-patch over C so it commutes with the shuffle (LN then gather == gather then LN).

3. **FC1**: `linear(A, fc1_weight, fc1_bias)` = `A @ fc1_weight^T + fc1_bias`,
   `[num_merged,6144] @ [6144,6144]^T → [num_merged,6144]`. cuBLAS returns **bf16**.

4. **GELU exact (erf)**: `F.gelu(..., approximate='none')` on the bf16 FC1 output → bf16.

5. **FC2**: `linear(gelu, fc2_weight, fc2_bias)` = `[num_merged,6144] @ [3584,6144]^T + bias
   → [num_merged,3584]` bf16 = **output**.

Key structural fact: `fc1_weight` columns are indexed by the merged 6144 axis, so
`FC1[m] = Σ_{β=0..3} ( LN(patch_β) @ fc1_weight[:, β*1536:(β+1)*1536]^T ) + bias`. The permutation
is over **rows/patches**, not columns, so it cannot be folded into the static weight — an explicit
gather is required.

## 2. Feedback workloads and derived quantities

| uuid# | num_patches | num_merged (M) | num_grids | atol | rtol |
|------:|------------:|---------------:|----------:|-----:|-----:|
| 1 | 2048 | 512 | 2 | 0.0022 | 0.05 |
| 2 | 6400 | 1600 | 1 | 0.0020 | 0.05 |
| 3 | 1024 | 256 | 4 | 0.0014 | 0.05 |
| 4 | 512  | 128 | 2 | 0.0021 | 0.05 |
| 5 | 4096 | 1024 | 4 | 0.0026 | 0.05 |

M ∈ {512, 1600, 256, 128, 1024} — all divisible by 128 except **256/128 boundaries fine**;
128 and 256 are the small, memory-bound cases. All M divisible by 64; 1600 divisible by 64 not 128
(1600 = 25*64 = 12.5*128) → a BLOCK_M of 128 leaves a half tile, 64 tiles evenly. Choose GEMM
block/masking that handles arbitrary M (mask the M tail) rather than assuming divisibility.

Compute / traffic estimates (per call):
- FC1 flops = `2*M*E*E = M*75.5e6`; FC2 flops = `2*M*E*O = M*44.0e6`.
  - M=1600: FC1 ≈ 120.8 GFLOP, FC2 ≈ 70.5 GFLOP, total ≈ 191 GFLOP.
  - M=128: total ≈ 15.3 GFLOP.
- Weight bytes (read at least once): fc1 = 6144*6144*2 = 75.5 MB, fc2 = 3584*6144*2 = 44.0 MB;
  combined ≈ 119.5 MB.
- Roofline (A800 ~312 TFLOPS bf16, HBM ~1.9 TB/s): weights dominate for **M ≲ ~200**
  (M=128 ≈ 63 µs weight-read vs ≈ 49 µs compute → memory-bound); M=1600 is compute-bound
  (~0.6 ms ideal). The two small workloads (M=128, 256) are where the reference's **host-side
  overhead** (Python per-grid loop with `.item()` syncs + list-append + `torch.cat`) is most
  visible and where fusion wins the most.

## 3. Where the speedup comes from (baseline inefficiencies)

The reference `run` is timed as the baseline. Its overheads over an optimal GPU pipeline:
1. **Host-side grid loop with `grid_thw[i,j].item()`** → up to `3*num_grids` device syncs per call,
   plus Python-level slicing, `permute/reshape` (materializing copies), list build, and `torch.cat`
   (an extra full read+write of the shuffled A matrix).
2. **fp32 LayerNorm materialization**: `hidden.float()` allocates a full fp32 copy of
   `[num_patches,1536]` and multiple intermediate tensors.
3. **Separate kernels** for LN, cast, cat, two GEMMs, bias adds, and GELU (each a launch + full
   HBM round trip of the activations).

Our plan removes host syncs (indices computed from `grid_thw` once), fuses LN+shuffle into a single
kernel, fuses bias+GELU into the FC1 epilogue and bias into the FC2 epilogue, and keeps activations
in bf16. The two GEMMs are the irreducible cost; the win is (a) eliminating overhead/extra passes
and (b) matching cuBLAS GEMM throughput as closely as possible with autotuned Triton.

**Risk:** cuBLAS bf16 GEMM on A800 is strong; for the compute-bound M=1600 case a Triton GEMM may
land at ~0.8–1.0× of cuBLAS. Overall geomean depends on winning big on small M (overhead-dominated)
while staying near-parity on large M. This is the central tradeoff to validate empirically.

## 4. Triton design space

### 4.1 LayerNorm + shuffle kernel (K1)
Produces `A_shuffled[num_merged, 6144]` bf16 directly (fused LN + gather), avoiding a separate cast
and `torch.cat`.
- **Layout choice A (one program per output patch)** — preferred: grid = `num_merged`. Each program
  loops β=0..3, gathers source patch row `src(m,β)`, loads its 1536 bf16 values, casts fp32, computes
  mean/var over 1536, applies `xhat*w+b`, stores 1536 bf16 contiguously at columns `[β*1536:(β+1)*1536]`.
  Writes are fully contiguous (coalesced 6144-wide row); 4 gathered strided loads. Per-program working
  set ~1536 fp32 = 6 KB (loop β, don't hold all 4 at once).
- **Layout choice B (one program per input patch, scatter)**: grid = `num_patches`; each normalizes one
  patch and scatters 1536 to its computed `(out_row, β)` slot. Same traffic; store is contiguous 1536.
  Slightly simpler indexing but requires input→dest map. Either is fine; pick A for contiguous 6144
  stores and simpler reduction.
- Reduction over C=1536: `1536 = 3*512`. Use `BLOCK_C` covering 1536 with masking, or a fixed 1536
  `tl.arange` (power-of-two pad to 2048 with mask). Two-pass (mean, then var) or one-pass
  Welford/`sum`+`sum(x^2)`; a single load into registers + `sum`/`sum(x*x)` is simplest and exact in fp32.
- eps added inside sqrt: `rstd = 1/sqrt(var + eps)`.

### 4.2 Index construction (avoid host sync)
Two options; both are "plumbing":
- **A (single small transfer)**: `grid_thw.to('cpu')` once (4×3 int64) → compute prefix sums and a
  gather-index int32 tensor `src_idx[num_merged, 4]` (or a base+arithmetic form) on host, upload once.
  One tiny sync — vastly cheaper than the reference's per-grid `.item()` loop. Simple, robust for
  `num_grids ∈ {1,2,4}`.
- **B (fully on-GPU)**: a trivial index kernel reads `grid_thw` on device, does a sequential scan over
  `num_grids (≤4)`, writes `src_idx`. Zero host sync. More code; keep as a later refinement if the
  single transfer shows up in timing.
- `num_merged` and the GEMM launch dims need **no** sync: `num_merged = hidden.shape[0] // 4` is known
  from tensor shape. Only the permutation mapping depends on `grid_thw` values.
- Correctness note: implement fully general `(t,h,w)` with h,w even; Σ over grids of `t*(h/2)*(w/2)`
  = num_patches/4 = num_merged by construction. Do not assume a single grid or square grids.

### 4.3 FC1 GEMM (K2): `A[M,6144] @ fc1_weight[6144,6144]^T + bias`, GELU epilogue
- Standard tiled bf16 GEMM, **fp32 accumulator** (matches cuBLAS accumulation).
- Weight is row-major `[N=out, K=in]`; accessing `W[n, k]` gives K contiguous along the K reduction —
  natural for `tl.dot(a[BM,BK], w[BK,BN])` after loading `W` tile as `[BK,BN]` via transpose indexing.
- Epilogue: `acc += fc1_bias[n]`; **match reference rounding**: reference rounds FC1 to bf16 *before*
  GELU (cuBLAS bf16 output), so cast `acc` (with bias) to bf16, then apply GELU, then keep bf16 for FC2
  input. (Applying GELU on fp32 then rounding differs by < bf16 ulp; will test both, prefer the
  reference-matching order to be safe against the tight atol.)
- **GELU exact**: `0.5*x*(1 + erf(x * 0.70710678118654752440))` using `tl.math.erf` (or libdevice
  erf). **Do not** use tanh approximation unless testing shows it stays within atol — erf is the
  reference and safer given atol as low as 0.0014.
- Autotune space: `BLOCK_M ∈ {64,128}`, `BLOCK_N ∈ {64,128,256}`, `BLOCK_K ∈ {32,64}`,
  `num_warps ∈ {4,8}`, `num_stages ∈ {3,4,5}`, `GROUP_M ∈ {4,8}` (L2 grouping / swizzle).
- **Occupancy for small M**: M=128 with BLOCK_M=128 → 1 M-tile; N=6144/BLOCK_N tiles. With BLOCK_N=128
  → 48 programs on 108 SMs (under-utilized). Mitigations: smaller BLOCK_M/BLOCK_N to raise tile count,
  or **split-K** (2–4 way) to fill SMs (needs fp32 atomic add or a second reduction pass). Consider a
  separate small-M config selected by M.

### 4.4 FC2 GEMM (K3): `gelu[M,6144] @ fc2_weight[3584,6144]^T + bias`
- Same GEMM template, `K=6144`, `N=3584`, bias epilogue only, output bf16.
- `3584 = 28*128 = 14*256` → divisible by 128 and 256; N tiling clean.
- Same autotune space; N smaller so fewer N-tiles → split-K more relevant for small M.

### 4.5 Fusion boundaries
- Cannot fuse FC1 and FC2 into one kernel (GELU nonlinearity + different K/N reductions). Keep 3
  kernels: K1 (LN+shuffle), K2 (FC1+bias+GELU), K3 (FC2+bias).
- Optional later: skip materializing `A_shuffled` and gather normalized rows inside the FC1 prologue.
  Blocked by LN needing the whole 1536 row to compute mean/var (a BLOCK_K < 1536 tile lacks the full
  row), and 1536 not being a power of two. Simpler and nearly-free to materialize A in K1 first
  (extra ≈ M*6144*2 bytes write+read; ≈ 19.6 MB at M=1600 vs 191 GFLOP — negligible). Prefer
  materialize-then-GEMM.

## 5. Numerical correctness & risks

1. **LayerNorm precision**: accumulate mean/var in fp32, `unbiased=False` (÷1536), affine in fp32,
   cast output to bf16 — exactly mirrors the reference. Risk: using bf16 accumulation would blow atol.
2. **GEMM accumulation**: fp32 accumulator required to match cuBLAS. bf16 inputs.
3. **GELU form**: exact erf. tanh-approx GELU differs by up to ~3e-4 in activation; after the FC2 sum
   over 6144 terms it could approach the atol (0.0014–0.0026). Use erf; keep tanh only as a tunable to
   test if it ever helps speed without failing tolerance.
4. **Intermediate rounding order (FC1→GELU)**: reference rounds to bf16 before GELU. Replicate to avoid
   a systematic bias near the tight atol.
5. **Output dtype**: bf16 store with round-to-nearest (Triton default).
6. **Value magnitudes**: LN output ~unit variance; FC1/FC2 outputs O(1) (weights scaled by
   1/sqrt(6144), bias N(0,1)). atol ~0.002 on O(1) values ≈ 0.2% absolute; bf16 has ~0.4% relative
   precision but fp32-accumulated GEMM keeps error well inside the combined `atol + rtol*|ref|`
   (rtol 0.05 dominates for |ref|≳0.04). Expect comfortable pass with fp32 accum + erf.
7. **Determinism**: no split-K atomics in the correctness-first candidate (atomic fp32 add reorders
   sums → small nondeterminism); if split-K is adopted for speed, re-verify tolerance.
8. **Edge/general handling**: arbitrary `num_grids ∈ {1,2,4}`, arbitrary even (h,w), t≥1; M-tail masking
   in GEMM (M not always a multiple of BLOCK_M, e.g. 1600 vs 128).

## 6. Validation strategy

Constraints: I cannot run CUDA, a profiler, or any harness directly; the **only** correctness+timing
signal is `./scripts/evaluate_candidate.sh feedback cNNN` over the fixed 5 workloads (one candidate =
all 5). Budget: 100 candidate evals; token soft/hard 1.0M/1.2M.

Plan:
- **c001 — correctness-first, conservative**: K1 LN+shuffle (per-output-patch, fp32 LN), K2 FC1 GEMM
  (moderate config e.g. BM128/BN128/BK64, warps8, stages3, fp32 accum, bias, erf GELU with
  bf16-before-GELU rounding), K3 FC2 GEMM (bias). Host-side index build (option A). Evaluate; confirm
  all 5 pass and record per-workload speedup + geomean as the anchor.
- **Iterate one dimension per candidate** (each a new immutable ID):
  - GEMM autotune configs (block sizes, stages, warps, GROUP_M/swizzle).
  - Small-M path (split-K or smaller blocks) for M=128/256; verify tolerance still holds after any
    atomic-based split-K.
  - K1 layout / vectorization tweaks; on-GPU index kernel (option B) if the single transfer shows cost.
  - Epilogue rounding-order A/B test; erf vs tanh (only if within tolerance).
- **Guardrails**: never regress a passing candidate's correctness for speed; a Triton failure is
  invalid (no Torch fallback). Track cumulative eval count; stop when geomean converges, then write
  `SEARCH_COMPLETE`. Never run `final` without operator approval.
- Record for every candidate in `candidates.jsonl`: parent, source hash, hypothesis, validation,
  per-workload result, geomean, decision, cumulative eval count, skill usage.

## 7. Skill usage

- `KernelWiki`: **not applicable** — Blackwell/Hopper only; target is A800/sm_80 (Ampere).
- `ncu-report-skill`: **not applicable / not permitted** — B200/sm_100 and requires running a profiler,
  which the isolation rules forbid.
- No other external knowledge sources are permitted; optimization relies on standard Ampere Triton
  GEMM practice and the evaluator feedback loop.

## 8. Open questions to resolve empirically via the evaluator

1. Does Triton FC1/FC2 reach cuBLAS parity at M=1600 (compute-bound)? If not, how large is the deficit
   vs the overhead win on small M in the geomean?
2. Is the single `grid_thw` host transfer negligible, or worth the on-GPU index kernel?
3. Best small-M strategy (split-K vs small blocks) and whether split-K stays within atol.
4. Does the bf16-before-GELU rounding order matter for tolerance vs a pure-fp32 epilogue?
