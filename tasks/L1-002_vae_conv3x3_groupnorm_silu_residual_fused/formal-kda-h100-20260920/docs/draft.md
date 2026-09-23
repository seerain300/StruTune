# Draft — L1/002 VAE Conv3x3 → GroupNorm → SiLU → (×2) → Residual Add

Task: `sol_execbench :: L1/002_vae_conv3x3_groupnorm_silu_residual_fused`
Target: NVIDIA H100 (`sm_90`), FP32. Primary metric: geomean speedup vs the PyTorch
reference, subject to per-workload correctness within the given tolerances.

This document is analysis only. No plan, no code. It defines the operation, the
constraints, numerical risks, the Triton design space, and how each candidate will be
validated.

---

## 1. Operation and signature

Entry point to implement in `solution/solution.py`:

```
run(x, conv1_weight, norm1_weight, norm1_bias,
    conv2_weight, norm2_weight, norm2_bias, eps) -> output
```

Reference semantics (from `task/definition.json`, `num_groups = 32`, `kernel = 3`):

```
residual = x
out = conv2d(x,   conv1_weight, bias=None, stride=1, padding=1)   # 3x3, C:256->256
out = group_norm(out, 32, norm1_weight, norm1_bias, eps)          # per (n, group) over 8ch x H x W
out = silu(out)                                                    # out * sigmoid(out)
out = conv2d(out, conv2_weight, bias=None, stride=1, padding=1)   # 3x3, C:256->256
out = group_norm(out, 32, norm2_weight, norm2_bias, eps)
out = silu(out)
output = out + residual
```

Fixed dtype: **float32** for every tensor and the output. Fixed constants:
`channels = 256`, `num_groups = 32` ⇒ **8 channels per group**, `kernel_size = 3`,
`stride = 1`, `padding = 1` (zero padding), so **spatial dimensions are preserved**
(H_out = H, W_out = W).

Shapes:
- `x`, `output`: `(B, 256, H, W)`, NCHW, contiguous.
- `conv1_weight`, `conv2_weight`: `(256, 256, 3, 3)` — 2.36 MB each (fits L2).
- `norm{1,2}_weight`, `norm{1,2}_bias`: `(256,)` — negligible.
- `eps`: python float scalar (1e-6 in every feedback workload).

GroupNorm detail that matters for the kernel: in NCHW a single group `g` for a fixed
batch element occupies channels `[8g, 8g+8)`, i.e. **8 consecutive H×W planes that are
contiguous in memory**. The normalization statistics (mean, var) are computed over those
`8·H·W` contiguous elements; `weight`/`bias` are per-channel (256), applied after
normalization.

---

## 2. Workload set (20 feedback shapes)

| B  | H    | W    | notes                    | M = B·H·W | group reduction = 8·H·W |
|----|------|------|--------------------------|-----------|--------------------------|
| 16 | 64   | 64   |                          | 65 536    | 32 768                   |
| 1  | 128  | 128  |                          | 16 384    | 131 072                  |
| 1  | 131  | 131  | **odd, non-multiple**    | 17 161    | 137 288                  |
| 32 | 128  | 128  | large                    | 524 288   | 131 072                  |
| 2  | 128  | 128  |                          | 32 768    | 131 072                  |
| 4  | 128  | 128  |                          | 65 536    | 131 072                  |
| 2  | 256  | 256  |                          | 131 072   | 524 288                  |
| 4  | 64   | 64   |                          | 16 384    | 32 768                   |
| 64 | 64   | 64   | largest batch            | 262 144   | 32 768                   |
| 2  | 64   | 64   |                          | 8 192     | 32 768                   |
| 1  | 1024 | 1024 | **huge single image**    | 1 048 576 | 8 388 608                |
| 1  | 293  | 293  | **odd/prime-ish**        | 85 849    | 686 792                  |
| 4  | 256  | 256  |                          | 262 144   | 524 288                  |
| 32 | 64   | 64   |                          | 131 072   | 32 768                   |
| 8  | 64   | 64   |                          | 32 768    | 32 768                   |
| 1  | 768  | 768  | large single image       | 589 824   | 4 718 592                |
| 4  | 128  | 96   | non-square               | 49 152    | 98 304                   |
| 4  | 96   | 128  | non-square               | 49 152    | 98 304                   |
| 8  | 64   | 128  | non-square               | 65 536    | 65 536                   |
| 2  | 256  | 192  | non-square               | 98 304    | 393 216                  |

Observations driving the design:
- **M (= B·H·W) is always large** (≥ 8k, up to ~1M). Conv-as-GEMM has a fat M dimension
  ⇒ tensor-core GEMMs will be well-fed on M.
- **N = C_out = 256** (modest but fine for 128- or 256-wide N tiles). **K = C_in·3·3 =
  2304** for the full im2col GEMM, or **K = 256** per shift for the 9-GEMM decomposition.
- **GroupNorm parallelism is bimodal.** Number of independent groups = `B·32`, ranging
  from **32** (B=1, 1024²) to **2048** (B=64, 64²). The 32-group cases each reduce
  **8.4 M elements** — a one-program-per-group reduction would leave 132 SMs almost idle.
  So the GroupNorm reduction must scale in *both* directions (split large groups across
  programs; keep many small groups cheap).
- **Odd / non-square / non-power-of-two spatial** (131, 293, 768, 1024, 96×128, 256×192,
  128×96) ⇒ all spatial index math and masking must be fully general; no assumptions of
  H·W divisibility by tile sizes.

---

## 3. Compute / memory characterization

Per conv: MACs = `M · C_out · C_in · 9`. Two convs dominate arithmetic. Example
B=32,128²: each conv ≈ 524288·256·256·9 ≈ 3.1e11 MAC ≈ 6.2e11 FLOP; two convs ≈ 1.24e12
FLOP. This is **compute-bound and essentially two large GEMMs** — the convs, not the
norm/act, set the FLOP floor.

Memory traffic argument (why a fused Triton kernel can beat the reference even if its conv
merely *matches* cuDNN). One full FP32 tensor at B=32,128² is `32·256·128·128·4 ≈ 0.5 GB`.
The eager reference launches ~8–10 kernels, each streaming full tensors:

- conv1 (read x + write t1), gn1 stats (read t1), gn1 apply (read t1 + write),
  silu1 (read + write), conv2 (read + write t2), gn2 stats (read t2),
  gn2 apply (read + write), silu2 (read + write), residual add (read t2 + read x + write).

That is ≈ **9–10 tensor passes ≈ 4.5–5 GB of DRAM traffic** for the non-conv work alone,
plus ~10 kernel launches. A fused implementation collapses the norm+act+add epilogues into
the conv boundaries:

- conv1: read x, write conv1_buf
- gn1+silu1: read conv1_buf (× reduction passes), write act1
- conv2: read act1, write conv2_buf
- gn2+silu2+residual: read conv2_buf (× reduction passes), read x, write output

≈ **4–5 passes** and **~4–5 launches**. On H100 HBM (~3.3 TB/s) the eliminated ~4–5 GB is
~1.3–1.5 ms per large workload that the reference pays and we do not. **Conclusion: the
conv quality is the decisive risk; the fusion/launch savings are the reliable edge.**

Weights (2.36 MB each) and norm params are tiny and L2-resident; they are not a bandwidth
concern.

---

## 4. Hard constraints (from CLAUDE.md / TASK.md)

- **Triton is the primary compute.** PyTorch allowed only for tensor metadata / launch
  plumbing (shapes, strides, `.view`/`.reshape` of weights, allocating output/temporaries,
  grid math). **No `F.conv2d`, `F.group_norm`, `F.silu`, or any Torch/CPU/NumPy/CUDA-ext
  compute fallback.** A failing Triton kernel is invalid and may **not** be swapped for a
  Torch path. This means **the convolution itself must be written in Triton** — the core
  difficulty.
- Immutable, sequential candidates `c001`, `c002`, … one source version at a time. Any
  meaningful source/config/launch change ⇒ new candidate ID; never reuse an ID.
- Evaluate only via `./scripts/evaluate_candidate.sh feedback <id>`. The full 20-workload
  feedback set counts as **one** evaluation. Budget: **100 evaluations**.
- Token budget: soft 9M / normal 10M / absolute 11M.
- Do **not** run CUDA, `nvidia-smi`, the external evaluator, or any alternate correctness
  harness directly. Profiling only via `./scripts/ncu_profile.sh` (ncu-report-skill
  workflow), and **never concurrently with an evaluation** (a foreign process on the locked
  GPU ⇒ return code 3, wasted evaluation).
- `final` only with explicit operator approval.

---

## 5. Tolerances and numerical risk

Per-workload tolerances are `max_atol ≈ 0.0027–0.0034`, `max_rtol = 1e-5`. Post-GroupNorm
activations are O(1) in magnitude (GroupNorm renormalizes to unit variance, then a per-
channel affine with random weight/bias, then SiLU), so `rtol·|ref| ≈ 1e-5` is negligible
and the effective budget is a **~0.003 absolute error per element**. That is generous and
strongly implies **TF32 tensor-core matmuls are acceptable** for the convolutions.

Risk register:

1. **TF32 vs FP32 in the conv GEMMs.** TF32 has a 10-bit mantissa (~5e-4 relative per
   product); FP32 accumulation over K=2304. Two chained convs could compound error, but
   GroupNorm after conv1 *renormalizes*, washing out much of conv1's absolute error, and
   the ~0.003 atol budget on O(1) outputs is comfortably above expected TF32 error.
   **Plan: use TF32 `tl.dot` (input_precision="tf32"), keep FP32 accumulators.** Keep an
   IEEE/FP32-matmul fallback variant in mind if any workload fails correctness. Do **not**
   use bf16/fp16 inputs — bf16's 8-bit mantissa (~4e-3 relative) risks exceeding atol.
2. **GroupNorm variance.** Reductions span up to **8.4 M elements** (B=1, 1024²) per group.
   `var = E[x²] − E[x]²` is catastrophic-cancellation prone at that scale. Use **Welford**
   or a **two-pass mean-then-sum-of-squared-deviations** in FP32; for split reductions,
   combine partials with the parallel/Welford merge formula (not naive E[x²]−E[x]²).
   Match the reference definition: **biased variance (divide by N)**, `1/sqrt(var+eps)`.
3. **SiLU stability.** `silu(x) = x·sigmoid(x)`. Use Triton's `tl.sigmoid` (numerically
   stable) rather than a hand-rolled `1/(1+exp(-x))` that can overflow for large negative
   x. Inputs here are O(1) post-norm so overflow is unlikely, but use the stable form.
4. **Zero padding / boundary.** 3×3, pad=1 ⇒ masked loads must return **exactly 0.0** for
   out-of-range spatial taps (not garbage, not NaN). Verify mask logic for all four edges
   and corners, and for odd dims (131, 293).
5. **GroupNorm affine ordering.** `y = ((x−mean)·rstd)·weight + bias`, weight/bias indexed
   per output channel; then SiLU on `y`. Applying affine before/after SiLU must match the
   reference (norm → affine → SiLU).
6. **Residual uses the original `x`** (the FP32 input), added *after* the second SiLU. Must
   preserve `x` unmodified (do not overwrite it with an in-place buffer).
7. **eps placement**: inside the sqrt, `rstd = rsqrt(var + eps)`, matching `F.group_norm`.

Because we cannot run an independent correctness harness, these must be reasoned out on
paper and confirmed by the evaluator's built-in tolerance check per candidate.

---

## 6. Triton design space

### 6.1 Convolution formulation (the crux)
Two candidate formulations, both expressible with tensor-core `tl.dot`:

- **(A) Full implicit im2col GEMM.** One conv = GEMM `[M × K] · [K × N]` with `M = B·H·W`,
  `K = C_in·9 = 2304`, `N = C_out = 256`. Weight is a metadata `reshape` of
  `(256,256,3,3)` → `(256, 2304)`. Per output tile, gather input taps for all 9 (kh,kw)
  with bounds masking. Pros: single big GEMM, best FLOP efficiency in principle. Cons:
  the input gather is non-contiguous / awkward in Triton and is exactly where naive
  conv-as-GEMM loses to cuDNN.
- **(B) 9× shifted 1×1 GEMM accumulation.** Loop over the 9 (kh,kw) offsets; each is a
  clean `[M × C_in] · [C_in × C_out]` GEMM on a spatially shifted input, accumulated into
  the output tile. K per step = 256. Pros: each sub-GEMM has contiguous channel access
  (especially in NHWC), simpler masking (shift + edge mask). Cons: input re-read up to 9×
  (mitigated by L2 / tiling), more accumulation.

Both need a decision on **data layout**. Input is NCHW. A GEMM wants the contraction
dimension (C_in) contiguous, i.e. **NHWC**. Options: (i) do index math directly in NCHW
(C_in stride = H·W) — simplest, no extra pass, but strided K loads; (ii) transpose to NHWC
once (a data-movement/plumbing pass) for coalesced K, transpose back at the end. Layout
choice is a first-class tuning axis; measure both.

### 6.2 Fusion boundaries
GroupNorm needs a full reduction over `8·H·W` that spans multiple conv output tiles, so it
**cannot** be fused into the conv's per-tile epilogue without a global reduction. Practical
fusion structure (per conv+norm+act stage):

1. Conv kernel → writes conv output to a temporary FP32 buffer (channels-last or -first per
   the layout decision).
2. GroupNorm reduction → mean/rstd per (batch, group).
3. Normalize + affine + SiLU (+ residual add on the *second* stage) fused into one kernel
   that reads the conv buffer once and writes the activated result.

The SiLU always fuses with the normalize/apply kernel (free). The residual add fuses into
the final apply kernel (read original `x`, add, write output). This already removes the
separate silu×2 and add passes and several launches vs the reference.

### 6.3 GroupNorm reduction strategy (must scale both ways)
- **Many small groups** (up to 2048): one program per (batch, group) can compute Welford
  over its contiguous `8·H·W` block, then a second sweep normalizes. Good occupancy when
  groups ≫ SMs.
- **Few huge groups** (as low as 32 for B=1, 1024²): one-program-per-group leaves ~100 SMs
  idle. Need a **split reduction** — grid = groups × spatial-tiles producing partial
  (sum, sumsq/M2, count), then a combine step, then normalize. Merge partials with the
  Welford/parallel-variance formula for stability.
- A robust approach: a two-kernel split reduction (partials → finalize mean/rstd), then a
  normalize+affine+SiLU(+residual) kernel — works for both regimes; small groups just have
  few tiles. Alternatively a one-pass persistent design; decide by profiling.

### 6.4 Tunables
Block sizes (BLOCK_M, BLOCK_N, BLOCK_K), `num_warps`, `num_stages`, GEMM tile for
N=256 (e.g. 128×256 or 128×128), reduction tile for GroupNorm, autotune configs keyed on
regime (small-spatial/large-batch vs large-spatial/small-batch). Consult **KernelWiki** for
Hopper (SM90) GEMM/conv tiling, warp specialization, and `tl.dot` TF32 guidance;
**ncu-report-skill** for occupancy / memory-vs-compute bound diagnosis.

### 6.5 Realistic expectation
A hand-written Triton FP32/TF32 conv is unlikely to *beat* cuDNN's conv FLOP-for-FLOP.
The geomean edge must come from (a) matching cuDNN reasonably on the convs via TF32 tensor
cores, and (b) decisively winning the memory-bound epilogues (norm+silu+add) and launch
overhead through fusion. If the conv is far off cuDNN on the compute-heavy shapes
(1024², 768², B=64), those workloads may not reach ≥1.0× even with perfect fusion — this is
the main threat to geomean and must be watched per-shape in the evaluator output.

---

## 7. Candidate roadmap (high-level intent — detailed steps go in plan.md)

- **c001 — correctness-first baseline.** Simplest fully-Triton implementation that is
  clearly correct: implicit-GEMM (or 9-shift) conv with TF32, straightforward scalable
  GroupNorm reduction, fused SiLU and residual. Goal: pass all 20 workloads and establish a
  speedup/latency baseline before optimizing. Establishes that the Triton-only path is
  numerically valid within tolerance.
- Subsequent candidates (planned, not yet fixed): layout NCHW↔NHWC, conv formulation A↔B,
  autotune tiles/warps/stages, GroupNorm reduction split strategy, fusing conv1's write
  with stage-1 stats, minimizing temporaries. Each is one immutable candidate, profiled
  with ncu (never during an eval), decided on measured geomean + per-shape correctness.

Convergence: stop when added candidates no longer improve geomean, or at budget; then write
`SEARCH_COMPLETE`. `final` only on operator approval.

---

## 8. Validation strategy

- **Correctness** is verified solely by the evaluator's built-in tolerance check
  (`./scripts/evaluate_candidate.sh feedback cNNN`) across all 20 workloads. No alternate
  correctness harness (forbidden). Each candidate must be reasoned correct on paper first
  (Section 5) to avoid burning evaluations on trivially wrong kernels.
- Design candidates to be **individually falsifiable**: change one axis at a time so an
  eval result attributes cleanly to that change.
- **Performance diagnosis** via `./scripts/ncu_profile.sh` (ncu-report-skill), run only when
  no evaluation is in flight, to confirm compute- vs memory-bound behavior, occupancy for
  the bimodal GroupNorm, tensor-core utilization of the conv GEMMs, and to guide tiling.
- **Record-keeping**: append one JSON object per evaluated candidate to `candidates.jsonl`
  (parent, source hash, hypothesis, validation/pass-fail, per-workload result, geomean,
  decision, cumulative eval count, skill usage); never rewrite prior records.
- **Watch metrics**: geomean speedup (primary) plus the worst per-shape speedup — the big
  compute-bound shapes (1024², 768², B=64·64²) are the likely geomean drag and the signal
  for whether conv tuning or fusion is the better next lever.

## 9. Open questions to resolve during planning/first evals

1. Does TF32 conv pass all 20 tolerances, or do any need IEEE/FP32 matmul? (Confirmed only
   by c001's eval.)
2. Formulation A (im2col K=2304) vs B (9× shifted 1×1) — which is faster on H100 for N=256?
3. NCHW-direct index math vs an explicit NHWC transpose pass — net win after the transpose
   cost?
4. Best GroupNorm reduction for the 32-group huge-image cases without hurting the
   2048-group cases.
5. Whether conv1's output write can be folded into stage-1 GroupNorm partials to save a
   pass, without hurting conv GEMM efficiency.
