# Draft — L2/015 Audio Sinusoidal Position Embedding with Conv Projection

Target HW: NVIDIA **A800 (sm_80, Ampere)** — A100-class. bf16 tensor cores via `mma.sync`
(ldmatrix), fp32 accumulation. **No** TMA / tcgen05 / CLC (those are Hopper/Blackwell). The
`KernelWiki` skill is Blackwell/Hopper-specific, so its concrete PRs are mostly *not*
transferable here; `ncu-report-skill` requires a profiler run, which is **forbidden** by the
task rules. So this task is driven by first-principles reasoning + the official feedback
evaluator.

---

## 1. Operation semantics (from `task/definition.json` reference)

`run(input_features, conv2d1_w, conv2d1_b, conv2d2_w, conv2d2_b, conv2d3_w, conv2d3_b,
conv_out_weight, positional_embedding, embed_scale)`:

1. **Conv1**: `F.conv2d(input_features[b,1,80,Tin], w1[384,1,3,3], b1, stride=2, pad=1)` →
   `[b,384,40,T1]`; then `F.gelu`.
2. **Conv2**: `F.conv2d(x, w2[384,384,3,3], b2, stride=2, pad=1)` → `[b,384,20,T2]`; `F.gelu`.
3. **Conv3**: `F.conv2d(x, w3[384,384,3,3], b3, stride=2, pad=1)` → `[b,384,10,T3]`; `F.gelu`.
4. **Reshape**: `permute(0,3,1,2).contiguous().view(b, T3, 384*10=3840)`.
   Flatten order is **channel-outer, freq-inner**: column `k = c*10 + f`.
5. **Linear (no bias)**: `x[b,T3,3840] @ conv_out_weight[1024,3840]^T` → `[b,T3,1024]`.
6. **Scale**: `x * embed_scale` (`embed_scale = sqrt(1024) = 32.0` in all feedback workloads).
7. **Add PE**: `x + positional_embedding[:T3].unsqueeze(0)` (broadcast over batch; indexed by
   time-after-conv `t`).

Output: `hidden_states[b, T3, 1024]`, **bf16**.

All conv/gelu/linear are executed in **bf16** in the reference (inputs are bf16); PyTorch/cuDNN
accumulate internally in fp32 but **round each intermediate activation to bf16**. This
"where does rounding happen" detail governs numerical matching (see §4).

### Constants
`d_model=1024`, `num_mel_bins=80`, `max_source_positions=1500`, `downsample_hidden=384`,
`freq_after_conv=10`, `conv_out_dim=3840`, `kernel=3`, `stride=2`, `pad=1`.
GELU here is `F.gelu` **default = exact erf** GELU: `0.5*x*(1+erf(x/√2))` (NOT tanh approx).

### Verified per-workload shapes  (Hout=⌊(Hin−1)/2⌋+1; freq always 40→20→10)

| wl | b  | Tin  | T1   | T2  | T3  | ref T3 | M_lin=b·T3 |
|----|----|------|------|-----|-----|--------|-----------|
| w1 | 8  | 3256 | 1628 | 814 | 407 | 407 ✓  | 3256 |
| w2 | 1  | 3000 | 1500 | 750 | 375 | 375 ✓  |  375 |
| w3 | 16 | 2048 | 1024 | 512 | 256 | 256 ✓  | 4096 |
| w4 | 2  | 1688 | 844  | 422 | 211 | 211 ✓  |  422 |
| w5 | 32 | 920  | 460  | 230 | 115 | 115 ✓  | 3680 |

Tolerances: `max_atol` 1.1 (w3: 1.3), `max_rtol` 0.05, `required_match_ratio` 0.98 — **loose**
(bf16 output scaled by 32 + PE ∈ [−1,1]).

---

## 2. Cost / roofline (hand-computed, A800 ≈ A100: ~312 bf16 TFLOP/s, ~2 TB/s)

FLOPs = 2·M_out·(Cin·9) with M_out=b·Hout·Wout; conv1 has Cin=1 (K=9).

| stage | w3 (largest) FLOPs | note |
|-------|--------------------|------|
| conv1 | ~4.5 GFLOP  | Cin=1, K=9 → memory/launch bound, NOT tensor-core efficient |
| **conv2** | **~435 GFLOP** | dominant; K=3456, tensor-core GEMM |
| conv3 | ~109 GFLOP  | K=3456, tensor-core GEMM |
| linear| ~32 GFLOP   | M=4096,K=3840,N=1024 GEMM |

**Compute is dominated by conv2, then conv3.** Ideal conv2 ≈ 1.4 ms; realistic 2–4 ms.

**Memory**: intermediate1 `[b,384,40,T1]` bf16 is *large*: w3 = 16·384·40·1024·2 ≈ **503 MB**
(largest across wls; w1 400 MB, w5 452 MB). conv2 re-reads it (9 taps, with L2 reuse ≈ once).
intermediate2 `[b,384,20,T2]` ≈ 126 MB (w3); intermediate3 ≈ 32 MB. So conv1 write +
conv2 read of intermediate1 is the main bandwidth term (~0.5–1 GB traffic).

### Why beating the reference is plausible
The reference launches ~10+ kernels: 3×(conv + gelu) = 6, a `permute→contiguous` copy (~1),
linear, `mul`, `add`. Our fused design uses **4 kernels** and removes the 32 MB
permute-contiguous copy + several elementwise passes. cuDNN also handles the awkward
**tiny freq dim** (Hout=20/10) which can push it to suboptimal algos/tiles; a purpose-built
implicit-GEMM with GELU/scale/PE fused into epilogues has real headroom even if per-conv we
only match cuDNN.

---

## 3. Triton design space

### 3.1 Chosen pipeline (baseline candidate c001): 4 kernels, implicit-GEMM convs

- **K-A `conv1+gelu`**: input `[b,1,80,Tin]` → intermediate1 stored **NHWC** `[b,40,T1,384]`.
  Cin=1 so K=9; direct kernel: each program computes a tile of output pixels × BLOCK_N out
  channels, gathers the 3×3 single-channel window (masked for pad), MAC over 9 taps
  (pad K→16 for a `tl.dot`, or explicit FMA since Cout vectorizes well). Add bias, GELU,
  round bf16.
- **K-B `conv2+gelu`**: reads intermediate1 NHWC → intermediate2 NHWC `[b,20,T2,384]`.
  Implicit GEMM: M=b·20·T2 output pixels, N=Cout=384, K=Cin·9=3456. Loop over the 9 taps;
  each tap = `tl.dot(A[BLOCK_M, 384], W_tap[384, BLOCK_N])` fp32-accum. NHWC makes the A-block
  read **contiguous along Cin** (coalesced). Bias + GELU + bf16 in epilogue.
- **K-C `conv3+gelu` (fused reshape)**: reads intermediate2 NHWC → writes **directly into the
  linear A-matrix** `[b, T3, 3840]` at column `cout*10 + f` (channel-outer/freq-inner). This
  *is* the reference `permute(0,3,1,2).view(...)`, so the permute-contiguous copy disappears.
  Same implicit-GEMM structure as K-B (M=b·10·T3).
- **K-D `linear + scale + PE`**: GEMM `A[M=b·T3, K=3840] @ Wt[3840,1024]` with epilogue
  `acc*embed_scale + positional_embedding[t]`, store bf16. A rows are contiguous in K
  (coalesced); PE row selected by time index `t` (same across batch).

**Layout decision**: store all intermediates **NHWC** (channels-last, contiguous C) so every
implicit-GEMM A-block load is coalesced. We *own* the intermediate buffers, so we pick layout
freely. The harness input is NCHW but Cin=1, so layout is irrelevant for conv1's input.

**Weight layout for `tl.dot`**: convs need `W_tap[Cin, Cout]` = `weight[cout, cin, kh, kw]`
(strided in the given `[Cout,Cin,3,3]`). Weight is tiny (2.65 MB, L2-resident and reused).
- Option A (pure Triton, preferred for correctness-first): load with computed strides inside
  the kernel — no torch compute, just strided global loads.
- Option B (later opt): pre-permute weights to `[9,Cin,Cout]` contiguous. A `.permute().
  contiguous()` is a **layout-only** transform (not the conv math) — arguably "launch
  plumbing," but it borders the "no torch computation" rule, so if used it will be done via a
  small Triton copy kernel to stay unambiguous.

### 3.2 Alternatives / later candidates
- **im2col + cuBLAS/`tl.dot` GEMM**: rejected — materialized im2col for conv2 is ~0.9 GB
  (M·3456), worse than implicit gather.
- **Fuse conv1 into conv2** (skip 0.5 GB intermediate1): conv2 pixel needs a 3×3 window of
  conv1 outputs, each conv1 output a 3×3 input window → up to 9× recompute of conv1. conv1 is
  cheap (K=9), so trading 0.5 GB traffic for recompute may win. Candidate for a later variant.
- **Fuse conv3 into linear**: linear row needs *all* 3840 K = all (c,f) of that time column of
  conv3 → would recompute conv3 fully per N-tile; likely not worth it. Keep separate.
- **Tap-blocking & tiling**: tune BLOCK_M∈{64,128}, BLOCK_N∈{64,128}, num_warps∈{4,8},
  num_stages∈{2,3,4}. Small freq means M is dominated by time; tile M over flattened
  (b,ho,wo). Boundary/pad handled by masking hi,wi at tap load.
- **conv1 vectorization**: since it's memory-bound, may fuse gelu and prefer a plain
  elementwise/tiled kernel over a padded `tl.dot`.

---

## 4. Numerical risks & mitigations

- **Match rounding sites**: reference rounds each conv output to bf16 *before* GELU and each
  intermediate to bf16. Mitigation: accumulate conv in **fp32**, round the pre-activation to
  bf16, compute GELU in fp32 from that bf16 value, round result to bf16 → mirrors the ref's
  bf16 intermediates and limits chained-error drift across the 3-conv cascade.
- **GELU flavor**: use **exact erf** (`tl.math.erf`/libdevice), not tanh approx, to match
  `F.gelu` default. tanh approx is likely within atol but erf is free-ish and safer.
- **Accumulation precision**: fp32 accumulators for all `tl.dot` (Ampere bf16 MMA accum is
  fp32 by default). Never accumulate in bf16.
- **embed_scale**: apply as fp32 (32.0). PE added as fp32 then rounded to bf16. Output scaled
  by 32 ⇒ absolute errors scale up, but `atol=1.1` (w3 1.3) already accounts for this.
- **Padding correctness**: stride-2, pad-1 → boundary output pixels read out-of-range hi/wi;
  mask to 0 (not the bias). Off-by-one here is the most likely correctness bug — assert Hout
  formula `⌊(Hin−1)/2⌋+1` in the wrapper and cover freq edges (Hout row 0 and last).
- **Flatten order**: `k = c*10 + f` (channel-outer). A swapped order silently passes shapes
  but fails values — pin this in K-C's write indexing and cross-check against the weight’s
  column semantics.
- **bf16 dynamic range**: activations after Kaiming/Xavier init are O(1); no overflow risk.
- **Tolerance headroom**: rtol 0.05, match_ratio 0.98 allow 2% outliers — comfortable for
  bf16, but the 3-conv chain compounds rounding; if a workload fails, first suspect
  rounding-site mismatch or pad masking, not raw precision.

---

## 5. Validation strategy

- **No external harness allowed**: cannot run torch reference, CUDA, profiler, `nvidia-smi`,
  or the evaluator directly. Correctness/speed come **only** from
  `./scripts/evaluate_candidate.sh feedback <cid>` over the 5 fixed workloads (= 1 evaluation).
- **In-code guards** (cheap, allowed): shape/stride asserts in the Python wrapper; assert
  computed T1/T2/T3 equal `⌊(n−1)/2⌋+1`; assert output shape `[b,T3,1024]` bf16; assert
  `conv_out_dim=3840`, `d_model=1024`.
- **Correctness-first ordering**: c001 = simplest fully-Triton correct pipeline (in-kernel
  strided weight loads, modest fixed tiles, exact erf, bf16 rounding at ref sites). Only after
  it passes all 5 do we tune tiles / fuse conv1→conv2 / weight pre-pack, each as a new
  immutable candidate ID with its own evaluation.
- **Regression discipline**: record per-workload pass + latency + geomean in `candidates.jsonl`;
  keep changes small per candidate so a geomean regression is attributable.
- **Fallback rule**: a failing Triton kernel is invalid — fix the Triton, never substitute a
  torch/CPU/numpy path.

### Success criteria
Geomean speedup > 1.0 vs reference with **all 5 workloads passing** correctness; iterate on
conv2/conv3 tiling (dominant cost) and the conv1→conv2 fusion for the memory-bound term.
