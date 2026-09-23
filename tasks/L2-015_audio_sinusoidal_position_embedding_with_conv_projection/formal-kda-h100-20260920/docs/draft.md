# Draft — L2/015 Audio Sinusoidal Position Embedding with Conv Projection

Target: NVIDIA H100 (`sm_90`), bfloat16. Submission `solution/solution.py::run(...)`.
Primary implementation **must be Triton**; PyTorch only for metadata / launch plumbing.
No Torch/CPU/NumPy/CUDA-extension computational fallback (so `F.conv2d`, `F.linear`,
`F.gelu` are all off-limits — every FLOP must be produced by a Triton kernel).

---

## 1. Operation semantics (from `task/definition.json` reference)

Signature:
```
run(input_features, conv2d1_weight, conv2d1_bias,
    conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
    conv_out_weight, positional_embedding, embed_scale)
```

Pipeline (all under `@torch.no_grad()`):

1. **Conv1 + GELU**: `x = F.conv2d(input_features, W1, b1, stride=2, padding=1)`; `x = F.gelu(x)`
   - in `(B, 1, 80, T)`, weight `(384, 1, 3, 3)` → out `(B, 384, 40, t1)`, `t1 = ceil(T/2)`.
2. **Conv2 + GELU**: weight `(384, 384, 3, 3)`, stride 2, pad 1 → `(B, 384, 20, t2)`, `t2 = ceil(t1/2)`.
3. **Conv3 + GELU**: weight `(384, 384, 3, 3)`, stride 2, pad 1 → `(B, 384, 10, t3)`, `t3 = ceil(t2/2)`.
4. **Flatten**: `x.permute(0,3,1,2).contiguous().view(B, t3, C*F)` with `C=384, F=10 → 3840`.
   - Flatten order is **channel-major, freq-minor**: column `k ∈ [0,3840)` decodes to
     `c = k // 10`, `f = k % 10`. This ordering is load-bearing for the linear step.
5. **Linear (no bias)**: `x = F.linear(x, conv_out_weight)`, `conv_out_weight` is `(1024, 3840)`;
   `out[m,n] = Σ_k x[m,k] · W[n,k]` → `(B, t3, 1024)`.
6. **Scale**: `x = x * embed_scale` (scalar, `= 32.0` in every feedback workload = `sqrt(1024)`).
7. **Add positional embedding**: `pos = positional_embedding[:t3, :]` (shape `(1500,1024)` sliced),
   broadcast over batch: `out[b,t,n] = x[b,t,n] + pos[t,n]`.

Output: `hidden_states` `(B, t3, 1024)` bfloat16.

**Important data facts from the feedback set**: `positional_embedding` is `{"type":"random"}`
and `embed_scale` is a passed scalar `32.0`. So there is **no sinusoid to generate at runtime** —
the "sinusoidal" name only describes how `get_inputs` builds `pe`; `run` just consumes the given
tensor. I only need to slice `[:t3]` and add. `seq_len = t3 ≤ 541 ≤ 1500`, so the slice is always
in-bounds (no masking needed on the pos add beyond `t3`).

---

## 2. Constants, shapes, and per-workload cost

Constants: `d_model=1024`, `num_mel_bins=80`, `max_source_positions=1500`,
`downsample_hidden_size=384`, `freq_dim_after_conv=10`, `conv_out_dim=3840`, `kernel_size=3`.

Derived: `t1=ceil(T/2)`, `t2=ceil(t1/2)`, `t3=ceil(t2/2)` (verified against every workload's
`time_after_conv`). Freq chain fixed `80→40→20→10`.

16 feedback workloads (the full set = one candidate evaluation). Derived time chain and the
heavy dimension `M2 = B·20·t2` (conv2 output pixels; conv2 has `K=384·9=3456` reduction):

| B  | T    | t1   | t2   | t3  | conv2 M=B·20·t2 | linear M=B·t3 |
|----|------|------|------|-----|-----------------|----------------|
| 2  | 1688 | 844  | 422  | 211 | 16,880          | 422            |
| 32 | 4328 | 2164 | 1082 | 541 | 692,480         | 17,312         |
| 1  | 1048 | 524  | 262  | 131 | 5,240           | 131            |
| 1  | 1808 | 904  | 452  | 226 | 9,040           | 226            |
| 32 | 920  | 460  | 230  | 115 | 147,200         | 3,680          |
| 64 | 128  | 64   | 32   | 16  | 40,960          | 1,024          |
| 16 | 2048 | 1024 | 512  | 256 | 163,840         | 4,096          |
| 16 | 384  | 192  | 96   | 48  | 30,720          | 768            |
| 4  | 2216 | 1108 | 554  | 277 | 44,320          | 1,108          |
| 2  | 256  | 128  | 64   | 32  | 2,560           | 64             |
| 2  | 2528 | 1264 | 632  | 316 | 25,280          | 632            |
| 8  | 1976 | 988  | 494  | 247 | 79,040          | 1,976          |
| 8  | 3256 | 1628 | 814  | 407 | 130,240         | 3,256          |
| 32 | 768  | 384  | 192  | 96  | 122,880         | 3,072          |
| 64 | 512  | 256  | 128  | 64  | 163,840         | 4,096          |
| 1  | 3000 | 1500 | 750  | 375 | 15,000          | 375            |

**Cost balance.** Per output pixel: conv2/conv3 do `3456` MACs, conv1 does `9`, linear does
`3840` per output row. Roughly conv2 ≫ conv3 > linear ≫ conv1. For the biggest workload
(B=32,T=4328): conv2 ≈ `692480·384·3456 ≈ 9.2e11` MACs (~1.8 TFLOP), conv3 ≈ `2.3e11`,
linear ≈ `6.8e10`, conv1 tiny. **Conv2 (and conv3) dominate; the whole task is really "make
the 384→384 stride-2 3×3 convs fast in Triton, then fuse everything else onto them."**

Regimes span: latency-bound tiny cases (B=1..2, t3≤131; B=64,T=128) where launch/occupancy
matters, and throughput-bound large cases (B=32,T=4328; B∈{16,64}) where MMA efficiency matters.
Geomean weighting means the small cases count as much as the big ones — I must not regress them.

---

## 3. Constraints & risks

- **Everything in Triton.** No `F.conv2d/linear/gelu`. Weight reshapes (contiguous → 2D) are
  metadata-only and allowed; `.contiguous()`/`.permute` used purely to hand a well-defined layout
  to a kernel are plumbing, but the actual compute copy (im2col / permute-materialize) should be
  done by *my* kernel where it matters for speed, not by an eager Torch copy that does real work.
- **A failed/incorrect Triton path may not be swapped for a Torch fallback.** So correctness of the
  Triton convs is non-negotiable; c001 must already be fully Triton and correct.
- **Validation channel is narrow**: only `./scripts/evaluate_candidate.sh feedback cNNN`. I may
  **not** run CUDA, `nvidia-smi`, the external evaluator directly, or build an alternate Torch/NumPy
  correctness harness. Bash compute is also blocked in this environment. Correctness confidence must
  come from careful derivation + the official feedback evaluator.
- **Profiling** only via `./scripts/ncu_profile.sh` (ncu-report-skill workflow), and **never**
  concurrently with an evaluation (a foreign process on the locked GPU → return code 3, wasted
  eval). Serialize profiling and evaluation strictly.
- **Budget**: 100 candidate evals; token soft/normal/hard = 9M/10M/11M. Final run operator-only.

### Numerical risks
- **Accumulation dtype.** cudnn/cublas accumulate convs and the GEMM in fp32. In Triton I must use
  fp32 accumulators (`tl.dot(..., out_dtype=tl.float32)` / fp32 acc) and only round to bf16 when
  writing an intermediate — mirroring the reference which stores bf16 between stages.
- **GELU flavor.** `F.gelu` default is the **exact erf** GELU: `0.5·x·(1+erf(x/√2))`, *not* the
  tanh approximation. I will use `libdevice.erf` in fp32 to match. The tanh approx might pass under
  the loose tolerance but risks the tight-error large workloads; erf is the safe default. GELU
  applied on the fp32 pre-round value vs on the bf16-rounded value differs slightly — reference does
  conv(bf16 out) then gelu(bf16), so I'll round conv result to bf16, then GELU (compute in fp32,
  store bf16) to stay closest.
- **Error propagation & the ×32 scale.** Errors from 3 stacked bf16 convs feed the linear, whose
  result is multiplied by `embed_scale=32`, amplifying absolute error before the pos add. Tolerances
  are per-workload `max_atol ∈ [0.92, 1.3]`, `max_rtol=0.05`, `required_match_ratio=0.98` — loose
  enough for bf16 storage (bf16 rel error ≈ 2⁻⁸ ≈ 0.004 ≪ 0.05) but I should keep all *reductions*
  in fp32 so error stays at storage-rounding level, not accumulation level. The 0.98 match ratio
  tolerates a few outliers, which is comforting for the amplified-scale rows.
- **Padding correctness.** stride-2, pad-1 3×3: input coord `ih = 2·of − 1 + kh`, `iw = 2·ot − 1 + kw`
  for `kh,kw ∈ {0,1,2}`. Out-of-range `[0,H)`/`[0,W)` taps must be masked to 0 (not wrapped). Off-by-one
  here is the most likely correctness bug; I'll derive once and reuse the same index helper for all 3 convs.
- **Bias dtype.** Conv biases are bf16; add in fp32 then round.
- **Boundary shapes.** Odd `t1/t2` (e.g. T=3000→t1=1500 even but many odd t2/t3) must be handled by
  masking the last output column; tiny `t3=16` and `M=64·t3` cases must still produce enough grid tiles.

---

## 4. Triton design space

Staged pipeline is the natural decomposition; fusion opportunities are the win vs the ~10-launch
eager reference (3 conv + 3 gelu + permute-contiguous copy + gemm + scale + add).

### 4.1 Conv formulation
- **Implicit GEMM (preferred).** Treat conv as GEMM: `M = B·OF·OT` output pixels, `N = OC=384`,
  `K = IC·9`. Weight `(OC, IC, 3, 3)` reshapes (free, contiguous) to `(OC, K)`; use as the B-operand
  (transposed inside `tl.dot`). Per K-block, compute the gathered input tile addresses on the fly
  (no materialized im2col). GELU + bias fused in the epilogue; write bf16.
  - For conv2/3, `K=3456` → tile K in chunks (e.g. 64/128) looping over `(ic_block × 9 taps)` or
    over `9 taps × ic_block`. `N=384` fits ~3 tiles of 128.
- **Tap-decomposed GEMM.** Alternative: accumulate 9 pointwise (1×1) GEMMs, each `(M,IC)×(IC,OC)`
  on a shifted/strided/masked input view. Cleaner K=384 contiguous reductions if activations are
  channel-last; padding via masking the shifted rows. Good MMA shape but 9 masked gathers.
- **Direct (non-MMA).** Only sensible for **conv1** (`IC=1, K=9`): load ≤9 masked scalars per pixel,
  multiply by `W(384×9)`. With K=9, `tl.dot` is awkward (K<16); either pad K→16 with zeros or do 9
  fused multiply-adds / small outer-products. Conv1 is cheap so simplicity > peak here.

### 4.2 Memory layout (the key perf lever)
Inputs are **NCHW**; reduction dim (IC) is *not* contiguous (T is). For MMA, K-contiguous is best.
Options:
- **Keep NCHW.** Simplest, matches inputs, but conv2/3 gather IC with stride `H·W` and time with
  stride 2 — poor coalescing / MMA-unfriendly K.
- **Channels-last (NHWC = `(B,F,T,C)`) for intermediates.** conv1 writes its output directly in
  NHWC; conv2/3 then read IC contiguously (great for `tl.dot` K), gathering only the 3×3 spatial
  taps. Freq H is tiny (40/20/10) so spatial tiling is cheap. This is the layout cudnn itself
  prefers for tensor cores, and is my leading candidate for the heavy convs.
- Flatten step: from conv3 NHWC `(B,10,t3,384)` I still need the `k=c·10+f` column order for the
  linear. I will **not** materialize the permute; instead the fused linear kernel decodes `k→(c,f)`
  and reads conv3 output at the right address (fusing away the reference's `.contiguous()` copy).

### 4.3 Fusion plan
- conv_i + bias + GELU fused (epilogue) — removes 3 separate gelu launches.
- **Flatten + Linear + scale + pos-add fused** into one GEMM kernel: `M=B·t3`, `K=3840`, `N=1024`;
  read conv3 output with `c=k//10,f=k%10`; epilogue `acc*embed_scale + pos[t]`; write bf16. Removes
  the permute-contiguous copy, the standalone scale, and the standalone add.
- Possible later fusion: conv3+GELU into the linear (linear row `(b,t3)` needs the full `c×f`
  conv3 slice = one conv3 output "time column"); high complexity, defer unless profiling demands.
- Net kernel count target: conv1, conv2, conv3, fused-linear ≈ **4 launches** vs ~10 in eager.

### 4.4 Tiling / autotune
- Autotune `BLOCK_M/BLOCK_N/BLOCK_K`, `num_warps`, `num_stages` per kernel, keyed on shape buckets
  so tiny (B≤2, B=64/T=128) and huge (B=32/T=4328) both get good configs.
- Ensure grid has enough tiles for H100's 132 SMs on mid/large shapes; accept latency-bound behavior
  on the genuinely tiny shapes (there's little compute to hide).
- Guard against `t3` not dividing block sizes (mask), and small-`M` GEMMs (linear M as low as 64).

---

## 5. Candidate roadmap (high level; details go in plan.md)

- **c001 — correctness-first, fully Triton.** Implicit-GEMM convs (conv1 special-cased), erf GELU,
  fp32 accum, conservative fixed block sizes, straightforward NCHW or single chosen layout, fused
  linear+scale+pos-add. Goal: pass all 16 workloads; establish a valid baseline speedup number.
- **c002+ — layout & fusion.** Move heavy convs to channels-last; fuse GELU/bias epilogues; confirm
  no regressions. 
- **c00x — autotuning & shape-bucketed configs**; targeted at conv2/conv3 (the cost center) and the
  large-B workloads, then re-check tiny cases.
- Iterate guided by geomean from the evaluator and, when a bottleneck is unclear, by `ncu_profile.sh`
  (serialized, never during an eval). Stop / write `SEARCH_COMPLETE` when geomean converges.

---

## 6. Validation strategy

1. **Derivation-first correctness.** Lock the index math (conv output = `ceil(n/2)`; input tap
   `2·o−1+k`; padding mask; flatten `k=c·10+f`; linear `out=Σ x·Wᵀ`; scale then pos-add) in a single
   shared helper reused across kernels. Re-derive against the reference before coding each kernel.
2. **Official feedback evaluator is the only correctness/perf oracle** — `./scripts/evaluate_candidate.sh
   feedback cNNN` over all 16 workloads = one eval. Watch that *every* selected workload passes
   correctness (a fail invalidates the candidate) and read the reported geomean speedup.
3. **No alternate harness / no direct CUDA / no nvidia-smi.** Reason about numerics instead
   (fp32 accum, erf GELU, bf16 storage matching reference).
4. **Profiling** only via `ncu_profile.sh` + ncu-report-skill, strictly serialized with evals, to
   localize conv2/conv3 bottlenecks (MMA utilization, memory throughput, occupancy) before spending
   evals on layout/tiling changes.
5. **Ledger discipline.** One immutable candidate per source version; append one complete JSON record
   per eval to `candidates.jsonl` (parent, source hash, hypothesis, per-workload result, geomean,
   decision, cumulative eval count, skill usage); never rewrite prior records; never reuse an ID for
   changed source.
6. **Guardrails to check every candidate:** all 16 pass; no tiny-shape regression; erf (not tanh)
   GELU unless proven equivalent within tolerance; padding masks correct on odd/boundary sizes;
   output dtype bf16 and shape `(B,t3,1024)`.

---

## 7. Open questions to resolve in plan.md / early candidates

- NCHW vs channels-last for conv2/3 — decide by profiling c001/c002.
- Whether K=9 conv1 is better as padded `tl.dot` vs manual FMA accumulation.
- Best K-loop order for conv2/3 (`taps×ic` vs `ic×taps`) for reuse of the gathered spatial window.
- Whether fusing conv3 into the linear pays off, given the linear is only ~4–13% of FLOPs.
- Shape-bucket autotune keys that cover both latency- and throughput-bound regimes without
  overfitting to the two largest workloads.
