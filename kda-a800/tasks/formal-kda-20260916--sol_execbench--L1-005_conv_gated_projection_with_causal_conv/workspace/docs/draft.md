# Draft — L1/005 `conv_gated_projection_with_causal_conv`

Target HW: NVIDIA **A800 (`sm_80`, Ampere)**. Primary impl **must be Triton**; PyTorch only for
metadata/launch plumbing. No Torch/CPU/NumPy/CUDA-ext computational fallback. Metric: **geomean
speedup vs the reference across the 5 fixed feedback workloads, with every selected workload passing
correctness**. Budget: 100 candidate evals; token soft 1.0M / hard 1.2M.

---

## 1. What the operation computes

Signature (all tensors `bfloat16`), constants `H = hidden_size = 2048`, `K_c = conv_kernel_size = 4`,
`3H = triple_hidden = 6144`:

```
x               : (B, S, H)
in_proj_weight  : (3H, H)      in_proj_bias : (3H,)
conv_weight     : (H, 1, K_c)  conv_bias    : (H,)      # depthwise, groups = H
out_proj_weight : (H, H)       out_proj_bias: (H,)
-> output       : (B, S, H)
```

Reference flow (from `task/definition.json`):

1. `BCx = F.linear(x, in_proj_weight, in_proj_bias)` → `(B, S, 3H)`  *(GEMM1)*
2. transpose to `(B, 3H, S)`, `chunk(3, dim=1)` → `Bgate, Cgate, Xproj`, each `(B, H, S)`
3. `Bx = Bgate * Xproj`  *(element-wise gating)*
4. `Bx_padded = F.pad(Bx, (K_c-1, 0))` then `F.conv1d(..., groups=H)` → `conv_out (B,H,S)` *(depthwise causal conv)*
5. `y = Cgate * conv_out`  *(output gating)*
6. transpose back to `(B, S, H)`, `.contiguous()`
7. `output = F.linear(y, out_proj_weight, out_proj_bias)` *(GEMM2)*

### 1.1 Exact index mapping (critical for correctness)

Flatten rows as `m = b*S + s`, with `M = B*S`, `m ∈ [0,M)`, `s = m mod S`, `b = m // S`.

After the transpose+chunk, for output channel `h ∈ [0,H)`:
- `Bgate[b,h,s] = BCx[b,s, 0*H + h]`
- `Cgate[b,h,s] = BCx[b,s, 1*H + h]`
- `Xproj[b,h,s] = BCx[b,s, 2*H + h]`

So the three chunks are **contiguous column blocks** of `BCx`: `[0,H)`, `[H,2H)`, `[2H,3H)`.

Gating: `Bx[m,h] = BCx[m,h] * BCx[m, 2H+h]`.

Causal depthwise conv (`F.pad` left = `K_c-1 = 3`, no right pad, stride 1, kernel 4). For tap
`k ∈ {0,1,2,3}` the padded input aligns so that output position `s` reads `Bx` at position `s-3+k`:

```
conv_out[m,h] = conv_bias[h] + Σ_{k=0..3} conv_weight[h,0,k] * Bx[b, h, s-3+k]
              where Bx[b,h,·] taps at s-3+k are ZERO when (s-3+k) < 0   (causal left pad)
```

- Tap `k=3` = **current** position `s` (weight `conv_weight[h,0,3]`).
- Tap `k=0` = oldest position `s-3` (weight `conv_weight[h,0,0]`).
- In flattened rows, position `s-3+k` is row `m-3+k`, valid only while it stays **in the same batch**
  (`s-3+k ≥ 0`). Rows crossing below `s=0` are the causal zero pad → must be masked to 0. Because rows
  are laid out `b*S + s`, staying `≥ b*S` is exactly the `s-3+k ≥ 0` condition, so the batch boundary
  and the causal pad coincide — a single `(s-3+k) ≥ 0` guard handles both.

Output gating: `y[m,h] = BCx[m, H+h] * conv_out[m,h]`.

GEMM2 (note `F.linear` uses `W^T`): `output[m, n] = out_proj_bias[n] + Σ_{h} y[m,h] * out_proj_weight[n,h]`.

Likewise GEMM1: `BCx[m, j] = in_proj_bias[j] + Σ_{i} x[m,i] * in_proj_weight[j,i]`.

**Layout win:** everything lives in `(M, ·)` row-major space (`x → (M,H)` is a free view of the
`(B,S,H)` contiguous tensor; `y → (M,H)`). The reference's two `transpose`s and `.contiguous()` copy
are only artifacts of routing data through `conv1d`'s `(B,C,S)` convention. If we run the conv along
`M`-rows directly we can **eliminate both transposes and the contiguous copy entirely**.

---

## 2. Workloads and problem sizes

| # | uuid (short) | B  | S    | M=B·S | atol   | rtol |
|---|--------------|----|------|-------|--------|------|
| 1 | b0a9e3f0     | 1  | 1024 | 1024  | 0.0130 | 0.05 |
| 2 | 8b678b13     | 4  | 256  | 1024  | 0.0130 | 0.05 |
| 3 | 58e3ac47     | 32 | 256  | 8192  | 0.0210 | 0.05 |
| 4 | 6033fd0f     | 2  | 4096 | 8192  | 0.0210 | 0.05 |
| 5 | bb262a64     | 1  | 2048 | 2048  | 0.0098 | 0.05 |

Observations:
- `H=2048`, `3H=6144`, `K_c=4`, `groups=H` are **compile-time constants** — bake them in and
  specialize kernels / `tl.constexpr`.
- Effective GEMM shapes: **GEMM1** is `(M×2048) · (2048×6144)`; **GEMM2** is `(M×2048) · (2048×2048)`.
- `M` takes only **three distinct values** {1024, 2048, 8192}. `S` takes {256,1024,2048,4096}. Small
  distinct set → autotune configs will generalize well; `M` is always a multiple of 256 and 1024,
  and `S` is always a power-of-two ≥ 256 (helpful for conv tiling / no ragged masks on `M` within a batch).
- Per-workload compute (FLOPs): GEMM1 ≈ `2·M·2048·6144`, GEMM2 ≈ `2·M·2048·2048` → **GEMM1 is 3×
  GEMM2**. The middle (gating+conv+gating) is tiny FLOPs but touches `~5·M·H` bf16 elements → memory-bound.
- rtol 5% is loose; atol is the tighter constraint (0.0098 on WL5, 0.013 on WL1/2, 0.021 on WL3/4).
  WL5 (B=1, S=2048, atol 0.0098) is the least forgiving.

---

## 3. Constraints & environment realities

- **Triton-only compute.** Both GEMMs must be Triton `tl.dot` kernels (cannot call `F.linear`/`torch.matmul`
  for compute — that would be a Torch computational fallback). This is the central performance risk:
  we must approach cuBLAS bf16 throughput on Ampere with a well-tuned Triton matmul, then win the rest
  via fusion.
- **No local execution / no profiler.** Per CLAUDE.md we may not run CUDA, a profiler, `nvidia-smi`,
  the external evaluator directly, or any alternate correctness harness. The **only** feedback channel
  is `./scripts/evaluate_candidate.sh feedback cNNN`, and each such run (5 workloads) consumes **one**
  of 100 evals and reports correctness + speedup. ⇒ correctness must be established **analytically**
  before spending an eval; a failed Triton run is wasted budget.
- **Skills:** `KernelWiki` targets Blackwell/Hopper (tcgen05/TMEM/CLC/FP4) and is **not applicable** to
  `sm_80`; `ncu-report-skill` requires running the profiler, which is disallowed. Both are effectively
  unusable here beyond generic reading; note in candidate `skill_usage` accordingly.
- **Immutability:** every meaningful source/config/launch change ⇒ new `cNNN`; never rewrite past records.

---

## 4. Numerical analysis & risks

### 4.1 Reference precision chain
Reference rounds to **bf16 at every stage**: `BCx` (fp32-accum cuBLAS → bf16), `Bx = Bgate*Xproj`
(bf16 op, internally fp32→round bf16), `conv_out` (cuDNN bf16, fp32-accum → bf16), `y = Cgate*conv_out`
(→ bf16), then GEMM2 (fp32-accum → bf16). So the "ground truth" itself carries bf16 rounding at 4
intermediate points.

### 4.2 Our precision plan
- **GEMMs:** load bf16 operands, accumulate in **fp32** (`tl.dot(..., out_dtype=fp32)` / fp32
  accumulator), add bias in fp32, cast to bf16 on store. Matches cuBLAS behavior.
- **Middle (gating+conv+gating):** compute entirely in **fp32** (promote loaded bf16 → fp32, multiply,
  conv-accumulate in fp32, multiply, then cast the final `y` to bf16). This is *more* accurate than the
  reference's intermediate bf16 roundings, and the differences are well inside the loose tolerances.
  - Key subtlety: the reference rounds `Bx` to **bf16 before** the conv. If our fp32 path ever drifts
    outside atol we can insert an explicit `Bx_bf16 = Bx.to(bf16).to(fp32)` round to mirror the
    reference; keep this as a fallback lever (WL5 with atol 0.0098 is where to watch).
- **Accumulation order:** conv is only 4 taps → order-insensitive. GEMM2 reduces over `H=2048`; use
  the standard `BLOCK_K` split with fp32 accumulator (cuBLAS also splits K). Differences here are
  random-sign and bounded by rtol 0.05 over 2048 terms.

### 4.3 Correctness pitfalls to get right the first time
1. **Tap order / causal direction:** `k=3` is the current sample; taps go backward in `s`. A mirrored
   kernel (using `conv_weight[...,K_c-1-k]` or reading `s+k`) will silently pass shape checks but fail
   numerically. Derive from `F.pad(left=3)` + `conv1d` cross-correlation semantics (PyTorch conv1d is
   cross-correlation, **not** flipped convolution).
2. **Batch boundaries:** taps at `s-3+k<0` must be zero; never read row `m-3+k` from the previous batch.
   Guard with `s - 3 + k ≥ 0` (equivalently `row ≥ b*S`).
3. **Chunk offsets:** `Bgate=[0,H)`, `Cgate=[H,2H)`, `Xproj=[2H,3H)` — off-by-`H` here is a classic bug.
4. **`F.linear` transpose:** weights are `(out, in)`; GEMM computes `Σ_in x·W[out,in]` (i.e. `x @ W^T`).
5. **Bias dtype:** biases are bf16; add in fp32 then round.
6. **Contiguity of `x`:** reshape `(B,S,H)→(M,H)` is only free if `x` is contiguous (it is per definition);
   assert strides / use `.reshape` guarded, but do not silently `.contiguous()` an already-contiguous tensor.

---

## 5. Triton design space

Baseline decomposition into **3 Triton kernels** (all in `(M,·)` space, no transposes):

### K1 — GEMM1 with bias: `BCx = x @ in_proj_weight^T + in_proj_bias`, shape `(M, 3H)`
- Standard tiled bf16 matmul: `BLOCK_M × BLOCK_N` output tile, `BLOCK_K` reduction over `H=2048`,
  fp32 accumulator, L2-friendly `GROUP_M` row-grouping, `num_warps`/`num_stages` autotuned.
- `N=6144` and `K=2048` are multiples of 64/128 → no N/K masking; `M∈{1024,2048,8192}` multiples of
  128 → no M masking for `BLOCK_M∈{64,128,256}`. Clean, mask-free fast path.
- Candidate tile menu: `BLOCK_M∈{64,128,256}`, `BLOCK_N∈{64,128,256}`, `BLOCK_K∈{32,64,128}`,
  `num_stages∈{3,4,5}`, `num_warps∈{4,8}`, `GROUP_M∈{4,8}`.

### K2 — fused gating + causal depthwise conv + gating: `y (M, H)`
- Reads `BCx`, computes `Bx=BCx[:, :H]*BCx[:, 2H:3H]`, the 4-tap causal conv (per-channel weights
  `conv_weight[h,0,·]`, `conv_bias[h]`), then `y = BCx[:, H:2H] * conv_out`. Pure element-wise + tiny
  stencil ⇒ **memory-bound**; goal is minimum passes.
- Tiling options:
  - (a) **Sequence-tiled**: block `(BLOCK_S rows within one batch) × (BLOCK_H channels)`; load a
    halo of 3 extra rows above the tile, compute `Bx` once, do the 4-tap stencil in registers/smem.
    Amortizes the `Bx` computation and the neighbor reads. Grid = `(B, ceil(S/BLOCK_S), ceil(H/BLOCK_H))`.
  - (b) **Row-parallel** (simple): one program per `(row-block, channel-block)`, recompute the ≤4
    neighbor `Bx` values by re-loading `Bgate,Xproj` for rows `m-3..m`. Simplest to prove correct;
    up to 4× redundant loads of `Bgate/Xproj` but conv is cheap. Good **c001** choice.
  - Batch-boundary guard via `s` index; conv weights broadcast over the row dimension (per-channel).
- Because `conv_weight`/`conv_bias` are `(H,·)` and reused for all `M` rows, keep them resident per
  channel-block.

### K3 — GEMM2 with bias: `output = y @ out_proj_weight^T + out_proj_bias`, shape `(M, H)`
- Same matmul template as K1 with `N=2048`.

### Fusion escalation ladder (later candidates)
- **F1: fold gating into GEMM1 epilogue.** Produce only `Bx (M,H)` and `Cgate (M,H)` (write `2H`
  instead of `3H`) by combining the `[0,H)` and `[2H,3H)` column tiles in the epilogue. Needs the
  epilogue to see both column blocks → either `BLOCK_N` spanning matched columns or a two-pass
  accumulator; evaluate cost/benefit (saves ~1/3 of the intermediate write+read).
- **F2: fuse conv into GEMM2's A-load.** GEMM2's `A = y (M,H)`; `y[m,h]` needs `conv_out[m,h]` which
  needs neighbor rows `m-1..m-3`. Feasible only if `BLOCK_M` rows are contiguous in `M` and we load a
  3-row halo into the K-loop; complex and risks correctness at batch/tile edges. Treat as a stretch goal
  only if K2 is shown to be a real bottleneck.
- **F3: merge K1+K2** so the gating/conv is the GEMM1 epilogue producing `y` directly. Blocked by the
  conv needing cross-row (cross-`BLOCK_M`) neighbors; likely not worth it. Keep K2 standalone.
- **GEMM1 restructuring:** single `N=3H` GEMM reuses each `A` tile across all three projections (best A
  reuse) — prefer over three separate GEMMs.

### Where the time goes (rough Ampere model, M=8192 case)
- GEMM1 ≈ `2·8192·2048·6144 ≈ 2.06e11` FLOP; GEMM2 ≈ `6.9e10` FLOP; total ≈ `2.75e11`.
- At ~1.5e14 bf16 FLOP/s realized → ~1.8 ms of unavoidable matmul; K2 memory traffic
  (`~5·M·H·2 B ≈ 0.17 GB`) at ~1.5 TB/s → ~0.1 ms. **GEMMs dominate.**
- ⇒ The realistic upside is: (i) our Triton GEMMs must land near cuBLAS, and (ii) we recover the
  reference's overheads — the slow **cuDNN depthwise `conv1d` with groups=2048**, two transposes, a
  `.contiguous()` copy, and several extra kernel launches / intermediate allocations. Grouped/depthwise
  conv over 2048 groups is a known cuDNN weak spot; that plus the transposes is our headroom.
- **Speedup expectation:** modest, roughly **1.1–1.5×** if GEMMs match cuBLAS; **<1×** risk if the
  Triton GEMMs underperform cuBLAS by more than the fusion savings. This risk ordering drives the
  candidate roadmap: first prove correctness + measure the GEMM gap, then tune GEMMs, then tighten fusion.

---

## 6. Validation strategy (given no local execution)

1. **Analytical correctness before any eval.** Re-derive every index map (§1.1, §4.3) and shape on
   paper; write kernels with `constexpr` H/3H/K_c so shapes are self-documenting. Add cheap host-side
   `assert`s on dtypes/shapes/contiguity in `run(...)` (metadata only — allowed).
2. **Start simple (c001 = correctness anchor):** 3 straightforward kernels (K1, K2 row-parallel variant
   b, K3), conservative single tile config, fp32 middle. The goal of c001 is a **passing, well-defined
   baseline** and a first speedup number, not peak performance.
3. **One variable per candidate.** After c001 passes, change exactly one thing per new `cNNN`
   (autotune GEMM1, then GEMM2, then K2 sequence-tiling, then F1 fusion), so each eval attributes its
   delta cleanly. Record parent/hash/hypothesis/per-workload result/geomean/decision/eval-count.
4. **Correctness watch-list:** WL5 (atol 0.0098) is the tightest — if a candidate fails only there,
   suspect the fp32-vs-bf16 `Bx` rounding (§4.2) or a conv tap/edge bug, not the GEMMs.
5. **Budget discipline:** 100 evals but token budget is the likely binding constraint; avoid
   speculative evals. Converge when geomean gains stall, then write `SEARCH_COMPLETE`. Never run `final`
   without operator approval.

---

## 7. Open questions to resolve empirically (via feedback evals)
- How close does a tuned Triton bf16 matmul get to cuBLAS on A800 for these `M,N,K`? (Sets the ceiling.)
- How large is the reference's `conv1d(groups=2048)`+transpose+contiguous overhead as a fraction of
  total? (Sets our headroom.) Inferred only from the end-to-end speedup deltas across candidates.
- Is K2 better as sequence-tiled (halo reuse) or row-parallel (simple)? Decide after c001’s number.
- Does F1 (gating-in-epilogue) pay for its added complexity, or is the standalone memory-bound K2
  already negligible vs the GEMMs?

## 8. Immediate next step
Write `docs/plan.md` with the executable candidate roadmap (c001 correctness anchor → GEMM autotune →
K2 tiling → fusion escalation), then implement `c001`.
