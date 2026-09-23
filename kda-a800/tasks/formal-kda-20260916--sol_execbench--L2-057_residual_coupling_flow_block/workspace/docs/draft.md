# Draft — L2/057 `residual_coupling_flow_block`

Task: optimize the official SOL-ExecBench task `L2/057_residual_coupling_flow_block`
on NVIDIA **A800 (sm_80, Ampere)**. Primary implementation must be **Triton**;
PyTorch allowed only for metadata / launch plumbing; **no Torch/CPU/NumPy/CUDA-ext
computational fallback**. Metric = geometric-mean speedup over the reference, with a
per-workload correctness gate.

---

## 1. What the operation actually computes

### 1.1 Reference structure (from `task/definition.json`)

Residual affine-coupling block used in a VITS/OpenVoice normalizing flow. Constants:

- `channels = 192`, `hidden_channels = 192`, `half_channels = 96`, `kernel_size = 5`, `n_layers = 4`.
- Inputs: `x [B, 192, T]` (float32), `x_mask [B, 1, T]` (float32), `reverse` (bool scalar),
  and 4 transforms × 3 conv1d weights+biases.

Per coupling layer `i` (of 4):

```
x0 = x[:, :96, :]          # first half   (conditioning)
x1 = x[:, 96:, :]          # second half  (updated)
h  = conv0_i(x0)           # Conv1d 96 -> 192, k=5, pad=2, + bias
h  = relu(h)
h  = conv1_i(h)            # Conv1d 192 -> 192, k=5, pad=2, + bias
h  = relu(h)
h  = conv2_i(h)            # Conv1d 192 -> 96,  k=5, pad=2, + bias
h  = h * x_mask
x1 = x1 + h                # forward   (or  x1 - h  for reverse)
x  = cat([x0, x1], dim=1)
x  = x * x_mask
```

Forward runs layers `0,1,2,3`; reverse runs `3,2,1,0` with subtraction.
Each conv1d uses `padding = kernel_size // 2 = 2` (SAME-length, zero padded).

### 1.2 Key algebraic simplification — the transforms are independent

`x0` (channels `0..95`) is **never written** by any layer: every layer only reads
`x[:, :96]` and writes back `x0` unchanged (aside from a mask multiply). Therefore the
conditioning input to *all four* `conv0_i` is the *same* `x0`. Because `x0` is constant,
`transform_i(x0)` does not depend on the running `x1`, and the four layers just accumulate:

```
delta = Σ_{i=0..3} ( transform_i(x0) * x_mask )
x1_out = x1_in ± delta            # + forward, − reverse
x0_out = x0_in                    # unchanged (mask multiply only)
out    = cat([x0_out, x1_out]) * x_mask
```

Consequences (all values identical to the reference, no approximation):
- **Order independence**: forward vs reverse differ only by the sign of `delta`; the
  reversed iteration order in the reference is irrelevant because the summands are constant.
- The 4 transforms can be **batched/grouped** into 3 wide convolutions:
  - `conv0`: one dense conv `96 → 4·192 = 768` on the shared `x0` → `H0 [B,4,192,T]`.
  - `conv1`: grouped conv (groups=4) `768 → 768` on `relu(H0)` → `H1 [B,4,192,T]`.
  - `conv2`: grouped conv (groups=4) `768 → 4·96 = 384` on `relu(H1)` → `[B,4,96,T]`,
    then sum over the 4 groups to get `delta [B,96,T]`.
- This turns the reference's ~12 convs + ~28 elementwise/copy/cat kernels into ~3 conv
  passes + a fused combine, cutting launch overhead and memory traffic dramatically —
  the primary lever for small workloads.

### 1.3 The mask is effectively all-ones (but treat carefully)

`get_inputs` builds `x_mask = torch.ones(B,1,T)`, and the feedback workloads mark
`x_mask` as `{"type": "random"}`, which in this harness means "use the value produced
by `get_inputs`" (only `reverse` is overridden as a scalar). So in practice **mask ≡ 1**
and every `* x_mask` is a no-op, and `x0_out == x0_in` exactly.

Risk mitigation: I will still **apply the mask** in the fused epilogue (cheap, correct if
mask were fractional) and I will **not** hard-code mask=1 into correctness reasoning; the
independence simplification above holds for mask=1 (verified by construction) and is what
the evaluator checks numerically. Because `mask` is binary/idempotent, applying it to
`x0`, to `delta`, and to the final output all commute and reduce to the reference result.
(If a future workload used a non-binary mask, the only subtlety is that the reference feeds
raw `x0` to `transform_0` but masked `x0` to `transform_{1,2,3}`; with mask∈{0,1} these are
identical. I will validate numerically rather than assume.)

---

## 2. Workloads & problem sizes

| WL | B  | T    | reverse | B·T     | atol   | rtol | match |
|----|----|------|---------|---------|--------|------|-------|
| 1  | 8  | 1721 | true    | 13 768  | 0.011  | 1e-5 | 0.98  |
| 2  | 4  | 541  | false   | 2 164   | 0.010  | 1e-5 | 0.98  |
| 3  | 16 | 2048 | true    | 32 768  | 0.012  | 1e-5 | 0.98  |
| 4  | 2  | 293  | true    | 586     | 0.0097 | 1e-5 | 0.98  |
| 5  | 64 | 8192 | true    | 524 288 | 0.012  | 1e-5 | 0.98  |

Observations:
- Both `reverse` values appear (WL2 forward, others reverse) — must handle the sign.
- `B·T` spans ~586 → ~524k: **three orders of magnitude**. A single fixed tiling will not
  be optimal everywhere. WL4/WL2 are tiny → **launch-overhead / occupancy bound**; WL5 is
  large → **compute/GEMM-throughput bound**; WL1/WL3 in between.
- Channels are tiny (96/192/768) — the GEMM K/N dims are small, so these are
  low-arithmetic-intensity convs; efficient tiling and epilogue fusion matter more than
  peak FLOPs.

### 2.1 Rough compute estimate

Per call, MACs ≈ `Σ_i (Cout·Cin·K) · B·T` over the 3 convs per transform × 4:
`(96·192 + 192·192 + 192·96)·5 · B·T · 4 = 4·5·(2·96·192 + 192²)·B·T`
`= 4·5·73728·B·T ≈ 1.47M · B·T` MACs → for WL5 (`B·T=524288`) ≈ **0.77 TMAC ≈ 1.5 TFLOP**.
On A800 FP32≈19.5 TF/s, TF32≈156 TF/s — so WL5 is meaningfully compute-bound and a naive
FP32 Triton GEMM may struggle vs cuDNN; small WLs are dominated by overheads where fusion
wins easily.

---

## 3. Constraints

- **Triton-only compute**; torch used only for allocation, views/slicing (metadata),
  weight packing/reshape, and kernel launch. Weight packing (concatenating the 4 conv0
  weights, stacking grouped conv1/conv2 weights) is *data movement on constants*, done once
  per call in torch — acceptable plumbing, but note it costs a little; consider whether the
  pack itself should be a tiny Triton/cheap copy or just `torch.cat`/`torch.stack` (these are
  memory ops on small weights `≤768·768·5`, negligible and not "computational" in the
  op-defining sense). I will keep packing to trivial reshapes/`cat` of provided weights.
- Immutable candidates `c001, c002, …`; one kernel version over all 5 WLs = one evaluation.
- Evaluate **only** via `./scripts/evaluate_candidate.sh feedback cNNN`. No direct CUDA /
  profiler / nvidia-smi / evaluator invocation. No changing the fixed workloads.
- Budget: 100 evals; token soft 1.0M / hard 1.2M. `final` only with operator approval.
- Output dtype float32, shape `[B,192,T]`, contiguous like input.

---

## 4. Numerical risks

1. **TF32 vs FP32.** The reference `F.conv1d` runs through cuDNN, which by default permits
   TF32 on Ampere (`torch.backends.cudnn.allow_tf32=True`). So the reference is effectively
   TF32-accumulated-in-FP32. Tolerances are loose (atol ≈ 0.01 on O(1) outputs, rtol 1e-5,
   98% match). Using `tl.dot(..., allow_tf32=True)` (TF32 inputs, FP32 accumulate) should
   both match reference *speed* and stay within tolerance. Full FP32 (`allow_tf32=False`) is
   more accurate but slower; keep as a correctness fallback if TF32 fails the gate.
2. **Magnitude sanity.** `x∼N(0,1)`; kaiming conv weights `∼N(0, 2/fan_in)`. conv0 output
   variance ≈ `fan_in · (2/fan_in) = 2` → std≈1.4 (+bias∼N(0,1)); ReLU; chained twice more;
   sum of 4 transforms → `delta = O(1)`, `x1 = O(1)`, `out = O(1)`. atol 0.01 ≈ 1% of a
   typical magnitude — TF32 relative error ~1e-3 per op, accumulated over K≤768 and a chain
   of 3 convs + 4-way sum → expected abs error a few×1e-3. Should pass with margin, but the
   *chained* convs are the compounding risk (conv1 amplifies conv0 error, etc.). Watch WL4
   (tightest atol 0.0097).
3. **Padding / boundary correctness.** Must zero-pad time at `t<0` and `t≥T` exactly as
   `F.conv1d(padding=2)`. Off-by-one in the tap offset is the most likely correctness bug.
4. **Reverse sign.** WL2 is forward (`+delta`), the rest reverse (`−delta`). A flipped sign
   is a silent, total correctness failure on 4/5 WLs.
5. **Group/channel indexing.** When batching the 4 transforms into grouped convs, mapping
   each output-channel block to the correct input-channel group and weight slice must be
   exact; an indexing slip mixes transforms and fails silently.
6. **Bias placement & ReLU order.** Bias is added *before* ReLU inside conv0/conv1, but
   conv2 has **no ReLU** after it (raw output feeds the mask+add). Fusing epilogues must
   respect: conv0→bias→relu, conv1→bias→relu, conv2→bias→(no relu)→mask→±add.
7. **Match-ratio semantics.** Only 98% of elements must satisfy atol/rtol, giving slack for
   a few outliers — but I should not rely on it; aim for full-tensor pass.

---

## 5. Triton design space

### 5.1 Data layout
`x` is `[B,192,T]` row-major → time contiguous (stride 1), channel stride `T`, batch stride
`192·T`. For a conv, for a fixed channel the output positions along `T` are contiguous
(coalesced loads/stores). The GEMM view is `M=T` (contiguous), `N=Cout`, `K=Cin`, with a
tap dimension `K_tap=5`. Two layout options:
- **Keep `[B,C,T]`** and index channels with stride `T` inside the kernel (no transpose).
- **Transpose to `[B,T,C]`** (channels contiguous) to make the dot's K-dim contiguous;
  costs an explicit transpose pass (memory). Likely not worth it given small C.

Plan to keep `[B,C,T]`.

### 5.2 Conv realization — two candidate kernels

**(A) Implicit-GEMM conv (preferred).**
Program grid over `(batch b, time-tile, out-channel-tile)`. Accumulate
`acc[BLOCK_T, BLOCK_COUT] += Σ_{k=0..4} Σ_{cin-tiles} Xk[BLOCK_T,BLOCK_CIN] · W[k][BLOCK_CIN,BLOCK_COUT]`
where `Xk` is `x` shifted by `k-2` in time with zero-pad masking at boundaries. Tile within a
single batch so time shifts stay in-bounds. Use `tl.dot` (TF32). Fuse `+bias`, `relu`,
mask, and `±` residual add into the epilogue of the relevant conv. K (Cin ≤ 768) is small
enough that the cin loop is short.

**(B) Shift-and-accumulate (5 GEMMs).**
Express conv as `Σ_{k} Wk @ shift(x, k-2)`; 5 plain GEMMs `[Cout,Cin]×[Cin,T]` per conv with
masked shifted A-operand. Simpler to reason about; more redundant loads but only K=5.

Both are viable; (A) reads each input slab once per cin-tile and is more memory-efficient.

### 5.3 Fusion / kernel-count strategy
Target pipeline (grouped-transform formulation from §1.2):
1. `k_conv0`: `x0 [B,96,T] → H0 [B,768,T]`, fused bias+ReLU. (dense, shared input)
2. `k_conv1`: grouped(4) `H0 → H1 [B,768,T]`, fused bias+ReLU.
3. `k_conv2_combine`: grouped(4) `H1 → [B,4,96,T]`, fused bias, **sum over the 4 groups**,
   apply mask, then `x1 ± delta`, write `out` (both halves) with mask.

Alternative granularities to explore as separate candidates:
- Keep 4 transforms separate (no grouping) — simpler, more launches; may still beat torch on
  small WLs via epilogue fusion; good **first correctness candidate**.
- Fuse conv0→relu→conv1→relu inside one kernel (persistent hidden in registers/SMEM) to
  avoid the `H0` round-trip — harder; consider only if memory-bound WLs justify it.
- A single fully-fused megakernel is likely over-engineering given time-tiling and the
  grouped reductions; defer.

### 5.4 Autotuning / occupancy
- `B·T` varies 586→524k: autotune `BLOCK_T ∈ {32,64,128,256}`, `BLOCK_COUT ∈ {32,64,128}`,
  `num_warps ∈ {2,4,8}`, `num_stages ∈ {2,3,4}`; key the autotune cache on `(B,T,reverse?)`
  or at least on size buckets. For tiny WL4/WL2 favor small tiles + many programs for
  occupancy; for WL5 favor larger tiles + software pipelining.
- Grid should cover `B` on one axis so small-T workloads still get enough CTAs (WL4:
  `B=2,T=293` → only ~a few time-tiles; batching the 4 transforms into wide Cout helps fill
  the machine).
- `tl.dot` requires the contracted dim ≥16; Cin∈{96,192,768} fine; pad tap/cin tiles as
  needed.

### 5.5 Reverse & mask handling
- Pass `reverse` as a compile-time-ish flag (kernel constexpr or a `sign` scalar `±1.0`)
  folded into the residual add: `x1_out = x1 + sign * (mask * delta)`.
- Multiply the whole output by mask once at store; write `x0` half straight through (× mask).

---

## 6. Baseline & incremental candidate plan (high level; detailed in plan.md later)

- **c001 — correctness-first.** Straightforward per-transform Triton conv (4×3 convs) with
  fused bias/relu/mask/add, `[B,C,T]` layout, implicit-GEMM or shift-accumulate, TF32.
  Goal: pass all 5 WLs; establish a correct, immutable reference point and measure speedup.
- **c00x — grouped/batched transforms** (§1.2 & §5.3) to slash launches/traffic; expect the
  biggest small-WL gains.
- **c00x — autotune** block sizes / warps / stages; size-bucketed configs.
- **c00x — deeper fusion** (fuse conv0→conv1 hidden in-kernel) only if profiling-by-proxy
  (the evaluator's per-WL timing) shows memory-bound behavior on mid WLs.
- **c00x — FP32 fallback** if TF32 fails the gate on any WL (esp. WL4).
- Stop when geomean converges or budget nears; write `SEARCH_COMPLETE`.

Each candidate: new immutable ID, record parent, source hash, hypothesis, per-WL
pass/speedup, geomean, decision, cumulative eval count, skill usage.

---

## 7. Validation strategy

- **Only sanctioned run path** is `./scripts/evaluate_candidate.sh feedback cNNN`, which
  invokes the official evaluator (correctness gate atol/rtol/match-ratio per WL + speedup).
  No direct CUDA/profiler/evaluator/nvidia-smi runs.
- **Correctness reasoning before eval** (I cannot run ad-hoc CUDA locally): check padding
  offset (`t+k-2`), bias-before-ReLU, no-ReLU after conv2, reverse sign, group→channel/weight
  indexing, and the §1.2 independence argument. Mentally trace shapes for each conv.
- **Incremental validation**: land a simple correct candidate (c001) *first*; only then
  refactor to grouped/fused variants, so any correctness regression is isolated to one change
  and the passing baseline is preserved (immutability keeps c001 as a fallback best).
- **Numerical guard**: start with TF32 (matches reference behavior + tolerances). If any WL
  (watch WL4, atol 0.0097) fails on precision, add an FP32-accumulate candidate. Never trade
  correctness for speed below the gate.
- **Per-WL diagnosis**: use the evaluator's per-workload pass/fail + timing to localize
  issues (e.g., a single failing WL usually means a size-dependent tiling/boundary bug, not a
  formula bug).
- **Regression discipline**: append one JSON record per candidate to `candidates.jsonl`;
  never rewrite; keep the best passing candidate as the standing champion.

---

## 8. Skill usage note

- `KernelWiki` targets **Blackwell (SM100) / Hopper (SM90)**; this task is **A800 / sm_80
  (Ampere)** — out of scope, so not consulted for the design.
- `ncu-report-skill` profiles on **B200 / sm_100** and requires running Nsight Compute, which
  is disallowed here and wrong-architecture — not used.
- No web/subagent/MCP/external tools used, per isolation rules. Design relies on the official
  definition, the five feedback workloads, and standard Triton/Ampere reasoning.

---

## 9. Open questions to resolve during implementation

1. Does grouped/batched packing of the 4 transform weights (via `torch.cat/stack`) count as
   acceptable "plumbing"? Treat as trivial constant reshapes; keep it minimal and outside the
   op's compute (all multiply-adds remain in Triton). If in doubt, keep transforms separate
   (c001) which needs no packing.
2. Is TF32 sufficient for WL4's tight atol, or is FP32 accumulate needed? Decide empirically.
3. Best kernel granularity per size bucket (per-transform vs grouped vs conv0+conv1 fused) —
   decide from per-WL eval timings.
4. Whether to write `out` in place / reuse `x` buffer for `x0` half to cut a copy.
