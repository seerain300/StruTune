# Draft — L2/030 Flux Concatenated Sequence Processing with Split

Run: `formal-kda-20260916--sol_execbench--L2-030_flux_concatenated_sequence_processing_with_split`
Target GPU: NVIDIA A800 (`sm_80`, Ampere). Framework: Triton primary; PyTorch only for
metadata/launch plumbing. No Torch/CPU/NumPy/CUDA-extension computational fallback.

---

## 1. Operation summary

Reference (`task/definition.json`) does, with `H = hidden_dim = 3072` (const):

```
concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
processed    = concatenated @ process_weight.t()                          # [B, T+I, H]
processed_encoder = processed[:, :T, :]                                   # [B, T, H]
processed_hidden  = processed[:, T:, :]                                   # [B, I, H]
return processed_encoder, processed_hidden
```

Symbols:
- `B` = `batch_size` (var), `T` = `text_seq_len` (var), `I` = `img_seq_len` (var).
- Inputs: `hidden_states` `[B, I, H]` fp32, `encoder_hidden_states` `[B, T, H]` fp32,
  `process_weight` `W` `[H, H]` fp32 (an `nn.Linear`-style weight, `out=H`, `in=H`, no bias).
- Outputs: `processed_encoder` `[B, T, H]` fp32, `processed_hidden` `[B, I, H]` fp32.

The task card calls this a **memory-bandwidth bottleneck** pattern from Flux dual-stream
transformer blocks: the `cat` materializes a fresh `[B, T+I, H]` tensor before the GEMM and the
`split` slices it afterward.

---

## 2. Core algebraic simplification (the main lever)

A linear projection is applied **independently per token row**. Concatenating along the sequence
(row) dimension, projecting, then splitting back along the same row boundaries is a pure
re-ordering that does **not** couple encoder rows with image rows. Therefore:

```
processed_encoder = encoder_hidden_states @ W.T      # [B, T, H]
processed_hidden  = hidden_states         @ W.T      # [B, I, H]
```

is **mathematically identical** (row for row, elementwise) to the concat→matmul→split reference,
with the exact same output ordering. Consequences:

- The `torch.cat` allocation + copy is **eliminated entirely** (this is the advertised bottleneck).
- The `split` becomes trivial — we write each GEMM result straight into its own output tensor;
  no post-hoc slicing/copy.
- Both streams share the **same weight `W`**, so `W` stays hot in cache/registers across both.

Both projections are the same GEMM shape family:
```
X [M, K] · W^T [K, N]  ->  Y [M, N],   with  K = N = H = 3072
```
where `X` is a stream flattened to 2-D (`[B*T, H]` for encoder, `[B*I, H]` for image), and
`W` is read as `W[n, k]` (i.e. `Y = X @ W.T`, per `nn.Linear`). Rows are fully independent, so
batch and sequence collapse into a single `M` dimension for contiguous inputs.

**Equivalence conditions to preserve:**
- Row order within each stream is preserved by flattening `[B, S, H] -> [B*S, H]` in C-contiguous
  order and reshaping the output back the same way.
- Encoder output uses encoder rows; image output uses image rows — never crossed.
- Must produce **two** separate output tensors of the exact reference shapes/dtype.

---

## 3. Workload analysis (fixed feedback set)

`H = 3072`, so every GEMM has `K = N = 3072`. `M_enc = B*T`, `M_img = B*I`,
`M_total = B*(T+I)` (the reference's concatenated row count).

| WL | uuid(prefix) | B | T | I | M_enc | M_img | M_total | atol | rtol | match |
|----|--------------|---|-----|-----|-------|-------|---------|-------|-------|-------|
| 1  | 7ae93ff7 | 2 | 1423 | 1489 | 2846 | 2978 | 5824 | 0.0021 | 1e-5 | 0.98 |
| 2  | 7a91d659 | 4 | 128  | 256  | 512  | 1024 | 1536 | 0.002  | 1e-5 | 0.98 |
| 3  | a9d58452 | 1 | 77   | 1024 | 77   | 1024 | 1101 | 0.0018 | 1e-5 | 0.98 |
| 4  | 9de8abcb | 2 | 128  | 256  | 256  | 512  | 768  | 0.002  | 1e-5 | 0.98 |
| 5  | e92f1fac | 1 | 77   | 4096 | 77   | 4096 | 4173 | 0.002  | 1e-5 | 0.98 |

Observations:
- `M_total` spans **768 → 5824** rows; `N = K = 3072` throughout. These are medium GEMMs, all
  compute-heavy (large K,N) but with modest-to-small M.
- Encoder streams can be **very small** (`M_enc = 77` in WL3/WL5) — a thin GEMM. The image stream
  dominates FLOPs there. A single tiling config must remain efficient for both a 77-row and a
  4096-row GEMM.
- FLOPs per workload `= 2 * M_total * N * K`:
  - WL1: `2·5824·3072·3072 ≈ 1.10e11` (110 GFLOP)
  - WL2: `≈ 2.90e10`, WL3: `≈ 2.08e10`, WL4: `≈ 1.45e10`, WL5: `≈ 7.87e10`.
- Tensor sizes (fp32, MB): `X_total = M_total·3072·4`; `W = 3072·3072·4 = 37.7 MB`.
  - WL1: `X ≈ 71.5 MB` per side (in+out ≈ 143 MB) plus `W` 37.7 MB.
  - `W` (37.7 MB) is close to A800 L2 (40 MB) — reuse of `W` across all `M` tiles is essential;
    large `BLOCK_M` and good L2 residency of `W` matter.

**Roofline sanity (A800):** peak fp32 SIMT ≈ 19.5 TFLOPS; TF32 tensor ≈ 156 TFLOPS; HBM ≈ 2 TB/s.
- True-fp32 SIMT GEMM for WL1 ≈ `1.10e11 / 19.5e12 ≈ 5.6 ms`.
- TF32-tensor GEMM for WL1 ≈ `0.7 ms` (but inaccurate — see §5).
- 3×TF32 emulation (`tf32x3`) ≈ 1/3 of TF32 throughput ≈ 52 TFLOPS eff. → WL1 ≈ `2.1 ms`.
- `cat` traffic for WL1 ≈ read(enc+img) + write(concat) ≈ `2 · 71.5 MB ≈ 143 MB` → ≈ `0.07 ms`
  of pure bandwidth that our fused path deletes; relatively larger for the small workloads.

---

## 4. Constraints

- **Dtype:** all fp32 in and out. Output tensors must be fp32 with reference shapes.
- **Triton-only compute:** the GEMM must run in a Triton kernel. PyTorch may allocate outputs,
  reshape/view, and launch. No `torch.matmul`/`F.linear`/cuBLAS as the computational path, no
  CPU/NumPy, no CUDA C extension, no "if Triton fails use torch" fallback. A failed Triton
  implementation is invalid, not fallback-able.
- **Arch:** `sm_80`. No `tcgen05`/TMEM/wgmma/Hopper-Blackwell features. Ampere `mma` (incl.
  TF32 tensor path) is the ceiling. `tl.dot` with fp32 inputs is the vehicle.
- **Evaluation:** only `./scripts/evaluate_candidate.sh feedback cNNN`. Five workloads = one
  candidate evaluation. Budget 100 evals; token soft 1.0M / hard 1.2M. Ranking = geomean speedup,
  every selected workload must pass correctness. No direct CUDA/profiler/nvidia-smi runs.

---

## 5. Numerical risks (decisive for design)

Random `~N(0,1)` inputs; each output element is a length-3072 dot product, so
`Var ≈ 3072`, i.e. output magnitude `≈ 55`. The pass rule (standard) is
`|out - ref| ≤ atol + rtol·|ref|` for ≥ 98% of elements. Effective budget for `|ref| ≈ 55`:
`≈ 0.002 + 1e-5·55 ≈ 0.00255`.

Precision options for `tl.dot` on fp32 inputs (Ampere):

1. **Plain TF32 (`input_precision="tf32"`)** — operands truncated to 10-bit mantissa
   (`~5e-4` relative). Error of a 3072-term dot ≈ `sqrt(3072)·(~7e-4)·O(1) ≈ 0.04`, i.e.
   **~0.04 absolute — ~16× over the 0.00255 budget on essentially every element.**
   → **Rejected.** Fast (~0.7 ms WL1) but fails correctness badly.

2. **3×TF32 (`input_precision="tf32x3"`)** — decomposes each fp32 operand into 3 TF32 terms;
   accuracy ≈ near-fp32 (relative `~1e-6`, i.e. absolute `~5e-5` at magnitude 55). **Comfortably
   inside 0.00255.** Throughput ≈ 1/3 of TF32 tensor ≈ 52 TFLOPS eff → clearly beats fp32 SIMT.
   → **Primary choice.** Best accuracy/speed trade-off.

3. **IEEE fp32 (`input_precision="ieee"`)** — true fp32 via CUDA-core FMA (no fp32 tensor core on
   Ampere), `~19.5` TFLOPS. Exactly matches the reference numerically; same speed class as cuBLAS
   fp32. Only wins from removing the `cat`. → **Fallback precision** if `tf32x3` is unavailable in
   the installed Triton, or as an accuracy-safety candidate.

**Why the reference must be true fp32 (not TF32):** with `atol ≈ 0.002` on magnitude-55 outputs,
no tensor-core approximation could reproduce a TF32 reference to that tolerance. The tight atol
is only self-consistent if the reference is computed in true fp32 and the atol is the *budget*
granted to approximate implementations. This is exactly the margin `tf32x3` needs and plain TF32
blows. It also implies our `tf32x3` result (≈ true fp32) will land near the reference — safe.

**Dependency / risk:** if, contrary to the above, the harness reference secretly ran with
`allow_tf32=True` (TF32), then a near-fp32 `tf32x3` result could differ from it by `~0.04` and
fail the 98% match. The tolerance analysis argues strongly against this, but the **first
evaluation of c001 is the check**: if `tf32x3` fails correctness, that scenario (or a strides/
shape bug) is the cause and we revisit. Accumulation is fp32 in all three modes (Triton accumulates
`tl.dot` in fp32), so accumulation order is not a concern.

Other numerical notes:
- No bias, no activation, no reductions across tokens → no additional cancellation beyond the dot.
- 2% mismatch slack (`0.98`) gives margin, but `tf32x3` should pass ~100% of elements.
- Keep everything fp32; do not down-cast the accumulator or outputs.

---

## 6. Triton design space

### 6.1 Kernel structure — tiled GEMM (`Y = X @ W.T`)
Standard 2-D tiled matmul:
- Program grid over `(M, N)` tiles: `BLOCK_M × BLOCK_N` output tile, reduce over `K` in
  `BLOCK_K` chunks; accumulate in fp32; `tl.dot(x_tile, w_tile, acc, input_precision=...)`.
- `W` read as `W[n, k]` so that `Y = X·W^T`; equivalently load `W` transposed tiles. Choose the
  load pattern (via strides) that keeps `W` accesses coalesced; `W` is `[H, H]` contiguous.
- `L2`-friendly tile scheduling (grouped-M / "super-grouping" of program ids) to maximize `W` and
  `X` reuse — important because `W` (37.7 MB) barely fits L2.
- Boundary masking on `M` (streams like `M=77` are not multiples of `BLOCK_M`); `N=K=3072` is a
  multiple of common block sizes (128/64), so N/K masking may be avoidable for speed.

### 6.2 Handling the two streams (launch strategy)
- **Option A (baseline, simplest): two kernel launches**, one per stream, each over its flattened
  `[M_side, H]`, writing directly into `processed_encoder` / `processed_hidden`. `W` is reused hot
  across both launches. Launch overhead is negligible vs GEMM. **Preferred starting point.**
- **Option B: one launch over a "virtual" concat.** A single grid of `M_total` row-tiles, where a
  tile reads from encoder or image tensor depending on the global row index, and writes to the
  matching output. Saves one launch but needs per-tile source selection (two base pointers +
  branch or index math). Marginal benefit; consider only if launch overhead shows up for the
  small workloads.
- **Option C: batched/grouped GEMM.** Treat `(B, stream)` as groups. Overkill here since flatten
  already collapses `B` into `M` for contiguous inputs; keep in reserve for non-contiguous cases.

For all options, avoid materializing the concatenation and avoid slicing copies.

### 6.3 Precision knob
- Start with `input_precision="tf32x3"` (accuracy-safe, tensor-core speed).
- Keep `input_precision="ieee"` as an alternative candidate for an accuracy floor / if `tf32x3`
  is missing in the installed Triton.
- Never use plain `"tf32"` for the committed solution (fails tolerance) — could be used only as a
  throwaway diagnostic to confirm the accuracy/speed trade-off, but not submitted.

### 6.4 Tiling / occupancy search axes (for later candidates, not this draft)
- `BLOCK_M ∈ {64, 128, 256}`, `BLOCK_N ∈ {64, 128, 256}`, `BLOCK_K ∈ {32, 64, 128}`.
- `num_warps ∈ {4, 8}`, `num_stages ∈ {2, 3, 4, 5}` (software pipelining of K-loop; watch shared
  memory pressure with fp32 tiles on `sm_80`, which has 164 KB/SM configurable smem).
- Grouped-M scheduling `GROUP_M ∈ {4, 8}` for L2 reuse.
- Consider a **thin-M specialization** for tiny encoder streams (`M=77`): small `BLOCK_M` (e.g. 64)
  to avoid wasting rows, or fold the small stream differently. Possibly different configs per
  stream (encoder vs image) since their `M` differ by 1–2 orders of magnitude.
- `tf32x3` roughly triples the effective K-work; `BLOCK_K` and pipelining interact with that.

### 6.5 What NOT to do
- No split-K unless a workload proves K-parallelism starved (K=3072 with these M/N is fine and
  split-K needs an fp32 atomic/second-pass reduction that risks accuracy/complexity).
- No fp16/bf16 down-cast (blows tolerance).
- No cuBLAS/`torch.matmul` compute path.

---

## 7. Baseline cost model & expected speedup

Reference cost ≈ `cat` (bandwidth) + `matmul` (true-fp32 SIMT, ~19.5 TFLOPS) + `split` (views,
≈ free). Our fused path ≈ `matmul` (`tf32x3`, ~52 TFLOPS eff) with **no** `cat`, writing outputs
directly.

- Matmul speedup ceiling ≈ `52 / 19.5 ≈ 2.7×` (if reference is true-fp32 SIMT and cuBLAS is not
  itself tensor-accelerated). Plus the deleted `cat` traffic (a few percent on large WL, more on
  the small WL where the GEMM is short and copies are relatively bigger).
- Realistic geomean target: meaningful (> ~1.5×) if the reference is true fp32; the exact number
  depends on how close cuBLAS fp32 SIMT is to peak and on our tiling efficiency at small M.
- **Falsifiable checkpoint:** if c001 (`tf32x3`, fused) is *slower* than the reference, the most
  likely explanation is that the reference matmul is already tensor-accelerated (TF32) — in which
  case our accuracy would also be at risk and we would re-examine assumptions rather than chase
  tiling.

---

## 8. Validation strategy

Constraints: no direct CUDA/profiler/nvidia-smi/alternate harness; the trusted evaluator is the
only correctness+timing oracle, invoked as `./scripts/evaluate_candidate.sh feedback cNNN` over
all five fixed workloads (= one evaluation).

Layered validation:
1. **Algebraic equivalence (static, this draft):** concat→matmul→split ≡ per-stream matmul; row
   order preserved by C-contiguous flatten/reshape; two outputs of exact reference shape/dtype.
   Documented above (§2).
2. **Shape/stride audit before each eval:** confirm handling of `[B, S, H]` contiguous inputs,
   flatten to `[B*S, H]`, reshape outputs back to `[B, S, H]`; handle `W` as `[H, H]` with correct
   transpose semantics; guard `T=0`/`I=0` (not in feedback set but cheap to guard). Prefer passing
   strides to the kernel over forcing `.contiguous()` copies; only copy if a stream is
   non-contiguous (feedback inputs are expected contiguous).
3. **Numerical margin reasoning (§5):** `tf32x3` error `~5e-5` ≪ budget `~0.00255`; expect ~100%
   element match. c001 doubles as the empirical confirmation that the reference is true fp32.
4. **Evaluator-driven loop:** each immutable candidate `cNNN` evaluated once over the 5 workloads;
   record per-workload correctness (must pass) and speedup, geomean, decision, cumulative eval
   count, parent, source hash, hypothesis, skill usage in `candidates.jsonl` (append-only).
5. **Regression discipline:** change one axis per candidate (precision → then tiling → then launch
   strategy) so each eval attributes cause cleanly; never reuse an id for changed source.
6. **Accuracy safety net:** keep an `ieee`-precision variant ready; if any workload fails
   correctness under `tf32x3`, evaluate the `ieee` variant to isolate accuracy vs. bug.

Success criteria: all 5 feedback workloads pass correctness, geomean speedup > 1.0 and improving,
convergence when tiling/launch tweaks stop yielding gains → then `SEARCH_COMPLETE`. Final 16-workload
eval is operator-only.

---

## 9. Candidate roadmap (sketch — full executable plan goes in docs/plan.md next)

- **c001** — Fused two-launch tiled Triton GEMM (`Y = X·W^T`), `input_precision="tf32x3"`, no
  cat/split, conservative tiling (e.g. `BM=128, BN=128, BK=32/64`, `warps=4`, `stages=3`),
  grouped-M scheduling. Establishes correctness + baseline speedup and validates the true-fp32
  reference assumption.
- **c002+** — Tiling/occupancy sweep around the winner (`BLOCK_*`, `num_warps`, `num_stages`,
  `GROUP_M`); possibly per-stream configs for the thin encoder (`M=77`) vs large image streams.
- **later** — launch-strategy alternative (single virtual-concat launch) if small-WL launch
  overhead matters; `ieee` accuracy variant kept as safety; consider `BLOCK_K`/pipeline tuning
  specific to the `tf32x3` 3× K-cost.
- Stop when improvement converges; write `SEARCH_COMPLETE`.

---

## 10. Skill applicability

- `KernelWiki` — scoped to Blackwell (SM100/B200) and Hopper (SM90/H100). This target is **A800 /
  sm_80 (Ampere)**; the Blackwell/Hopper-specific techniques (tcgen05, TMEM, wgmma, CLC, 2-SM,
  NVFP4, FA4) do not apply. Not invoked for this draft.
- `ncu-report-skill` — profiling on **B200 / sm_100**, and direct profiler execution is disallowed
  here anyway. Not applicable.
- Skill usage for this task: **none / not applicable** (to be recorded as such in candidate rows).

---

## 11. Open questions / risks to watch

1. **Reference precision** (true fp32 vs TF32) — governs both our speedup ceiling and accuracy
   safety. Strongly argued to be true fp32 by the tight atol; confirmed empirically by c001.
2. **`tf32x3` availability** in the installed Triton — if absent, drop to `ieee` (still removes
   cat; smaller but positive win) and note it.
3. **Small-M efficiency** — `M=77` encoder streams (WL3/WL5) may underutilize; watch whether they
   drag geomean and whether a thin-M config/launch helps.
4. **L2 pressure from `W` (37.7 MB ≈ L2)** — scheduling for `W` reuse across `M` tiles is
   important; grouped-M and large `BLOCK_M`.
5. **Contiguity assumptions** — if any feedback input is non-contiguous, handle via strides rather
   than a copy that reintroduces bandwidth cost.
6. **Two outputs, exact shapes/dtype** — must return `(processed_encoder [B,T,H], processed_hidden
   [B,I,H])` fp32, order and slicing semantics matching the reference exactly.
