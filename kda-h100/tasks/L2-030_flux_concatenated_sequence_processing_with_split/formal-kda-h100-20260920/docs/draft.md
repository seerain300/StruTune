# Draft — L2/030 Flux Concatenated Sequence Processing With Split

Task: optimize `sol_execbench` task `L2/030_flux_concatenated_sequence_processing_with_split`
on H100 (`sm_90`). Submission is `solution/solution.py` exposing `run(...)`, primary
implementation in Triton, no Torch/CPU/NumPy/CUDA-extension computational fallback.

This is the **draft only**. No `docs/plan.md` and no solution code are produced this turn.

---

## 1. The operation

Reference (`task/definition.json`), fp32 throughout, `@torch.no_grad()`:

```python
def run(hidden_states, encoder_hidden_states, process_weight):
    text_seq_len = encoder_hidden_states.shape[1]      # T
    img_seq_len  = hidden_states.shape[1]              # I
    concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
    processed    = torch.matmul(concatenated, process_weight.t())            # [B, T+I, H] @ [H, H]
    processed_encoder = processed[:, :text_seq_len, :]  # [B, T, H]
    processed_hidden  = processed[:, text_seq_len:, :]  # [B, I, H]
    return processed_encoder, processed_hidden
```

Shapes / dtypes:
- `hidden_states`        `[B, I, H]`, fp32   (image latent)
- `encoder_hidden_states` `[B, T, H]`, fp32   (text conditioning)
- `process_weight`       `[H, H]`, fp32       (H = 3072, constant)
- outputs `processed_encoder [B, T, H]`, `processed_hidden [B, I, H]`, both fp32.

### 1.1 Key algebraic simplification (the whole trick)

The linear projection is applied **independently to every sequence position**. Concatenation
along the sequence dim and the subsequent split therefore do **not** couple any rows:

```
processed[b, s, :] = concatenated[b, s, :] @ W.T        # row-wise, s independent
```

Hence, exactly and identically to the reference:

```
processed_encoder = encoder_hidden_states @ W.T         # [B, T, H] @ [H, H].T
processed_hidden  = hidden_states         @ W.T         # [B, I, H] @ [H, H].T
```

The `torch.cat` produces a fresh `[B, T+I, H]` buffer purely so a single `matmul` can be
issued; **it contributes no arithmetic and can be eliminated**. This matches the task
description ("represents a memory bandwidth bottleneck") — the concat is redundant traffic.
The split is a pair of views (free). So the optimized operation is simply **two independent
row-major GEMMs that share the same weight `W`**, writing directly into the two output tensors.

Because `[B, S, H]` inputs are contiguous, `reshape(-1, H)` is a zero-copy view, so each GEMM
is a plain 2-D `A[M, K] @ Bmat[K, N]` with:
- `K = N = H = 3072` (fixed),
- `Bmat = W.T` (i.e. `Bmat[k, n] = W[n, k]`), a `[K, N]` matrix accessed transposed from a
  row-major `[N, K]` `W`,
- `M_text = B·T` for the encoder GEMM, `M_img = B·I` for the image GEMM.

`W` is symmetric in *shape* (`[3072, 3072]`) so the transposed access is a stride swap only.

---

## 2. Workload inventory (`task/feedback_workloads.jsonl`)

The file contains **16** workloads. (Note: `README.md` says "five fixed feedback workloads",
which is stale; `TASK.md` says feedback runs the FULL set and the final is a 16-workload run.
The JSONL file is authoritative: 16 rows, and the feedback set equals the final set.)
`H = 3072`; per-workload total FLOPs `= 2·M_tot·H² = M_tot · 1.887e7`.

| # | B | T | I | M_text=B·T | M_img=B·I | M_tot | GFLOP | atol |
|---|---|---|---|-----------|-----------|-------|-------|------|
| 1 | 2 | 128 | 256 | 256 | 512 | 768 | 14.5 | 0.002 |
| 2 | 1 | 256 | 512 | 256 | 512 | 768 | 14.5 | 0.002 |
| 3 | 2 | 131 | 293 | 262 | 586 | 848 | 16.0 | 0.002 |
| 4 | 1 | 128 | 256 | 128 | 256 | 384 | 7.2 | 0.0019 |
| 5 | 1 | 77 | 4096 | 77 | 4096 | 4173 | 78.8 | 0.002 |
| 6 | 1 | 512 | 2048 | 512 | 2048 | 2560 | 48.3 | 0.002 |
| 7 | 1 | 77 | 1024 | 77 | 1024 | 1101 | 20.8 | 0.0018 |
| 8 | 4 | 77 | 1024 | 308 | 4096 | 4404 | 83.1 | 0.002 |
| 9 | 1 | 1024 | 4096 | 1024 | 4096 | 5120 | 96.6 | 0.0021 |
| 10 | 16 | 128 | 256 | 2048 | 4096 | 6144 | 116 | 0.0021 |
| 11 | 32 | 256 | 512 | 8192 | 16384 | 24576 | 464 | 0.0021 |
| 12 | 4 | 256 | 512 | 1024 | 2048 | 3072 | 58.0 | 0.002 |
| 13 | 2 | 1423 | 1489 | 2846 | 2978 | 5824 | 110 | 0.0021 |
| 14 | 1 | 1087 | 1163 | 1087 | 1163 | 2250 | 42.5 | 0.002 |
| 15 | 4 | 211 | 449 | 844 | 1796 | 2640 | 49.8 | 0.002 |
| 16 | 4 | 128 | 256 | 512 | 1024 | 1536 | 29.0 | 0.002 |

Observations:
- **`M` spans two orders of magnitude**: from `M_text=77` (tiny, GPU-underutilizing) up to
  `M_img=16384` (large, strongly compute-bound). A single fixed tile shape will not be optimal
  everywhere → heuristic/autotune over `M` is warranted.
- `N=K=3072 = 2^10·3` is divisible by 32/64/128/256/384/512 → **no N/K masking needed** for the
  usual block sizes; only the `M` dimension needs masking (many `M` are not multiples of 128,
  e.g. 77, 262, 308, 844, 1087, 2846).
- Workload #11 alone (M_tot=24576, 464 GFLOP) dominates absolute runtime; #5/#8/#9/#10/#13 are
  the next tier. Small ones (#4, #1, #2, #3) matter for geomean but are cheap in wall time.
- Several workloads pair a **tiny text GEMM with a large image GEMM** (#5: M_text=77 vs
  M_img=4096; #8: 308 vs 4096) — the text GEMM there is essentially launch/weight-load bound.

---

## 3. Constraints (from CLAUDE.md / TASK.md)

- **Triton primary**; Torch only for metadata/launch plumbing (shape, reshape views, output
  allocation, grid computation). No Torch/CPU/NumPy/CUDA-extension computational fallback; a
  failing Triton kernel is invalid and must not be papered over with `torch.matmul`.
- Candidates are **immutable and sequential** (`c001`, `c002`, …). Any meaningful source /
  config / launch change requires a new ID; never reuse an ID or rewrite earlier records.
- Evaluate **only** via `./scripts/evaluate_candidate.sh feedback <id>`; the full 16-workload
  set = one candidate evaluation. Budget: 100 evaluations. Token soft/normal/hard limits:
  9M / 10M / 11M.
- `final` is operator-only.
- Profiling only via `./scripts/ncu_profile.sh …` (ncu-report-skill workflow), **never**
  concurrently with an evaluation (foreign process on the locked GPU → discarded measurement,
  return code 3, wasted budget). Never call `ncu`/`nvidia-smi`/CUDA directly.
- Isolation: work only in this workspace; only KernelWiki + ncu-report-skill as external
  knowledge; no subagents/MCP/web.

---

## 4. Numerical analysis — precision is the pivotal design choice

### 4.1 What the reference computes

PyTorch fp32 `matmul` with (default) `allow_tf32 = False` performs a **true IEEE fp32**
matmul (cuBLAS CUDA-core path, fp32 accumulation). The evaluator compares our output to
*this* fp32 reference with `max_atol ∈ [0.0018, 0.0021]`, `max_rtol = 1e-5`,
`required_match_ratio = 0.98` (≤2% of elements may exceed tolerance). The fact that the
tolerance is a *finite* `~2e-3` (not near machine-eps) is strong evidence the reference is
true fp32 and that the author intends lower-precision-but-tolerant implementations to pass.
(If the reference were itself TF32, a 2e-3 tolerance would be near-impossible to hit for any
non-bit-identical implementation — so we can rule that out.)

### 4.2 Output magnitude and error budget

For random inputs (`~N(0,1)`-scale), an output element is `sum_{k=1..3072} A[m,k]·W[n,k]`,
i.e. a sum of 3072 zero-mean unit-ish products ⇒ magnitude `≈ sqrt(3072) ≈ 55`. With
`atol ≈ 2e-3` and `rtol = 1e-5`, the allowed error per element is
`≈ 2e-3 + 1e-5·55 ≈ 2.55e-3`, and the absolute `atol` term dominates.

### 4.3 `tl.dot` precision modes on Hopper

Triton's `tl.dot(a, b, input_precision=…)` offers three fp32 paths:

| mode | mechanism | throughput (H100, rough) | accuracy (K=3072, out ~55) |
|------|-----------|--------------------------|-----------------------------|
| `"tf32"` | 1× TF32 tensor-core MMA (10-bit mantissa) | ~495 TFLOPS | per-product rel err ~2⁻¹¹ ⇒ **abs err ~0.027** |
| `"tf32x3"` | 3× TF32 MMA (hi·hi + hi·lo + lo·hi) | ~150 TFLOPS eff. | per-product rel err ~2⁻²¹ ⇒ **abs err ~3e-5** |
| `"ieee"` | fp32 FMA on CUDA cores | ~67 TFLOPS | matches reference to fp32 eps |

Estimates:
- **`tf32` fails**: abs error ~0.027 ≫ 2.55e-3. Fraction of elements within tolerance would be
  ~8% ≪ 98%. Unusable.
- **`tf32x3` passes with wide margin**: it keeps the two leading TF32 limbs of each operand and
  drops only the `lo·lo` cross term (~2⁻²²·|prod|), giving per-product relative error ~2⁻²¹.
  Accumulated absolute error ≈ `2⁻²¹ · sqrt(3072) · 1 ≈ 2.7e-5`, i.e. a **~70–90× margin** below
  even the tightest `atol=0.0018`. The 0.98 match-ratio adds further slack for stray outliers.
- **`ieee`** is safest numerically but offers **no compute speedup** over the fp32 reference
  (same CUDA-core FMA path); its only win vs baseline would be eliminating the concat.

**Conclusion: `tf32x3` is the target compute path** — it is the only tensor-core path that is
both accuracy-safe against a true-fp32 reference and materially faster than the fp32 baseline.
`ieee` is the conservative fallback-within-Triton if `tf32x3` unexpectedly misses on some
boundary workload (it will still be a valid Triton implementation, just slower).

The precision margin is essentially **shape-independent** here: `K=3072` is fixed for every
workload, and relative error scales with `sqrt(K)` identically across all 16 cases, so a
`tf32x3` result that passes one workload should pass all of them. This must still be
**confirmed empirically** by the evaluator (correctness is checked on every feedback run).

---

## 5. Performance analysis (roofline / bottleneck)

H100 SXM ballpark: fp32 (CUDA core) ≈ 67 TFLOPS; TF32 tensor core ≈ 495 TFLOPS
(⇒ tf32x3 ≈ 150 TFLOPS effective); HBM3 BW ≈ 3.35 TB/s; L2 ≈ 50 MB.

- **Weight fits in L2.** `W` is `3072²·4 B = 37.7 MB < 50 MB`. Across the two GEMMs (and across
  M-tiles) `W` is loaded once from HBM and thereafter served from L2. This blunts the classic
  "read the weight twice" penalty of splitting into two launches, and makes swizzled tile
  scheduling for L2 reuse worthwhile.
- **Arithmetic-intensity knee.** fp32 roofline knee ≈ 67e12/3.35e12 ≈ 20 FLOP/B; tf32x3 knee
  ≈ 150e12/3.35e12 ≈ 45 FLOP/B. A GEMM's AI ≈ `2·M·N·K / (4·(M·K + K·N + M·N))`. With N=K=3072:
  - Large `M` (≥~2048): AI is high (hundreds) ⇒ **compute-bound**; tf32x3 ~2× over fp32 baseline.
  - Tiny `M` (≤~256, e.g. text GEMM M=77/128/262): AI ≈ tens, below the tf32x3 knee ⇒
    **memory-bound on the 37.7 MB weight read** (~11 µs floor at 3.35 TB/s, shared by baseline).
    Compute precision barely matters here; the win comes from removing the concat and from not
    wasting a whole second launch on a tiny tile.
- **Concat overhead removed.** Baseline `torch.cat` writes+reads a `[B,(T+I),H]·4 B` buffer:
  e.g. workload #11 = 24576·3072·4 ≈ 302 MB write (+read) ≈ tens of µs of pure overhead that we
  delete entirely.
- **Where wall-clock lives.** Workload #11 (464 GFLOP) is ~3.1 ms at tf32x3 vs ~6.9 ms fp32; it
  dwarfs the rest. #5/#8/#9/#10/#13 are the second tier. The small workloads are cheap in wall
  time but count equally in the **geomean**, so we must not *regress* them (launch overhead, tiny
  tiles) even though their absolute cost is low.

Expected outcome: compute-bound workloads ~1.8–2.5× (tf32x3 + no concat); memory/launch-bound
tiny workloads ~1.0–1.4× (concat removal + fused launch). Geomean plausibly ~1.4–2.0×,
contingent on tiling and small-M handling.

---

## 6. Triton design space

### 6.1 GEMM formulation
Standard tiled `A[M,K] @ Bmat[K,N]` with `C[M,N]` output, fp32 accumulator, fp32 store:
- `A` = reshaped input `[M, H]`, row-major (`stride_am=H, stride_ak=1`).
- `Bmat = W.T`: read `W` `[N, K]` row-major with `stride_bk=1, stride_bn=K` (transposed access;
  contiguous along `k`, which is the reduction dim — good for the inner load).
- `C` = output `[M, H]`, row-major.
- Compute: `acc += tl.dot(a_tile, b_tile, input_precision="tf32x3")` over K-tiles.

### 6.2 Launch strategy — three candidates to explore
1. **Two independent launches** (simplest baseline candidate): one GEMM for encoder, one for
   image, sharing `W` (hot in L2). Clean, exact outputs, no concat/split. Downside: doubles
   launch overhead and wastes a launch on tiny text GEMMs (M=77/128).
2. **Single fused two-region launch** (preferred): one grid whose program-id maps into either
   the *encoder region* (rows `[0, M_text)`, reading `encoder`, writing `processed_encoder`) or
   the *image region* (rows `[0, M_img)`, reading `hidden`, writing `processed_hidden`). Each
   region's M is padded up to a whole number of `BLOCK_M` tiles; a per-tile branch selects the
   base pointers. Removes the concat, uses one launch, keeps `W` hot across both regions. This
   is a 2-group grouped-GEMM sharing `B=W.T`, `N`, `K`.
3. **Persistent / swizzled scheduler** (for the large workloads, esp. #11): persistent grid of
   `~#SMs` programs iterating over a swizzled tile list for L2 reuse of `W` and better tail
   behaviour. Consider only if profiling shows tail-effect / L2-miss on the big cases.

### 6.3 Tiling / config knobs
- `BLOCK_M ∈ {32, 64, 128, 256}`, `BLOCK_N ∈ {64, 128, 256}`, `BLOCK_K ∈ {32, 64, 128}`.
  Since `N=K=3072` is divisible by all of these, only `M` gets masked.
- `num_warps ∈ {4, 8}`, `num_stages ∈ {3, 4, 5}` (tf32x3 issues 3 MMAs per K-step, so enough
  stages to hide TMA/global loads matters).
- Group-M swizzle (`GROUP_M`) for L2 locality on `W` and `A`.
- Because `M` varies so widely, prefer **`triton.autotune`** keyed on `M` (and/or a small
  hand-tuned M-bucketed heuristic) so tiny-M and huge-M pick different tiles. Autotune configs
  are part of the immutable source — fine, but each config-set change = new candidate ID.
- Accumulate in fp32; store fp32 (output dtype must equal reference).

### 6.4 Plumbing / correctness details
- `reshape(-1, H)` on contiguous `[B,S,H]` is a free view; outputs are contiguous `[B,S,H]`.
- Guard: `.contiguous()` on inputs/weight is a no-op copy when already contiguous, but adds a
  copy if not — prefer passing strides to the kernel over forcing `.contiguous()`; verify the
  evaluator supplies contiguous tensors before relying on it.
- Handle `M_text` or `M_img` possibly `0`? Not in these workloads (all T,I ≥ 77), but a guard is
  cheap.
- Mask only the M dimension: `offs_m < M`. K/N loops are exact for the chosen blocks.

---

## 7. Risks & mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `tf32x3` misses tolerance on some boundary workload | Low (70–90× margin, K fixed) | Evaluator checks every run; if any fail, fall back to `ieee` for that path (still Triton, still valid) or investigate per-workload. |
| Small-M workloads (77/128) regress due to 2× launch overhead | Medium | Fused single-launch (design 6.2 #2); pick small `BLOCK_M` with masking; avoid a dedicated tiny launch. |
| Large-M #11 tail effect / L2 thrash | Medium | Group-M swizzle; consider persistent scheduler; verify with ncu (not during eval). |
| One fixed tile config is suboptimal across 2-orders-of-magnitude M | High | Autotune / M-bucketed heuristic. |
| Non-contiguous inputs break reshape assumption | Low | Pass strides; or `.reshape` (copies only if needed) as metadata plumbing. |
| Accidental use of `input_precision="tf32"` (fast but wrong) | — | Explicitly set `"tf32x3"`; never leave it defaulted. |
| Wasting eval budget on profiling collisions | — | Never profile while an eval runs; strictly serialize (rule in CLAUDE.md). |

---

## 8. Validation strategy

1. **A-priori numerical reasoning** (done, §4): `tf32x3` has a ~70–90× absolute-error margin
   vs the tightest `atol`, shape-independent because `K=3072` is fixed. This gives high
   confidence before spending any evaluation.
2. **Evaluator is the sole correctness+timing oracle.** Every `feedback` run reports per-workload
   pass/fail (within `atol/rtol` at 0.98 match ratio) and timing over the full 16-workload set,
   counting as one candidate evaluation. Direct CUDA / alternate harnesses are forbidden, so I
   will not attempt a local numerical check.
3. **Candidate progression (to be detailed in `docs/plan.md` next turn, not now):**
   - `c001` — correctness-first, simplest correct Triton implementation (two straightforward
     tiled GEMMs, `tf32x3`, no concat). Establishes a passing baseline and a speedup datum.
   - subsequent candidates — fused single-launch, autotune/M-bucketing, swizzle/persistent for
     the large workloads; each an immutable new ID with recorded parent/hash/hypothesis/result.
4. **Profiling** only via `./scripts/ncu_profile.sh` following the ncu-report-skill workflow,
   strictly serialized with evaluations, used to confirm compute- vs memory-bound behaviour and
   to guide tile/stage choices for the dominant workloads (#11, #5, #8, #9, #10, #13).
5. **Stopping / convergence:** record geomean per candidate in `candidates.jsonl`; stop at budget
   or when improvement converges; write `SEARCH_COMPLETE` with a reason; never run `final`
   without operator approval.

---

## 9. Open questions to resolve empirically

- Actual input distribution/scale (`type: "random"`) — affects output magnitude and thus the
  precise (not the qualitative) error margin. The `tf32x3` margin is large enough that this
  should not change the decision, but confirm on the first `c001` eval.
- Are the evaluator-supplied tensors guaranteed contiguous? (Governs reshape-vs-copy.)
- Does the fused single-launch actually beat two launches on the tiny-M workloads, or is the
  branch overhead not worth it? (Measure `c00x` two-launch vs fused.)
- Best tile/stage configuration for the compute-bound tier and whether a persistent/swizzled
  scheduler helps workload #11 specifically (ncu-guided).
- Confirm `tl.dot(..., input_precision="tf32x3")` is available and lowers as expected on this
  Triton build for `sm_90`; if not, evaluate `"ieee"` (correctness-safe, concat-removal win only).

---

### Summary of the plan-of-attack (analysis-level, not the executable plan)
The operation reduces exactly to **two weight-shared row-major GEMMs `A @ W.T`** with the
concat eliminated and the split replaced by writing to two output tensors. The decisive lever
is **precision**: `tf32x3` tensor-core matmul is accuracy-safe against the true-fp32 reference
(~70–90× margin) and ~2× faster on compute-bound workloads, while concat removal helps
everywhere and especially the memory/launch-bound tiny-M cases. Start correctness-first, then
fuse to a single launch and tune tiling/scheduling for the M range, validating every step
through the evaluator.
