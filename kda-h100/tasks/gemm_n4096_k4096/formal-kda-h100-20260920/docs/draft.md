# Draft — `gemm_n4096_k4096` (FlashInfer, H100 / sm_90)

Status: draft only. No `plan.md`, no solution code created this turn.

---

## 1. Operation and exact semantics

From `task/definition.json`:

```python
def run(A, B):
    C = torch.matmul(A, B.T)   # C = A @ B.T
    return C
```

- `A`: shape `[M, K]`, dtype `float16`, row-major (K contiguous).
- `B`: shape `[N, K]`, dtype `float16`, row-major (K contiguous).
- `C`: shape `[M, N]`, dtype `float16`.
- Constants: `N = 4096`, `K = 4096`. Only `M` varies.
- Provenance: Llama 3.1 8B `attn.o_proj` (hidden 4096 → 4096). `M` is the token/row count.

Mathematically `C[m,n] = sum_k A[m,k] * B[n,k]`. Both operands are indexed along the
contraction dimension `K` on their **contiguous** axis. This is the "NT" GEMM layout
(A: M×K row-major, B: N×K row-major, B logically transposed). NT is the *most
favorable* layout for tensor-core GEMM: both A and B tiles are loaded coalesced along K,
and no physical transpose/copy is needed (`B.T` is a stride-swap view — cuBLAS handles it
with a transpose flag, so the reference pays no copy cost either). Our Triton kernel will
read `B` directly in `[N, K]` layout and contract over K.

The reference is `torch.matmul` → cuBLAS(Lt) `hgemm` with FP32 accumulation and FP16
output. **The speedup metric is `t_reference / t_candidate`, i.e. we must beat cuBLAS on
H100 for every shape while staying correct.** That is a high bar for large, compute-bound
M and a genuine opportunity for small, memory/occupancy-bound M.

---

## 2. Workload analysis (this is where the geomean lives)

`task/feedback_workloads.jsonl` has **43 workloads**, all identical except `M`:

- A dense small-M sweep: `M ∈ {256, 248, 240, …, 16, 8}` in steps of 8, then
  `{24, 16, 8, 4, 2, 1}`, plus scattered small values `{7, 35, 70, 15}`.
- Only **4 large** values: `M ∈ {972, 2053, 2379, 8192}`.

Counting: **~39 of 43 workloads have M ≤ 256**; only 4 exceed 972. Because the ranking
metric is the **geometric mean of per-workload speedups**, the small-M regime dominates
the score. Optimizing M=8192 in isolation cannot move the geomean much; consistent wins
across the many small-M shapes can.

### 2.1 Roofline / regime crossover (H100 SXM assumed)

Reference numbers used for modeling (H100 SXM5): 132 SMs, ~3.35 TB/s HBM3, **50 MB L2**,
~989 TFLOPS FP16 tensor (FP32-accumulate), 228 KB smem/SM.

- FLOPs = `2·M·N·K = 2·M·4096·4096 ≈ M · 3.36e7`.
- Bytes to read B once = `N·K·2 = 33.5 MB` (independent of M). A and C are small for small M.
- Time to stream B once from HBM ≈ `33.5e6 / 3.35e12 ≈ 10 µs`.
- Compute time ≈ `M·3.36e7 / 989e12 ≈ M · 34 ns` (ideal).
- **Crossover** (compute-time = B-stream-time): `M·34 ns = 10 µs ⇒ M ≈ 290`.

So essentially **every feedback workload except {972, 2053, 2379, 8192} is
memory/occupancy-bound**, and the four large ones are compute-bound.

### 2.2 L2 residency changes the small-M picture

`B` is 33.5 MB, which **fits in H100's 50 MB L2**. The evaluator runs warmup 2 + 10 timed
iterations on the *same* fixed A/B tensors, so after the first iteration B (and small A)
are L2-resident. For small M the timed iterations are therefore **L2-bandwidth bound**,
not HBM-bound — B is re-read from L2 (~multiple TB/s, higher than HBM) each call. The
practical consequences:

- The floor for small M is a few µs, set by (a) L2→SM bandwidth and (b) kernel launch /
  wave-quantization overhead — **not** the 10 µs HBM figure.
- Winning small-M requires **saturating L2 read bandwidth**, which requires **enough
  concurrently-resident CTAs** across the 132 SMs, and **minimal launch/tail overhead**.

### 2.3 The core small-M problem: not enough tiles → low occupancy

With a single M-tile (M ≤ BLOCK_M) the CTA grid is just the number of N-tiles:

| BLOCK_N | N-tiles (N=4096) | CTAs (1 M-tile) | Occupancy vs 132 SMs |
|--------:|-----------------:|----------------:|---------------------:|
| 256     | 16               | 16              | 12%                  |
| 128     | 32               | 32              | 24%                  |
| 64      | 64               | 64              | 48%                  |
| 32      | 128              | 128             | 97%                  |

A naïve `BLOCK_N=256` GEMM leaves 88% of SMs idle for M≤256 → the L2/HBM read of B is
issued from only 16 CTAs and cannot saturate bandwidth. **Raising the active CTA count is
the primary lever for the bulk of the score.** Two independent knobs do this:

1. **Smaller BLOCK_N** (more N-tiles). Cheap, no reduction, but small BLOCK_N lowers MMA
   efficiency (irrelevant while memory-bound) and shrinks per-CTA work.
2. **Split-K** (partition the K=4096 contraction across `SPLIT_K` CTAs, then reduce
   partials). Multiplies CTA count by `SPLIT_K`; classic answer for skinny GEMM. Adds a
   reduction cost that is negligible when the output `M·N` is tiny.

Both keep total B traffic at exactly one pass (each B element read once across all CTAs).
K=4096 = 2¹² divides cleanly by 32/64/128/256 and by split factors 2/4/8/16, so tiling and
split-K produce no ragged K remainder.

---

## 3. Constraints (from CLAUDE.md / TASK.md)

- **Triton is the primary compute path.** PyTorch only for allocation, metadata, launch
  plumbing. No Torch/CPU/NumPy/CUDA-extension fallback — a failing Triton kernel is invalid
  and must not be swapped for `torch.matmul`.
- Entry point: `solution/solution.py` exposing `run(A, B) -> C` (fp16).
- Immutable candidates `c001, c002, …`; any meaningful source/config/launch change ⇒ new
  ID; never reuse an ID for changed source.
- Evaluate only via `./scripts/evaluate_candidate.sh feedback cNNN`; one full 43-workload
  pass = **one** of the **100** candidate evaluations.
- Token budget: soft 9M / normal 10M / hard 11M.
- Profiling only via `./scripts/ncu_profile.sh` (ncu-report-skill workflow), **never
  concurrent with an evaluation** (foreign process on the locked GPU ⇒ rc=3, wasted eval).
- Final full 43-workload eval is operator-only; do not run `final` without approval.
- Do not read parent dirs, other workspaces, baselines, evaluator internals, etc.

Implications for search: eval budget is generous (100) but each eval costs wall-clock and
tokens; profile with ncu *between* evals to form hypotheses instead of brute-forcing
configs. Correctness is only observable through the evaluator, so the first candidate must
be a conservative, provably-correct GEMM to anchor the baseline geomean before tuning.

---

## 4. Numerical risks and mitigations

1. **Accumulation precision.** K=4096 terms. FP16 accumulation would drift far outside
   tolerance. Mitigation: accumulate in **FP32** (`tl.dot` default `out_dtype=float32`;
   `acc` buffer FP32), cast to FP16 only at the epilogue store. This matches cuBLAS's
   FP32-accumulate → FP16-output behavior, so rounding should agree to ~1 ULP-fp16.
2. **Split-K reduction precision.** Partial sums must be combined in **FP32**, never FP16.
   Options: (a) `tl.atomic_add` into an **FP32** workspace `C_f32[M,N]`, then a cheap cast
   kernel FP32→FP16; (b) two-stage — write `[SPLIT_K, M, N]` FP32 partials, then a
   reduction kernel sums along split and casts. Atomic-add ordering is nondeterministic but
   FP32 partial magnitudes are well-conditioned, so run-to-run differences are ≪ fp16 tol.
   Avoid FP16 atomics (precision + hardware-path pitfalls). Only enable split-K where it
   pays (small/medium M); large-M single-K path avoids atomics entirely.
3. **Output overflow.** Random ~N(0,1) inputs give `C[m,n] ~ N(0, K)`, std ≈ 64, so values
   ~±300 typically — safely inside fp16 max (65504). Even if the generator uses a larger
   scale, the reference overflows identically, so *matching* (FP32 acc + round-to-nearest
   cast) is what matters, not avoiding overflow.
4. **Masking / boundaries.** N=4096 and K=4096 are divisible by all candidate block sizes
   (no N/K tail), but **M is arbitrary** (down to 1, and odd values 7/15/35). Must mask A
   loads and C stores on the M dimension (`offs_m < M`). For split-K, ensure the K split
   covers K exactly (choose SPLIT_K | 4096, or guard the last chunk). Tiny M (1,2,4) still
   requires BLOCK_M ≥ 16 for the MMA path — padded rows are masked; wasted MMA lanes are
   free while memory-bound.
5. **NaN propagation.** With masking, out-of-range A/B loads use `other=0.0`; zero-padding
   contributes 0 to the FP32 sum — safe. Confirm no `1/0` or log/exp anywhere (pure GEMM,
   none).

Validation of these is only observable through the evaluator's correctness gate; the first
candidate is designed to be unambiguously correct so a green result confirms the numerical
approach before any aggressive tuning.

---

## 5. Triton design space

### 5.1 Baseline: classic tiled GEMM (`tl.dot`)
Grid over (M-tiles × N-tiles), K-loop with FP32 accumulator, mask on M, store FP16.
This is the safe anchor (c001). Layout note: load `a = A[offs_m, offs_k]` as
`[BLOCK_M, BLOCK_K]`, load `b = B[offs_n, offs_k]` as `[BLOCK_N, BLOCK_K]` (coalesced
along K), and compute `acc += tl.dot(a, tl.trans(b))` → `[BLOCK_M, BLOCK_N]`. On sm_90
`tl.dot` lowers to `wgmma.mma_async` with FP32 accumulate.

### 5.2 Tile-shape / scheduling knobs
- **BLOCK_M**: small (16–32) for the skinny regime (M mostly ≤256, so one M-tile);
  larger (128) only for the 4 big-M shapes.
- **BLOCK_N**: primary occupancy knob for small M (see §2.3 table) — likely 64/128 with
  split-K, or 32 without.
- **BLOCK_K**: 32/64/128; larger amortizes loop overhead, costs smem per stage.
- **num_stages** (software pipelining): 3–5 to overlap TMA/`cp.async` loads with MMA
  (KernelWiki `technique-pipeline-stages`: 695→940 TFLOPS from pipelining on the tutorial
  GEMM). Bounded by 228 KB smem.
- **num_warps**: 4–8.
- **L2 swizzle / group-M raster** (`GROUP_M`): reorder tiles to improve L2 locality for the
  large-M shapes (KernelWiki `technique-tile-scheduling`). Low value here since B fits in
  L2 anyway, but harmless and standard.

### 5.3 Split-K (the key small-M lever)
Partition K into `SPLIT_K` chunks; grid becomes (M-tiles × N-tiles × SPLIT_K). Each CTA
accumulates its K-slice in FP32 and `atomic_add`s into an FP32 workspace; a lightweight
cast pass produces FP16 C. Choose `SPLIT_K` so that
`n_mtiles · n_ntiles · SPLIT_K ≈ 132…264` (1–2 waves). For M ≤ 256 (1 M-tile) and
BLOCK_N=128 (32 N-tiles), `SPLIT_K ∈ {4, 8}` reaches 128/256 CTAs. Guard: disable split-K
(SPLIT_K=1, no atomics) once M is large enough to already fill the machine, to avoid
reduction overhead and non-determinism on the compute-bound shapes.

### 5.4 Dedicated GEMV / tiny-M path (candidate for M ≤ ~8)
For M=1..8 the MMA path wastes ≥50–94% of tensor-core rows. An alternative is a
reduction-style kernel (`tl.sum(a[:,None,:]*b, axis=…)` or per-row dot via `tl.dot` with
BLOCK_M=16 masked) tiled over N with split-K for occupancy. Given the regime is L2/BW-bound
(compute waste is hidden), a well-occupied `tl.dot` split-K kernel may already match a
hand GEMV; keep GEMV as a fallback experiment only if profiling shows the MMA tiny-M path
is launch/issue bound rather than bandwidth bound. (Must stay Triton — no cuBLAS gemv.)

### 5.5 Persistent kernel
A Hopper static-stride persistent kernel (grid = #SMs, loop over tiles) can cut launch
overhead and smooth wave quantization (KernelWiki `technique-persistent-kernels`,
`pattern-tail-effect`). CLC is SM100-only — not available on H100 — so we'd use the
software static-stride form. Marginal benefit for single-wave small-M grids; more relevant
if we fuse the split-K reduction or chase launch overhead. Treat as a later experiment.

### 5.6 Config selection: heuristic dispatch vs autotune
- **`triton.autotune`** over a curated config list keyed on an M-bucket is the simplest way
  to get near-optimal configs across the 43 shapes; downside is first-call benchmarking
  cost (lands in warmup, not timed) and less determinism.
- **Manual heuristic dispatch** on M (tiny / small / medium / large → fixed
  tile+split params) is deterministic, has no autotune overhead, and is easy to reason
  about for immutability. Because the workload set and hardware are fixed, a small hand
  table is attractive. Likely approach: start with autotune to *discover* good configs via
  profiling, then optionally freeze into a heuristic in a later candidate.

### 5.7 What we will NOT rely on
- Blackwell-only features (tcgen05, TMEM, CLC, 2-SM cooperative, `tl.dot_scaled`) — this is
  sm_90. KernelWiki Hopper-relevant items (static-stride persistence, pipelining,
  swizzled raster, split-K/stream-K) are the applicable subset.
- FP16 accumulation, FP16 atomics, any non-Triton compute fallback.

---

## 6. Performance model and per-regime targets

| Regime | M | Bottleneck | Strategy | Realistic target vs cuBLAS |
|---|---|---|---|---|
| Tiny | 1–8 | launch + L2 BW, MMA under-fill | high-occupancy split-K (or GEMV) | ≥1.0×, upside if cuBLAS uses a generic small kernel |
| Small | 16–256 | L2 read BW / occupancy | small BLOCK_M + BLOCK_N + split-K to fill 132 SMs | 1.0–1.3× (main geomean driver) |
| Medium | ~500–1000 (972) | transition | moderate tiles, maybe split-K | ~1.0× |
| Large | 2053–8192 | compute (tensor cores) | big tiles, pipelining, group-M swizzle, no split-K | ~1.0× (cuBLAS is strong; avoid regressions) |

Strategic emphasis: **defend the large-M shapes at ~parity** (do not regress) while
**pushing occupancy on the ≥39 small-M shapes**, since the geomean is small-M-weighted. A
plausible outcome is a geomean modestly above 1.0× driven almost entirely by small M.
Beating cuBLAS is not guaranteed; part of the search value is quantifying where headroom
actually exists via profiling.

---

## 7. Validation strategy

Correctness is only observable through `evaluate_candidate.sh feedback`, so:

1. **c001 = conservative correct baseline** (plain autotuned tiled `tl.dot`, FP32 acc,
   M-masking, no split-K). A green run confirms: entry-point contract, dtype/layout
   handling, masking on all M (incl. 1/7/15/35), and the FP32-accumulate numerical model.
   It also yields the reference geomean to measure every later idea against.
2. **Change one thing per candidate** (immutable IDs), record hypothesis → result, so a
   regression is attributable. Never rewrite earlier `candidates.jsonl` records.
3. **Profile between evals, never during.** Use `./scripts/ncu_profile.sh` with the
   ncu-report-skill on a representative small-M shape (e.g. M=16 or M=64) and a large-M
   shape (M=8192) to confirm the *hypothesized* bottleneck before spending eval budget:
   - Small-M: check `dram__throughput` / `lts__throughput` (L2), achieved occupancy, and
     active-CTA count. Expect memory/occupancy-bound → validates the split-K lever.
   - Large-M: check `sm__pipe_tensor_op` utilization and pipeline stalls → validates tile
     size / stages.
   Strictly sequence profiling and evaluation (never overlap on the locked GPU → rc=3).
4. **Numerical spot-checks are indirect**: since we cannot run a private correctness
   harness, rely on the evaluator's tolerance gate and keep the numerics conservative
   (FP32 everywhere but the final cast; FP32 split-K reduction). If a split-K candidate
   ever fails correctness, suspect FP16 accumulation/atomics or a K-coverage/mask bug, not
   tolerance.
5. **Convergence / stop**: stop when successive candidates no longer improve the geomean
   meaningfully, or at budget; then write `SEARCH_COMPLETE` with the reason. `final` only
   on explicit operator approval.

---

## 8. Key risks and pitfalls

- **Beating cuBLAS on large M is hard**; over-investing there wastes budget. Prioritize the
  small-M geomean drivers; hold large-M at parity.
- **Split-K atomics**: correctness (FP32 workspace + cast), determinism within tolerance,
  and workspace allocation cost per call (allocate once / reuse; keep it out of the timed
  hot path where possible — but allocation counts, so prefer sizing it minimally, only for
  the M·N actually needed).
- **Autotune first-call cost** could interact with the warmup=2 window; verify it does not
  inflate timed iters (autotune benchmarking happens at compile/first-call). A frozen
  heuristic sidesteps this.
- **Wave quantization / tail** for medium M where n_tiles·splitk is just over 132 — tune
  split factor to land near 1 or 2 full waves.
- **Immutability discipline**: every tile/split/launch change is a new candidate ID; do not
  mutate an evaluated source in place.
- **GPU-sharing rule**: absolutely no profiler while an eval is timing on the locked GPU.

---

## 9. Candidate roadmap (high-level sketch — full plan goes in `docs/plan.md` next)

1. **c001** — correct anchor: autotuned tiled `tl.dot`, FP32 acc, M-mask, group-M raster,
   no split-K. Establish baseline geomean + confirm numerics.
2. **c002** — add split-K path (FP32 atomic workspace + cast) gated to small/medium M;
   tune SPLIT_K / BLOCK_N for ~1–2 waves. Expected main small-M win.
3. **c003+** — refine per-M-bucket configs (freeze heuristic vs autotune), tune stages /
   num_warps, evaluate a dedicated tiny-M (M≤8) path and optional persistent kernel, each
   informed by ncu profiling. Iterate until geomean converges.

(Details, exact config tables, and stop criteria will be specified in `docs/plan.md`.)
