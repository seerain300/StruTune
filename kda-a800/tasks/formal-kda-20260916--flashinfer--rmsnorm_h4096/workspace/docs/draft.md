# Draft — `rmsnorm_h4096` (FlashInfer, A800 / sm_80)

## 0. Purpose & scope of this document

This is the analysis draft required before any code or executable plan. It captures:
the operation semantics, the exact numerics to reproduce, the workload regimes,
hardware roofline, the Triton design space, numerical risks, and a validation
strategy given the tight feedback loop. No solution code and no `plan.md` are created
here. The concrete, ordered candidate steps belong in `docs/plan.md` (next turn).

---

## 1. Operation semantics

RMSNorm (Root Mean Square Layer Normalization), row-wise over the hidden axis.

Reference (authoritative, from `task/definition.json`):

```python
@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape      # hidden_size == 4096 (asserted)
    EPS = 1e-5
    x = hidden_states.to(torch.float32)                # upcast bf16 -> fp32
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)       # scale, then weight; fp32 math
    return y.to(hidden_states.dtype)                   # downcast fp32 -> bf16
```

Per row `i` (length `H = 4096`):

```
ms_i     = (1/H) * Σ_j x_ij^2            # mean of squares, fp32
inv_rms_i = 1 / sqrt(ms_i + 1e-5)        # fp32
y_ij     = (x_ij * inv_rms_i) * w_j      # fp32
out_ij   = bf16(y_ij)
```

Key exact-semantics facts I must reproduce to stay inside tolerance:

- **Accumulation dtype is fp32.** Input is bf16, upcast to fp32 *before* squaring.
- **Reduction denominator is `H = 4096`** (a true mean, not sum). Epsilon `1e-5` is
  added to the *mean of squares* (i.e. to `E[x^2]`), not to a Bessel-corrected
  variance and not to the RMS itself.
- **No mean-subtraction** — this is RMSNorm, not LayerNorm; there is no centering.
- **Multiply order:** `(x * inv_rms) * weight`, all in fp32; `weight` is upcast to
  fp32 first. Only the final result is cast back to bf16.
- **Epsilon is a fixed constant `1e-5`** (also stated in the description). Hardcode it.

### Shapes / dtypes
- `hidden_states`: `[batch_size, 4096]`, **bf16**.
- `weight`: `[4096]`, **bf16**.
- `output`: `[batch_size, 4096]`, **bf16**.
- `hidden_size` is a compile-time constant (`4096`, a power of two → nice for tiling,
  no reduction-mask needed).
- `batch_size` is the only variable axis.

---

## 2. Workload analysis

Five fixed feedback workloads (only `batch_size` varies):

| uuid (short) | batch_size | rows | input bytes (bf16) | regime |
|---|---|---|---|---|
| 9d40…5001 | 15 | 15 | 15·4096·2 ≈ 120 KB | tiny / latency-bound |
| 1453… | 64 | 64 | ≈ 512 KB | small / latency-bound |
| 33bf… | 7 | 7 | ≈ 57 KB | tiny / latency-bound |
| 841b… | 14509 | 14509 | ≈ 113 MB | large / bandwidth-bound |
| f0f5… | 14418 | 14418 | ≈ 113 MB | large / bandwidth-bound |

**Two clearly distinct regimes**, and the geomean is over both, so both matter:

- **Tiny/small (7, 15, 64 rows).** Far below the ~108-SM A800's ability to fill.
  With "one program per row" the grid is 7–64 blocks → most SMs idle. But the total
  work is minuscule (tens–hundreds of KB), so wall-time is dominated by **kernel-launch
  latency and memory latency**, not throughput. The dominant win here is **fusion**:
  the torch reference launches many kernels and materializes fp32 intermediates,
  whereas a single fused Triton kernel is one launch reading bf16 once / writing once.
- **Large (14418, 14509 rows).** ~113 MB read + ~113 MB written ⇒ ~226 MB of HBM
  traffic (weight is 8 KB, cached, negligible). This regime is strictly
  **memory-bandwidth-bound**; the objective is to hit the bf16-in/bf16-out roofline
  and avoid the reference's extra fp32-intermediate traffic.

**Implication:** a single, lean, fused, one-pass kernel should win in *both* regimes
against the un-fused reference. The optimization headroom is (a) not leaving bandwidth
on the table for the large cases and (b) minimizing overhead for the tiny cases.

---

## 3. Hardware roofline (A800, sm_80)

- A800 = GA100 (108 SMs), sm_80, HBM2e. Peak HBM bandwidth ≈ **1.5–2.0 TB/s**
  (≈2.0 TB/s for the 80 GB SKU, ≈1.55 TB/s for the 40 GB SKU); 40 MB L2.
- Arithmetic intensity of RMSNorm is very low: ~2 FLOP-ish per element for the
  square + a couple of multiplies, against 2 bytes read + 2 bytes written per element.
  → **Memory bound.** FP32 ALU throughput is irrelevant at these sizes.

**Ideal large-case time** (bf16 in + bf16 out, weight negligible):
`226 MB / 1.8 TB/s ≈ 125 µs` (≈100 µs at 2.0 TB/s, ≈145 µs at 1.55 TB/s).
This is the target floor for batch≈14.5k. Any fused kernel that reads each input
element once and writes each output once, with coalesced accesses, should approach it.

**Reference inefficiency:** the torch reference upcasts to fp32 (`x = …to(float32)`
is a 4096·B·4-byte materialization = 2× the input traffic at fp32), then `x.pow(2)`,
then `.mean(...)`, then broadcast-mul twice, then downcast — multiple passes over
large fp32 buffers plus several kernel launches. So the *speedup vs reference* on the
large cases can be substantial purely from fusion + staying in bf16 for I/O.

---

## 4. Constraints (from CLAUDE.md / TASK.md)

- **Triton is the primary implementation.** PyTorch only for metadata / launch
  plumbing (shape, dtype, output allocation, grid). **No** Torch compute fallback,
  **no** CPU/NumPy fallback, **no** CUDA-extension fallback, **no** alternate-impl
  fallback. A failing Triton kernel is invalid and must be fixed as Triton.
- **Immutable, sequential candidates** `c001, c002, …`; any meaningful source/config/
  launch change ⇒ new candidate ID; never reuse an ID or rewrite past records.
- **Evaluation only** via `./scripts/evaluate_candidate.sh feedback <id>`; the five
  fixed workloads = one candidate evaluation. Budget: 100 evals; token soft limit
  1.0M / hard 1.2M. `final` is operator-approved only.
- **I cannot run CUDA, a profiler, `nvidia-smi`, Python, or the evaluator directly.**
  Local Bash is in "don't ask" / denied mode. ⇒ The *only* empirical signal is the
  feedback evaluation. First-try correctness matters a lot; wasted evals cost tokens.
- Skills `KernelWiki` (Blackwell/Hopper) and `ncu-report-skill` (B200/sm_100) are
  **not applicable** to this sm_80 task and are not used.

---

## 5. Numerical risks & mitigations

| Risk | Cause | Mitigation |
|---|---|---|
| Wrong accumulation precision | Squaring/summing in bf16 loses ~8 mantissa bits; 4096-term sum drifts | Load bf16 → `.to(tl.float32)` **before** squaring; reduce with fp32 `tl.sum`. |
| Epsilon placement | Adding eps to RMS or to variance instead of to `E[x²]` | Compute `ms = sum(x²)/H`; `inv = rsqrt(ms + 1e-5)`. Match reference exactly. |
| Wrong denominator | Using `sum` not `mean`, or wrong H | Divide by `H = 4096` (constexpr). |
| Accidental centering | Copy-pasting a LayerNorm kernel | RMSNorm has **no** mean subtraction — do not subtract row mean. |
| Multiply-order drift | `x*w*inv_rms` vs `(x*inv_rms)*w` giving ULP diffs | Follow reference order `(x*inv)*w`; keep all fp32 until final cast. Differences are sub-bf16-ULP and safe, but match to be conservative. |
| Weight precision | Multiplying in bf16 | Upcast `weight` to fp32 before multiply (as reference does). |
| Final cast / rounding | Writing fp32 or wrong rounding | Cast result to bf16 (Triton default round-to-nearest matches torch) and store to a bf16 output tensor. |
| `rsqrt` accuracy | Fast-approx rsqrt could miss tolerance | Use fp32 `tl.rsqrt` (well within bf16 tolerance) or `1/tl.sqrt`; both are fp32-accurate enough. |
| Non-contiguous input | Evaluator hands a strided tensor | Pass row stride explicitly (`x.stride(0)`), or `.contiguous()` as a metadata-only safeguard. Prefer stride args to avoid a copy. |
| Overflow | `x²` overflow | bf16 range is huge; random inputs ⇒ no overflow. fp32 square is safe. |
| Div-by-zero / all-zero row | `ms = 0` | `+1e-5` guarantees a finite `inv_rms`. |

**Tolerance note:** `definition.json` does not expose explicit `rtol/atol`. Because I
cannot see the checker thresholds and cannot test locally, the safe policy is to
**bit-for-bit mirror the reference math** (fp32 accumulate, fp32 weight, same eps
placement, same op order, bf16 output). This maximizes the chance of passing whatever
bf16 tolerance the official evaluator uses.

---

## 6. Triton design space

All designs share the fused body: load a row of bf16, upcast fp32, square, fp32
reduce, `rsqrt(mean+eps)`, scale, multiply fp32 weight, cast bf16, store.

### A. One-program-per-row, single tile (baseline; classic FlashInfer/vLLM/tutorial form)
- `grid = (batch_size,)`; `BLOCK_SIZE = 4096` (= H, power of two → **no reduction mask**).
- Each program: `offs = tl.arange(0, 4096)`; load row, `xf=x.to(fp32)`; `s=tl.sum(xf*xf)`;
  `inv=tl.rsqrt(s/4096 + 1e-5)`; `y=(xf*inv)*w.to(fp32)`; store bf16.
- Weight loaded once per program (resident in L2 across programs → cheap).
- **Pros:** simplest, coalesced, one launch, bandwidth-optimal for large batch.
  **Cons:** low SM occupancy for tiny batch (but that regime is latency-bound anyway).
- **Tuning axes:** `num_warps ∈ {4, 8, 16}` (128/256/512 threads ⇒ 32/16/8 elems per
  thread), `num_stages`. This is the primary workhorse; likely `c001`.

### B. Autotuned variant of A
- `triton.autotune` over `num_warps`/`num_stages` keyed on `batch_size` bucket, so the
  tiny and large regimes each get their best config. Risk: autotune runs configs at
  first call — must ensure the evaluator's timing excludes warmup, else noise. Treat as
  a follow-up candidate once a good static config is known.

### C. Multiple rows per program (row blocking)
- Each program handles `R` rows (2D tile `[R, 4096]` or an inner loop), `grid = ceil(B/R)`.
- **Pro:** fewer, larger blocks → can improve memory-level parallelism and amortize
  per-program setup for the large cases; reduces grid size.
- **Con:** for tiny batch it *reduces* parallelism (fewer blocks); register/SRAM
  pressure grows with `R·4096`. Useful mainly as a large-batch tuning knob.

### D. Split-reduction (split-K, two-pass) for the tiny cases
- Split H into `K` chunks; pass 1 writes `K` partial sums per row (grid `(B, K)`),
  pass 2 normalizes. For batch 7, `K=16` → 112 blocks → fills the machine.
- **Con:** adds a second kernel launch and a global round-trip. Since the tiny regime
  is *launch-latency* bound, a 2nd launch likely **hurts**, not helps. Keep as a
  contingency only if the tiny-batch speedup underperforms and profiling-by-inference
  says the reduction (not launch) is the bottleneck. **Low priority.**

### E. Vectorization / access width
- Rows are contiguous (stride 1), so `tl.arange(0,4096)` loads are naturally coalesced;
  Triton vectorizes bf16 loads (up to 128-bit) automatically. Ensure the pointer math
  uses the row stride and unit inner stride. Little to gain beyond A if already
  coalesced, but verify no accidental masking/guard kills vectorization.

### F. Grid-stride / persistent kernel
- One block per SM looping over rows. Marginal benefit here; grid of 14.5k blocks is
  already fine for A800 scheduling. Not worth the complexity unless launch-bound at
  large batch (unlikely). **Deprioritized.**

**Chosen direction:** start with **A** (static, well-chosen `num_warps`), confirm
correctness + measure both regimes, then explore **B/C** as tuning refinements. Only
consider **D** if the tiny regime clearly lags.

---

## 7. Entry point / integration contract

`solution/solution.py` must expose `run(hidden_states, weight)` returning the bf16
output tensor (mirroring the reference signature). Plumbing responsibilities (PyTorch
allowed here, metadata only):

- Read `B, H = hidden_states.shape`; assert `H == 4096`.
- Allocate `out = torch.empty_like(hidden_states)` (bf16, same device).
- Pass `stride(0)` for the row stride (both x and out), unit inner stride.
- Compute `grid` from `B`; launch the Triton kernel; return `out`.
- Do **not** synchronize unnecessarily; let the evaluator handle timing/sync.
- Keep `EPS`, `H`, `BLOCK_SIZE` as constants/constexpr where possible.

Edge cases to keep safe: `B` can be 7 (grid must be ≥1 and correct), and very large
(14509 — well within 1D grid limits). No masking needed on the hidden axis because
`BLOCK_SIZE == H == 4096`; batch axis uses `pid < B` implicitly via grid size.

---

## 8. Validation strategy (given no local execution)

Because the **only** empirical channel is `evaluate_candidate.sh feedback`, and every
eval costs tokens, I will front-load correctness by construction:

1. **Static review before each eval:** re-check dtype casts (bf16→fp32 before square),
   eps placement, denominator `H`, op order, bf16 store, stride/pointer math, grid,
   constexpr usage, and that no Torch-compute fallback path exists.
2. **Mirror reference math exactly** (Section 5) to de-risk the unseen tolerance.
3. **c001 = correctness anchor:** simplest correct design A with a sane `num_warps`
   (likely 8). Its purpose is to (a) confirm all five workloads PASS and (b) establish
   baseline speedups per regime. Record everything in `candidates.jsonl`.
4. **Optimize only after correctness is proven.** Change one variable per candidate
   (`num_warps`, then `num_stages`, then row-blocking, then autotune) so each eval
   yields an unambiguous signal. New ID per source change; never rewrite past records.
5. **Read the evaluator output per workload** (pass/fail + speedup), attribute wins to
   regimes (tiny vs large), and stop when the geomean converges or budget nears.
6. **Convergence / stop:** when successive candidates no longer improve geomean
   meaningfully (large case near the bandwidth roofline; tiny case near launch-latency
   floor), write `SEARCH_COMPLETE` with the rationale. Never run `final` without
   operator approval.

**Interpreting results:** for the large cases, compare achieved time against the
~100–145 µs roofline (Section 3) to know how much headroom remains. For the tiny
cases, expect the fused single-launch kernel to already dominate the multi-launch
reference; further kernel tuning yields little there, so don't overspend evals on it.

---

## 9. Risk register & open questions

- **Unknown tolerance:** mitigated by exact-math mirroring; if a workload fails
  correctness unexpectedly, first suspect eps placement, denominator, or a stray bf16
  intermediate accumulation — not the tuning knobs.
- **Autotune warmup vs. timing:** unclear whether the evaluator isolates warmup; prefer
  a strong *static* config first, adopt autotune only if it demonstrably helps.
- **Occupancy vs. work for tiny batch:** likely launch-bound; avoid over-engineering
  split-K unless evidence says otherwise.
- **Bandwidth SKU (40 vs 80 GB):** changes the absolute roofline but not the strategy;
  the fused one-pass design is optimal regardless.
- **Contiguity assumption:** inputs are freshly generated randoms (very likely
  contiguous); still pass explicit strides so a strided input cannot silently break.

---

## 10. Direction summary (detail deferred to `docs/plan.md`)

- **c001:** fused, one-pass, one-program-per-row Triton kernel, `BLOCK_SIZE=4096`,
  fp32 accumulate, exact reference math, `num_warps=8` — correctness anchor + baseline.
- **Then:** single-variable tuning — sweep `num_warps` (4/16) and `num_stages`;
  consider row-blocking (design C) and/or autotune (design B) for the large regime.
- **Contingency:** split-reduction (design D) only if the tiny regime underperforms.
- **Stop** at geomean convergence (large near roofline, tiny near launch floor) and
  write `SEARCH_COMPLETE`; `final` only with operator approval.
</content>
</invoke>
