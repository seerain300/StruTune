# Draft — L1/007 `hyena_fft_size_padding_rfft`

Target: NVIDIA H100 (`sm_90`). Submission: `solution/solution.py` exposing `run(x)`.
Primary compute **must be Triton**; PyTorch only for metadata/launch. No Torch/CuFFT/CPU/NumPy
computational fallback.

---

## 1. What the operation actually computes

Reference (from `task/definition.json`):

```python
batch, channels, seqlen = x.shape          # channels == d_model == 256 (const)
fft_size = 2 * seqlen                       # N = 2L
x_f32   = x.to(torch.float32)               # input is already float32
x_freq  = torch.fft.rfft(x_f32, n=fft_size) # implicit zero-pad L -> N, then N-point rfft
x_freq  = x_freq / fft_size                 # normalize by N
return x_freq.real.contiguous(), x_freq.imag.contiguous()
```

- Input `x`: `(batch, 256, seqlen)`, `float32`.
- Output: **two** real `float32` tensors `x_freq_real`, `x_freq_imag`, each
  `(batch, 256, seqlen+1)`. (The op cannot return a complex dtype, so real/imag are split.)
- `freq_len = seqlen + 1 = N/2 + 1`, exactly the rfft output length for `N = 2*seqlen`.

### 1.1 The structural key: the transform is a *partial* DFT (half the input is zero)

`rfft(x, n=2L)` zero-pads the length-`L` signal to length `N = 2L`, then takes the real DFT.
Because samples `L..2L-1` are **identically zero**, the definition collapses to a sum over only
the `L` real input samples:

```
X[k] = Σ_{n=0}^{L-1} x[n] · exp(-2πi · k·n / N),   N = 2L,   k = 0 … L   (that's L+1 bins)
```

Writing `θ_{k,n} = 2π·k·n / N = π·k·n / L`:

```
real_out[m,k] = (1/N) · Σ_{n=0}^{L-1} x[m,n] · cos(θ_{k,n})
imag_out[m,k] = (1/N) · Σ_{n=0}^{L-1} x[m,n] · (−sin(θ_{k,n}))
```

Flatten the leading dims to `M = batch·256` independent rows. Then the whole op is **two real
matrix products** sharing the left operand:

```
R_real = (1/N) · X · C          X : (M × L)     C : (L × P)  with C[n,k] =  cos(θ_{k,n})
R_imag = (1/N) · X · S          S : (L × P)     with S[n,k] = −sin(θ_{k,n})
P = L+1 output bins
```

This is the single most important observation: **the operator is a GEMM against a (data-independent)
DFT matrix**, and it exploits that only `L` of the `N` inputs are nonzero (cuFFT, the reference,
transforms all `N` points and cannot skip the zeros). This is the natural Triton formulation.

### 1.2 Endpoint invariants (free correctness checks)

- **DC bin** `k=0`: `θ=0` ⇒ `imag_out[:,0] = 0` exactly; `real_out[:,0] = (Σ_n x[n]) / N`.
- **Nyquist bin** `k=L` (index `N/2`): `θ_{L,n} = π·n` ⇒ `sin = 0` ⇒ `imag_out[:,L] = 0`;
  `real_out[:,L] = (Σ_n (−1)^n x[n]) / N`.
- Both endpoint imag parts must be exactly `0`. Any leakage there flags an argument-reduction bug.

---

## 2. Constraints and hard requirements

- **Triton-only compute.** No `torch.fft`, no torch matmul substitute, no cuFFT, no CPU/NumPy path.
  Shape-based dispatch *between Triton kernels* is allowed (it is "launch plumbing"); a Torch numeric
  path is not.
- **Dtypes:** in/out all `float32`. Cast is a no-op (already f32) but the accumulation dtype inside
  the kernel matters a lot (see §4).
- **Tolerances:** every workload uses `max_atol = 1e-5`, `max_rtol = 1e-5`. Assuming the standard
  `|a−b| ≤ atol + rtol·|b|` test, and output magnitudes ~`1/(2√L)` (see §4.1), the **effective budget
  is ≈ 1e-5 absolute**. This is tight and is the dominant design constraint after correctness.
- **Output layout:** two separate contiguous `(batch,256,L+1)` tensors. Kernel must write both;
  keep row-major contiguity (channel/bin are the fastest-moving axis of the flattened `M×P`).
- **Every feedback workload is checked for correctness**, including the tiny boundary shapes
  (`L=131`, `L=211`, `L=256`) and the huge ones (`L=32768`). A single failing shape invalidates the
  candidate, so the kernel must be *uniformly* correct across all 16 shapes, not just the fast ones.
- Budget: 100 candidate evaluations; token soft/normal/hard = 9M / 10M / 11M. One full feedback set =
  one evaluation. Profiling only via `./scripts/ncu_profile.sh`, never concurrent with an evaluation.

---

## 3. Workload characterization (all 16 feedback shapes)

`M = batch·256` rows, `N = 2L`, `P = L+1` output bins. "pow2 N" ⇒ cuFFT is on its fast radix path;
"2·prime" ⇒ cuFFT must Bluestein-pad (to a power of two ≥ `2N−1`), doing several oversized FFTs, so
its effective work per useful output is much higher — the prime shapes are where a custom kernel can
win most.

| # | batch |     L | N=2L  | P=L+1 |     M | N class      | dense MAC ≈ M·L·P |
|---|------:|------:|------:|------:|------:|--------------|------------------:|
| 1 |     8 |  1024 |  2048 |  1025 |  2048 | 2¹¹ (pow2)   |            2.1e9  |
| 2 |    16 |  1423 |  2846 |  1424 |  4096 | 2·1423 prime |            8.3e9  |
| 3 |     2 |   773 |  1546 |   774 |   512 | 2·773 prime  |            3.1e8  |
| 4 |    64 |  1571 |  3142 |  1572 | 16384 | 2·1571 prime |            4.0e10 |
| 5 |     8 |  1321 |  2642 |  1322 |  2048 | 2·1321 prime |            3.6e9  |
| 6 |     2 |   256 |   512 |   257 |   512 | 2⁹  (pow2)   |            3.4e7  |
| 7 |    64 |  8192 | 16384 |  8193 | 16384 | 2¹⁴ (pow2)   |            1.1e12 |
| 8 |     1 |   131 |   262 |   132 |   256 | 2·131 prime  |            4.4e6  |
| 9 |    32 |  4096 |  8192 |  4097 |  8192 | 2¹³ (pow2)   |            1.4e11 |
|10 |     8 |  1879 |  3758 |  1880 |  2048 | 2·1879 prime |            7.2e9  |
|11 |    32 |  1489 |  2978 |  1490 |  8192 | 2·1489 prime |            1.8e10 |
|12 |    16 |  2048 |  4096 |  2049 |  4096 | 2¹² (pow2)   |            1.7e10 |
|13 |     4 |  1801 |  3602 |  1802 |  1024 | 2·1801 prime |            3.3e9  |
|14 |     4 |   512 |  1024 |   513 |  1024 | 2¹⁰ (pow2)   |            2.7e8  |
|15 |     2 | 32768 | 65536 | 32769 |   512 | 2¹⁶ (pow2)   |            5.5e11 |
|16 |     2 |   211 |   422 |   212 |   512 | 2·211 prime  |            2.3e7  |

(Every "prime" `L` above was checked by hand: 773, 1321, 1423, 1489, 1571, 1801, 1879, 131, 211 are
all prime, so `N=2L` has a large prime factor.)

Class split: **7 power-of-two** shapes (#1,6,7,9,12,14,15) and **9 "2·prime"** shapes (rest).

### 3.1 Cost regimes and where the win/loss lives

Dense DFT is `Θ(M·L·P) ≈ Θ(M·L²)` MACs per output-pair; cuFFT is `Θ(M·N·log N)`. So:

- **Small / medium shapes** (#3,6,8,13,14,16 and to a degree #2,5,10): dense MAC ≤ ~1e10; a single
  tensor-core GEMM finishes in tens of µs, while cuFFT pays fixed launch + (for the primes) Bluestein
  overhead. **These are expected wins**, and the prime ones can be large wins.
- **Large power-of-two** (#7 ≈ 1.1e12, #15 ≈ 5.5e11, #9 ≈ 1.4e11, #12 ≈ 1.7e10): dense DFT is
  hundreds of GFLOP–multi-TFLOP. Even at ideal TF32 tensor throughput (~5e14 FLOP/s) #7 is ~9 ms
  whereas cuFFT is memory-bound at well under 1 ms. **Dense DFT loses hard here** (#15/#7 could be
  10–70× slower). This is the crux of the whole task.
- **Large prime** (#4 ≈ 4.0e10): borderline; dense is ~0.3 ms at TF32, cuFFT Bluestein-pads 3142→8192
  and does 3 oversized transforms per row × 16384 rows — plausibly a win but must be measured.

**Implication for the metric.** Geometric-mean speedup is dominated as much by the worst losses as
by the best wins. A pure dense-DFT kernel can plausibly land near or a bit above 1.0 (many prime wins
offsetting a few pow2 losses), but the large power-of-two losses (#7,#15 especially) are the ceiling.
Genuine geomean gains require making those large pow2 shapes competitive — i.e. a real FFT-structured
Triton kernel — not just a bigger GEMM.

---

## 4. Numerical risk analysis

### 4.1 Output magnitude and the real tolerance budget
For random input of order-1 samples, `X[k] = Σ_n x[n]·e^{-iθ}` behaves like a random walk of `L`
unit steps ⇒ `|X[k]| ~ √L`. After `/N = /2L`, `|out| ~ √L/(2L) = 1/(2√L)`. Concretely ~`1.6e-2`
(L=1024) down to ~`2.8e-3` (L=32768). With `rtol·|b| ≈ 1e-5·(few e-3) ≈ 1e-8`, the passing test is
governed almost entirely by `atol = 1e-5`. **We must hold ~1e-5 absolute error.** The DC bin is an
exception: `real_out[:,0] = Σx/N` is also ~`1/(2√L)` scale for zero-mean input, similar budget.

### 4.2 fp32 accumulation is fine; TF32 is the danger
- **True fp32 accumulation of the length-`L` sum:** partial sums grow like `√j`, rounding error
  accumulates to `~ε·L` on the *un-normalized* `X[k]`; after `/N=/2L` the error is `~ε/2 ≈ 6e-8`,
  **independent of `L`**. So an fp32-accumulated DFT is comfortably within 1e-5 even at L=32768.
- **TF32 tensor cores (10-bit mantissa, ε≈5e-4):** quantizing the cos/sin matrix and/or accumulating
  in TF32 gives relative error ~`5e-4` per element ⇒ `X[k]` error `~5e-4·√L`, normalized
  `~5e-4/(2√L)`. That is ~`6e-6` at L=1571 (marginal) but ~`2.2e-5` at L=131 and worse for the
  smallest shapes — **plain TF32 will fail the small shapes**. Also TF32 on the *twiddle values*
  (which are exactly representable-ish `cos/sin ∈[-1,1]`) still loses mantissa on the products.
- **Mitigations, in order of preference:**
  1. `tl.dot(..., input_precision="ieee")` — true fp32 MMA (slower, ~1/8 TF32 rate on H100) but
     safe everywhere. Good default for correctness-first c001.
  2. `input_precision="tf32x3"` (3-pass TF32, ~19-bit effective) — near-fp32 accuracy at ~3× TF32
     cost; likely the best accuracy/speed compromise once correctness is established.
  3. Non-`tl.dot` fp32 FMA reduction (no tensor cores) — safest numerically, but throughput-bound;
     only viable for the small shapes.
- **Decision:** start with the numerically safe path (ieee or explicit fp32 accumulate), confirm all
  16 shapes pass, then trade down to tf32x3 / TF32 *per shape* only where the error margin allows and
  measure the speedup, never blindly.

### 4.3 Argument reduction for the twiddles — the subtle killer
`θ_{k,n} = π·k·n / L`. With `k,n` up to 32768, `k·n` reaches ~`1.07e9` and `θ` reaches ~`3.4e9`
radians. **fp32 cannot hold such an angle** (ulp at 1e9 is ~64, so `sin/cos` of the naive product is
noise). This breaks any on-the-fly twiddle computed as `sin(pi*k*n/L)` in fp32.
- **Fix:** reduce the index product modulo the period *in integers first*:
  `r = (k·n) mod N` with `k·n` computed in **int64** (fits: 1.07e9 ≪ 2⁶³), then
  `θ = 2π · (r / N)`, `r ∈ [0, N)`, so the fp32 angle is in `[0, 2π)` and `cos/sin` are accurate.
  (int32 is unsafe: `32768·32768 = 2³⁰` overflows near the int32 limit for larger products — use
  int64 for the multiply, reduce, then cast the small residue to fp32.)
- This same integer-mod trick is what lets us avoid materializing giant cos/sin matrices (see §5.2)
  and is mandatory for numerical validity at large `L`.

### 4.4 Precompute vs. on-the-fly memory
Materializing `C,S` costs `2·L·P·4` bytes: 0.54 GB at L=8192 (#7) and **8.6 GB at L=32768 (#15)** —
infeasible to store, and pure overhead to write/read. ⇒ Twiddles must be generated **inside** the
kernel per output tile (via §4.3), never precomputed to global memory for the large shapes. For small
shapes a cached matrix in constant/shared memory is an option but probably unnecessary.

### 4.5 Other numerical notes
- Endpoint imag parts must be *exactly* 0 (§1.2). The integer-mod path gives `r=0` at k=0 and
  `r=n·L mod 2L ∈ {0,L}` at k=L ⇒ `sin(π)=0` up to fp32 rounding (~1e-7), safely under tolerance.
- Normalization `1/N` should be applied as a single multiply on the fp32 accumulator (exact power of
  two for pow2 shapes; benign otherwise). Fold into the epilogue, not per-term.
- Keep reductions in fp32 regardless of MMA input precision (`tl.dot` accumulates in fp32 — good).

---

## 5. Triton design space

### 5.1 Plan A — Dense DFT as a fused GEMM (correctness-first baseline)
One kernel computing both `R_real` and `R_imag`. Standard GEMM tiling:
- Grid over `(M tiles × P tiles)`; each program owns a `BLOCK_M × BLOCK_P` output tile.
- Reduce over `n = 0..L-1` in `BLOCK_K` chunks: load `X` tile `(BLOCK_M × BLOCK_K)`; generate the
  twiddle tiles `Ccos, Csin = (BLOCK_K × BLOCK_P)` on the fly via §4.3 (int64 `k·n mod N`, then
  `cos`, `−sin`); two `tl.dot` accumulations into fp32 `acc_real`, `acc_imag`.
- Epilogue: multiply by `1/N`, store to the two output tensors.
- Precision knob = `input_precision` (ieee → tf32x3 → tf32), chosen per §4.2.
- Pros: dead-simple, provably correct, exploits the L-vs-N zero structure, tensor-core friendly,
  no giant matrices. Wins the small/medium and (likely) most prime shapes.
- Cons: `Θ(M·L²)` — loses badly on #7/#9/#15/#12 (the large pow2 shapes).
- This is the natural **c001**: get all 16 shapes green, establish the geomean floor.

### 5.2 Plan B — precision/throughput tuning of Plan A
Same kernel, tuned: block sizes, `num_warps`/`num_stages`, and *per-shape* precision (ieee for small
L where margin is tight and cost is low; tf32x3/tf32 for large L where the error budget allows and the
speedup matters). Possibly a separate small-shape variant that skips tensor cores. Cheap iterations,
several candidate IDs.

### 5.3 Plan C — FFT-structured Triton kernel for the power-of-two shapes (the real upside)
To make #7/#9/#12/#15 competitive we need `Θ(M·N·log N)` not `Θ(M·L²)`. Options:
- **Radix-2 / mixed-radix Cooley–Tukey (Stockham autosort)** on `N=2L`, exploiting the length-`L`
  real, zero-padded input. Because the second half is zero, the first butterfly stage is trivial and
  a real-input (rfft) packing halves the work again. Twiddles via §4.3 integer reduction per stage.
- **In-SRAM per-row FFT** when the working set fits H100 shared memory (228 KB/SM): `N=2048`→16 KB,
  `8192`→64 KB, `16384`→128 KB fit; **`N=32768`→256 KB and `65536`→512 KB do not** ⇒ #15 needs a
  multi-block / global-memory Stockham or a 4-step (split-radix / two-pass) FFT.
- **Cooley–Tukey factorization of the DFT-matmul** (a few radix stages to shrink the reduction, then
  a residual GEMM) — a middle ground reusing Plan A machinery with far fewer MACs.
- High risk / high effort; only justified after measurements confirm the large pow2 shapes are the
  binding constraint on geomean. Correctness bar (1e-5, all shapes) is unforgiving for hand-rolled
  FFTs.

### 5.4 Plan D — shape-dispatched hybrid (expected final form)
`run(x)` inspects `(batch, seqlen)` and dispatches to the best *Triton* kernel: Plan A/B GEMM for
small + prime shapes (where it beats cuFFT), Plan C FFT for large power-of-two shapes. All paths are
Triton; the dispatch is pure launch plumbing and is allowed. This is the likely end state if Plan C
proves viable; otherwise the deliverable is a well-tuned Plan B.

### 5.5 Kernel-count / launch considerations
- Two outputs can be produced by one fused kernel (two accumulators) — avoids re-reading `X`.
- Contiguity: flatten `(batch,256)→M`; `X` is `(M×L)` row-major, outputs `(M×P)` row-major, then
  view back to `(batch,256,P)` — a free reshape, no copy.
- Autotune over `BLOCK_M/BLOCK_P/BLOCK_K`, `num_warps`, `num_stages`, keyed on `L` (and maybe `M`)
  so tiny shapes (L=131) and huge shapes (L=32768) each get sensible configs.

---

## 6. Validation strategy

- **Primary correctness gate:** `./scripts/evaluate_candidate.sh feedback cNNN` over the full 16-shape
  set (one evaluation). This is the authoritative check; all 16 must pass at `atol=rtol=1e-5`.
- **Pre-evaluation offline reasoning (no GPU, no alternate harness):**
  - Verify the math identity §1.1 symbolically and the endpoint invariants §1.2 (imag=0 at k=0 and
    k=L) — cheap structural self-checks to catch indexing/sign errors before spending an evaluation.
  - Confirm bin count `P=L+1` and output shapes/contiguity match the definition exactly.
  - Sanity the sign convention: `exp(-iθ)` ⇒ `real=+cos`, `imag=−sin` (matches `torch.fft` forward).
  - Argument-reduction audit: ensure `k·n` is formed in int64 before `mod N` for the largest shape.
- **Precision budgeting before relaxing to TF32:** use §4.2 estimates to predict per-shape error and
  only lower precision where the predicted error is safely `< ~1e-5`; verify with a feedback eval.
- **Profiling (separate from evaluation, via `./scripts/ncu_profile.sh`):** after c001 is correct,
  profile the large pow2 shapes to confirm the dense-DFT is compute-bound (motivating Plan C) and the
  small/prime shapes to confirm they are already winning. Never run profiling concurrently with an
  evaluation (return-code-3 discard risk).
- **Immutability discipline:** each meaningful source/precision/dispatch change ⇒ new candidate ID;
  record parent, source hash, hypothesis, per-workload pass/latency, geomean, decision, cumulative
  eval count, and skill usage in `candidates.jsonl`.

---

## 7. Risks, unknowns, and open questions

1. **Random input distribution/scale** is unspecified. Magnitude/tolerance analysis (§4.1) assumes
   order-1 samples; if inputs are larger the absolute-error budget scales with them (helps), if tiny
   it tightens. First feedback eval will reveal actual pass margins.
2. **cuFFT baseline behavior on 2·prime sizes** (Bluestein vs. Rader) determines the size of the prime
   wins — must be measured, not assumed.
3. **Plan C feasibility/ROI** — a correct, fast Triton FFT (esp. the out-of-SRAM L=32768 case) is the
   main technical risk; may not be worth the token/eval budget if Plan B already clears geomean > 1.
4. **TF32 correctness margin on small L** — likely unusable for L≲512; ieee/tf32x3 fallback needed
   there, costing speed exactly where cuFFT is weakest anyway (acceptable).
5. **`tl.dot` with on-the-fly generated operand** — need to confirm Triton generates efficient code
   when one GEMM operand is computed (cos/sin of an int-reduced index) rather than loaded; if not,
   a two-kernel (materialize-tile-in-SRAM then dot) structure or the FMA path may be required.

---

## 8. Direction into the plan

1. **c001 = Plan A** (fused dense-DFT GEMM, int64 argument reduction, fp32/ieee accumulate): prove all
   16 shapes correct and record the geomean floor.
2. **c002+ = Plan B** tuning: block sizes / warps / stages / autotune keys, and per-shape precision
   step-down (ieee → tf32x3 → tf32) guided by §4.2 and profiling.
3. **Decide on Plan C** only if profiling confirms the large power-of-two shapes are the binding
   geomean constraint and the eval/token budget supports an FFT-structured kernel; land it as a
   shape-dispatched hybrid (Plan D).
4. Stop and write `SEARCH_COMPLETE` when geomean improvement converges or budget is exhausted.

Consult the `KernelWiki` skill before writing Triton for H100-specific GEMM/tensor-core and
warp-specialization guidance, and `ncu-report-skill` for the profiling passes above.
