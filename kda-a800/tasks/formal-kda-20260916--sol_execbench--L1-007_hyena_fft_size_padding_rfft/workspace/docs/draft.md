# Draft — L1/007 `hyena_fft_size_padding_rfft`

Run: `formal-kda-20260916--sol_execbench--L1-007_hyena_fft_size_padding_rfft`
Target HW: **NVIDIA A800, `sm_80` (Ampere)**. Primary implementation must be **Triton**; PyTorch only for
metadata / launch plumbing; **no** Torch / CPU / NumPy / CUDA-extension computational fallback.

This document is analysis only. No `plan.md`, no solution code is produced in this step.

---

## 1. What the operation actually computes

Reference (`task/definition.json`):

```python
batch, channels, seqlen = x.shape          # x: float32, contiguous (b, c, s)
fft_size = 2 * seqlen                        # N = 2*seqlen
x_freq = torch.fft.rfft(x.float(), n=fft_size)   # implicit zero-pad seqlen -> N
x_freq = x_freq / fft_size                   # normalize by N
return x_freq.real.contiguous(), x_freq.imag.contiguous()   # each (b, c, seqlen+1)
```

Key structural facts:

- **Zero-padding is exactly 2×.** The input is length `seqlen`; the transform length is `N = 2*seqlen`.
  Only the first `seqlen` samples are nonzero; samples `seqlen … 2*seqlen-1` are implicit zeros.
- **Output length `freq_len = seqlen + 1`.** For an rfft of a length-`N` signal the output has
  `N//2 + 1 = seqlen + 1` bins — i.e. the output is the *full* one-sided spectrum, `k = 0 … seqlen`.
  There is **no truncation** we can exploit; we must produce every non-redundant bin.
- **Normalization** is division by `N = 2*seqlen` (a plain scale, not `1/sqrt(N)`).
- Output is returned as **two real tensors** (real, imag), not a complex tensor.

### 1.1 Exact math (the important reformulation)

Let `W_N = exp(-2πi / N)` with `N = 2*seqlen`. Because the tail is zero, the padded DFT collapses to a
sum over only the real data samples:

```
X[k] = Σ_{n=0}^{seqlen-1} x[n] · W_N^{k n},        k = 0 … seqlen
```

So each output bin is a linear combination of the `seqlen` input samples. Splitting real/imag and folding
in the `1/N` normalization:

```
R[k] = (1/N) · Σ_{n=0}^{seqlen-1} x[n] · cos(2π k n / N)
I[k] = -(1/N) · Σ_{n=0}^{seqlen-1} x[n] · sin(2π k n / N)
```

Flattening `(batch, channels)` into `M = batch · channels` rows (the input is contiguous `(b,c,s)`, so it is
already a row-major `M × seqlen` matrix), this is **two real matrix products**:

```
Let C[k,n] = cos(2π k n / N),  S[k,n] = sin(2π k n / N)      # shape (seqlen+1) × seqlen
R = ( Xmat @ Cᵀ ) / N                                        # (M × seqlen) @ (seqlen × seqlen+1)
I = -( Xmat @ Sᵀ ) / N
```

This "DFT-as-GEMM" view is the conceptual backbone: it is trivially correct, fully parallel, and maps
cleanly onto Triton `tl.dot`. Its weakness is arithmetic cost (Section 4).

### 1.2 Structural invariants (free correctness checks)

- `M = batch · d_model = batch · 256`.
- **DC bin** `k=0`: `W^0 = 1` ⇒ `R[0] = (Σ x[n]) / N`, `I[0] = 0` exactly.
- **Nyquist bin** `k = seqlen` (this is the *last* output bin, since `N` is even and `N/2 = seqlen`):
  `W_N^{seqlen·n} = e^{-iπ n} = (-1)^n`, purely real ⇒ `I[seqlen] = 0` exactly.
- These two `imag = 0` columns are hard invariants; any implementation that emits nonzero values there is
  buggy. Good cheap internal sanity signals.

---

## 2. The five feedback workloads (and what they imply)

`d_model = 256` (const), `batch` and `seqlen` vary. `M = batch·256`. `N = 2·seqlen`.

| WL | batch | seqlen | N = 2·seqlen | seqlen factorization | N structure | M = rows | out bins |
|----|------:|-------:|-------------:|----------------------|-------------|---------:|---------:|
| 1  | 4     | 1801   | 3602         | **1801 prime**       | 2 · prime   | 1024     | 1802     |
| 2  | 2     | 211    | 422          | **211 prime**        | 2 · prime   | 512      | 212      |
| 3  | 64    | 8192   | 16384 = 2¹⁴  | 8192 = 2¹³           | power of 2  | 16384    | 8193     |
| 4  | 8     | 1024   | 2048 = 2¹¹   | 1024 = 2¹⁰           | power of 2  | 2048     | 1025     |
| 5  | 8     | 1321   | 2642         | **1321 prime**       | 2 · prime   | 2048     | 1322     |

Tolerance for **all** five: `max_atol = 1e-5`, `max_rtol = 1e-5` (tight — see Section 5).

Two distinct regimes emerge, and they matter enormously:

- **Power-of-two `N` (WL3, WL4):** cuFFT (the reference) is *extremely* good here — genuine `O(N log N)`
  radix-2. WL3 in particular is large.
- **`N = 2·prime` (WL1, WL2, WL5):** cuFFT must fall back to **Bluestein** (chirp-z) for the prime factor,
  which pads to a larger convenient length (`≥ 2N−1`, typically the next power of two) and runs **three**
  FFTs plus pointwise multiplies. This carries a large constant factor and plan overhead → these are cuFFT's
  *weak* sizes and our best opportunities.

**Geomean intuition.** The ranking metric is geometric mean speedup across selected workloads, and every
workload must pass correctness. Because it is a geomean, one catastrophic workload dominates: e.g. big wins
on WL1/2/5 (say 3×) but a 1000× loss on WL3 gives `(3·3·3·a·1000⁻¹)^(1/5) ≪ 1`. **We cannot afford a
blow-up on WL3.** Any pure `O(N²)` approach that tanks WL3 is disqualifying for geomean even if it wins the
other four.

---

## 3. Constraints and environment

- **Triton-only compute.** No `torch.fft`, no torch matmul substitute for the core math, no NumPy/CPU path.
  PyTorch is allowed for shape/stride/dtype handling and kernel launch only. A failing Triton kernel is
  invalid and must not be silently replaced by a Torch fallback.
- **dtype:** input and outputs are `float32`. Accumulation must be fp32 (or higher effective precision).
- **A800 / sm_80:** Ampere. Peak fp32 (non-tensor-core) ~19.5 TFLOP/s; TF32 tensor cores ~156 TFLOP/s but
  **TF32 has only ~10 mantissa bits (~1e-3 rel error)** → unusable at `1e-5` tolerance. HBM ~2 TB/s.
- **Budget:** 100 candidate evaluations; token soft/hard 1.0M / 1.2M. Each eval = all five feedback
  workloads under one immutable kernel.
- **Final** = 16-workload eval, operator-approved only. The hidden 11 extra workloads may include `seqlen`
  values that are neither power-of-two nor `2·prime` → generality matters for the eventual `final`, even
  though the feedback set is conveniently structured.

### 3.1 Memory / bandwidth floor (why WL3 is special)

For WL3: input `16384 × 8192 × 4 B = 537 MB`; output real+imag `16384 × 8193 × 2 × 4 B ≈ 1.07 GB`. Reading
input + writing output ≈ 1.6 GB ⇒ at ~2 TB/s a hard floor of **~0.8 ms** regardless of algorithm. cuFFT
already lives near this floor. A DFT-as-GEMM for WL3 needs `M·(seqlen+1)·seqlen ≈ 1.1e12` MACs **per
matrix** (×2 matrices) ⇒ `~4.4e12` FLOP ⇒ even at a generous 15 TFLOP/s fp32 that is **~290 ms**, i.e.
~300× over the memory floor and hopelessly compute-bound. **Conclusion: WL3 mandates a true `O(N log N)`
FFT, not a dense DFT.**

---

## 4. Triton design space

Ordered roughly from simplest/most-robust to fastest/most-complex. The realistic winning path is a
**size-adaptive hybrid**.

### 4.1 Dense DFT-as-GEMM (`Xmat @ Cᵀ`, `Xmat @ Sᵀ`) — baseline
- **Idea:** materialize / stream the `(seqlen+1) × seqlen` cos & sin twiddle matrices; two fp32 GEMMs via
  `tl.dot(..., input_precision="ieee")`; fuse `1/N` scale and the real/imag split into the epilogue.
- **Pros:** trivially correct; simple; excellent on *small* sizes (WL2, and plausibly WL1/WL5 where cuFFT
  pays Bluestein overhead). Blocked `tl.dot` accumulation is numerically well-behaved (tree reduction).
- **Cons:** `O(M · seqlen²)` — catastrophic on WL3 (Section 3.1); large twiddle-matrix memory
  (WL3: `8193×8192×4 ≈ 268 MB` each, ×2 = 536 MB — feasible on 80 GB but wasteful of bandwidth if reread).
- **Role:** **c001 correctness/measurement baseline.** Establishes ground truth and shows the WL-by-WL gap
  before investing in FFT complexity. Expect: win WL2, competitive WL1/WL5, big loss WL3, loss/near WL4.

### 4.2 Twiddle generation & caching (cross-cutting)
- Twiddle matrices/tables depend only on `seqlen` (and device/dtype), **not** on `x`. Generate once and
  **memoize** at module scope keyed by `seqlen`. Amortized over benchmark timing iterations this removes
  `O(seqlen²)` trig from the hot path.
- **Generate the tables with a Triton kernel** (or as part of the compute kernel), *not* with `torch.cos`
  — computing twiddles is arithmetic and belongs on the Triton side to respect the "no Torch compute" rule.
  Only index/stride bookkeeping uses torch.
- **Risk to flag:** if the evaluator times a *single* cold `run()` including first-call setup, the one-time
  `O(seqlen²)` generation is exposed and could dominate small workloads. Must confirm empirically that
  caching helps rather than hurts (warmup semantics unknown to us). Keep generation itself efficient and, if
  possible, table-based (`O(seqlen)` twiddle vector + on-the-fly index math) rather than a full dense matrix.

### 4.3 Radix-2 split + prime-DFT (natural fit for `N = 2·seqlen`)
- The 2× zero-padding makes one radix-2 decimation-in-time stage exact and cheap. Splitting the padded
  length-`N` signal into even/odd subsequences yields **two length-`seqlen`** sub-transforms (each itself a
  half-zero-padded length-`seqlen` sequence), combined with twiddles `e^{-2πi k/N}`.
  - If `seqlen` is a power of two (WL3, WL4) ⇒ **recurse** → a full radix-2 FFT.
  - If `seqlen` is an odd prime (WL1, WL2, WL5) ⇒ the two sub-transforms are length-`seqlen` DFTs done as
    **small matmuls** (`O(seqlen²)` but on `seqlen`, not `N`, and reused for both halves).
- **Pros:** unifies both regimes; converts cuFFT's Bluestein weakness into a plain prime-size DFT we control;
  much cheaper than dense DFT on the prime sizes (works on `seqlen`, shares twiddles across even/odd).
- **Cons:** more kernel complexity; for the prime sizes it is still `O(M·seqlen²)` on `seqlen` (WL1:
  `~2·1024·1801² ≈ 6.6e9` MAC — comparable to dense but with better structure); does **not** by itself fix
  WL3 unless the pow2 branch recurses into a real FFT.

### 4.4 Full Triton FFT for power-of-two `N` (needed for WL3, WL4)
- Implement an iterative **Stockham / Cooley–Tukey radix-2 (or radix-4)** FFT along the (padded) transform
  axis, with `M` rows in parallel. Exploit real input + Hermitian symmetry (compute only `seqlen+1` bins).
  Optionally the classic **real-FFT-via-half-length-complex-FFT** trick (pack real signal into `N/2` complex,
  one complex FFT of length `N/2`, split/recombine) to halve work — but watch numerics at the split.
- **Pros:** the only way to be competitive on WL3/WL4; approaches the memory floor.
- **Cons:** substantial engineering; register/shared-memory tiling of butterflies in Triton is fiddly; bit-
  reversal or Stockham index management; real-FFT recombination adds numerical care. Highest risk/highest
  reward. **cuFFT is a very strong baseline on pow2 — beating it on WL3 is uncertain; matching within ~0.7–1×
  while winning big on the Bluestein sizes may already yield geomean > 1.**
- **Mixed radix / Bluestein in Triton** for arbitrary sizes (for the hidden final set) is a further, larger
  step — deferred; feedback set does not require it.

### 4.5 Epilogue fusion (cross-cutting, cheap win)
- The reference materializes a complex tensor, then does **three** extra full passes: `/N`, `.real
  .contiguous()`, `.imag.contiguous()`. Whatever core we use, we can **fuse** normalization and the
  real/imag split directly into the kernel epilogue, writing the two output tensors in one pass. Pure
  bandwidth savings, always beneficial, and part of why even a "match cuFFT compute" kernel can still net a
  speedup.

### 4.6 Realistic strategy
1. **c001:** dense DFT-as-GEMM (fp32 `tl.dot`, fused epilogue, cached twiddles) — correctness + baseline map.
2. Iterate on the **Bluestein-size wins** (WL1/2/5): tune tiling, `BLOCK_M/N/K`, num_warps, twiddle
   generation, and try the radix-2 + prime-DFT split to cut work.
3. Attack **WL3/WL4** with a real power-of-two FFT branch (Section 4.4) so WL3 stops being a geomean sink.
4. Assemble a **size-adaptive dispatch**: pow2 → FFT; `2·prime`/other → DFT-matmul (or radix-2+prime-DFT).
   All branches remain pure Triton.

---

## 5. Numerical risks (tolerance is tight: 1e-5 atol & rtol vs cuFFT)

The comparison target is cuFFT's own rfft (itself carrying FFT rounding). We must land within `1e-5`
absolute **and** relative. Output magnitudes are *small* (normalized by `N`), e.g. DC bin `~ Σx / N` with
`x ~ N(0,1)` is `O(1/√seqlen)` (~6e-3 for WL3), so `rtol=1e-5` bites on the larger components while
`atol=1e-5` bounds the near-zero bins. Risks:

1. **fp32 accumulation over many terms.** Dense DFT sums `seqlen` products (up to 8192). Naive sequential
   accumulation error `~ seqlen · eps` is dangerous; `tl.dot`'s blocked/tree reduction gives `~ log(seqlen)
   · eps · ‖·‖` (~1e-6 rel) which should pass but is **borderline** at the tightest bins. Prefer fp32
   accumulators everywhere; consider Kahan/split accumulation if a workload fails by a hair.
2. **TF32 is forbidden.** Must set `input_precision="ieee"` (true fp32) on every `tl.dot`. TF32's ~1e-3
   error fails instantly. This is the single easiest way to accidentally fail correctness — guard it.
3. **Twiddle phase argument reduction.** `cos/sin(2π k n / N)` with `k·n` up to `~seqlen² ≈ 6.7e7` (WL3)
   **exceeds fp32's exact-integer range (2²⁴ ≈ 1.6e7)**. Computing `2π·k·n/N` directly in fp32 loses phase
   accuracy → wrong twiddles. **Must reduce `k·n mod N` in int32/int64 first**, then form `angle = 2π·m/N`
   with `m ∈ [0,N)` (`angle ∈ [0,2π)`, well-conditioned for `tl.sin/tl.cos`). This is mandatory.
4. **`tl.sin/tl.cos` accuracy.** Triton's device trig is generally good to ~1 ulp for reduced args in
   `[0,2π)`; still verify. If insufficient, use a twiddle *table* of `N` roots-of-unity generated once
   (indexed by reduced `k·n mod N`) so each twiddle is a single table lookup rather than a live trig eval.
5. **Real-FFT recombination (if used).** The half-length-complex trick and radix butterflies introduce
   subtractions of similar-magnitude terms → potential cancellation; validate the Nyquist/DC bins and the
   `imag=0` invariants (Section 1.2) specifically.
6. **`N` even ⇒ Nyquist imag exactly 0.** Emit exact `0.0` for `I[0]` and `I[seqlen]`; do not let tiny trig
   residue leak nonzero there (could still be within atol, but clean is safer).
7. **Determinism / reduction order** across tile shapes must be stable enough to stay within tolerance for
   all `M, seqlen`.

---

## 6. Validation strategy

- **Only** `./scripts/evaluate_candidate.sh feedback <cNNN>` is used for correctness/timing (the trusted
  controller runs the official evaluator over the five fixed workloads). No direct CUDA, profiler,
  `nvidia-smi`, or alternate harness (per CLAUDE.md).
- **Pre-eval reasoning gates** (before spending an evaluation):
  - Shapes/dtypes: outputs `(b, 256, seqlen+1)` float32, contiguous, two tensors real+imag.
  - Invariants: `I[:,:,0] == 0`, `I[:,:,seqlen] == 0`, `R[:,:,0] == Σ_n x/N`.
  - fp32 path only (`input_precision="ieee"`), no TF32; twiddle args reduced mod `N`.
- **Per-candidate record** (append-only to `candidates.jsonl`): parent, source hash, hypothesis, validation
  status, per-workload pass/speedup, geomean, decision, cumulative eval count, skill usage.
- **Sequencing:** immutable `c001`, `c002`, … one source version each; any meaningful source/config/launch
  change ⇒ new ID; never rewrite earlier records.
- **Interpreting results by regime:** treat WL2 (tiny, Bluestein) as the "easy win" signal, WL1/WL5 as the
  "medium Bluestein" opportunity, and WL3/WL4 (pow2) as the "must-not-blow-up" constraint. Track geomean
  but watch the WL3 factor as the gating risk.
- **Convergence:** stop when geomean improvement plateaus or budget nears; write `SEARCH_COMPLETE` with the
  reason. `final` only on explicit operator approval.

### 6.1 Skill applicability note
The two permitted skills (`KernelWiki`, `ncu-report-skill`) are scoped to **Blackwell/Hopper (SM90/SM100,
H100/B200)**. This task runs on **A800 / sm_80 (Ampere)**, so their architecture-specific guidance
(tcgen05/TMEM/CLC/NVFP4, `ncu` on sm_100) is **not directly applicable**. I will not force-invoke them for
Ampere kernel tuning; if a genuinely transferable, architecture-neutral point arises (e.g. general Triton
GEMM tiling heuristics), I will consult and cite it, and record usage per candidate.

---

## 7. Open questions / assumptions to resolve during implementation

1. **Benchmark warmup semantics** — does timing include a cold first `run()` (exposing twiddle-cache build)?
   Determines whether table memoization is a net win. Resolve empirically via c001 vs a no-cache variant.
2. **Can we beat cuFFT on WL3 pow2 at all**, or is the play "match within ~0.7× on pow2, win big on
   Bluestein sizes for geomean > 1"? Measure with c001 first.
3. **Twiddle representation** — live `tl.sin/tl.cos` with int argument reduction vs a precomputed
   root-of-unity table (`O(N)` memory, exact index lookup). Pick per accuracy/speed after c001.
4. **Generality for `final`** — the hidden 11 workloads may need mixed-radix/Bluestein in Triton; ensure the
   dispatch degrades to a correct (if slower) DFT-matmul for arbitrary `seqlen` rather than failing.

---

## 8. Summary

The op is a **normalized, 2×-zero-padded real DFT** producing the full one-sided spectrum (`seqlen+1` bins),
split into real/imag. Mathematically it is two real GEMMs against fixed cos/sin twiddle matrices. The five
feedback workloads split into **cuFFT-strong power-of-two sizes (WL3 large, WL4)** and **cuFFT-weak
`2·prime` Bluestein sizes (WL1, WL2, WL5)**. The geomean metric forbids blowing up WL3, which — by the
`O(seqlen²)` cost and memory-floor analysis — rules out a pure dense DFT for that size and mandates a true
`O(N log N)` FFT for pow2. Plan of attack: **c001 dense DFT-as-GEMM** (correct, fused epilogue, cached
fp32-accurate twiddles) to map the landscape and grab the Bluestein-size wins, then add a **power-of-two
Triton FFT branch** and a **size-adaptive dispatch**. The dominant numerical hazards are TF32 leakage
(forbidden), `k·n` phase overflow beyond fp32 integer range (must reduce mod `N`), and fp32 accumulation at
the `1e-5` tolerance — all mitigable with fp32 IEEE `tl.dot`, integer argument reduction, and (if needed)
table-based twiddles. Validation is strictly through the official evaluator, gated by shape/invariant/
precision reasoning to conserve the 100-eval budget.
