# Draft — L2/051 `seqlen-finetuned-reconstructed_hyena_complete_forward_block`

Target: NVIDIA H100 (`sm_90`). Primary implementation must be Triton; PyTorch only for
tensor metadata / launch plumbing. No Torch/CPU/NumPy/CUDA-extension computational fallback.
Submission is `solution/solution.py` exposing `run(...)` with the exact reference signature.

## 1. Operation overview

A complete Hyena block forward pass fused end-to-end. Fixed constants (from
`task/definition.json`):

| name | value |
|------|-------|
| `d_model` | 256 |
| `d_inner` | 1024 |
| `order` | 2 |
| `l_max` | 32768 |
| `short_filter_order` | 3 |
| `filter_order` | 64 |
| `emb_dim` | 5 |
| `inner_width = d_model*(order+1)` | 768 |

Variable axes: `batch_size` (B) and `seq_len` (L). All tensors are `float32`.
Because every workload has `L <= 4096 << l_max = 32768`, `l_filter = min(L, l_max) = L`
**always**, so the final zero-pad-back branch (`l_filter < seq_len`) is dead code and
never executes. I will assume `l_filter == L` throughout and treat the pad branch as
unreachable (a defensive assert / passthrough is fine but it never triggers on the
feedback or final set).

### 1.1 Exact dataflow (traced from the reference)

Let `x = hidden_states` shape `[B, L, 256]`, fp32.

1. **Residual capture**: `residual = x` (fp32). This exact tensor is reused twice later.
2. **LayerNorm1** over last dim (256), `unbiased=False` variance:
   `n1 = (x - mean)/sqrt(var + eps) * norm1_weight + norm1_bias`. In `get_inputs`
   `norm1_weight=ones`, `norm1_bias=zeros`, but they are runtime inputs → implement the
   general affine.
3. **Input projection**: `u = Linear(n1, in_proj_weight[768,256], in_proj_bias[768])`
   → `[B, L, 768]`, then logically `transpose(1,2)` → `[B, 768, L]`.
4. **Short depthwise conv** (groups=768, kernel 3): reference pads `(2,2)` then `conv1d`
   with no padding and truncates to `L`. Net effect is a **causal FIR** with taps at
   offsets `{-2,-1,0}`:
   `uc[b,c,t] = w[c,0]*u[b,c,t-2] + w[c,1]*u[b,c,t-1] + w[c,2]*u[b,c,t] + scb[c]`,
   with out-of-range `u[...,<0]` treated as 0. (`short_conv_weight` is `[768,1,3]`.)
5. **Split** `uc` `[B,768,L]` into three `[B,256,L]` chunks along channel:
   `x0 = uc[:, 0:256]`, `x1 = uc[:, 256:512]`, `v0 = uc[:, 512:768]`.
   (In reference terms: `x = (x0, x1)`, `v = v0`.)
6. **Implicit filter generation** (batch-independent, depends only on L):
   - `t = linspace(0,1,L)` → `[L]` (the `t[:,None]`/`[None,:,None]` broadcasts).
   - `t_resc = linspace(0, L-1, L) = [0,1,...,L-1]`; `w = 2*pi*t_resc/L`.
   - `f = linspace(1e-4, 1, 2) = [1e-4, 1.0]` (bands=2).
   - positional embedding `z[l] = [ t[l], cos(-f0*w[l]), cos(-f1*w[l]),
     sin(-f0*w[l]), sin(-f1*w[l]) ]` → `[L, 5]` (order: `t`, then both cosines, then both
     sines — from `cat([t, cos(-f*w), sin(-f*w)])`).
   - filter MLP (per-position, 5→64→64→64→256):
     `h = sin(sin_freq * (z @ fl1^T + b1))`; `h = sin(sin_freq*(h @ fl2^T + b2))`;
     `h = sin(sin_freq*(h @ fl3^T + b3))`; `h = h @ fl_final^T` (no bias) → `[L, 256]`.
     `sin_freq` is `[1,64]` (all ones in `get_inputs` but treat as a runtime per-feature
     scale).
   - exponential modulation: `decay[l,c] = exp(-t[l]*|exp_mod_deltas[c]|)`;
     `h = h * (decay + exp_mod_shift)`.
   - add filter bias: `h = h + filter_bias[c]` → `h` shape `[L, 256]`.
7. **Scramble reshape into k** — **critical subtlety**. Reference does
   `k = h.transpose(0,1).reshape(1, 256, L)` where `h` is logically `[1, L, 256]`.
   `transpose(0,1)` yields a non-contiguous `[L,1,256]` view; `.reshape` forces a
   **contiguous copy in the logical row-major order of `[L,1,256]`**, i.e. the flat buffer
   is `B[l*256 + c] = h[l, c]` (same as flattening `h[L,256]` row-major). Re-viewing that
   buffer as `[1,256,L]` gives
   `k[0, a, b] = B[a*L + b] = h[(a*L+b)//256, (a*L+b)%256]`.
   **This is NOT `k[a,b] = h[b,a]` (a clean transpose) in general** — it is a genuine
   index scramble. To match the reference bit-tolerantly, my kernel must reproduce exactly
   this mapping: filter row `a`, column `b` reads the flattened `h` at position `a*L+b`.
   The safest implementation is to materialize `h` as a contiguous `[L,256]` buffer and
   index it flat as `hflat[a*L+b]` when building `k[256,L]`. (Confirm empirically via the
   evaluator — this scramble is the #1 correctness risk.)
8. **Order-2 FFT convolution** (loop runs exactly once, `o=0`, `x_i = x1`):
   - `g = v0 * x1` (elementwise gate), `[B,256,L]`.
   - `fft_size = 2L`. `k_f = rfft(k[0], n=2L)/(2L)`, `v_f = rfft(g, n=2L)`,
     `y = irfft(v_f*k_f, n=2L, norm='forward')[..., :L]`.
   - **Normalization**: `rfft` uses default `backward` (no scale) but `k_f` is explicitly
     divided by `2L`; `irfft` uses `norm='forward'` (no `1/n` on inverse). Net result is a
     properly scaled **linear (causal) convolution** of `g` with the length-L filter,
     truncated to the first L taps:
     `y[b,c,t] = sum_{s=0}^{t} k[c,s] * g[b,c,t-s]`, `t ∈ [0,L)`.
     Because the FFT is zero-padded to `>= 2L-1`, the first-L outputs are the exact linear
     conv — so an FFT of size `2L` and a **direct causal FIR of full length L give the same
     math** (only rounding differs). This is the key that lets me avoid FFT entirely.
   - skip/residual with per-channel filter bias:
     `vc = y + g * filter_bias[c]` (the `bias_reshaped[o]` term; note it multiplies the
     **gated** `g`, not the raw `v0`).
9. **Final gate + out projection**: `y2 = vc * x0` → logically `[B,256,L]`, transpose to
   `[B,L,256]`, then `hyena_out = Linear(y2, out_proj_weight[256,256], out_proj_bias)`.
10. **Residual add #1**: `r = hyena_out + residual` (residual = original fp32 `x`).
11. **LayerNorm2** over 256 (same formula, `norm2_*`).
12. **MLP**: `m = gelu_tanh(Linear(n2, fc1[1024,256], b1))`;
    `mlp_out = Linear(m, fc2[256,1024], b2)`. GELU is the tanh approximation.
13. **Residual add #2 / output**: `output = mlp_out + r` → `[B,L,256]`.

Equivalently `output = mlp_out + hyena_out + hidden_states`.

## 2. Cost / shape analysis (feedback set)

Workloads `(B, L)`: (1,1024) (2,4096) (16,512) (1,512) (4,541) (2,2048) (8,997) (32,256)
(1,2048) (2,128) (2,2053) (4,256) (1,4096) (1,256) (1,131) (2,512). L includes several
**non-power-of-two** values (541, 997, 2053, 131) — this rules out a naive radix-2 FFT.

Dominant compute (MAC counts, per workload, scale with `N = B*L` tokens):
- in_proj: `N*256*768`
- out_proj: `N*256*256`
- MLP fc1+fc2: `N*256*1024*2` (largest dense term)
- filter MLP: `L*(5*64+64*64+64*64+64*256)` — tiny, batch-independent.
- causal conv (depthwise): `~ B*256*L^2/2`.

Worst dense case `B=2,L=4096` (N=8192): MLP ≈ 4.3 GMAC, in_proj ≈ 1.6 GMAC,
conv ≈ `2*256*4096^2/2 ≈ 4.3 GMAC`. All within a few GFLOP → sub-millisecond on H100
even at modest efficiency. The many small shapes (e.g. 2×128, 1×131, 1×256, 32×256) are
**launch/overhead-bound** — fusion to reduce kernel count and intermediate DRAM traffic is
the main lever there.

## 3. Constraints & isolation

- Triton-only compute. `torch.fft`, `torch.nn.functional.linear/conv1d`, `torch.mm`, etc.
  are **computational** and disallowed as the implementation. PyTorch usage limited to
  `.shape`, allocation of output/scratch, dtype/stride metadata, and kernel launch.
- No fallback of any kind if a Triton kernel fails — must fix the kernel.
- Cannot use `torch.fft` → the FFT convolution **must** be reimplemented. Implementing a
  general (non-pow2, Bluestein/mixed-radix) FFT in Triton is impractical; therefore the
  plan is a **direct causal FIR convolution in Triton** (mathematically identical to the
  reference's zero-padded FFT conv on the first-L outputs).
- Evaluation only via `./scripts/evaluate_candidate.sh feedback <id>`; full 16-workload set
  = one candidate evaluation. Budget 100 evals; token soft limit 9M.
- Profiling only via `./scripts/ncu_profile.sh` (ncu-report-skill workflow), never
  overlapping an evaluation, never `ncu`/`nvidia-smi`/CUDA directly.
- **Local ad-hoc `python`/numpy execution is blocked in this environment.** I cannot run
  private correctness scripts, so kernels must be correct by construction and the scramble
  / normalization semantics verified through the evaluator (spend evals deliberately).

## 4. Numerical risks

Ranked by likelihood of causing a miss (tolerances: `atol ≈ 2.4e-4…3.2e-4`,
`rtol = 1e-5`, `match_ratio ≥ 0.98`):

1. **The `k` scramble reshape (§1.7).** Getting it wrong (e.g. implementing a clean
   transpose `h[b,a]`) silently corrupts every filter tap. Highest-risk item. Mitigation:
   replicate the flat `a*L+b` indexing exactly; validate on the smallest shape first.
2. **FFT-conv vs direct-conv rounding.** Direct fp32 summation of up to L≈4096 terms vs
   cuFFT. Both fp32; values are small (0.02-scale weights × decay). Expected abs error
   well under `3e-4`. The final output is dominated by the passed-through `hidden_states`
   (O(1)) plus small (~0.01–0.1) hyena/MLP corrections, so effective relative tolerance on
   the corrections is loose (~1e-3…1e-2). Use fp32 accumulation; consider pairwise/blocked
   summation if a shape drifts. `match_ratio 0.98` tolerates 2% outliers.
3. **Conv boundary handling.** Taps `{-2,-1,0}` for the short conv and the causal
   truncation for the long conv must zero out-of-range indices, not wrap. Off-by-one on the
   left pad (2 zeros) or on the first-L slice would shift the whole filter.
4. **LayerNorm variance.** `unbiased=False` (divide by N=256). Using N-1 would bias var.
5. **GELU flavor.** Must be `approximate="tanh"`
   (`0.5*x*(1+tanh(√(2/π)*(x+0.044715 x^3)))`), not the erf form.
6. **Positional embedding ordering.** `z = [t, cos(-f0 w), cos(-f1 w), sin(-f0 w),
   sin(-f1 w)]`; both cosines precede both sines. `f = [1e-4, 1.0]`. Wrong column order
   feeds `fl1` incorrectly.
7. **`sin_freq` scaling** applied inside every `sin` (before the linear? no — after: it's
   `sin(sin_freq * h)` where `h` is the linear output). All-ones in data but implement the
   multiply.
8. **`decay + exp_mod_shift` then `+ filter_bias`, and the bias term multiplies the gated
   `g` not raw `v0`.** Easy to misplace.
9. **Magnitude/overflow**: `|delta| ∈ [3.07, 15.35]`, `decay = exp(-t|delta|) ∈ (2e-7, 1]`
   — no overflow; sins bounded. Safe in fp32.

## 5. Triton design space

Kernels needed (all compute in Triton; torch only allocates + launches):

- **K1 — LN1 + in_proj + short conv + gate build.** Options:
  (a) fully fused per-token: one program per `(b, L-block)` computes LN1 over 256, then the
  768-wide projection (256→768 GEMV per token), then the depthwise conv needs neighbor
  tokens (t-1,t-2) → requires either recompute of projection for neighbors or a two-pass
  split. Cleaner: **K1a** LN1+in_proj → `u[B,768,L]` (channel-major for conv-friendly L
  access); **K1b** depthwise short conv + split + gate `g = v0*x1`, keeping `x0`, `x1`,
  `g`. Fusing LN1 into the GEMM saves a full `[B,L,256]` round-trip. The 256→768 matmul is
  small K=256; a standard Triton GEMM (BLOCK_M×BLOCK_N over K=256) works, weight resident.
- **K2 — filter generation** `[L,256]` (batch-independent, compute once per L): per-L-block
  program builds `z[L,5]`, runs the 3× (64-wide sin-MLP) + final 64→256 projection, applies
  decay + shift + filter_bias, and writes the **scrambled `k[256,L]`** directly (write
  `h[l,c]` to the flat address `l*256+c`, then a second view reads it as `[256,L]`; or fold
  the scramble into the address computation). Weights fit in SRAM
  (fl2/fl3 4KB each, fl_final 64KB — may keep final as a separate small GEMM). This kernel
  is cheap and reused across the batch.
- **K3 — depthwise causal FIR** `y[b,c,t] = Σ_{s≤t} k[c,s] g[b,c,t-s]`. Independent per
  `(b,c)`; block over output positions `t` (BLOCK_T) and stream the sum over `s` in tiles,
  loading `k[c,·]` and `g[b,c,·]` sub-tiles. This is a triangular (Toeplitz) reduction, no
  cross-channel reduction. Compute `O(L^2)` but simple FMA; fp32 accumulate. For very small
  L, a single-block direct loop suffices. Then fuse `vc = y + g*filter_bias[c]` and the
  final gate `y2 = vc * x0` and the transpose to `[B,L,256]` layout for K4.
  - Optimization axis: block the `s`-reduction; exploit that for `t < BLOCK` the triangle
    is small; optionally split into a "diagonal band" fast path. Possible future
    alternative: chunked/overlap-add short-FFT — deferred (non-pow2 lengths make it hard).
- **K4 — out_proj + residual1 + LN2 + fc1 + gelu + fc2 + residual2.** The tail is a
  per-token pipeline (all reductions are within a token's 256/1024 features). Candidate
  fusion: **K4a** out_proj(256→256)+add residual → `r`; **K4b** LN2+fc1(256→1024)+gelu;
  **K4c** fc2(1024→256)+add `r` → output. fc1/fc2 dominate; standard Triton GEMMs with
  K=256 and K=1024. Fusing LN2 into fc1 and the two residual adds into the epilogues cuts
  several `[B,L,256]`/`[B,L,1024]` DRAM passes.

Layout decisions:
- Keep `x` / residual / output in `[B,L,256]` row-major (matches I/O contract).
- Produce `u`, `x0`, `x1`, `g` in **channel-major `[B,256,L]`** so the conv reads
  contiguously along `L`; convert back to token-major at K3's epilogue (transpose fold into
  the write, or let out_proj K4a read strided).
- `k` stored `[256,L]` contiguous along `L`.

Precision: accumulate all GEMMs, LN, conv, and MLP reductions in fp32 (inputs already fp32;
no tensor-core fp16/tf32 downcast unless a later candidate shows it stays within tolerance —
tf32 on the matmuls is a tempting speed lever but risks the 1e-5 rtol / small-correction
accuracy, so start with fp32/IEEE math and only try tf32 as a measured optimization).

Autotuning axes: BLOCK_M/N/K for the GEMMs, BLOCK_T / reduction tiling for the conv,
num_warps, num_stages. Different regimes for tiny (L≤256, overhead-bound → fewer/bigger
fused launches) vs large (L≥2048, compute-bound → tuned GEMM/conv tiles).

## 6. Candidate roadmap (sketch; detailed plan later in docs/plan.md)

- **c001 — correctness-first, moderately fused.** Straightforward Triton kernels per §5
  (K1a/K1b, K2 with the scramble, K3 direct conv, K4a/b/c), fp32 IEEE throughout, safe
  block sizes. Goal: pass all 16 workloads within tolerance; establish the baseline speedup
  and confirm the scramble/normalization semantics empirically.
- Subsequent candidates: fuse LN into adjacent GEMM epilogues; tune GEMM/conv tiles per
  size regime; fuse the K4 tail; consider tf32 matmuls (guarded by tolerance); optimize the
  conv triangle (banded / block-skip). Each is a new immutable candidate ID.

Convergence: stop when geomean improvement plateaus or budgets (100 evals / 9M tokens soft)
are hit, then write `SEARCH_COMPLETE`.

## 7. Validation strategy

- **Primary**: `./scripts/evaluate_candidate.sh feedback cNNN` on the full 16-workload set
  (one eval). Every workload must pass `atol/rtol/match_ratio`. This is the only trusted
  correctness signal available (local python is blocked).
- **Economy**: because each check consumes an eval, build kernels correct-by-construction;
  reason through the scramble, normalization, boundary, and activation details (§1, §4)
  before the first eval. Order first-eval attention on the smallest shape's failure modes.
- **Debugging without local run**: if c001 misses, bisect by hypothesis using the ranked
  risks (§4) — scramble first, then conv boundaries, then LN/GELU/positional-order — each
  fix is a new candidate. Keep intermediate-stage structure so a wrong stage is
  localizable.
- **Performance**: after correctness, profile with `./scripts/ncu_profile.sh --set ...`
  (ncu-report-skill), never overlapping an eval; use KernelWiki for H100/Hopper GEMM +
  fusion + warp-specialization guidance on tuning K1/K4 GEMMs and the conv.
- Record for every candidate in `candidates.jsonl`: parent, source hash, hypothesis,
  validation, per-workload result, geomean, decision, cumulative eval count, skill usage.
