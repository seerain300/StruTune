# KDA A800 best solution: L1/007_hyena_fft_size_padding_rfft
# candidate: c001  |  feedback: 0.08x  |  final (authoritative): 0.10x
# campaign formal-kda-20260916 (A800, g0056)  |  evaluations: 1
# source: tasks/formal-kda-20260916--sol_execbench--L1-007_hyena_fft_size_padding_rfft/control/candidates/c001/solution.py (sha256-frozen snapshot)

"""
L1/007 hyena_fft_size_padding_rfft  --  candidate c001

Dense DFT-as-GEMM baseline (pure Triton compute).

The reference computes, for input x : (batch, channels, seqlen), float32:
    N = 2 * seqlen                       (fft_size, 2x zero-padding)
    X = rfft(x, n=N) / N                 -> (batch, channels, seqlen+1) complex
    return X.real.contiguous(), X.imag.contiguous()

Because only the first `seqlen` samples are non-zero, the padded DFT collapses to
    X[k] = sum_{n=0}^{seqlen-1} x[n] * exp(-2*pi*i * k*n / N),   k = 0 .. seqlen
    R[k] = (1/N) * sum_n x[n] * cos(2*pi*k*n/N)
    I[k] = -(1/N) * sum_n x[n] * sin(2*pi*k*n/N)

Flattening (batch, channels) into M = batch*channels rows, this is two real GEMMs
    R = ( Xmat @ Cos^T ) / N ,   I = -( Xmat @ Sin^T ) / N
with twiddle matrices Cos[k,n]=cos(2*pi*k*n/N), Sin[k,n]=sin(2*pi*k*n/N).

Implementation notes / correctness gates:
  * Compute is Triton only.  PyTorch is used only for shape/stride/dtype bookkeeping
    and kernel launch.  No Torch/CPU/NumPy computational fallback.
  * fp32 accumulation everywhere; every tl.dot uses input_precision="ieee" (TF32 is
    forbidden -- its ~1e-3 error would fail the 1e-5 tolerance instantly).
  * Twiddle phase is argument-reduced in *integer* arithmetic: m = (k*n) mod N is
    formed in int32 (k*n <= ~6.7e7 < 2^31), then the twiddle is looked up from an
    O(N) root-of-unity table built once by a Triton kernel and memoized per seqlen.
    This avoids forming 2*pi*k*n/N in fp32 where k*n overflows the 2^24 exact-int range.
  * sin is snapped to exactly 0 at m=0 (DC) and m=N/2 (Nyquist, N even) so the
    imaginary output is exactly 0 there, matching the Hermitian invariants.
"""

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Twiddle root-of-unity table generation (Triton compute, done once per seqlen)
# ---------------------------------------------------------------------------
@triton.jit
def _twiddle_table_kernel(cos_ptr, sin_ptr, N, TWO_PI, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    m = offs.to(tl.float32)
    # angle in [0, 2*pi); m and N are exact in fp32 (N <= 2^15 here)
    angle = TWO_PI * (m / N)
    c = tl.cos(angle)
    s = tl.sin(angle)
    # Snap sin to exactly 0 at DC (m=0) and Nyquist (2*m == N) to keep imag clean.
    is_zero = offs == 0
    is_half = (2 * offs) == N
    s = tl.where(is_zero | is_half, 0.0, s)
    tl.store(cos_ptr + offs, c, mask=mask)
    tl.store(sin_ptr + offs, s, mask=mask)


# ---------------------------------------------------------------------------
# Dense DFT-as-GEMM kernel: R = (Xmat @ Cos^T)/N , I = -(Xmat @ Sin^T)/N
# ---------------------------------------------------------------------------
@triton.jit
def _dft_gemm_kernel(
    x_ptr, cos_ptr, sin_ptr, r_ptr, i_ptr,
    M, KC, F, N, INV_N,
    stride_xm, stride_xk,
    stride_om, stride_of,
    BLOCK_M: tl.constexpr, BLOCK_F: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_f = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)

    acc_r = tl.zeros((BLOCK_M, BLOCK_F), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_F), dtype=tl.float32)

    f_row = offs_f[None, :]  # [1, BLOCK_F]

    for k0 in range(0, KC, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # load x tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < KC)
        a = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # twiddle index tile [BLOCK_K, BLOCK_F]: m = (n * f) mod N, integer arithmetic
        n_col = offs_k[:, None]                # [BLOCK_K, 1]
        prod = n_col * f_row                   # [BLOCK_K, BLOCK_F] int32, <= ~6.7e7
        midx = prod % N                        # in [0, N)
        cosb = tl.load(cos_ptr + midx)         # always in-bounds (0 <= midx < N)
        sinb = tl.load(sin_ptr + midx)

        acc_r = tl.dot(a, cosb, acc_r, input_precision="ieee")
        acc_i = tl.dot(a, sinb, acc_i, input_precision="ieee")

    r = acc_r * INV_N
    im = acc_i * (-INV_N)

    out_mask = (offs_m[:, None] < M) & (offs_f[None, :] < F)
    o_off = offs_m[:, None] * stride_om + offs_f[None, :] * stride_of
    tl.store(r_ptr + o_off, r, mask=out_mask)
    tl.store(i_ptr + o_off, im, mask=out_mask)


# ---------------------------------------------------------------------------
# Twiddle cache (depends only on seqlen + device; not on x values)
# ---------------------------------------------------------------------------
_TWIDDLE_CACHE = {}


def _get_twiddle(seqlen, device):
    N = 2 * seqlen
    key = (seqlen, torch.device(device).type, torch.device(device).index)
    entry = _TWIDDLE_CACHE.get(key)
    if entry is not None:
        return entry
    cos_t = torch.empty(N, device=device, dtype=torch.float32)
    sin_t = torch.empty(N, device=device, dtype=torch.float32)
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _twiddle_table_kernel[grid](cos_t, sin_t, N, 2.0 * math.pi, BLOCK=BLOCK)
    _TWIDDLE_CACHE[key] = (cos_t, sin_t)
    return cos_t, sin_t


# tiling
_BLOCK_M = 64
_BLOCK_F = 64
_BLOCK_K = 64


@torch.no_grad()
def run(x: torch.Tensor):
    assert x.dim() == 3, "expected (batch, channels, seqlen)"
    batch, channels, seqlen = x.shape
    N = 2 * seqlen
    F = seqlen + 1
    M = batch * channels

    x = x.contiguous()
    xmat = x.view(M, seqlen)

    cos_t, sin_t = _get_twiddle(seqlen, x.device)

    r = torch.empty((M, F), device=x.device, dtype=torch.float32)
    im = torch.empty((M, F), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(F, _BLOCK_F))
    _dft_gemm_kernel[grid](
        xmat, cos_t, sin_t, r, im,
        M, seqlen, F, N, 1.0 / N,
        xmat.stride(0), xmat.stride(1),
        r.stride(0), r.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_F=_BLOCK_F, BLOCK_K=_BLOCK_K,
        num_warps=4, num_stages=2,
    )

    x_freq_real = r.view(batch, channels, F)
    x_freq_imag = im.view(batch, channels, F)
    return x_freq_real, x_freq_imag
