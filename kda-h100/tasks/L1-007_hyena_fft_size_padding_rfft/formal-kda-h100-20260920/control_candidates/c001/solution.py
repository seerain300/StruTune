"""
KDA candidate c001 — L1/007 hyena_fft_size_padding_rfft

Operation (see docs/draft.md §1):
    x : (batch, channels=256, seqlen=L)  float32
    N  = 2*L (fft_size),  P = L+1 (rfft output bins)
    rfft(x, n=N)/N  ==  a *partial* DFT over the L nonzero samples:

        real_out[m,k] = (1/N) * sum_{n=0}^{L-1} x[m,n] * cos(pi*k*n/L)
        imag_out[m,k] = (1/N) * sum_{n=0}^{L-1} x[m,n] * (-sin(pi*k*n/L))
        for k = 0 .. L  (P = L+1 bins)

    This is two real GEMMs sharing the left operand X:(M x L) against the
    on-the-fly DFT matrices C:(L x P) [cos] and S:(L x P) [-sin], with
    M = batch*256.

Numerics (docs/draft.md §4):
  * fp32 accumulation (tl.dot input_precision="ieee") -> normalized error
    ~eps/2, independent of L, safely under the 1e-5 tolerance on all shapes.
  * Argument reduction: theta = pi*k*n/L reaches ~3.4e9 rad at L=32768, far
    beyond fp32 range. We form the index product k*n in int64, reduce
    r = (k*n) mod N first, then theta = 2*pi*r/N with r in [0,N) so the
    fp32 angle stays in [0, 2*pi) and cos/sin are accurate.

Triton-only compute; PyTorch is used solely for metadata / launch plumbing.
No Torch / cuFFT / CPU / NumPy computational fallback.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dft_kernel(
    x_ptr,            # *f32  (M, L)
    real_ptr,         # *f32  (M, P)
    imag_ptr,         # *f32  (M, P)
    M, L, P, N,       # sizes; N = 2*L (fft_size)
    two_pi_over_n,    # f32   2*pi / N
    inv_n,            # f32   1 / N
    stride_xm, stride_xk,
    stride_om, stride_op,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_p = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)       # row index m in [0, M)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)       # bin index k in [0, P)
    offs_p_i64 = offs_p.to(tl.int64)

    acc_real = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
    acc_imag = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)

    for k0 in range(0, L, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)               # sample index n in [0, L)
        # X tile (BLOCK_M x BLOCK_K); zero-padded outside [0, L) so masked
        # samples contribute nothing to either GEMM.
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < L)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=x_mask,
            other=0.0,
        )

        # Twiddle tile (BLOCK_K x BLOCK_P): r = (n*k) mod N in int64, then
        # theta = 2*pi*r/N in [0, 2*pi).
        prod = offs_k[:, None].to(tl.int64) * offs_p_i64[None, :]
        r = prod % N
        theta = r.to(tl.float32) * two_pi_over_n
        cos_t = tl.cos(theta)
        nsin_t = -tl.sin(theta)

        acc_real += tl.dot(x_tile, cos_t, input_precision="ieee")
        acc_imag += tl.dot(x_tile, nsin_t, input_precision="ieee")

    acc_real = acc_real * inv_n
    acc_imag = acc_imag * inv_n

    out_mask = (offs_m[:, None] < M) & (offs_p[None, :] < P)
    out_off = offs_m[:, None] * stride_om + offs_p[None, :] * stride_op
    tl.store(real_ptr + out_off, acc_real, mask=out_mask)
    tl.store(imag_ptr + out_off, acc_imag, mask=out_mask)


@torch.no_grad()
def run(x: torch.Tensor):
    """Fused FFT-size-padding + rfft (normalized) for Hyena, Triton kernel.

    Args:
        x: (batch, channels, seqlen) float32 CUDA tensor.
    Returns:
        (x_freq_real, x_freq_imag), each (batch, channels, seqlen+1) float32.
    """
    assert x.dim() == 3, "expected (batch, channels, seqlen)"
    batch, channels, seqlen = x.shape
    L = int(seqlen)
    N = 2 * L
    P = L + 1
    M = batch * channels

    x = x.contiguous()
    x2d = x.view(M, L)

    real2d = torch.empty((M, P), dtype=torch.float32, device=x.device)
    imag2d = torch.empty((M, P), dtype=torch.float32, device=x.device)

    BLOCK_M = 64
    BLOCK_P = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(P, BLOCK_P))

    _dft_kernel[grid](
        x2d,
        real2d,
        imag2d,
        M, L, P, N,
        float(2.0 * math.pi / N),
        float(1.0 / N),
        x2d.stride(0), x2d.stride(1),
        real2d.stride(0), real2d.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_P=BLOCK_P,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    x_freq_real = real2d.view(batch, channels, P)
    x_freq_imag = imag2d.view(batch, channels, P)
    return x_freq_real, x_freq_imag
