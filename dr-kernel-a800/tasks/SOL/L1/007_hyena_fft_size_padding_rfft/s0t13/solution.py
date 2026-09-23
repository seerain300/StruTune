import math
import torch
import triton
import triton.language as tl


@triton.jit
def bitreverse_pairs_kernel(t_ptr, HALF: tl.constexpr):
    """
    In-place bit-reverse pairing for interleaved real/imag time-domain vector t of length 2*HALF elements.
    For i in [0, HALF), compute rev = bit_reverse(i, HALF) and swap t[2*i] with t[2*rev],
    and t[2*i+1] with t[2*rev+1].
    Note: t_ptr points to either t_real or t_imag buffer of length 2*HALF.
    """
    i = tl.program_id(axis=0)
    # Only run for i < HALF
    # Compute bit-reverse of i in [0, HALF)
    rev = tl.zeros((), dtype=tl.int32)
    j = tl.zeros((), dtype=tl.int32)
    # Assuming HALF fits within 16 bits (enough for typical seqlen); bit-reverse
    while j < 16:
        b = (i >> (15 - j)) & 1
        rev ^= b << j
        j += 1
    # Swap real and imag at i and rev
    base_i = 2 * i
    base_rev = 2 * rev
    real_i = tl.load(t_ptr + base_i)
    imag_i = tl.load(t_ptr + base_i + 1)
    real_rev = tl.load(t_ptr + base_rev)
    imag_rev = tl.load(t_ptr + base_rev + 1)
    tl.store(t_ptr + base_i, real_rev)
    tl.store(t_ptr + base_i + 1, imag_rev)
    tl.store(t_ptr + base_rev, real_i)
    tl.store(t_ptr + base_rev + 1, imag_i)


@triton.jit
def real_fft_stage_kernel(t_ptr, N: tl.constexpr, k: tl.constexpr, ccos: tl.constexpr, cssin: tl.constexpr):
    """
    Perform one stage of radix-2 Cooley-Tukey FFT on interleaved real/imag t of length 2*N.
    Stage k: for i in [0, N//2), compute j = i ^ k, and update pairs (i, j).
    Note: N here is actually HALF length processed. For k in {N, N//2, ...}, j = i ^ k < N.
    """
    HALF = N  # we process indices i in [0, HALF), corresponding to first half in 2*N
    i = tl.program_id(axis=0)
    while i < HALF:
        j = i ^ k
        # Load u = t_real[i] + i t_imag[i], v = t_real[j] + i t_imag[j]
        u_real = tl.load(t_ptr + 2 * i)
        u_imag = tl.load(t_ptr + 2 * i + 1)
        v_real = tl.load(t_ptr + 2 * j)
        v_imag = tl.load(t_ptr + 2 * j + 1)
        # theta = 2*pi*k/(2*HALF) = 2*pi*k/(2*N) where N=HALF after padding
        # But we passed N as total length for cos/sin; safer to compute using 2*HALF overall:
        # We can use N passed in as 2*HALF length; angle is consistent since t_ptr is of length 2*N.
        theta = (2.0 * 3.141592653589793 * k) / (2 * HALF)
        c = ccos  # cos(theta)
        s = cssin  # sin(theta)
        # alpha = u*c + v*s ; beta = -u*s + v*c
        alpha_real = u_real * c + v_real * s
        alpha_imag = u_imag * c + v_imag * s
        beta_real = -u_real * s + v_real * c
        beta_imag = -u_imag * s + v_imag * c
        # Store back to i and j (alpha for both i and j)
        tl.store(t_ptr + 2 * i, alpha_real)
        tl.store(t_ptr + 2 * i + 1, alpha_imag)
        tl.store(t_ptr + 2 * j, alpha_real)
        tl.store(t_ptr + 2 * j + 1, alpha_imag)
        i += 1


@triton.jit
def divide_inplace_kernel(x_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise divide in-place: x[i] = x[i] / scale
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = x / scale
    tl.store(x_ptr + offsets, x, mask=mask)


def _run_rfft_triton(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Compute rFFT of real input x via Triton: output real and imaginary parts of shape (batch, channels, seqlen+1).
    Returns out_real, out_imag (float32 tensors).
    """
    # Ensure CUDA and float32
    assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
    assert x.dtype == torch.float32, "Input must be float32 for numerical stability."

    batch, channels, seqlen = x.shape
    N = 2 * seqlen  # padded length for rfft with n
    HALF = seqlen   # we process first half

    # Flatten input to 1D for simplicity
    x_flat = x.reshape(-1).contiguous()
    # Prepare time-domain buffers: t_real of length N, t_imag of length N (zeros)
    t_real = torch.empty(N, dtype=torch.float32, device=x.device)
    t_imag = torch.empty(N, dtype=torch.float32, device=x.device)

    # Copy input into first half and zero the second half
    t_real[:HALF] = x_flat[:HALF]
    t_imag[:HALF] = 0.0
    # Second half zeros for real input
    t_real[HALF:] = 0.0
    t_imag[HALF:] = 0.0

    # Launch bit-reverse pairing kernel (interleaved real/imag)
    grid_bit = (HALF,)
    bitreverse_pairs_kernel[grid_bit](t_real, HALF)
    bitreverse_pairs_kernel[grid_bit](t_imag, HALF)

    # Perform stages: k = N//2, N//4, ..., 1
    # Note: N is 2*HALF, so k values are HALF, HALF//2, ..., 1
    k = HALF
    while k >= 1:
        theta = (2.0 * 3.141592653589793 * k) / (2 * HALF)  # angle for 2*HALF
        c = math.cos(theta)
        s = math.sin(theta)
        grid_stage = (HALF,)
        real_fft_stage_kernel[grid_stage](t_real, HALF, k, c, s)
        real_fft_stage_kernel[grid_stage](t_imag, HALF, k, c, s)
        k //= 2

    # Extract first HALF bins: out_real[i] = t_real[2*i], out_imag[i] = t_imag[2*i]
    out_real = torch.empty(HALF, dtype=torch.float32, device=x.device)
    out_imag = torch.empty(HALF, dtype=torch.float32, device=x.device)
    for i in range(HALF):
        out_real[i] = t_real[2 * i]
        out_imag[i] = t_imag[2 * i]

    # Normalize by N = 2*seqlen using Triton
    scale = float(N)
    grid_div = (triton.cdiv(HALF, 1024),)
    divide_inplace_kernel[grid_div](out_real, HALF, scale, BLOCK_SIZE=1024)
    divide_inplace_kernel[grid_div](out_imag, HALF, scale, BLOCK_SIZE=1024)

    # Reshape to (batch, channels, seqlen+1)
    out_real = out_real.view(batch, channels, seqlen + 1)
    out_imag = out_imag.view(batch, channels, seqlen + 1)
    return out_real, out_imag


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure CUDA and float32; use Triton kernels for all computation
        if not x.is_cuda:
            x = x.cuda()
        x = x.to(torch.float32)
        out_real, out_imag = _run_rfft_triton(x)
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
