import torch
import triton
import triton.language as tl


@triton.jit
def bitreverse_pairs_kernel(real_ptr, imag_ptr, S: tl.constexpr):
    """
    Bit-reverse pair in-place for the real and imaginary time-domain halves:
    For i in [0, S), swap (i, rev) where rev is the bit-reversed index of i in [S, 2S).
    real[rev + S] and imag[rev + S] are paired with real[i] and imag[i].
    """
    HALF = S
    i = tl.program_id(axis=0)
    while i < HALF:
        rev = tl.zeros((), dtype=tl.int32)
        j = tl.zeros((), dtype=tl.int32)
        # 16-bit bit-reverse for indices up to 65535
        while j < 16:
            b = (i >> (15 - j)) & 1
            rev ^= b << j
            j += 1
        # Load pairs
        a = tl.load(real_ptr + i)
        c = tl.load(imag_ptr + i)
        b = tl.load(real_ptr + rev + HALF)
        d = tl.load(imag_ptr + rev + HALF)
        # Store swapped pairs
        tl.store(real_ptr + i, b)
        tl.store(imag_ptr + i, d)
        tl.store(real_ptr + rev + HALF, a)
        tl.store(imag_ptr + rev + HALF, c)
        i += 1


@triton.jit
def real_complex_fft_stages_kernel(real_ptr, imag_ptr, N: tl.constexpr):
    """
    In-place complex Cooley-Tukey FFT for a time vector of length N (complex).
    Iterates k = 1, 2, 4, ... until k >= N. For each stage, i in [0, N/2):
      idx = i & (k/2)
      theta = 2*pi * idx * k / N
      c = cos(theta), s = sin(theta)
      a = real[i], b = real[i + N/2], ca = imag[i], da = imag[i + N/2]
      ar = a + b, br = (a - b) * c + (ca - da) * s
      ai = (da - ca) * c + (a - b) * s, bi = b (not used for update)
      real[i] = ar, real[i + N/2] = br
      imag[i] = ai, imag[i + N/2] = bi
    Assumes N is power-of-two up to 65536.
    """
    HALF = N // 2
    k = 1
    while k < N:
        step = k // 2
        i = tl.zeros((), dtype=tl.int32)
        while i < HALF:
            idx = i & step
            theta = (2.0 * 3.141592653589793 * idx * k) / N
            c = tl.cos(theta)
            s = tl.sin(theta)
            a = tl.load(real_ptr + i)
            b = tl.load(real_ptr + i + HALF)
            ca = tl.load(imag_ptr + i)
            da = tl.load(imag_ptr + i + HALF)
            ar = a + b
            diff_real = a - b
            diff_imag = ca - da
            br = diff_real * c + diff_imag * s
            ai = diff_imag * c - diff_real * s  # (da - ca)*c + (a - b)*s is same as diff_imag*c - diff_real*s
            # Update outputs
            tl.store(real_ptr + i, ar)
            tl.store(imag_ptr + i, ai)
            tl.store(real_ptr + i + HALF, br)
            tl.store(imag_ptr + i + HALF, 0.0)  # bi is not used for real output; leave 0 for safety
            i += 1
        k *= 2


@triton.jit
def normalize_divide_kernel(out_ptr, n_elements, scale: tl.float32, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise division: out[i] = out[i] / scale for i in [0, n_elements).
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(out_ptr + offsets, mask=mask, other=0.0)
    x = x / scale
    tl.store(out_ptr + offsets, x, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only forward:
        - Convert input x (float32) and construct real/imag time-domain vectors for rfft of length N = 2*seqlen.
        - Bit-reverse pair the first half with the second half.
        - Perform complex FFT in Triton (radix-2 stages) on the vector.
        - Extract the first seqlen+1 complex bins: real and imaginary parts.
        - Normalize by 2*seqlen using Triton elementwise kernel.
        Returns:
          x_freq_real: (batch, channels, seqlen+1), float32
          x_freq_imag: (batch, channels, seqlen+1), float32
        """
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        batch, channels, seqlen = x.shape
        N = 2 * seqlen

        # Ensure float32 input for stability
        x_f32 = x.to(torch.float32).contiguous()

        # Allocate real and imag time-domain buffers (complex input for rfft)
        real = torch.empty(N, dtype=torch.float32, device=x.device)
        imag = torch.zeros(N, dtype=torch.float32, device=x.device)

        # Initialize real[0:S] = x_f32, real[S:2S] = 0, imag[:] = 0
        real[:seqlen] = x_f32
        imag.zero_()

        # Bit-reverse pairing: one program per index in first half
        HALF = seqlen
        grid_bitrev = (HALF,)
        bitreverse_pairs_kernel[grid_bitrev](real, imag, S=seqlen)

        # Complex FFT stages: single grid (we iterate vectors within kernel)
        grid_stages = (1,)
        real_complex_fft_stages_kernel[grid_stages](real, imag, N=N)

        # Extract first seqlen+1 bins: real[0:S], imag[0:S]
        out_real = real[:seqlen + 1].contiguous()
        out_imag = imag[:seqlen + 1].contiguous()

        # Normalize by 2*seqlen using Triton kernel
        n_real = out_real.numel()
        n_imag = out_imag.numel()
        scale = float(N)

        BLOCK_SIZE = 1024
        grid_norm = (triton.cdiv(n_real, BLOCK_SIZE),)
        normalize_divide_kernel[grid_norm](out_real, n_real, scale, BLOCK_SIZE=BLOCK_SIZE)
        grid_norm2 = (triton.cdiv(n_imag, BLOCK_SIZE),)
        normalize_divide_kernel[grid_norm2](out_imag, n_imag, scale, BLOCK_SIZE=BLOCK_SIZE)

        # Reshape to (batch, channels, seqlen + 1)
        x_freq_real = out_real.view(batch, channels, seqlen + 1)
        x_freq_imag = out_imag.view(batch, channels, seqlen + 1)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
