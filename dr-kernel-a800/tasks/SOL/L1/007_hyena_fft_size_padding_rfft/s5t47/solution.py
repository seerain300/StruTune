import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_bins(x_ptr, out_ptr, N, invN,
                    BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for bins j = 0..seqlen:
      real_out[j] = (1/N) * sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N)
    x_ptr: pointer to padded input vector, length N
    out_ptr: pointer to output vector, length seqlen+1
    N: int, padded length (2 * seqlen)
    invN: float, 1.0 / N
    """
    pid = tl.program_id(0)  # one program per (b, c) row
    # Loop over j bins
    # Triton supports dynamic for-loops; we use a range to iterate bins.
    # However, Triton requires static control flow. Instead, we launch this kernel with grid=(M,) and have
    # each program compute all j bins sequentially. We rely on a fixed iteration count by using tl.static_range
    # if we pass the number of bins as a constexpr. Here, we pass seqlen+1 as a constexpr via launch-time meta.

    # We cannot access constexprs from dynamic control flow directly; thus we provide a wrapper that passes
    # the number of bins as meta. Triton doesn't allow arbitrary meta parameters in function signature; so
    # we restructure: launch one program per row, and inside compute j loops using Python range. Triton will
    # treat range as a compile-time unrolled loop if the bounds are constexpr. To ensure, we instead launch
    # separate kernels per bin j.

    # Since Triton kernels cannot accept dynamic loops with Python range, we provide a kernel that computes
    # one bin per launch. Therefore, we will define a different kernel below that computes a single j.

@triton.jit
def rfft_real_one_j(x_ptr, out_ptr_j, N, j, invN,
                    BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for a fixed j across all rows:
      real_out[j] = (1/N) * sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N)
    x_ptr: pointer to padded input vector, length N
    out_ptr_j: pointer to scalar output for this j (1D vector of length M)
    N: int, padded length (2 * seqlen)
    j: int, bin index (0..seqlen)
    invN: float, 1.0 / N
    """
    pid = tl.program_id(0)  # one program per row
    # Accumulator
    acc = 0.0
    # Loop over k in chunks
    for k0 in range(0, N, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < N
        xk = tl.load(x_ptr + offs, mask=mask, other=0.0)  # xk is float32
        angle = 2.0 * 3.141592653589793 * j * offs / N
        cosv = tl.cos(angle)
        prod = xk * cosv
        # Reduce across the block
        block_sum = tl.sum(prod, axis=0)
        acc += block_sum

    out_val = acc * invN
    # Store scalar to out_ptr_j[pid]
    tl.store(out_ptr_j + pid, out_val)


@triton.jit
def rfft_imag_one_j(x_ptr, out_ptr_j, N, j, invN,
                    BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for a fixed j across all rows:
      imag_out[j] = (1/N) * sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N)
    x_ptr: pointer to padded input vector, length N
    out_ptr_j: pointer to scalar output for this j (1D vector of length M)
    N: int, padded length (2 * seqlen)
    j: int, bin index (1..seqlen-1)
    invN: float, 1.0 / N
    """
    pid = tl.program_id(0)  # one program per row
    acc = 0.0
    for k0 in range(0, N, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < N
        xk = tl.load(x_ptr + offs, mask=mask, other=0.0)
        angle = 2.0 * 3.141592653589793 * j * offs / N
        sinv = tl.sin(angle)
        prod = xk * sinv
        block_sum = tl.sum(prod, axis=0)
        acc += block_sum

    out_val = acc * invN
    tl.store(out_ptr_j + pid, out_val)


def _compute_bins_triton(x: torch.Tensor, batch: int, channels: int, seqlen: int):
    """
    Compute real and imaginary parts of rfft bins using Triton kernels.
    Returns:
      real_out: tensor of shape (batch, channels, seqlen+1), float32
      imag_out: tensor of shape (batch, channels, seqlen+1), float32
    """
    device = x.device
    # Flatten rows
    M = batch * channels

    real_out = torch.empty((batch, channels, seqlen + 1), device=device, dtype=torch.float32)
    imag_out = torch.empty((batch, channels, seqlen + 1), device=device, dtype=torch.float32)

    # Precompute invN
    N = 2 * seqlen
    invN = 1.0 / N

    # Launch kernels per bin j
    # Triton kernels require grid to be specified. Each kernel computes one j-bin for all rows.
    # We set grid=(M,) where M is the number of rows (batch*channels).
    BLOCK_K = 256

    # Real bins: j in 0..seqlen
    for j in range(0, seqlen + 1):
        out_j = torch.empty((M,), device=device, dtype=torch.float32)
        rfft_real_one_j[(M,)](x, out_j, N, j, invN, BLOCK_K=BLOCK_K, num_warps=2)
        # Scatter back to (batch, channels) layout
        # out_j shape (M,); map pid -> (b, c)
        b_idx = torch.div(torch.arange(M, device=device), channels, rounding_mode='floor')
        c_idx = torch.remainder(torch.arange(M, device=device), channels)
        real_out[b_idx, c_idx, j] = out_j

    # Imag bins: j in 1..seqlen-1
    for j in range(1, seqlen):
        out_j = torch.empty((M,), device=device, dtype=torch.float32)
        rfft_imag_one_j[(M,)](x, out_j, N, j, invN, BLOCK_K=BLOCK_K, num_warps=2)
        b_idx = torch.div(torch.arange(M, device=device), channels, rounding_mode='floor')
        c_idx = torch.remainder(torch.arange(M, device=device), channels)
        imag_out[b_idx, c_idx, j] = out_j

    # Set imag_out[0] and imag_out[seqlen] to zero
    imag_out[:, :, 0] = 0
    imag_out[:, :, -1] = 0

    return real_out, imag_out


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Fused FFT size padding and real FFT computation for Hyena convolution.
        Args:
            x: Input tensor of shape (batch, channels, seqlen), float32 on CUDA.
        Returns:
            x_freq_real: Real part of normalized frequency domain output (batch, channels, seqlen+1), float32
            x_freq_imag: Imaginary part of normalized frequency domain output (batch, channels, seqlen+1), float32
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        assert x.dtype == torch.float32, "Input must be float32 for numerical stability."
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)."

        batch, channels, seqlen = x.shape

        # Build padded input per row on host using PyTorch (no torch math beyond indexing/allocations).
        # We need to zero-pad each row to N=2*seqlen.
        # To ensure Triton can read these vectors, we allocate a temporary


def run(*args):
    return ModelNew()(*args)
