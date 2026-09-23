import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded_kernel(x_ptr, padded_ptr, L: tl.int32, N: tl.int32):
    # One program per (b, c) slice: x_ptr points to the start of the row (length L),
    # padded_ptr points to the start of the corresponding padded row (length N).
    pid = tl.program_id(0)
    # Only copy first L elements; padded[0:L] = x[0:L], padded[L:] remains zero (host pre-zeroed)
    for i in tl.static_range(0, L):
        val = tl.load(x_ptr + i)
        tl.store(padded_ptr + i, val)


@triton.jit
def _accumulate_real_kernel(padded_ptr, out_real_ptr, L: tl.int32, N: tl.int32, K: tl.int32):
    # Compute y_real[K] = sum_{j=0}^{N-1} padded[j] * cos(2*pi*K*j/N)
    inv_N = 1.0 / N
    acc = 0.0
    # Loop over j
    for j in tl.static_range(0, N):
        v = tl.load(padded_ptr + j)
        angle = 2.0 * 3.141592653589793 * (K * j) / N
        c = tl.cos(angle)
        acc += v * c
    # Normalize by N (original code divides by 2*L)
    acc = acc * inv_N
    tl.store(out_real_ptr + K, acc)


@triton.jit
def _accumulate_imag_kernel(padded_ptr, out_imag_ptr, L: tl.int32, N: tl.int32, K: tl.int32):
    # Compute y_imag[K] = sum_{j=0}^{N-1} padded[j] * sin(2*pi*K*j/N)
    inv_N = 1.0 / N
    acc = 0.0
    for j in tl.static_range(0, N):
        v = tl.load(padded_ptr + j)
        angle = 2.0 * 3.141592653589793 * (K * j) / N
        s = tl.sin(angle)
        acc += v * s
    acc = acc * inv_N  # normalize by N (divide by 2*L)
    tl.store(out_imag_ptr + K, acc)


def triton_rfft_pad_and_compute(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    x: (B, C, L) float32 tensor on CUDA
    Returns (real_out, imag_out): both (B, C, L+1) float32 tensors
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernels"
    assert x.dtype == torch.float32, "Input must be float32"

    B, C, L = x.shape
    N = 2 * L  # original code pads to 2*L

    # Prepare padded input: zeros for tail
    padded = torch.zeros((B, C, N), device=x.device, dtype=torch.float32)

    # Copy x[:, :, :] into padded[:, :, 0:L]
    # One program per (b, c) slice
    grid_copy = (B * C,)
    _copy_row_to_padded_kernel[grid_copy](
        x.view(B * C, L),
        padded.view(B * C, N),
        L, N
    )

    # Output buffers
    real_out = torch.zeros((B, C, L + 1), device=x.device, dtype=torch.float32)
    imag_out = torch.zeros((B, C, L + 1), device=x.device, dtype=torch.float32)

    # Accumulate real and imaginary parts for k in [0..L]
    # One program per k
    for k in range(L + 1):
        _accumulate_real_kernel[(1,)](padded.view(B * C, N), real_out.view(B * C, L + 1), L, N, k)
        _accumulate_imag_kernel[(1,)](padded.view(B * C, N), imag_out.view(B * C, L + 1), L, N, k)

    return real_out, imag_out


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure CUDA and dtype float32, as original code casts to float32
        if not x.is_cuda:
            raise AssertionError("Input must be on CUDA for Triton kernels")
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        real_out, imag_out = triton_rfft_pad_and_compute(x)
        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
