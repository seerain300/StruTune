import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded_kernel(x_row_ptr, padded_ptr, L, n, BLOCK: tl.constexpr):
    """
    x_row_ptr: *float32, length L, row of input for a (batch, channel) slice
    padded_ptr: *float32, length n, output buffer with x[0..L-1] followed by zeros
    """
    j = tl.program_id(axis=0)
    if j < L:
        tl.store(padded_ptr + j, tl.load(x_row_ptr + j))


@triton.jit
def _direct_rfft_real_padded_kernel(padded_ptr, out_real_ptr, L, n, BLOCK_K: tl.constexpr):
    """
    Compute real part of rFFT for k in [0..L] using direct summation over j=0..n-1:
    y[k] = sum_{j=0}^{n-1} padded[j] * cos(2*pi*k*j / n)
    padded_ptr: *float32, length n = 2*L, containing x[0..L-1] followed by zeros
    out_real_ptr: *float32, length (L+1), stores y[0..L]
    L: int, original seqlen
    n: int, 2*L
    """
    k0 = tl.program_id(axis=0) * BLOCK_K
    ks = k0 + tl.arange(0, BLOCK_K)
    mask_k = ks <= L

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Sum over j from 0 to n-1
    for j in range(0, n):
        xj = tl.load(padded_ptr + j)
        angle = 2.0 * 3.141592653589793 * ks * j * (1.0 / n)
        cosv = tl.cos(angle)
        acc += xj * cosv

    tl.store(out_real_ptr + ks, acc, mask=mask_k)


@triton.jit
def _direct_rfft_imag_padded_kernel(padded_ptr, out_imag_ptr, L, n, BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rFFT for k in [0..L] using direct summation over j=0..n-1:
    y[k] = sum_{j=0}^{n-1} padded[j] * sin(2*pi*k*j / n)
    padded_ptr: *float32, length n = 2*L, containing x[0..L-1] followed by zeros
    out_imag_ptr: *float32, length (L+1), stores y[0..L]
    L: int, original seqlen
    n: int, 2*L
    """
    k0 = tl.program_id(axis=0) * BLOCK_K
    ks = k0 + tl.arange(0, BLOCK_K)
    mask_k = ks <= L

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Sum over j from 0 to n-1
    for j in range(0, n):
        xj = tl.load(padded_ptr + j)
        angle = 2.0 * 3.141592653589793 * ks * j * (1.0 / n)
        sinv = tl.sin(angle)
        acc += xj * sinv

    tl.store(out_imag_ptr + ks, acc, mask=mask_k)


@triton.jit
def _normalize_divide_kernel(in_ptr, out_ptr, numel, scale, BLOCK: tl.constexpr):
    """
    out[i] = in[i] / scale
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < numel
    vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    vals = vals / scale
    tl.store(out_ptr + offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect input as (batch, channels, seqlen)
        if len(args) != 1:
            raise ValueError("ModelNew.forward expects a single tensor argument (batch, channels, seqlen)")
        x = args[0]
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if x.dim() != 3:
            raise ValueError("Input must be a 3D tensor (batch, channels, seqlen)")
        B, C, L = x.shape
        n = 2 * L

        # Output tensors (real and imaginary parts)
        out_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Process each (batch, channel) slice independently
        for b in range(B):
            for c in range(C):
                # Flatten the row of length L
                x_row = x[b, c, :].contiguous()
                # Allocate padded buffer of length n and initialize to zeros
                padded = torch.zeros(n, dtype=torch.float32, device=x.device)

                # Launch pad copy kernel: one program per element j < L
                grid_copy = (L,)
                _copy_row_to_padded_kernel[grid_copy](x_row, padded, L, n)

                # Compute real part via direct DFT over padded input
                out_real_bc = out_real[b, c, :]  # length L+1
                grid_k_real = (triton.cdiv(L, 128),)  # BLOCK_K controls per-program k range
                _direct_rfft_real_padded_kernel[grid_k_real](padded, out_real_bc, L, n, BLOCK_K=128)

                # Compute imaginary part via direct DFT over padded input
                out_imag_bc = out_imag[b, c, :]  # length L+1
                grid_k_imag = (triton.cdiv(L, 128),)
                _direct_rfft_imag_padded_kernel[grid_k_imag](padded, out_imag_bc, L, n, BLOCK_K=128)

                # Normalize by 2*L using elementwise division
                grid_div_real = (triton.cdiv(L + 1, 1024),)
                _normalize_divide_kernel[grid_div_real](out_real_bc, out_real_bc, L + 1, 2 * L, BLOCK=1024)

                grid_div_imag = (triton.cdiv(L + 1, 1024),)
                _normalize_divide_kernel[grid_div_imag](out_imag_bc, out_imag_bc, L + 1, 2 * L, BLOCK=1024)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
