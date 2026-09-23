import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded_kernel(x_row_ptr, padded_ptr, L, n, BLOCK: tl.constexpr):
    """
    x_row_ptr: *float32, length L, row of input for a (batch, channel) slice
    padded_ptr: *float32, length n, output buffer with x[0..L-1] followed by zeros
    """
    pid = tl.program_id(axis=0)
    j = pid  # one program per element in the row
    if j < L:
        tl.store(padded_ptr + j, tl.load(x_row_ptr + j))
    # padded_ptr[j] is zero for j >= L by construction in host code


@triton.jit
def _direct_rfft_real_padded_kernel(padded_ptr, out_real_ptr, L, n, BLOCK_K: tl.constexpr):
    """
    Compute real part of rFFT for k in [0..L] using direct summation over j=0..n-1:
    y_real[k] = sum_{j=0}^{n-1} padded[j] * cos(2*pi*k*j / n)
    padded_ptr: *float32, length n = 2*L, containing x[0..L-1] followed by zeros
    out_real_ptr: *float32, length (L+1), stores y[0..L]
    L: int, original seqlen
    n: int, 2*L
    """
    pid = tl.program_id(axis=0)
    k0 = pid * BLOCK_K
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
    y_imag[k] = sum_{j=0}^{n-1} padded[j] * sin(2*pi*k*j / n)
    padded_ptr: *float32, length n = 2*L, containing x[0..L-1] followed by zeros
    out_imag_ptr: *float32, length (L+1), stores y[0..L]
    L: int, original seqlen
    n: int, 2*L
    """
    pid = tl.program_id(axis=0)
    k0 = pid * BLOCK_K
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
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    vals = tl.load(in_ptr + offs, mask=mask, other=0.0)
    vals = vals / scale
    tl.store(out_ptr + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Expect input of shape (batch, channels, seqlen)
        assert x.dim() == 3, "Input must be 3D: (batch, channels, seqlen)"
        B, C, L = x.shape

        # Cast to float32 for numerical stability, contiguous
        x_f32 = x.to(torch.float32).contiguous()
        n = 2 * L  # padding to 2*L as in original code

        # Output buffers: real and imag parts, shape (B, C, L+1)
        out_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # For each (batch, channel) slice, process the row of length L
        for b in range(B):
            for c in range(C):
                # 1) Zero-padded buffer: copy x[b, c, :] to first L elements
                row = x_f32[b, c, :].contiguous()  # length L
                padded = torch.zeros(n, dtype=torch.float32, device=x.device)
                # Copy first L elements
                padded[:L] = row

                # 2) Compute real part via direct DFT over j=0..n-1
                out_real_row = torch.empty(L + 1, dtype=torch.float32, device=x.device)
                BLOCK_K = 128
                grid_k = (triton.cdiv(L, BLOCK_K),)
                _direct_rfft_real_padded_kernel[grid_k](padded, out_real_row, L, n, BLOCK_K=BLOCK_K)

                # 3) Compute imaginary part via direct DFT over j=0..n-1
                out_imag_row = torch.empty(L + 1, dtype=torch.float32, device=x.device)
                _direct_rfft_imag_padded_kernel[grid_k](padded, out_imag_row, L, n, BLOCK_K=BLOCK_K)

                # 4) Normalize by 2*L using Triton
                scale = 2.0 * L
                out_real_tmp = torch.empty_like(out_real_row, device=x.device)
                _normalize_divide_kernel[(L + 1,)](out_real_row, out_real_tmp, L + 1, scale, BLOCK=1)
                out_imag_tmp = torch.empty_like(out_imag_row, device=x.device)
                _normalize_divide_kernel[(L + 1,)](out_imag_row, out_imag_tmp, L + 1, scale, BLOCK=1)

                # 5) Write to output tensor at last dimension index 0..L
                out_real[b, c, :L + 1] = out_real_tmp
                out_imag[b, c, :L + 1] = out_imag_tmp

        # Return real and imaginary parts as required by original run function
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
