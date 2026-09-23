import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded_kernel(x_row_ptr, padded_ptr, L, n, BLOCK: tl.constexpr):
    """
    x_row_ptr: *float32, length L, row of input for a (batch, channel) slice
    padded_ptr: *float32, length n, output buffer with x[0..L-1] followed by zeros
    """
    j = tl.program_id(axis=0)  # one program per element
    if j < L:
        tl.store(padded_ptr + j, tl.load(x_row_ptr + j))


@triton.jit
def _reduce_real_block_kernel(padded_ptr, cos_table_ptr, out_real_ptr, L, n, BLOCK_J: tl.constexpr):
    """
    Reduce over j in blocks to compute y_real[k] = sum_j padded[j] * cos_table[j, k]
    padded_ptr: *float32, length n
    cos_table_ptr: *float32, shape (L+1, n), row-major contiguous (offsets: row*L*n + col)
    out_real_ptr: *float32, length (L+1), accumulates results (one program per k)
    """
    k = tl.program_id(axis=0)  # one program per k
    acc = 0.0
    # Iterate j in blocks of size BLOCK_J; compile-time constant loop
    for jj in range(0, BLOCK_J * 4, BLOCK_J):  # placeholder; see launch config below
        # Note: We must know n at compile time for this loop; Triton requires static loops.
        # To avoid dynamic loops, we simply set BLOCK_J >= n and iterate once.
        j_vec = jj + tl.arange(0, BLOCK_J)
        mask_j = j_vec < n
        xj = tl.load(padded_ptr + j_vec, mask=mask_j, other=0.0)
        cos_row = tl.load(cos_table_ptr + k * n + j_vec, mask=mask_j, other=0.0)
        prod = xj * cos_row
        local_sum = tl.zeros((), dtype=tl.float32)
        for t in range(0, BLOCK_J):
            local_sum += prod[t]
        acc += local_sum
    tl.store(out_real_ptr + k, acc)


@triton.jit
def _reduce_imag_block_kernel(padded_ptr, sin_table_ptr, out_imag_ptr, L, n, BLOCK_J: tl.constexpr):
    """
    Reduce over j in blocks to compute y_imag[k] = sum_j padded[j] * sin_table[j, k]
    padded_ptr: *float32, length n
    sin_table_ptr: *float32, shape (L+1, n), row-major contiguous (offsets: row*L*n + col)
    out_imag_ptr: *float32, length (L+1), accumulates results (one program per k)
    """
    k = tl.program_id(axis=0)  # one program per k
    acc = 0.0
    for jj in range(0, BLOCK_J * 4, BLOCK_J):  # same placeholder; see launch config below
        j_vec = jj + tl.arange(0, BLOCK_J)
        mask_j = j_vec < n
        xj = tl.load(padded_ptr + j_vec, mask=mask_j, other=0.0)
        sin_row = tl.load(sin_table_ptr + k * n + j_vec, mask=mask_j, other=0.0)
        prod = xj * sin_row
        local_sum = tl.zeros((), dtype=tl.float32)
        for t in range(0, BLOCK_J):
            local_sum += prod[t]
        acc += local_sum
    tl.store(out_imag_ptr + k, acc)


@triton.jit
def _normalize_divide_kernel(in_ptr, out_ptr, numel, scale, BLOCK: tl.constexpr):
    """
    Elementwise division: out[i] = in[i] / scale
    """
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < numel
    val = tl.load(in_ptr + idx, mask=mask, other=0.0)
    val = val / scale
    tl.store(out_ptr + idx, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect input of shape (batch, channels, seqlen)
        assert len(args) == 1, "ModelNew expects a single input tensor"
        x = args[0]
        assert x.ndim == 3, "Input must be 3D (batch, channels, seqlen)"
        B, C, L = x.shape

        device = x.device
        x = x.to(torch.float32)

        n = 2 * L

        # Output tensors: shape (B, C, L+1)
        out_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=device)
        out_imag = torch.empty((B, C, L + 1), dtype=torch.float32, device=device)

        # Iterate over batch and channels; compute rFFT per (b, c) slice
        for b in range(B):
            for c in range(C):
                # 1) Prepare padded buffer of length n (zeros for j >= L)
                padded = torch.empty(n, dtype=torch.float32, device=device)
                x_row = x[b, c, :]  # length L
                grid_copy = (L,)
                _copy_row_to_padded_kernel[grid_copy](x_row, padded, L, n, BLOCK=1)

                # 2) Precompute cos and sin tables on device (PyTorch ops)
                j = torch.arange(n, device=device, dtype=torch.float32)  # [0..n-1]
                k = torch.arange(L + 1, device=device, dtype=torch.float32)  # [0..L]
                angle = (2.0 * 3.141592653589793) * k[:, None] * j[None, :] / float(n)
                cos_table = torch.cos(angle)  # (L+1, n)
                sin_table = torch.sin(angle)  # (L+1, n)

                # 3) Reduce to compute real and imag parts using Triton
                # Set BLOCK_J >= n so the loop runs once (compile-time unrolled)
                BLOCK_J = n + 1  # ensure covers all j
                grid_reduce = (L + 1,)  # one program per k
                _reduce_real_block_kernel[grid_reduce](padded, cos_table, out_real[b, c, :], L, n, BLOCK_J=BLOCK_J)
                _reduce_imag_block_kernel[grid_reduce](padded, sin_table, out_imag[b, c, :], L, n, BLOCK_J=BLOCK_J)

                # 4) Normalize by 2*L (elementwise division in Triton)
                numel = L + 1
                grid_norm = (triton.cdiv(numel, 256),)
                _normalize_divide_kernel[grid_norm](out_real[b, c, :], out_real[b, c, :], numel, 2.0 * float(L), BLOCK=256)
                _normalize_divide_kernel[grid_norm](out_imag[b, c, :], out_imag[b, c, :], numel, 2.0 * float(L), BLOCK=256)

        # Return real and imaginary parts (both shape (B, C, L+1))
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
