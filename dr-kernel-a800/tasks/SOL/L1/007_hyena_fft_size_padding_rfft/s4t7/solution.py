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
    # padded_ptr[j] for j >= L is assumed to be zeroed by host code


@triton.jit
def _compute_cos_table_kernel(j_idx_ptr, k_idx_ptr, out_ptr, L, n, BLOCK_J: tl.constexpr):
    """
    Precompute cos(2*pi*k*j / n) for j in [0..n-1], k in [0..L] into out_ptr of shape (L+1, n).
    out_ptr is row-major contiguous: out[k, j] = out_ptr[k*n + j].
    j_idx_ptr: *int32, length n
    k_idx_ptr: *int32, length L+1
    """
    k = tl.program_id(axis=0)  # over rows (k index)
    j = tl.program_id(axis=1)  # over columns (j index)

    # Guard j
    if j >= n:
        return

    j_val = tl.load(j_idx_ptr + j)  # should be j itself
    k_val = tl.load(k_idx_ptr + k)  # should be k itself

    angle = 2.0 * 3.141592653589793 * k_val * j_val * (1.0 / n)
    cosv = tl.cos(angle)
    tl.store(out_ptr + k * n + j, cosv)


@triton.jit
def _compute_sin_table_kernel(j_idx_ptr, k_idx_ptr, out_ptr, L, n, BLOCK_J: tl.constexpr):
    """
    Precompute sin(2*pi*k*j / n) for j in [0..n-1], k in [0..L] into out_ptr of shape (L+1, n).
    out_ptr is row-major contiguous: out[k, j] = out_ptr[k*n + j].
    j_idx_ptr: *int32, length n
    k_idx_ptr: *int32, length L+1
    """
    k = tl.program_id(axis=0)  # over rows (k index)
    j = tl.program_id(axis=1)  # over columns (j index)

    if j >= n:
        return

    j_val = tl.load(j_idx_ptr + j)  # should be j itself
    k_val = tl.load(k_idx_ptr + k)  # should be k itself

    angle = 2.0 * 3.141592653589793 * k_val * j_val * (1.0 / n)
    sinv = tl.sin(angle)
    tl.store(out_ptr + k * n + j, sinv)


@triton.jit
def _reduce_real_block_kernel(padded_ptr, cos_table_ptr, out_real_ptr, L, n, BLOCK_J: tl.constexpr):
    """
    Reduce to y_real[k] = sum_j padded[j] * cos_table[j, k] for k in [0..L].
    We iterate over j in blocks. out_real_ptr[k] accumulates via atomic_add.
    """
    k = tl.program_id(axis=0)
    acc = 0.0
    # Iterate j in chunks of BLOCK_J
    for jj in range(0, n, BLOCK_J):
        j_vec = jj + tl.arange(0, BLOCK_J)
        mask_j = j_vec < n
        xj = tl.load(padded_ptr + j_vec, mask=mask_j, other=0.0)
        cos_row = tl.load(cos_table_ptr + k * n + j_vec, mask=mask_j, other=0.0)
        prod = xj * cos_row
        # sum vector to scalar
        local_sum = tl.zeros((), dtype=tl.float32)
        for t in range(0, BLOCK_J):
            local_sum += prod[t]
        acc += local_sum
    tl.atomic_add(out_real_ptr + k, acc)


@triton.jit
def _reduce_imag_block_kernel(padded_ptr, sin_table_ptr, out_imag_ptr, L, n, BLOCK_J: tl.constexpr):
    """
    Reduce to y_imag[k] = sum_j padded[j] * sin_table[j, k] for k in [0..L].
    We iterate over j in blocks. out_imag_ptr[k] accumulates via atomic_add.
    """
    k = tl.program_id(axis=0)
    acc = 0.0
    for jj in range(0, n, BLOCK_J):
        j_vec = jj + tl.arange(0, BLOCK_J)
        mask_j = j_vec < n
        xj = tl.load(padded_ptr + j_vec, mask=mask_j, other=0.0)
        sin_row = tl.load(sin_table_ptr + k * n + j_vec, mask=mask_j, other=0.0)
        prod = xj * sin_row
        local_sum = tl.zeros((), dtype=tl.float32)
        for t in range(0, BLOCK_J):
            local_sum += prod[t]
        acc += local_sum
    tl.atomic_add(out_imag_ptr + k, acc)


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

        # Output tensors
        out_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=device)
        out_imag = torch.empty((B, C, L + 1), dtype=torch.float32, device=device)

        # Indices for kernels (int32 on device)
        j_idx = torch.arange(0, n, dtype=torch.int32, device=device)
        k_idx = torch.arange(0, L + 1, dtype=torch.int32, device=device)

        # Precompute cos and sin tables using Triton kernels
        cos_table = torch.empty((L + 1, n), dtype=torch.float32, device=device)
        sin_table = torch.empty((L + 1, n), dtype=torch.float32, device=device)

        # Launch kernels to fill cos/sin tables
        grid_cos = (L + 1, n)
        _compute_cos_table_kernel[grid_cos](j_idx, k_idx, cos_table, L, n, BLOCK_J=64)
        grid_sin = (L + 1, n)
        _compute_sin_table_kernel[grid_sin](j_idx, k_idx, sin_table, L, n, BLOCK_J=64)

        # For each (batch, channel) slice, compute rFFT and normalize
        for b in range(B):
            for c in range(C):
                # Prepare padded buffer of length n
                padded = torch.empty(n, dtype=torch.float32, device=device)
                # Copy input row to padded
                grid_copy = (L,)
                _copy_row_to_padded_kernel[grid_copy](x[b, c, :], padded, L, n, BLOCK=L)

                # Reduce to real and imaginary parts
                out_real_row = torch.zeros((L + 1,), dtype=torch.float32, device=device)
                out_imag_row = torch.zeros((L + 1,), dtype=torch.float32, device=device)

                # Launch reduction kernels
                grid_reduce = (L + 1,)
                _reduce_real_block_kernel[grid_reduce](padded, cos_table, out_real_row, L, n, BLOCK_J=n)
                _reduce_imag_block_kernel[grid_reduce](padded, sin_table, out_imag_row, L, n, BLOCK_J=n)

                # Normalize by 2*L
                numel = L + 1
                grid_norm = (triton.cdiv(numel, 256),)
                _normalize_divide_kernel[grid_norm](out_real_row, out_real_row, numel, 2.0 * float(L), BLOCK=256)
                _normalize_divide_kernel[grid_norm](out_imag_row, out_imag_row, numel, 2.0 * float(L), BLOCK=256)

                # Write into outputs (B, C, L+1)
                out_real[b, c, :] = out_real_row
                out_imag[b, c, :] = out_imag_row

        # Return real and imaginary parts (both shape (B, C, L+1))
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
