import torch
import triton
import triton.language as tl


@triton.jit
def _zero_padded_kernel(padded_ptr, n, BLOCK: tl.constexpr):
    """
    Zero-initialize a buffer of length n using vectorized stores.
    padded_ptr: *float32, length n
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(padded_ptr + offsets, 0.0, mask=mask)


@triton.jit
def _copy_row_to_padded_kernel(x_row_ptr, padded_ptr, L, BLOCK: tl.constexpr):
    """
    Copy a row of length L into padded[0:L] of a pre-zeroed padded buffer.
    x_row_ptr: *float32, length L
    padded_ptr: *float32, length 2*L, already zero-initialized
    """
    pid = tl.program_id(axis=0)
    j = pid  # one element per program; j in [0, L)
    if j < L:
        tl.store(padded_ptr + j, tl.load(x_row_ptr + j))


@triton.jit
def _reduce_real_block_kernel(padded_ptr, cos_table_ptr, out_real_ptr, L, n, BLOCK_J: tl.constexpr):
    """
    Compute a partial sum for y_real[k] over a block of j indices.
    padded_ptr: *float32, length n
    cos_table_ptr: *float32, shape (L+1, n), row-major contiguous
    out_real_ptr: *float32, length (L+1), accumulates results
    """
    # One program per k
    k = tl.program_id(axis=0)
    acc = 0.0
    # Iterate over j in chunks of BLOCK_J
    for jj in range(0, n, BLOCK_J):
        j_vec = jj + tl.arange(0, BLOCK_J)
        mask_j = j_vec < n
        xj = tl.load(padded_ptr + j_vec, mask=mask_j, other=0.0)
        cos_row = tl.load(cos_table_ptr + k * n + j_vec, mask=mask_j, other=0.0)
        prod = xj * cos_row
        local_sum = tl.zeros((), dtype=tl.float32)
        # Reduce vector prod to scalar
        for t in range(0, BLOCK_J):
            local_sum += prod[t]
        acc += local_sum
    # Atomic add into out_real[k] (multiple programs per k could be used; here we use one program per k)
    tl.atomic_add(out_real_ptr + k, acc)


@triton.jit
def _reduce_imag_block_kernel(padded_ptr, sin_table_ptr, out_imag_ptr, L, n, BLOCK_J: tl.constexpr):
    """
    Compute a partial sum for y_imag[k] over a block of j indices.
    padded_ptr: *float32, length n
    sin_table_ptr: *float32, shape (L+1, n), row-major contiguous
    out_imag_ptr: *float32, length (L+1), accumulates results
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

        # Iterate over batch and channels; compute rFFT per (b, c) slice
        for b in range(B):
            for c in range(C):
                # Prepare padded buffer of length n
                padded = torch.empty(n, dtype=torch.float32, device=device)


def run(*args):
    return ModelNew()(*args)
