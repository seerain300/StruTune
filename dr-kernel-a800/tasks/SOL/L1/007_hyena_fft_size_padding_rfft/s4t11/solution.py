import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded_kernel(
    x_ptr, out_ptr,
    B, C, L,
    stride_xb, stride_xc, stride_xl,
    stride_outb, stride_outc, stride_outl,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    k = tl.program_id(2)

    if b >= B or c >= C or k >= L:
        return

    x_row_ptr = x_ptr + b * stride_xb + c * stride_xc + k * stride_xl
    out_row_ptr = out_ptr + b * stride_outb + c * stride_outc

    j = tl.arange(0, BLOCK_K)
    mask_j = j < (2 * L)
    vals = tl.load(x_row_ptr + j * stride_xl, mask=mask_j, other=0.0)
    tl.store(out_row_ptr + j * stride_outl, vals, mask=mask_j)


@triton.jit
def _compute_cos_table_kernel(
    out_ptr,
    N, K,
    stride_outj, stride_outk,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    j = tl.program_id(0)
    k = tl.program_id(1)
    if j >= N or k >= K:
        return

    idx_j = j * stride_outj
    idx_k = k * stride_outk
    base = out_ptr + idx_j + idx_k

    # Compute cos(2*pi*k*j/N) for this (j, k)
    j_vec = j + tl.arange(0, BLOCK_J)
    k_vec = k + tl.arange(0, BLOCK_K)
    mask_j = j_vec < N
    mask_k = k_vec < K

    angle = 2.0 * 3.141592653589793 * k_vec * j_vec / N  # vectorized angle
    cos_vals = tl.cos(angle)

    # Store per-k vector into row j: out_ptr[j, k_vec]
    # We store a [BLOCK_K] slice for each k in the grid; mask ensures bounds.
    tl.store(out_ptr + j * stride_outj + k_vec * stride_outk, cos_vals, mask=mask_k)


@triton.jit
def _compute_sin_table_kernel(
    out_ptr,
    N, K,
    stride_outj, stride_outk,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    j = tl.program_id(0)
    k = tl.program_id(1)
    if j >= N or k >= K:
        return

    idx_j = j * stride_outj
    idx_k = k * stride_outk
    base = out_ptr + idx_j + idx_k

    j_vec = j + tl.arange(0, BLOCK_J)
    k_vec = k + tl.arange(0, BLOCK_K)
    mask_j = j_vec < N
    mask_k = k_vec < K

    angle = 2.0 * 3.141592653589793 * k_vec * j_vec / N
    sin_vals = tl.sin(angle)

    tl.store(out_ptr + j * stride_outj + k_vec * stride_outk, sin_vals, mask=mask_k)


@triton.jit
def _reduce_real_kernel(
    padded_ptr, cos_ptr, out_ptr,
    N, K,
    stride_pj, stride_pk,  # strides for padded
    stride_cj, stride_ck,  # strides for cos_table
    stride_ol,             # stride for output along K
    BLOCK_J: tl.constexpr,
):
    k = tl.program_id(0)
    if k >= K:
        return
    acc = tl.zeros((), dtype=tl.float32)

    j = 0
    while j < N:
        j_vec = j + tl.arange(0, BLOCK_J)
        mask_j = j_vec < N

        # Load padded segment
        padded_row_ptrs = padded_ptr + j_vec * stride_pj + k * stride_pk
        vals = tl.load(padded_row_ptrs, mask=mask_j, other=0.0)

        # Load cos segment for this k
        cos_row_ptrs = cos_ptr + j_vec * stride_cj + k * stride_ck
        cos_vals = tl.load(cos_row_ptrs, mask=mask_j, other=0.0)

        prod = vals * cos_vals
        acc += tl.sum(prod, axis=0)
        j += BLOCK_J

    tl.store(out_ptr + k * stride_ol, acc)


@triton.jit
def _reduce_imag_kernel(
    padded_ptr, sin_ptr, out_ptr,
    N, K,
    stride_pj, stride_pk,  # strides for padded
    stride_sj, stride_sk,  # strides for sin_table
    stride_ol,             # stride for output along K
    BLOCK_J: tl.constexpr,
):
    k = tl.program_id(0)
    if k >= K:
        return
    acc = tl.zeros((), dtype=tl.float32)

    j = 0
    while j < N:
        j_vec = j + tl.arange(0, BLOCK_J)
        mask_j = j_vec < N

        # Load padded segment
        padded_row_ptrs = padded_ptr + j_vec * stride_pj + k * stride_pk
        vals = tl.load(padded_row_ptrs, mask=mask_j, other=0.0)

        # Load sin segment for this k
        sin_row_ptrs = sin_ptr + j_vec * stride_sj + k * stride_sk
        sin_vals = tl.load(sin_row_ptrs, mask=mask_j, other=0.0)

        prod = vals * sin_vals
        acc += tl.sum(prod, axis=0)
        j += BLOCK_J

    tl.store(out_ptr + k * stride_ol, acc)


@triton.jit
def _divide_kernel(
    x_ptr, out_ptr, numel, scale,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < numel
    vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
    vals = vals / scale
    tl.store(out_ptr + idx, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We expect a single tensor input: (B, C, L)
        if len(args) == 0:
            raise RuntimeError("ModelNew expects at least one tensor input")
        x = args[0]
        if not isinstance(x, torch.Tensor):
            raise RuntimeError("ModelNew expects a torch.Tensor as input")

        # Assume single input of shape (B, C, L)
        if x.dim() != 3:
            raise RuntimeError("ModelNew expects input of shape (B, C, L)")
        B, C, L = x.shape
        device = x.device
        dtype = torch.float32  # ensure float32 for compute

        # Padded input buffer: (B, C, 2*L)
        N = 2 * L
        padded = torch.empty((B, C, N), dtype=dtype, device=device)

        # Strides
        stride_xb, stride_xc, stride_xl = x.stride()
        stride_outb, stride_outc, stride_outl = padded.stride()

        # Launch copy kernel: one program per (b, c, k)
        grid_copy = (B, C, L)
        _copy_row_to_padded_kernel[grid_copy](
            x, padded,
            B, C, L,
            stride_xb, stride_xc, stride_xl,
            stride_outb, stride_outc, stride_outl,
            BLOCK_K=1,  # minimal per-k; masked by 2*L check
        )

        # Allocate cos/sin tables: (N, L), float32
        cos_table = torch.empty((N, L), dtype=dtype, device=device)
        sin_table = torch.empty((N, L), dtype=dtype, device=device)

        stride_cj, stride_ck = cos_table.stride()
        stride_sj, stride_sk = sin_table.stride()

        # Compute cos table: grid (N, L)
        grid_cos = (N, L)
        _compute_cos_table_kernel[grid_cos](
            cos_table,
            N, L,
            stride_cj, stride_ck,
            BLOCK_J=1, BLOCK_K=1,
        )

        # Compute sin table: grid (N, L)
        _compute_sin_table_kernel[grid_cos](
            sin_table,
            N, L,
            stride_sj, stride_sk,
            BLOCK_J=1, BLOCK_K=1,
        )

        # Allocate outputs: (B, C, L+1)
        out_real = torch.empty((B, C, L + 1), dtype=dtype, device=device)
        out_imag = torch.empty((B, C, L + 1), dtype=dtype, device=device)

        # Strides for outputs
        # We index along K by stride=1 for contiguous output (L+1 dim), but we store per k
        stride_ol = 1  # out tensors are contiguous along last dim

        # Launch reduction kernels: one program per k
        grid_reduce = (L,)
        _reduce_real_kernel[grid_reduce](
            padded, cos_table, out_real,
            N, L,
            stride_pj=1, stride_pk=1,  # padded strides (per-element pointer arithmetic)
            stride_cj=stride_cj, stride_ck=stride_ck,
            stride_ol=stride_ol,
            BLOCK_J=256,
        )
        _reduce_imag_kernel[grid_reduce](
            padded, sin_table, out_imag,
            N, L,
            stride_pj=1, stride_pk=1,
            stride_sj=stride_sj, stride_sk=stride_sk,
            stride_ol=stride_ol,
            BLOCK_J=256,
        )

        # Normalize by 2*L
        scale = 2.0 * float(L)
        numel_real = out_real.numel()
        grid_div = ((numel_real + 1023) // 1024,)
        _divide_kernel[grid_div](out_real, out_real, numel_real, scale, BLOCK=1024)
        numel_imag = out_imag.numel()
        _divide_kernel[grid_div](out_imag, out_imag, numel_imag, scale, BLOCK=1024)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
