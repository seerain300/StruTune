import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded_kernel(
    x_ptr, out_ptr,
    B, C, L,
    stride_xb, stride_xc, stride_xl,
    stride_ob, stride_oc, stride_ol,
):
    # Each program handles one (b, c) row
    b = tl.program_id(0)
    c = tl.program_id(1)
    if b >= B or c >= C:
        return

    # Input row pointer
    x_row_ptr = x_ptr + b * stride_xb + c * stride_xc

    # Output row pointer for padded buffer of length 2*L
    out_row_ptr = out_ptr + b * stride_ob + c * stride_oc

    # k in [0..L-1]
    # We launch the reduction kernel with L+1 programs, so this copy kernel only needs to handle k in [0..L)
    # Here, we copy the original row into positions 0..L-1; positions L..2L-1 remain zero.
    k = tl.arange(0, 256)  # vectorized lanes, masked by L
    mask = k < L
    vals = tl.load(x_row_ptr + k * stride_xl, mask=mask, other=0.0)
    tl.store(out_row_ptr + k * stride_ol, vals, mask=mask)


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

    j_vec = j + tl.arange(0, BLOCK_J)
    k_vec = k + tl.arange(0, BLOCK_K)
    mask_j = j_vec < N
    mask_k = k_vec < K

    angle = 2.0 * 3.141592653589793 * k_vec * j_vec / N
    cos_vals = tl.cos(angle)
    tl.store(out_ptr + j_vec * stride_outj + k_vec * stride_outk, cos_vals, mask=mask_k)


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

    j_vec = j + tl.arange(0, BLOCK_J)
    k_vec = k + tl.arange(0, BLOCK_K)
    mask_j = j_vec < N
    mask_k = k_vec < K

    angle = 2.0 * 3.141592653589793 * k_vec * j_vec / N
    sin_vals = tl.sin(angle)
    tl.store(out_ptr + j_vec * stride_outj + k_vec * stride_outk, sin_vals, mask=mask_k)


@triton.jit
def _reduce_real_kernel(
    padded_ptr, cos_ptr, out_ptr,
    N, K,
    stride_pj, stride_pk,  # strides for padded: (stride along batch, stride along length)
    stride_cj, stride_ck,  # strides for cos_table (j stride, k stride)
    stride_ol,             # stride for output along K
    BLOCK_J: tl.constexpr,
):
    # One program per k
    k = tl.program_id(0)
    if k >= K:
        return
    acc = tl.zeros((), dtype=tl.float32)

    j = 0
    while j < N:
        j_vec = j + tl.arange(0, BLOCK_J)
        mask_j = j_vec < N

        # Load padded values for this block of j: shape (BLOCK_J,)
        pvals = tl.load(padded_ptr + j_vec * stride_pj, mask=mask_j, other=0.0)  # padded[b, c, j_vec]

        # Load cos block for this k across j_vec: shape (BLOCK_J,)
        cvals = tl.load(cos_ptr + j_vec * stride_cj + k * stride_ck, mask=mask_j, other=0.0)  # cos_table[j_vec, k]

        prod = pvals * cvals
        prod = tl.where(mask_j, prod, 0.0)
        acc += tl.sum(prod, axis=0)

        j += BLOCK_J

    # Accumulate across (b, c) into out_ptr[k]
    tl.atomic_add(out_ptr + k * stride_ol, acc)


@triton.jit
def _reduce_imag_kernel(
    padded_ptr, sin_ptr, out_ptr,
    N, K,
    stride_pj, stride_pk,
    stride_cj, stride_ck,
    stride_ol,
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

        pvals = tl.load(padded_ptr + j_vec * stride_pj, mask=mask_j, other=0.0)
        svals = tl.load(sin_ptr + j_vec * stride_cj + k * stride_ck, mask=mask_j, other=0.0)

        prod = pvals * svals
        prod = tl.where(mask_j, prod, 0.0)
        acc += tl.sum(prod, axis=0)

        j += BLOCK_J

    tl.atomic_add(out_ptr + k * stride_ol, acc)


@triton.jit
def _divide_kernel(
    out_ptr, out_div_ptr,
    numel, divisor,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < numel
    vals = tl.load(out_ptr + idx, mask=mask, other=0.0)
    vals = vals / divisor
    tl.store(out_div_ptr + idx, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Fused FFT size padding and real FFT computation for Hyena convolution.
        Args:
            x: Input tensor of shape (batch, channels, seqlen)
        Returns:
            x_freq_real: Real part of normalized frequency domain output (batch, channels, seqlen+1)
            x_freq_imag: Imaginary part of normalized frequency domain output (batch, channels, seqlen+1)
        """
        # Ensure float32 and contiguous
        x = x.contiguous().to(torch.float32)
        B, C, L = x.shape
        device = x.device

        # Padded buffer: (B, C, 2*L)
        N = 2 * L
        out_pad = torch.empty((B, C, N), dtype=torch.float32, device=device)

        # Allocate cos/sin tables: (N, L)
        cos_table = torch.empty((N, L), dtype=torch.float32, device=device)
        sin_table = torch.empty((N, L), dtype=torch.float32, device=device)

        # Launch copy row to padded buffer kernel
        grid_copy = (B, C)
        _copy_row_to_padded_kernel[grid_copy](
            x, out_pad,
            B, C, L,
            x.stride(0), x.stride(1), x.stride(2),
            out_pad.stride(0), out_pad.stride(1), out_pad.stride(2),
            num_warps=2, num_stages=1,
        )

        # Compute cos and sin tables using Triton
        BLOCK_J = 128
        BLOCK_K = 64
        grid_cos = ((N + BLOCK_J - 1) // BLOCK_J, (L + BLOCK_K - 1) // BLOCK_K)
        _compute_cos_table_kernel[grid_cos](
            cos_table,
            N, L,
            cos_table.stride(0), cos_table.stride(1),
            BLOCK_J, BLOCK_K,
            num_warps=4, num_stages=2,
        )
        _compute_sin_table_kernel[grid_cos](
            sin_table,
            N, L,
            sin_table.stride(0), sin_table.stride(1),
            BLOCK_J, BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Accumulate outputs: (B, C, L+1), using atomics across (b,c)
        out_real = torch.zeros((B, C, L + 1), dtype=torch.float32, device=device)
        out_imag = torch.zeros((B, C, L + 1), dtype=torch.float32, device=device)

        # Reduction kernels: one program per k in [0..L]
        grid_reduce = (L + 1,)
        _reduce_real_kernel[grid_reduce](
            out_pad, cos_table, out_real,
            N, L,
            out_pad.stride(0), out_pad.stride(2),  # stride along batch and length
            cos_table.stride(0), cos_table.stride(1),
            out_real.stride(2),
            BLOCK_J=128,
            num_warps=4, num_stages=2,
        )
        _reduce_imag_kernel[grid_reduce](
            out_pad, sin_table, out_imag,
            N, L,
            out_pad.stride(0), out_pad.stride(2),
            sin_table.stride(0), sin_table.stride(1),
            out_imag.stride(2),
            BLOCK_J=128,
            num_warps=4, num_stages=2,
        )

        # Normalize by 2*L: apply to each (b,c,k)
        # Flatten outputs to 1D for division kernel
        out_real_flat = out_real.view(-1)
        out_imag_flat = out_imag.view(-1)
        total = (B * C) * (L + 1)
        grid_div = (triton.cdiv(total, 1024),)
        _divide_kernel[grid_div](
            out_real_flat, out_real_flat,
            total, 2.0 * float(N),
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )
        _divide_kernel[grid_div](
            out_imag_flat, out_imag_flat,
            total, 2.0 * float(N),
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
