import torch
import triton
import triton.language as tl


@triton.jit
def _row_norm_kernel(
    x_ptr,            # *pointer to hidden_states
    inv_rms_ptr,      # *pointer to per-row inverse r.m.s. (float32)
    B,                # batch size (rows)
    H,                # hidden size (columns)
    EPS,              # epsilon (float32)
    stride_x_row,     # stride for row in x
    stride_x_col,     # stride for col in x
    BLOCK_SIZE: tl.constexpr,  # base chunk size along columns (e.g., 512)
    VEC: tl.constexpr,         # columns processed per iteration (e.g., 16)
):
    row_id = tl.program_id(0)
    if row_id >= B:
        return
    x_row_ptr = x_ptr + row_id * stride_x_row

    sumsq = 0.0
    # Iterate over columns in chunks of BLOCK_SIZE * VEC
    for col_start in range(0, H, BLOCK_SIZE * VEC):
        offs = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = offs < H
        x = tl.load(x_row_ptr + offs * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv_rms)


@triton.jit
def _row_scale_kernel(
    x_ptr,            # *pointer to hidden_states
    weight_ptr,       # *pointer to weight (float32)
    inv_rms_ptr,      # *pointer to per-row inverse r.m.s. (float32)
    out_ptr,          # *pointer to output
    B,                # batch size (rows)
    H,                # hidden size (columns)
    OUT_DTYPE: tl.constexpr,  # output dtype (tl.float16 or tl.bfloat16)
    BLOCK_SIZE: tl.constexpr, # base chunk size along columns (e.g., 512)
    VEC: tl.constexpr,        # columns processed per iteration (e.g., 16)
):
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    x_row_ptr = x_ptr + row_id * stride_x_row
    out_row_ptr = out_ptr + row_id * stride_out_row
    inv_rms = tl.load(inv_rms_ptr + row_id)  # fp32 scalar

    # We will process columns in chunks; typically H == BLOCK_SIZE * VEC for our choice
    for col_start in range(0, H, BLOCK_SIZE * VEC):
        offs = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = offs < H

        x = tl.load(x_row_ptr + offs * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w  # fp32 compute

        if OUT_DTYPE == tl.float16:
            y_cast = y.to(tl.float16)
        elif OUT_DTYPE == tl.bfloat16:
            y_cast = y.to(tl.bfloat16)
        else:
            y_cast = y  # default fp32
        tl.store(out_row_ptr + offs * stride_out_col, y_cast, mask=mask)


def _run_triton(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Ensure CUDA tensors and contiguous
    assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    x = hidden_states.contiguous()
    w = weight.contiguous()

    B, H = x.shape
    out = torch.empty_like(x)

    # Tile parameters: process 512 * 16 = 8192 columns per iteration
    BLOCK_SIZE = 512
    VEC = 16
    chunk = BLOCK_SIZE * VEC

    # Grid: one program per row
    grid = (B,)

    # Compute strides (assume row-major)
    stride_x_row, stride_x_col = x.stride(0), x.stride(1)
    stride_out_row, stride_out_col = out.stride(0), out.stride(1)

    # 1) Compute per-row inv_rms
    inv_rms = torch.empty((B,), device=x.device, dtype=torch.float32)
    _row_norm_kernel[grid](
        x, inv_rms,
        B, H, 1e-5,
        stride_x_row, stride_x_col,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC=VEC,
        num_warps=8,
        num_stages=2,
    )

    # 2) Scale rows using inv_rms and weight
    _row_scale_kernel[grid](
        x, w, inv_rms, out,
        B, H,
        OUT_DTYPE=tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16 if x.dtype == torch.float16 else tl.float32,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC=VEC,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Triton path: ensure CUDA and use kernels
        return _run_triton(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
