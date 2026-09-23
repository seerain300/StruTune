import torch
import triton
import triton.language as tl


@triton.jit
def fused_row_norm_scale_kernel(
    x_ptr,            # *pointer to hidden_states (B, H)
    weight_ptr,       # *pointer to weight (H,)
    out_ptr,          # *pointer to output (B, H)
    B,                # batch size (rows)
    H,                # hidden size (columns)
    EPS,              # epsilon (float32)
    stride_x_row,     # stride for row in x (elements)
    stride_x_col,     # stride for col in x (elements)
    stride_out_row,   # stride for row in out (elements)
    stride_out_col,   # stride for col in out (elements)
    OUT_DTYPE: tl.constexpr,  # output dtype (tl.float16 or tl.bfloat16)
    BLOCK_SIZE: tl.constexpr, # base chunk size along columns (e.g., 512)
    VEC: tl.constexpr,        # columns processed per iteration (e.g., 16)
    NUM_ITERS: tl.constexpr,  # number of iterations = ceil_div(H, BLOCK_SIZE * VEC)
):
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    base_x = x_ptr + row_id * stride_x_row
    # 1) First pass: compute sum of squares across the row
    sum_sq = 0.0  # scalar float32
    for it in tl.static_range(NUM_ITERS):
        col_start = it * (BLOCK_SIZE * VEC)
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = col_offsets < H
        x_vals = tl.load(base_x + col_offsets * stride_x_col, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)

    # 2) Second pass: scale and store output
    base_out = out_ptr + row_id * stride_out_row
    for it in tl.static_range(NUM_ITERS):
        col_start = it * (BLOCK_SIZE * VEC)
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = col_offsets < H

        x_vals = tl.load(base_x + col_offsets * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        y = x_vals * inv_rms * w_vals

        # Cast to output dtype (match input hidden_states dtype)
        if OUT_DTYPE == tl.float16:
            y = y.to(tl.float16)
        elif OUT_DTYPE == tl.bfloat16:
            y = y.to(tl.bfloat16)
        # else default float32

        tl.store(base_out + col_offsets * stride_out_col, y, mask=mask)


def _run_triton(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Ensure CUDA tensors
    assert hidden_states.is_cuda and weight.is_cuda, "Tensors must be on CUDA device"
    B, H = hidden_states.shape
    assert H == 4096, "hidden_size must be 4096 as per original assertion"

    # Prepare output dtype for Triton (match input hidden_states dtype)
    if hidden_states.dtype == torch.float16:
        OUT_DTYPE = tl.float16
    elif hidden_states.dtype == torch.bfloat16:
        OUT_DTYPE = tl.bfloat16
    else:
        # Default to bfloat16 for bfloat16 inputs; for float32, we can store float32
        OUT_DTYPE = tl.float32

    # Allocate output
    out = torch.empty_like(hidden_states)

    # Strides in elements
    stride_x_row = hidden_states.stride(0)
    stride_x_col = hidden_states.stride(1)
    stride_out_row = out.stride(0)
    stride_out_col = out.stride(1)

    # Tile configuration: 8192 columns per iteration for H=4096 => NUM_ITERS=1
    BLOCK_SIZE = 512
    VEC = 16
    NUM_ITERS = (H + (BLOCK_SIZE * VEC) - 1) // (BLOCK_SIZE * VEC)

    # Launch one program per row
    grid = (B,)
    fused_row_norm_scale_kernel[grid](
        hidden_states, weight, out,
        B, H, 1e-5,
        stride_x_row, stride_x_col,
        stride_out_row, stride_out_col,
        OUT_DTYPE=OUT_DTYPE,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC=VEC,
        NUM_ITERS=NUM_ITERS,
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
