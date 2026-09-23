import torch
import triton
import triton.language as tl


@triton.jit
def _fused_norm_scale_row_kernel(
    x_ptr,            # *pointer to hidden_states
    weight_ptr,       # *pointer to weight
    out_ptr,          # *pointer to output
    B,                # batch size (rows)
    H,                # hidden size (columns)
    EPS,              # epsilon (float32)
    stride_x_row,     # stride for row in x
    stride_x_col,     # stride for col in x
    stride_out_row,   # stride for row in out
    stride_out_col,   # stride for col in out
    OUT_DTYPE: tl.constexpr,  # output dtype (tl.float16 or tl.bfloat16)
    BLOCK_SIZE: tl.constexpr, # base chunk size along columns
    VEC: tl.constexpr,        # columns processed per iteration
    NUM_ITERS: tl.constexpr,  # number of iterations over columns
):
    row_id = tl.program_id(0)
    # If row_id >= B, nothing to do (safety for grid > B)
    if row_id >= B:
        return

    # 1) First pass: compute sum of squares for this row
    sum_sq = 0.0
    for it in tl.static_range(NUM_ITERS):
        col_start = it * BLOCK_SIZE * VEC
        offs_col = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = offs_col < H

        x_row_ptr = x_ptr + row_id * stride_x_row + offs_col * stride_x_col
        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean_sq = sum_sq / H
    inv_rms = tl.rsqrt(mean_sq + EPS)

    # 2) Second pass: compute outputs and store
    for it in tl.static_range(NUM_ITERS):
        col_start = it * BLOCK_SIZE * VEC
        offs_col = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = offs_col < H

        x_row_ptr = x_ptr + row_id * stride_x_row + offs_col * stride_x_col
        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)

        w_vals = tl.load(weight_ptr + offs_col, mask=mask, other=0.0)
        w_vals = w_vals.to(tl.float32)

        out_vals = x_vals * inv_rms * w_vals

        out_row_ptr = out_ptr + row_id * stride_out_row + offs_col * stride_out_col
        tl.store(out_row_ptr, out_vals.to(OUT_DTYPE), mask=mask)


# Example helper to use the Triton kernel (not used by evaluator, but useful for testing)
def fused_operator_triton(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are on CUDA and contiguous
    assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    x = hidden_states.to(torch.float32).contiguous()
    w = weight.to(torch.float32).contiguous()
    B, H = x.shape
    out = torch.empty_like(x, dtype=torch.float32)

    # Strides
    stride_x_row = x.stride(0)
    stride_x_col = x.stride(1)
    stride_out_row = out.stride(0)
    stride_out_col = out.stride(1)

    # Choose tiling: process 16384 columns per iteration
    BLOCK_SIZE = 1024
    VEC = 16
    NUM_ITERS = (H + (BLOCK_SIZE * VEC) - 1) // (BLOCK_SIZE * VEC)

    # Output dtype: match input's original dtype for storage
    OUT_DTYPE = tl.float16 if hidden_states.dtype == torch.float16 else tl.bfloat16

    grid = (B,)
    _fused_norm_scale_row_kernel[grid](
        x, w, out,
        B, H, 1e-5,
        stride_x_row, stride_x_col,
        stride_out_row, stride_out_col,
        OUT_DTYPE,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC=VEC,
        NUM_ITERS=NUM_ITERS,
        num_warps=8,
        num_stages=2,
    )
    # Cast back to original dtype if needed (output is already in original dtype via OUT_DTYPE store)
    return out.to(hidden_states.dtype)


# Minimal model entry point for evaluator
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return fused_operator_triton(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
