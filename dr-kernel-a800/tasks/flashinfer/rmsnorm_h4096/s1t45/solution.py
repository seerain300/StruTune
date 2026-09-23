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
    NUM_ITERS: tl.constexpr,  # iterations = ceil_div(H, BLOCK_SIZE * VEC)
):
    # Each program handles one row (hidden_states[i, :])
    row_id = tl.program_id(0)
    # Accumulate sum of squares across columns
    sum_sq = 0.0
    # First pass: compute inv_rms[i]
    for it in tl.static_range(NUM_ITERS):
        col_start = it * BLOCK_SIZE * VEC
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H
        x = tl.load(x_ptr + row_id * stride_x_row + cols * stride_x_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
    mean_sq = sum_sq / H
    inv_rms = tl.rsqrt(mean_sq + EPS)

    # Second pass: compute outputs y = x * inv_rms * weight
    for it in tl.static_range(NUM_ITERS):
        col_start = it * BLOCK_SIZE * VEC
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H

        x = tl.load(x_ptr + row_id * stride_x_row + cols * stride_x_col, mask=mask, other=0.0)
        x = x.to(tl.float32)

        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        w = w.to(tl.float32)

        y = x * inv_rms * w

        # Store in original dtype
        if OUT_DTYPE == tl.float16:
            tl.store(out_ptr + row_id * stride_out_row + cols * stride_out_col, y.to(tl.float16), mask=mask)
        else:
            tl.store(out_ptr + row_id * stride_out_row + cols * stride_out_col, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
        # Make sure inputs are contiguous
        x = hidden_states.contiguous()
        w = weight.contiguous()

        B, H = x.shape
        # Output tensor in the same dtype as input hidden_states
        out = torch.empty_like(x)

        # Grid: one program per row
        grid = (B,)

        # Tuned parameters: each iteration processes 16384 columns; for H=4096, NUM_ITERS=1
        BLOCK_SIZE = 512
        VEC = 32
        NUM_ITERS = (H + (BLOCK_SIZE * VEC) - 1) // (BLOCK_SIZE * VEC)

        # Choose output dtype for the kernel
        out_dtype = tl.float16 if x.dtype == torch.float16 else tl.bfloat16

        _fused_norm_scale_row_kernel[grid](
            x, w, out,
            B, H, 1e-5,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            out_dtype,
            BLOCK_SIZE, VEC, NUM_ITERS,
            num_warps=8, num_stages=2
        )
        return out


def run(*args):
    return ModelNew()(*args)
