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
    # One Triton program per row
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    # First pass: compute sum of squares for this row to get inv_rms
    sum_sq = 0.0
    for it in tl.static_range(NUM_ITERS):
        col_start = it * BLOCK_SIZE * VEC
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H
        # Load x[row, cols] as float32
        x_vals = tl.load(
            x_ptr + row_id * stride_x_row + cols * stride_x_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        # Accumulate sum of squares
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
    mean = sum_sq / H
    inv_rms = tl.rsqrt(mean + EPS)  # float32 scalar

    # Second pass: compute output y[row, cols] = x[row, cols] * inv_rms * weight[cols]
    for it in tl.static_range(NUM_ITERS):
        col_start = it * BLOCK_SIZE * VEC
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H
        # Load x[row, cols]
        x_vals = tl.load(
            x_ptr + row_id * stride_x_row + cols * stride_x_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        # Load weight[cols]
        w_vals = tl.load(
            weight_ptr + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        # Compute output
        y_vals = x_vals * inv_rms * w_vals
        # Cast to output dtype
        if OUT_DTYPE == tl.float16:
            y_vals = y_vals.to(tl.float16)
        else:
            y_vals = y_vals.to(tl.bfloat16)
        # Store
        tl.store(
            out_ptr + row_id * stride_out_row + cols * stride_out_col,
            y_vals,
            mask=mask,
        )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        x = hidden_states.contiguous()
        w = weight.contiguous()

        B, H = x.shape
        out = torch.empty_like(x)

        # Grid: one program per row
        grid = (B,)

        # Fixed high-performance tiling (earlier best performer)
        BLOCK_SIZE = 512
        VEC = 16
        NUM_ITERS = (H + BLOCK_SIZE * VEC - 1) // (BLOCK_SIZE * VEC)

        # Strides
        stride_x_row = x.stride(0)
        stride_x_col = x.stride(1)
        stride_out_row = out.stride(0)
        stride_out_col = out.stride(1)

        # Determine OUT_DTYPE based on input dtype
        if x.dtype == torch.float16:
            OUT_DTYPE = tl.float16
        elif x.dtype == torch.bfloat16:
            OUT_DTYPE = tl.bfloat16
        else:
            # Fallback: keep as float32 if unexpected dtype
            OUT_DTYPE = tl.float32

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
        return out


def run(*args):
    return ModelNew()(*args)
