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
    NUM_ITERS: tl.constexpr,  # number of iterations across columns
):
    # One program per row
    row_id = tl.program_id(0)
    # Guard: if grid > B, return (safety)
    if row_id >= B:
        return

    # First pass: compute sum of squares across the row
    sum_sq = 0.0
    for it in tl.static_range(NUM_ITERS):
        col_start = it * (BLOCK_SIZE * VEC)
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H

        x_row_ptr = x_ptr + row_id * stride_x_row + cols * stride_x_col
        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)
        x_vals_f32 = x_vals.to(tl.float32)
        sum_sq += tl.sum(x_vals_f32 * x_vals_f32)

    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)

    # Second pass: compute outputs and store
    for it in tl.static_range(NUM_ITERS):
        col_start = it * (BLOCK_SIZE * VEC)
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H

        x_row_ptr = x_ptr + row_id * stride_x_row + cols * stride_x_col
        w_ptr = weight_ptr + cols
        out_row_ptr = out_ptr + row_id * stride_out_row + cols * stride_out_col

        x_vals = tl.load(x_row_ptr, mask=mask, other=0.0)
        x_vals_f32 = x_vals.to(tl.float32)

        w_vals = tl.load(w_ptr, mask=mask, other=0.0)
        w_vals_f32 = w_vals.to(tl.float32)

        y_vals = x_vals_f32 * inv_rms * w_vals_f32

        # Cast to output dtype
        if OUT_DTYPE == tl.float16:
            y_cast = y_vals.to(tl.float16)
        else:
            y_cast = y_vals.to(tl.bfloat16)

        tl.store(out_row_ptr, y_cast, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Expect [B, 4096] for hidden_states and [4096] for weight
        B, H = hidden_states.shape
        assert H == 4096, "hidden_size must be 4096"

        # Compute in float32
        x = hidden_states.to(torch.float32).contiguous()
        w = weight.to(torch.float32).contiguous()

        out = torch.empty_like(x, device=x.device, dtype=torch.float32)

        # Strides
        stride_x_row = x.stride(0)
        stride_x_col = x.stride(1)
        stride_out_row = out.stride(0)
        stride_out_col = out.stride(1)

        # Tiling parameters
        BLOCK_SIZE = 1024
        VEC = 32
        NUM_ITERS = (H + BLOCK_SIZE * VEC - 1) // (BLOCK_SIZE * VEC)

        # Grid: one program per row
        grid = (B,)

        # Output dtype for casting
        out_dtype = torch.bfloat16 if hidden_states.dtype == torch.bfloat16 else torch.float16

        # Launch kernel (ensure full argument list to avoid NameError)
        _fused_norm_scale_row_kernel[grid](
            x, w, out,
            B, H, 1e-5,
            stride_x_row, stride_x_col,
            stride_out_row, stride_out_col,
            OUT_DTYPE=tl.bfloat16 if out_dtype == torch.bfloat16 else tl.float16,
            BLOCK_SIZE=BLOCK_SIZE,
            VEC=VEC,
            NUM_ITERS=NUM_ITERS,
            num_warps=8,
            num_stages=2,
        )

        # Cast back to original dtype of hidden_states
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
