import torch
import triton
import triton.language as tl


@triton.jit
def _row_rms_kernel(
    x_ptr,            # *pointer to hidden_states (B, H), float32
    inv_rms_ptr,      # *pointer to per-row inverse RMS, float32
    B,                # batch size (rows)
    H,                # hidden size (columns)
    EPS,              # epsilon (float32)
    stride_x_row,     # row stride for x
    stride_x_col,     # col stride for x
    NUM_ITERS: tl.constexpr,  # number of column tile iterations
    BLOCK_SIZE: tl.constexpr, # base chunk size along columns
    VEC: tl.constexpr,        # columns processed per iteration
):
    # One program per row
    row_id = tl.program_id(0)
    # Guard: if row_id >= B, exit (safety in case grid > B)
    if row_id >= B:
        return

    # Base pointer for this row
    row_x_ptr = x_ptr + row_id * stride_x_row

    # Accumulator for sum of squares in FP32
    sum_sq = 0.0

    # First pass: accumulate sum of squares across all columns
    for it in range(NUM_ITERS):
        col_start = it * (BLOCK_SIZE * VEC)
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H

        # Load x[row, cols] as float32
        x_vals = tl.load(row_x_ptr + cols * stride_x_col, mask=mask, other=0.0)
        # Sum of squares for this tile
        chunk_sums = tl.sum(x_vals * x_vals, axis=0)
        # Accumulate
        sum_sq += chunk_sums

    # Compute mean and inverse RMS
    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)

    # Store inv_rms[row] as FP32
    tl.store(inv_rms_ptr + row_id, inv_rms)


@triton.jit
def _row_scale_kernel(
    x_ptr,            # *pointer to hidden_states (B, H), float32
    weight_ptr,       # *pointer to weight (H,), float32
    out_ptr,          # *pointer to output (B, H), float32
    inv_rms_ptr,      # *pointer to inv_rms (B,), float32
    B,                # batch size (rows)
    H,                # hidden size (columns)
    stride_x_row,     # row stride for x
    stride_x_col,     # col stride for x
    stride_out_row,   # row stride for out
    stride_out_col,   # col stride for out
    NUM_ITERS: tl.constexpr,  # number of column tile iterations
    BLOCK_SIZE: tl.constexpr, # base chunk size along columns
    VEC: tl.constexpr,        # columns processed per iteration
    OUT_DTYPE: tl.constexpr,  # output dtype (tl.float16 or tl.bfloat16)
):
    # One program per row
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    # Load inv_rms[row] (FP32)
    inv_rms = tl.load(inv_rms_ptr + row_id)

    row_x_ptr = x_ptr + row_id * stride_x_row
    row_out_ptr = out_ptr + row_id * stride_out_row

    # Second pass: compute and store outputs
    for it in range(NUM_ITERS):
        col_start = it * (BLOCK_SIZE * VEC)
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H

        # Load x[row, cols] and weight[cols] as float32
        x_vals = tl.load(row_x_ptr + cols * stride_x_col, mask=mask, other=0.0)
        w_vals = tl.load(weight_ptr + cols, mask=mask, other=0.0)

        # Compute y = x * inv_rms * weight in FP32
        y_vals = x_vals * inv_rms * w_vals

        # Cast to requested output dtype
        if OUT_DTYPE == tl.float16:
            y_cast = y_vals.to(tl.float16)
        elif OUT_DTYPE == tl.bfloat16:
            y_cast = y_vals.to(tl.bfloat16)
        else:
            # default keep FP32 (shouldn't happen, but safe)
            y_cast = y_vals

        # Store
        tl.store(row_out_ptr + cols * stride_out_col, y_cast, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        x = hidden_states.contiguous()
        w = weight.contiguous()

        # Compute in float32 for numerical stability
        x_fp32 = x.to(torch.float32)
        w_fp32 = w.to(torch.float32)

        B, H = x_fp32.shape
        # Prepare output buffer in FP32
        out_fp32 = torch.empty((B, H), device=x.device, dtype=torch.float32)

        # Choose tiling parameters
        BLOCK_SIZE = 1024
        VEC = 32
        NUM_ITERS = (H + (BLOCK_SIZE * VEC) - 1) // (BLOCK_SIZE * VEC)

        # Strides (in elements, since we pass pointers)
        stride_x_row = x_fp32.stride(0)
        stride_x_col = x_fp32.stride(1)
        stride_out_row = out_fp32.stride(0)
        stride_out_col = out_fp32.stride(1)

        # Allocate per-row inv_rms in FP32
        inv_rms = torch.empty(B, device=x.device, dtype=torch.float32)

        # Grid: one program per row
        grid = (B,)

        # Kernel 1: compute per-row inverse RMS
        _row_rms_kernel[grid](
            x_fp32, inv_rms, B, H, 1e-5,
            stride_x_row, stride_x_col,
            NUM_ITERS=NUM_ITERS,
            BLOCK_SIZE=BLOCK_SIZE,
            VEC=VEC,
            num_warps=8, num_stages=2
        )

        # Kernel 2: compute output y = x * inv_rms * weight
        # Determine output dtype based on original hidden_states dtype
        out_dtype = torch.float16 if x.dtype == torch.float16 else torch.bfloat16

        _row_scale_kernel[grid](
            x_fp32, w_fp32, out_fp32, inv_rms, B, H,
            stride_x_row, stride_x_col,
            stride_out_row, stride_out_col,
            NUM_ITERS=NUM_ITERS,
            BLOCK_SIZE=BLOCK_SIZE,
            VEC=VEC,
            OUT_DTYPE=tl.float16 if out_dtype == torch.float16 else tl.bfloat16,
            num_warps=8, num_stages=2
        )

        # Cast to original dtype of hidden_states and return
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
