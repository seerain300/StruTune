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
    NUM_ITERS: tl.constexpr,  # number of column iterations
):
    # One program per row
    row_id = tl.program_id(0)

    # First pass: compute per-row sum of squares and inv_rms
    sumsq = 0.0
    for it in range(NUM_ITERS):
        col_start = it * BLOCK_SIZE * VEC
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H
        x = tl.load(x_ptr + row_id * stride_x_row + cols * stride_x_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)

    # Second pass: compute outputs
    for it in range(NUM_ITERS):
        col_start = it * BLOCK_SIZE * VEC
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H

        # Load row slice
        x = tl.load(x_ptr + row_id * stride_x_row + cols * stride_x_col, mask=mask, other=0.0)
        x = x.to(tl.float32)

        # Load weight slice and broadcast
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        w = w.to(tl.float32)

        y = x * inv_rms * w

        # Cast to output dtype
        if OUT_DTYPE == tl.float16:
            y = y.to(tl.float16)
        elif OUT_DTYPE == tl.bfloat16:
            y = y.to(tl.bfloat16)

        tl.store(out_ptr + row_id * stride_out_row + cols * stride_out_col, y, mask=mask)


def _choose_kernel_config(H: int):
    # Choose a tile size that minimizes NUM_ITERS while keeping register pressure reasonable.
    # For H=4096, BLOCK_SIZE * VEC >= H to make NUM_ITERS == 1.
    # Use BLOCK_SIZE=512 and VEC=128 to process 65536 columns per iteration.
    BLOCK_SIZE = 512
    VEC = 128
    NUM_ITERS = (H + BLOCK_SIZE * VEC - 1) // (BLOCK_SIZE * VEC)
    NUM_ITERS = max(1, NUM_ITERS)
    grid = (B,)  # one program per row
    return BLOCK_SIZE, VEC, NUM_ITERS, grid


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Expect 2D hidden_states [B, H] and 1D weight [H]
        assert hidden_states.ndim == 2, "hidden_states must be 2D [B, H]"
        assert weight.ndim == 1, "weight must be 1D [H]"
        B, H = hidden_states.shape
        assert H == 4096, "This optimized kernel assumes hidden size == 4096"

        # Ensure contiguous for simple stride handling
        x = hidden_states.contiguous()
        w = weight.contiguous()
        # Output tensor
        out = torch.empty((B, H), dtype=x.dtype, device=x.device)

        # Strides
        stride_x_row = x.stride(0)
        stride_x_col = x.stride(1)
        stride_out_row = out.stride(0)
        stride_out_col = out.stride(1)

        # Determine output dtype for Triton
        OUT_DTYPE = tl.float16 if x.dtype == torch.float16 else tl.bfloat16

        # Choose kernel config
        BLOCK_SIZE, VEC, NUM_ITERS, grid = _choose_kernel_config(H)

        # Launch Triton kernel
        _fused_norm_scale_row_kernel[grid](
            x, w, out,
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


def run(*args):
    return ModelNew()(*args)
