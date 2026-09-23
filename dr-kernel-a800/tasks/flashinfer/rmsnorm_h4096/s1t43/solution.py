import torch
import triton
import triton.language as tl


@triton.jit
def _compute_inv_rms_rows_kernel(
    x_ptr,            # *pointer to hidden_states (B,H)
    inv_rms_ptr,      # *pointer to output inv_rms (B,)
    B,                # batch size (rows)
    H,                # hidden size (columns)
    EPS,              # epsilon (float32)
    stride_x_row,     # stride for row in x
    stride_x_col,     # stride for col in x
    BLOCK_SIZE: tl.constexpr,  # columns processed per iteration
    VEC: tl.constexpr,         # vector length along columns per iteration
    NUM_ITERS: tl.constexpr,   # number of iterations to cover H (ceil_div(H, BLOCK_SIZE * VEC))
):
    # One program per row
    row = tl.program_id(0)
    if row >= B:
        return

    # Accumulate sum of squares across columns
    sumsq = 0.0
    # Loop over column chunks (NUM_ITERS is compile-time constant, so Triton can unroll)
    for it in range(0, NUM_ITERS):
        cols = it * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        col_mask = cols < H
        x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=col_mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row, inv_rms)


@triton.jit
def _scale_rows_with_weight_kernel(
    x_ptr,            # *pointer to hidden_states (B,H)
    weight_ptr,       # *pointer to weight (H,)
    inv_rms_ptr,      # *pointer to inv_rms (B,)
    out_ptr,          # *pointer to output (B,H)
    B,                # batch size (rows)
    H,                # hidden size (columns)
    stride_x_row,     # stride for row in x
    stride_x_col,     # stride for col in x
    stride_out_row,   # stride for row in out
    stride_out_col,   # stride for col in out
    BLOCK_SIZE: tl.constexpr,  # columns processed per iteration
    VEC: tl.constexpr,         # vector length along columns per iteration
    NUM_ITERS: tl.constexpr,   # number of iterations to cover H
    OUT_DTYPE: tl.constexpr,   # output dtype (tl.float16 or tl.bfloat16)
):
    # One program per row
    row = tl.program_id(0)
    if row >= B:
        return

    inv = tl.load(inv_rms_ptr + row)  # float32 scalar per row

    for it in range(0, NUM_ITERS):
        cols = it * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        col_mask = cols < H

        # Load x chunk and weight chunk as float32
        x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=col_mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

        # Compute output in fp32, then cast to desired dtype
        y = x * inv * w  # fp32

        # Cast to output dtype (match input hidden_states dtype)
        # Note: OUT_DTYPE is a tl.constexpr derived in host code
        if OUT_DTYPE == tl.bfloat16:
            y = y.to(tl.bfloat16)
        elif OUT_DTYPE == tl.float16:
            y = y.to(tl.float16)
        else:
            # Keep fp32 if some other dtype is passed (shouldn't happen here)
            pass

        tl.store(out_ptr + row * stride_out_row + cols * stride_out_col, y, mask=col_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure device consistency
        x = hidden_states.contiguous()
        w = weight.contiguous()

        B, H = x.shape
        # The original assertion: hidden_size == 4096
        assert H == 4096, "This implementation expects hidden size H=4096"

        # Output tensor same dtype as input
        out = torch.empty_like(x)

        # Strides
        stride_x_row = x.stride(0)
        stride_x_col = x.stride(1)
        stride_out_row = out.stride(0)
        stride_out_col = out.stride(1)

        # Output dtype for Triton (match input dtype)
        out_dtype = tl.float16 if x.dtype == torch.float16 else tl.bfloat16

        # Allocate inv_rms vector (FP32 for numerical stability)
        inv_rms = torch.empty(B, device=x.device, dtype=torch.float32)

        # Kernel launch parameters
        BLOCK_SIZE = 512
        VEC = 16  # columns processed per iteration (512 * 16 = 8192)
        NUM_ITERS = (H + BLOCK_SIZE * VEC - 1) // (BLOCK_SIZE * VEC)

        # Grid: one program per row
        grid = (B,)

        # Kernel 1: compute inv_rms
        _compute_inv_rms_rows_kernel[grid](
            x, inv_rms, B, H, 1e-5,
            stride_x_row, stride_x_col,
            BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, NUM_ITERS=NUM_ITERS,
            num_warps=8, num_stages=2,
        )

        # Kernel 2: scale rows using weight and inv_rms
        _scale_rows_with_weight_kernel[grid](
            x, w, inv_rms, out, B, H,
            stride_x_row, stride_x_col,
            stride_out_row, stride_out_col,
            BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, NUM_ITERS=NUM_ITERS, OUT_DTYPE=out_dtype,
            num_warps=8, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
