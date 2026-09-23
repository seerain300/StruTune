import torch
import triton
import triton.language as tl


@triton.jit
def _row_rms_kernel(
    x_ptr,          # *pointer to hidden_states (contiguous)
    inv_rms_ptr,    # *pointer to per-row inverse RMS
    B,              # batch size (rows)
    H,              # hidden size (columns)
    EPS,            # epsilon (float32)
    stride_x_row,   # stride for row in x
    stride_x_col,   # stride for col in x
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)
    # Accumulate sum of squares in fp32
    sumsq = tl.zeros((), dtype=tl.float32)

    for i in range(0, NUM_ITERS):
        col_start = i * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]
        offs = row * stride_x_row + cols * stride_x_col
        mask = cols < H
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x)

    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row, inv_rms)


@triton.jit
def _scale_row_kernel(
    x_ptr,          # *pointer to hidden_states (contiguous)
    weight_ptr,     # *pointer to weight (contiguous)
    inv_rms_ptr,    # *pointer to per-row inverse RMS
    out_ptr,        # *pointer to output (contiguous)
    B,              # batch size (rows)
    H,              # hidden size (columns)
    OUT_DTYPE: tl.constexpr,  # output dtype (tl.float16 or tl.bfloat16)
    stride_x_row,   # stride for row in x
    stride_x_col,   # stride for col in x
    stride_out_row, # stride for row in out
    stride_out_col, # stride for col in out
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)
    inv_rms = tl.load(inv_rms_ptr + row).to(tl.float32)

    for i in range(0, NUM_ITERS):
        col_start = i * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]
        x_offs = row * stride_x_row + cols * stride_x_col
        w_offs = cols
        mask = cols < H
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + w_offs, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w
        # Cast to desired output dtype
        if OUT_DTYPE == tl.float16:
            y = y.to(tl.float16)
        else:
            y = y.to(tl.bfloat16)  # default bfloat16
        out_offs = row * stride_out_row + cols * stride_out_col
        tl.store(out_ptr + out_offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure CUDA tensors and contiguity
        assert hidden_states.is_cuda and weight.is_cuda, "Triton kernels require CUDA tensors"
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        B, H = hidden_states.shape
        out = torch.empty_like(hidden_states)

        # Per-row inv_rms in fp32
        inv_rms = torch.empty(B, dtype=torch.float32, device=hidden_states.device)

        # Strides
        stride_x_row = hidden_states.stride(0)
        stride_x_col = hidden_states.stride(1)
        stride_out_row = out.stride(0)
        stride_out_col = out.stride(1)

        # Tuned parameters: process 16384 columns per iteration
        BLOCK_SIZE = 2048
        VEC = 8
        NUM_ITERS = (H + BLOCK_SIZE - 1) // BLOCK_SIZE

        # 1) Compute per-row inv_rms
        grid_norm = (B,)
        _row_rms_kernel[grid_norm](
            hidden_states, inv_rms, B, H, 1e-5,
            stride_x_row, stride_x_col,
            BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, NUM_ITERS=NUM_ITERS,
            num_warps=4, num_stages=2
        )

        # 2) Scale and write output
        if hidden_states.dtype == torch.float16:
            OUT_DTYPE = tl.float16
        else:
            OUT_DTYPE = tl.bfloat16  # default for bfloat16 inputs

        grid_scale = (B,)
        _scale_row_kernel[grid_scale](
            hidden_states, weight, inv_rms, out, B, H,
            OUT_DTYPE,
            stride_x_row, stride_x_col,
            stride_out_row, stride_out_col,
            BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, NUM_ITERS=NUM_ITERS,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
