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
    NUM_ITERS: tl.constexpr,  # number of column chunks
):
    # One program per row
    row_id = tl.program_id(0)

    # First pass: accumulate sum of squares to compute inv_rms
    sum_sq = 0.0  # fp32 accumulator
    for it in tl.static_range(NUM_ITERS):
        col_start = it * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H

        # Load x[row, cols] and cast to fp32
        x_row_ptr = x_ptr + row_id * stride_x_row
        x_vals = tl.load(x_row_ptr + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)

        # Sum of squares for this chunk
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)

    # Second pass: compute and store outputs
    for it in tl.static_range(NUM_ITERS):
        col_start = it * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H

        # Load x[row, cols] and weight[cols], cast to fp32
        x_row_ptr = x_ptr + row_id * stride_x_row
        w_ptr = weight_ptr
        x_vals = tl.load(x_row_ptr + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Output: x * inv_rms * weight
        out_vals = x_vals * inv_rms * w_vals

        # Cast to output dtype
        if OUT_DTYPE == tl.bfloat16:
            out_vals = out_vals.to(tl.bfloat16)
        else:
            out_vals = out_vals.to(tl.float16)

        # Store to output
        out_row_ptr = out_ptr + row_id * stride_out_row
        tl.store(out_row_ptr + cols * stride_out_col, out_vals, mask=mask)


def run(hidden_states, weight):
    # hidden_states: [B, 4096], weight: [4096]
    assert hidden_states.shape[1] == 4096, "hidden size must be 4096"
    assert weight.shape[0] == 4096, "weight size must match hidden size"

    EPS = 1e-5

    # Ensure CUDA tensors and contiguous
    x = hidden_states.contiguous()
    w = weight.contiguous()

    # Output tensor in the same dtype/shape as input
    out = torch.empty_like(x)

    # Strides in elements
    stride_x_row = x.stride(0)
    stride_x_col = x.stride(1)
    stride_out_row = out.stride(0)
    stride_out_col = out.stride(1)

    # Tile configuration
    BLOCK_SIZE = 256
    VEC = 32
    NUM_ITERS = (4096 + BLOCK_SIZE - 1) // BLOCK_SIZE  # for H=4096, NUM_ITERS=2

    # Launch Triton kernel: one program per row
    grid = (x.shape[0],)
    _fused_norm_scale_row_kernel[grid](
        x, w, out,
        x.shape[0], 4096, EPS,
        stride_x_row, stride_x_col,
        stride_out_row, stride_out_col,
        OUT_DTYPE=tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16,
        BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, NUM_ITERS=NUM_ITERS,
        num_warps=8, num_stages=2,
    )
    return out


def get_inputs():
    hidden_states = torch.randn([1, 4096], dtype=torch.bfloat16, device='cuda')
    weight = torch.randn([4096], dtype=torch.bfloat16, device='cuda')
    return [hidden_states, weight]


def fused_operator(tensor_0, tensor_1):
    _out = run(tensor_0, tensor_1)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return fused_operator(*args)


def run(*args):
    return ModelNew()(*args)
