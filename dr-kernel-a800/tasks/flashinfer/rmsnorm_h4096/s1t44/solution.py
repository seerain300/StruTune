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
    OUT_DTYPE: tl.constexpr,  # output dtype selector
    BLOCK_SIZE: tl.constexpr, # base chunk size along columns
    VEC: tl.constexpr,        # columns processed per iteration
    NUM_ITERS: tl.constexpr,  # number of iterations = ceil_div(H, BLOCK_SIZE * VEC)
):
    # One program per row
    row = tl.program_id(0)
    if row >= B:
        return

    # Accumulator for sum of squares across the row
    sum_sq = 0.0  # fp32 scalar

    # First pass: compute sum of squares for this row and derive inv_rms
    for it in range(NUM_ITERS):
        start = it * (BLOCK_SIZE * VEC)
        offs_col = start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = offs_col < H

        # Load x[row, offs_col] in fp32
        x_vals = tl.load(
            x_ptr + row * stride_x_row + offs_col * stride_x_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        # Accumulate sum of squares
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    # Compute inv_rms = 1 / sqrt(mean + EPS)
    mean = sum_sq / H
    inv_rms = tl.math.rsqrt(mean + EPS)

    # Second pass: compute outputs and store
    for it in range(NUM_ITERS):
        start = it * (BLOCK_SIZE * VEC)
        offs_col = start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = offs_col < H

        # Load x[row, offs_col] in fp32
        x_vals = tl.load(
            x_ptr + row * stride_x_row + offs_col * stride_x_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        # Load weight[offs_col] in fp32
        w_vals = tl.load(
            weight_ptr + offs_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        # Compute y = x * inv_rms * w (fp32)
        y_vals = x_vals * inv_rms * w_vals

        # Cast to output dtype and store
        if OUT_DTYPE == 0:  # float16
            y_cast = y_vals.to(tl.float16)
        else:  # bfloat16
            y_cast = y_vals.to(tl.bfloat16)

        tl.store(
            out_ptr + row * stride_out_row + offs_col * stride_out_col,
            y_cast,
            mask=mask,
        )


def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Expect 2D hidden states and 1D weight
    assert hidden_states.dim() == 2, "hidden_states must be 2D"
    assert weight.dim() == 1, "weight must be 1D"
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 4096, "hidden_size must be 4096"

    # Compute in float32
    x = hidden_states.to(torch.float32)
    weight_f32 = weight.to(torch.float32)

    # Output tensor in the same dtype and device as input
    out = torch.empty_like(hidden_states, dtype=x.dtype, device=x.device)

    # Kernel launch parameters: make NUM_ITERS == 1 for H=4096
    BLOCK_SIZE = 512
    VEC = 32
    NUM_ITERS = (hidden_size + (BLOCK_SIZE * VEC) - 1) // (BLOCK_SIZE * VEC)

    # Select output dtype for the kernel
    OUT_DTYPE = 0 if hidden_states.dtype == torch.float16 else 1  # 0: fp16, 1: bf16

    grid = (batch_size,)
    _fused_norm_scale_row_kernel[grid](
        x, weight_f32, out,
        batch_size, hidden_size, 1e-5,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        OUT_DTYPE=OUT_DTYPE,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC=VEC,
        NUM_ITERS=NUM_ITERS,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # If not CUDA, fallback to torch implementation
        if not hidden_states.is_cuda or not weight.is_cuda:
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)
        # Triton kernel performs all computation
        return run(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
