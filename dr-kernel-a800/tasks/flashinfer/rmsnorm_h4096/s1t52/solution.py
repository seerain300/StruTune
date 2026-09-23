import torch
import triton
import triton.language as tl


@triton.jit
def _fused_row_norm_scale_kernel(
    x_ptr,            # *pointer to hidden_states (B, H)
    weight_ptr,       # *pointer to weight (H,)
    out_ptr,          # *pointer to output (B, H)
    B,                # batch size (rows)
    H,                # hidden size (columns)
    EPS,              # epsilon (float32)
    stride_x_row,     # stride for row in x (elements)
    stride_x_col,     # stride for col in x (elements)
    stride_out_row,   # stride for row in out (elements)
    stride_out_col,   # stride for col in out (elements)
    OUT_DTYPE: tl.constexpr,   # output dtype (tl.bfloat16 or tl.float16)
    BLOCK_SIZE: tl.constexpr,  # base chunk size along columns (e.g., 256)
    VEC: tl.constexpr,         # columns processed per iteration (e.g., 64)
    NUM_ITERS: tl.constexpr,   # ceil_div(H, BLOCK_SIZE * VEC)
):
    row_id = tl.program_id(0)
    # Bounds check: if row_id >= B, return (safety in case grid > B)
    if row_id >= B:
        return

    # 1) Compute sum of squares over the row to get inv_rms
    sum_sq = 0.0
    for it in tl.static_range(NUM_ITERS):
        cols = it * BLOCK_SIZE * VEC + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H
        x_ptrs = x_ptr + row_id * stride_x_row + cols * stride_x_col
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
    mean = sum_sq / H
    inv_rms = tl.math.rsqrt(mean + EPS)  # 1 / sqrt(mean + EPS)

    # 2) Produce output: y[row, j] = x[row, j] * inv_rms * weight[j]
    for it in tl.static_range(NUM_ITERS):
        cols = it * BLOCK_SIZE * VEC + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H
        x_ptrs = x_ptr + row_id * stride_x_row + cols * stride_x_col
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        w_ptrs = weight_ptr + cols * 1  # weight is 1D; stride = 1
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

        out_vals = x_vals * inv_rms * w_vals

        # Cast to desired output dtype
        if OUT_DTYPE == tl.bfloat16:
            out_vals = out_vals.to(tl.bfloat16)
        elif OUT_DTYPE == tl.float16:
            out_vals = out_vals.to(tl.float16)
        else:
            # default to float32 if unexpected dtype
            out_vals = out_vals.to(tl.float32)

        out_ptrs = out_ptr + row_id * stride_out_row + cols * stride_out_col
        tl.store(out_ptrs, out_vals, mask=mask)


def _run_triton(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of:
      x = hidden_states.to(torch.float32)
      inv_rms = 1 / sqrt(mean(x^2, dim=-1, keepdim=True) + EPS)
      y = (x * inv_rms) * weight.to(torch.float32)
      return y.to(hidden_states.dtype)
    Assumes hidden_states is [B, H], weight is [H]. Computes in FP32, stores in hidden_states.dtype.
    """
    assert hidden_states.dim() == 2, "hidden_states must be 2D [B, H]"
    assert weight.dim() == 1, "weight must be 1D [H]"
    B, H = hidden_states.shape

    # Allocate output tensor with same shape and dtype as input
    out = torch.empty((B, H), device=hidden_states.device, dtype=hidden_states.dtype)

    # Ensure inputs are contiguous for simple stride arithmetic
    x = hidden_states.contiguous()
    w = weight.contiguous()

    # Select output dtype for Triton kernel
    if hidden_states.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        out_dtype = tl.float16
    else:
        # default to float32 if unexpected dtype
        out_dtype = tl.float32

    # Grid: one program per row
    grid = (B,)

    # Tuning parameters: process 16384 columns per iteration to minimize NUM_ITERS
    BLOCK_SIZE = 256
    VEC = 64
    NUM_ITERS = triton.cdiv(H, BLOCK_SIZE * VEC)

    _fused_row_norm_scale_kernel[grid](
        x, w, out,
        B, H, 1e-5,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        OUT_DTYPE=out_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC=VEC,
        NUM_ITERS=NUM_ITERS,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Triton path: ensure CUDA and use kernel
        return _run_triton(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
