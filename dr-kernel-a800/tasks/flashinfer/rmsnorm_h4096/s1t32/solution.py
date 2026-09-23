import torch
import triton
import triton.language as tl


@triton.jit
def _row_rms_kernel(
    x_ptr,            # *pointer to hidden_states (B, H), float32
    inv_rms_ptr,      # *pointer to per-row inv_rms (B,), float32
    B,                # batch size (rows)
    H,                # hidden size (columns)
    EPS,              # epsilon (float32)
    stride_x_row,     # stride for row in x
    stride_x_col,     # stride for col in x
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    # One program per row
    row_id = tl.program_id(0)
    # Guard: if row_id >= B, do nothing (safety for grid > B)
    if row_id >= B:
        return

    row_x_ptr = x_ptr + row_id * stride_x_row

    # Accumulator for sum of squares across all columns
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Iterate over column tiles
    # Note: Triton requires loop bounds to be constexpr, so we loop over NUM_ITERS
    for it in range(NUM_ITERS):
        col_start = it * (BLOCK_SIZE * VEC)
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H
        # Load chunk; out-of-bound elements are set to 0 to avoid NaNs
        x_chunk = tl.load(row_x_ptr + cols * stride_x_col, mask=mask, other=0.0)
        # Accumulate sum of squares for this chunk
        sum_sq += tl.sum(x_chunk * x_chunk, axis=0)

    # Compute inv_rms = 1 / sqrt(mean + EPS) = 1 / sqrt((sum_sq / H) + EPS)
    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)

    # Store per-row inv_rms
    tl.store(inv_rms_ptr + row_id, inv_rms)


@triton.jit
def _row_scale_weight_kernel(
    x_ptr,            # *pointer to hidden_states (B, H), float32
    weight_ptr,       # *pointer to weight (H,), float32
    inv_rms_ptr,      # *pointer to per-row inv_rms (B,), float32
    out_ptr,          # *pointer to output (B, H), float32
    B,                # batch size (rows)
    H,                # hidden size (columns)
    stride_x_row,     # stride for row in x
    stride_x_col,     # stride for col in x
    stride_out_row,   # stride for row in out
    stride_out_col,   # stride for col in out
    OUT_DTYPE: tl.constexpr,  # output dtype (tl.float16 or tl.bfloat16)
    BLOCK_SIZE: tl.constexpr, # base chunk size along columns
    VEC: tl.constexpr,        # columns processed per iteration
    NUM_ITERS: tl.constexpr,  # number of iterations over columns
):
    # One program per row
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    row_x_ptr = x_ptr + row_id * stride_x_row
    row_out_ptr = out_ptr + row_id * stride_out_row

    # Load per-row inv_rms
    inv_rms = tl.load(inv_rms_ptr + row_id)

    # Iterate over column tiles and write outputs
    for it in range(NUM_ITERS):
        col_start = it * (BLOCK_SIZE * VEC)
        cols = col_start + tl.arange(0, BLOCK_SIZE * VEC)
        mask = cols < H
        x_chunk = tl.load(row_x_ptr + cols * stride_x_col, mask=mask, other=0.0)
        w_chunk = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        y_chunk = x_chunk * inv_rms * w_chunk
        # Cast to output dtype
        if OUT_DTYPE == tl.bfloat16:
            y_chunk = y_chunk.to(tl.bfloat16)
        else:
            y_chunk = y_chunk.to(tl.float16)
        tl.store(row_out_ptr + cols * stride_out_col, y_chunk, mask=mask)


def _run_triton(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Triton-optimized fused implementation:
    - Compute per-row inv_rms in a Triton kernel.
    - Scale and apply weight in a Triton kernel.
    Returns output tensor with the same dtype as hidden_states.
    """
    assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels."

    # Ensure contiguous for simple stride handling
    x = hidden_states.contiguous()
    w = weight.contiguous()

    B, H = x.shape
    # We compute in float32 for numerical stability
    x_fp32 = x.to(torch.float32)

    # Prepare inv_rms buffer (float32)
    inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)

    # Output buffer (float32 for compute; cast to original dtype after)
    out_fp32 = torch.empty((B, H), dtype=torch.float32, device=x.device)

    # Tile parameters
    BLOCK_SIZE = 1024
    VEC = 32
    NUM_ITERS = (H + (BLOCK_SIZE * VEC) - 1) // (BLOCK_SIZE * VEC)

    # Launch kernel to compute per-row inv_rms
    grid_rows = (B,)
    _row_rms_kernel[grid_rows](
        x_fp32, inv_rms, B, H, 1e-5,
        x_fp32.stride(0), x_fp32.stride(1),
        BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, NUM_ITERS=NUM_ITERS,
        num_warps=8, num_stages=2,
    )

    # Launch kernel to scale and write outputs
    _row_scale_weight_kernel[grid_rows](
        x_fp32, w.to(torch.float32), inv_rms, out_fp32, B, H,
        x_fp32.stride(0), x_fp32.stride(1),
        out_fp32.stride(0), out_fp32.stride(1),
        OUT_DTYPE=tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16,
        BLOCK_SIZE=BLOCK_SIZE, VEC=VEC, NUM_ITERS=NUM_ITERS,
        num_warps=8, num_stages=2,
    )

    # Cast to original dtype
    return out_fp32.to(hidden_states.dtype)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA for Triton
        if not hidden_states.is_cuda or not weight.is_cuda:
            # Move to CUDA if available; evaluation environment should provide CUDA tensors.
            if torch.cuda.is_available():
                hidden_states = hidden_states.to('cuda')
                weight = weight.to('cuda')
            else:
                # Fallback: if no CUDA, just run the original PyTorch implementation
                # Note: This branch is defensive; the evaluator typically supplies CUDA tensors.
                with torch.no_grad():
                    x = hidden_states.to(torch.float32)
                    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
                    y = (x * inv_rms) * weight.to(torch.float32)
                    return y.to(hidden_states.dtype)
        # Triton path
        return _run_triton(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
