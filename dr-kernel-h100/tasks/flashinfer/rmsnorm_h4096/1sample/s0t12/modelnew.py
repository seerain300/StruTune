import torch
import triton
import triton.language as tl


# Triton reduction kernel:
# For each row, compute inv_rms = rsqrt(mean(x^2, dim=-1) + EPS) in float32.
@triton.jit
def _row_rms_inv_kernel(
    hidden_ptr,           # *const fp16/bf16/fp32, shape [B, H]
    inv_rms_ptr,          # *float32, shape [B]
    B: tl.int32,          # batch size
    H: tl.int32,          # hidden size
    EPS: tl.float32,      # epsilon
    stride_h_row: tl.int32,  # hidden row stride (elements)
    stride_h_col: tl.int32,  # hidden col stride (elements)
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
        # Load row slice; cast to float32 for accumulation
        x = tl.load(hidden_ptr + row * stride_h_row + offs * stride_h_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
        col += BLOCK_SIZE

    mean = sum_sq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row, inv_rms)


# Triton elementwise apply kernel:
# For each row, load inv_rms[row], then for each column block:
# out[row, col] = hidden[row, col] * inv_rms[row] * weight[col] (all in fp32).
@triton.jit
def _apply_two_scales_kernel(
    hidden_ptr,           # *const fp16/bf16/fp32, shape [B, H]
    inv_rms_ptr,          # *float32, shape [B]
    weight_ptr,           # *float32, shape [H]
    out_ptr,              # *float32, shape [B, H]
    B: tl.int32,
    H: tl.int32,
    W: tl.int32,          # weight length (== H in our case)
    stride_h_row: tl.int32,  # hidden row stride
    stride_h_col: tl.int32,  # hidden col stride
    stride_out_row: tl.int32, # out row stride
    stride_out_col: tl.int32, # out col stride
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    if row >= B:
        return

    col_start = col_block * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load row-wise scaling and column-wise weights
    inv_r = tl.load(inv_rms_ptr + row)  # scalar float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)  # vector float32

    # Load input row slice and compute output
    x = tl.load(hidden_ptr + row * stride_h_row + offs * stride_h_col, mask=mask, other=0.0)
    x = x.to(tl.float32)
    y = x * inv_r * w

    # Store output
    tl.store(out_ptr + row * stride_out_row + offs * stride_out_col, y, mask=mask)


def run_triton(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are on CUDA
    assert hidden_states.is_cuda and weight.is_cuda, "Triton kernels require CUDA tensors."

    # Shapes
    B, H = hidden_states.shape
    # Weight length should match hidden size (assert for safety)
    assert weight.numel() == H, "Weight length must match hidden size."

    # Compute in float32 for stability
    hidden = hidden_states.contiguous()
    weight_f32 = weight.to(torch.float32).contiguous()

    # Allocate inv_rms (one per row)
    inv_rms = torch.empty((B,), dtype=torch.float32, device=hidden.device)

    # Launch reduction kernel
    BLOCK_SIZE_R = 1024
    grid_reduce = (B,)
    _row_rms_inv_kernel[grid_reduce](
        hidden, inv_rms, B, H, 1e-5, hidden.stride(0), hidden.stride(1),
        BLOCK_SIZE=BLOCK_SIZE_R,
        num_warps=8, num_stages=2
    )

    # Allocate output buffer in fp32 for compute
    out_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

    # Choose elementwise tile and warps based on H
    if H >= 4096:
        BLOCK_SIZE_E = 4096
        num_warps_e = 16
    elif H >= 2048:
        BLOCK_SIZE_E = 2048
        num_warps_e = 16
    else:
        BLOCK_SIZE_E = 1024
        num_warps_e = 8

    # Launch elementwise apply: 2D grid over rows and column blocks
    grid_apply = (B, triton.cdiv(H, BLOCK_SIZE_E))
    _apply_two_scales_kernel[grid_apply](
        hidden, inv_rms, weight_f32, out_f32,
        B, H, H,
        hidden.stride(0), hidden.stride(1),
        out_f32.stride(0), out_f32.stride(1),
        BLOCK_SIZE=BLOCK_SIZE_E,
        num_warps=num_warps_e, num_stages=2
    )

    # Cast back to original dtype
    return out_f32.to(hidden_states.dtype)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return run_triton(hidden_states, weight)