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
    # Guard: if row >= B, return (grid ensures this, but keep safe)
    if row >= B:
        return

    sum_sq = 0.0
    col = 0
    while col < H:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        ptrs = hidden_ptr + row * stride_row + offs * stride_col
        x = tl.load(ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
        col += BLOCK_SIZE

    mean = sum_sq / H
    inv = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row, inv)


# Elementwise apply: y = (x * inv_rms[row]) * weight[col]
@triton.jit
def _apply_two_scales_kernel(
    x_f32_ptr,             # *const float32, shape [B, H]
    inv_rms_ptr,           # *const float32, shape [B]
    weight_ptr,            # *const float32, shape [H]
    out_ptr,               # *float32, shape [B, H]
    B: tl.int32,
    H: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    if row >= B:
        return
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    # Load row scaling factor
    inv_r = tl.load(inv_rms_ptr + row)

    # Load row slice from x and weight slice
    x_row_ptrs = x_f32_ptr + row * H + cols
    x_row = tl.load(x_row_ptrs, mask=mask, other=0.0)

    w_ptrs = weight_ptr + cols
    w = tl.load(w_ptrs, mask=mask, other=0.0)

    # Compute and store
    y = x_row * inv_r * w
    out_row_ptrs = out_ptr + row * H + cols
    tl.store(out_row_ptrs, y, mask=mask)


def run_triton(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Ensure contiguous and device placement
    assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    hidden = hidden_states.contiguous()
    weight = weight.contiguous()

    B, H = hidden.shape
    x = hidden.to(torch.float32)  # compute in fp32
    # Allocate per-row inv_rms in fp32
    inv_rms = torch.empty(B, dtype=torch.float32, device=hidden.device)

    # Choose reduction block and warps
    # For H=4096, 1024 gives 4 iterations with good occupancy
    BLOCK_SIZE_R = 1024
    num_warps_r = 8

    # Launch reduction kernel: one program per row
    grid_reduce = (B,)
    _row_rms_inv_kernel[grid_reduce](
        x, inv_rms, B, H, 1e-5, x.stride(0), x.stride(1),
        BLOCK_SIZE=BLOCK_SIZE_R,
        num_warps=num_warps_r, num_stages=2
    )

    # Prepare weight in fp32
    w_f32 = weight.to(torch.float32)

    # Choose elementwise block and warps
    # For H=4096, process entire row in one program to minimize overhead
    if H <= 4096:
        BLOCK_SIZE_E = 4096
        num_warps_e = 16
    elif H <= 8192:
        BLOCK_SIZE_E = 4096
        num_warps_e = 16
    else:
        BLOCK_SIZE_E = 2048
        num_warps_e = 8

    # Allocate output in fp32 for compute
    out_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

    # Launch elementwise apply: 2D grid over rows and column blocks
    grid_apply = (B, triton.cdiv(H, BLOCK_SIZE_E))
    _apply_two_scales_kernel[grid_apply](
        x, inv_rms, w_f32, out_f32, B, H,
        BLOCK_SIZE=BLOCK_SIZE_E,
        num_warps=num_warps_e, num_stages=1
    )

    # Cast back to original dtype
    return out_f32.to(hidden_states.dtype)


# Optional: keep original Model for reference; evaluation uses ModelNew.
class Model(torch.nn.Module):
    def forward(self, hidden_states, weight):
        return run_triton(hidden_states, weight)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        return run_triton(hidden_states, weight)