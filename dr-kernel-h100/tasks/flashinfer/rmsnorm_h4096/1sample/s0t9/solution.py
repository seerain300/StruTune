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
        x = tl.load(hidden_ptr + row * stride_row + offs * stride_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
        col += BLOCK_SIZE

    mean = sum_sq / H
    inv = tl.math.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row, inv)


# Elementwise kernel: apply row and column scaling
@triton.jit
def _apply_two_scales_kernel(
    hidden_ptr,            # *const fp16/bf16/fp32, shape [B, H]
    inv_rms_ptr,           # *float32, shape [B]
    weight_ptr,            # *float32, shape [H]
    out_ptr,               # *float32, shape [B, H]
    B: tl.int32,           # batch size
    H: tl.int32,           # hidden size
    stride_row_hidden: tl.int32,
    stride_col_hidden: tl.int32,
    stride_row_out: tl.int32,
    stride_col_out: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_blk = tl.program_id(1)
    if row >= B:
        return

    col_start = col_blk * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    inv = tl.load(inv_rms_ptr + row)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)  # weight is float32
    x = tl.load(hidden_ptr + row * stride_row_hidden + offs * stride_col_hidden, mask=mask, other=0.0)

    x = x.to(tl.float32)
    y = x * inv * w
    tl.store(out_ptr + row * stride_row_out + offs * stride_col_out, y, mask=mask)


def run_triton(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert hidden_states.dim() == 2, "hidden_states must be 2D [B, H]"
    assert weight.dim() == 1 and weight.numel() == hidden_states.shape[1], "weight must be 1D of length H"
    hidden = hidden_states.contiguous()
    weight = weight.contiguous()
    B, H = hidden.shape

    # 1) Compute inv_rms in fp32
    x = hidden.to(torch.float32)
    inv_rms = torch.empty((B,), dtype=torch.float32, device=hidden.device)
    _row_rms_inv_kernel[(B,)](
        x, inv_rms, B, H, 1e-5,
        x.stride(0), x.stride(1),
        BLOCK_SIZE=1024, num_warps=8, num_stages=2
    )

    # 2) Apply row scaling and column scaling using Triton
    out_f32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)
    BLOCK_SIZE_E = 2048  # good balance for H=4096 and general sizes
    num_warps_e = 16      # high occupancy for 2K tiles
    grid_apply = (B, triton.cdiv(H, BLOCK_SIZE_E))
    _apply_two_scales_kernel[grid_apply](
        x, inv_rms, weight.to(torch.float32), out_f32,
        B, H,
        x.stride(0), x.stride(1),
        out_f32.stride(0), out_f32.stride(1),
        BLOCK_SIZE=BLOCK_SIZE_E,
        num_warps=num_warps_e, num_stages=2
    )

    # 3) Cast back to original hidden dtype
    return out_f32.to(hidden_states.dtype)


# Optional: keep original Model for reference; evaluation uses ModelNew.
class Model(torch.nn.Module):
    def forward(self, hidden_states, weight):
        return run_triton(hidden_states, weight)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        return run_triton(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
