import torch
import math

import triton
import triton.language as tl


@triton.jit
def rms_kernel(
    x_ptr,              # *float32, [B, H]
    out_inv_rms_ptr,    # *float32, [B]
    B, H, EPS,          # int, float32
    stride_x_row, stride_x_col,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return

    # Accumulate sum of squares in fp32
    sum_sq = 0.0
    col_start = 0
    while col_start < H:
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = cols < H
        x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
        col_start += BLOCK_N

    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)
    tl.store(out_inv_rms_ptr + row, inv_rms)


@triton.jit
def scale_kernel(
    x_ptr,              # *float32, [B, H]
    weight_ptr,         # *float32, [H]
    out_ptr,            # *float32, [B, H] (we store fp32 here)
    B, H,               # int
    inv_rms_ptr,        # *float32, [B]
    stride_x_row, stride_x_col,
    stride_out_row, stride_out_col,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return

    inv_rms = tl.load(inv_rms_ptr + row)  # fp32 scalar per row
    col_start = 0
    while col_start < H:
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = cols < H
        x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        w = w.to(tl.float32)
        y = x * inv_rms * w  # fp32 compute
        tl.store(out_ptr + row * stride_out_row + cols * stride_out_col, y, mask=mask)
        col_start += BLOCK_N


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Fallback to original PyTorch if not on CUDA
        if hidden_states.device.type != 'cuda':
            batch_size, hidden_size = hidden_states.shape
            assert hidden_size == 4096
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure contiguous inputs
        x = hidden_states.contiguous()
        weight = weight.contiguous()

        # Compute in fp32 for numerical stability
        x_fp32 = x.to(torch.float32)
        weight_fp32 = weight.to(torch.float32)

        B, H = x_fp32.shape

        # Per-row inv_rms buffer (fp32)
        inv_rms = torch.empty(B, dtype=torch.float32, device=x.device)

        # Output buffer (fp32; we cast to original dtype after kernels)
        out_fp32 = torch.empty_like(x_fp32)

        # Strides
        stride_x_row = x_fp32.stride(0)
        stride_x_col = x_fp32.stride(1)
        stride_out_row = out_fp32.stride(0)
        stride_out_col = out_fp32.stride(1)

        # Launch RMS kernel: one program per row
        BLOCK_N = 256  # good default for H up to 4096
        grid = (B,)
        rms_kernel[grid](
            x_fp32, inv_rms,
            B, H, 1e-5,
            stride_x_row, stride_x_col,
            BLOCK_N=BLOCK_N,
        )

        # Launch scale kernel: one program per row
        scale_kernel[grid](
            x_fp32, weight_fp32, out_fp32,
            B, H,
            inv_rms,
            stride_x_row, stride_x_col,
            stride_out_row, stride_out_col,
            BLOCK_N=BLOCK_N,
        )

        # Cast back to original hidden_states dtype
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
