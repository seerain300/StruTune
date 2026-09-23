import torch
import triton
import triton.language as tl

# Reduction: compute inv_rms for each row (per batch element)
@triton.jit
def _row_rms_inv_kernel(x_ptr, out_inv_ptr, B, H, EPS, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    # Each program handles one row
    # Accumulate sum of squares in float32
    sumsq = 0.0
    col = 0
    while col < H:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(x_ptr + row_id * H + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
        col += BLOCK_SIZE
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(out_inv_ptr + row_id, inv_rms)


# Elementwise apply: y[row, col] = x[row, col] * inv_rms[row] * weight[col]
@triton.jit
def _apply_two_scales_kernel(x_ptr, inv_ptr, weight_ptr, out_ptr,
                              B, H, OUT_DTYPE_CODE: tl.constexpr,
                              BLOCK_SIZE: tl.constexpr):
    # 2D grid: (row, col_block)
    row_id = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    # Load inv_rms for this row
    inv = tl.load(inv_ptr + row_id)

    # Load the corresponding weight slice
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    # Compute in float32
    inv = inv.to(tl.float32)
    w = w.to(tl.float32)

    # Load input slice for this row
    x = tl.load(x_ptr + row_id * H + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)

    y = x * inv * w

    # Cast to desired output dtype
    # 0: bf16, 1: fp16, 2: fp32
    y_cast = y
    if OUT_DTYPE_CODE == 0:
        y_cast = y.to(tl.bfloat16)
    elif OUT_DTYPE_CODE == 1:
        y_cast = y.to(tl.float16)
    else:
        y_cast = y  # float32

    tl.store(out_ptr + row_id * H + cols, y_cast, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Expect hidden_states: [B, H], weight: [H]
        # We keep dtype handling minimal and perform compute in float32 for stability.
        assert hidden_states.dim() == 2, "hidden_states must be 2D [B, H]"
        assert weight.dim() == 1, "weight must be 1D [H]"
        B, H = hidden_states.shape

        # Ensure contiguity for Triton
        x = hidden_states.contiguous()
        w = weight.contiguous()

        # Compute in float32
        x_f32 = x.to(torch.float32)
        EPS = 1e-5

        # 1) Compute per-row inv_rms
        out_inv = torch.empty((B,), device=x.device, dtype=torch.float32)
        # Tune BLOCK_SIZE for reduction; 1024 works broadly
        BLOCK_SIZE_R = 1024
        grid_reduce = (B,)
        _row_rms_inv_kernel[grid_reduce](
            x_f32, out_inv, B, H, EPS,
            BLOCK_SIZE=BLOCK_SIZE_R,
            num_warps=8,
            num_stages=2
        )

        # 2) Apply two scales in a single pass
        # Allocate output in the original dtype; we'll cast inside kernel
        out = torch.empty((B, H), device=x.device, dtype=x.dtype)

        # Determine OUT_DTYPE_CODE for kernel
        if x.dtype == torch.bfloat16:
            out_dtype_code = 0  # bf16
        elif x.dtype == torch.float16:
            out_dtype_code = 1  # fp16
        elif x.dtype == torch.float32:
            out_dtype_code = 2  # fp32
        else:
            raise RuntimeError(f"Unsupported dtype: {x.dtype}")

        # Elementwise kernel launch: 2D grid over rows and column blocks
        BLOCK_SIZE_E = 2048  # Larger block for better throughput on 4096-wide rows
        grid_apply = (B, triton.cdiv(H, BLOCK_SIZE_E))
        _apply_two_scales_kernel[grid_apply](
            x_f32, out_inv, w.to(torch.float32), out,
            B, H, out_dtype_code,
            BLOCK_SIZE=BLOCK_SIZE_E,
            num_warps=8,
            num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
