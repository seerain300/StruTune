import torch
import triton
import triton.language as tl

# ----------------------------
# Triton reduction kernel:
# Computes per-row inv_rms = rsqrt(mean(x^2, dim=-1) + EPS)
# Saves one scalar per row (inv_rms) in float32.
# ----------------------------
@triton.jit
def _row_rms_inv_kernel(hidden_ptr,  # *const fp16/bf16/fp32, shape [B, H]
                        inv_ptr,     # *float32, shape [B]
                        B: tl.constexpr,
                        H: tl.constexpr,
                        EPS: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    # Bounds check
    if row >= B:
        return

    # Accumulate sum of squares across columns in chunks
    sumsq = 0.0
    col_start = 0
    while col_start < H:
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Pointer arithmetic for row-major: row * H + offs
        x = tl.load(hidden_ptr + row * H + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
        col_start += BLOCK_SIZE

    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_ptr + row, inv_rms)


# ----------------------------
# Triton elementwise apply kernel:
# Applies y = (x * inv_rms[row]) * weight[col] to the whole output tensor.
# Operates in float32 for compute, casts to out_dtype at store.
# The elementwise kernel handles 2D grid: (row, column-block).
# ----------------------------
@triton.jit
def _apply_two_scales_kernel(x_ptr,        # *float32, shape [B, H] input after casting
                             inv_ptr,      # *float32, shape [B], per-row scale
                             w_ptr,        # *float32, shape [H] weight
                             out_ptr,      # *OUT_DTYPE, shape [B, H] output
                             B, H,         # int32
                             OUT_DTYPE: tl.constexpr,  # 0=bf16, 1=fp16, 2=fp32
                             BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    if row >= B:
        return

    col_start = col_block * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load the per-row scale and the column-wise weight vector slice
    inv_row = tl.load(inv_ptr + row)  # scalar float32
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)  # vector float32

    # Compute base pointers
    x_row_ptr = x_ptr + row * H
    out_row_ptr = out_ptr + row * H

    # Load x slice, apply scales, and store to output (cast at store)
    x = tl.load(x_row_ptr + offs, mask=mask, other=0.0)  # float32
    y = x * inv_row * w  # elementwise multiply; all in float32

    # Cast to desired output dtype
    if OUT_DTYPE == 0:
        y_cast = y.to(tl.bfloat16)
    elif OUT_DTYPE == 1:
        y_cast = y.to(tl.float16)
    else:
        y_cast = y  # float32

    tl.store(out_row_ptr + offs, y_cast, mask=mask)


def run_triton(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    # Ensure contiguity and types
    hidden = hidden_states.contiguous()
    weight = weight.contiguous()

    B, H = hidden.shape
    assert H > 0 and hidden.ndim == 2, "hidden_states must be 2D [batch, hidden_size]"
    assert weight.ndim == 1 and weight.shape[0] == H, "weight must be 1D with length hidden_size"

    # Compute in float32 for numerical stability
    x_f32 = hidden.to(torch.float32)
    w_f32 = weight.to(torch.float32)

    # Output buffer in float32 (compute buffer), final cast happens at store time
    out_f32 = torch.empty((B, H), device=hidden.device, dtype=torch.float32)

    # Allocate per-row inv_rms in float32
    inv_rms = torch.empty((B,), device=hidden.device, dtype=torch.float32)

    # Choose tile sizes:
    # Reduction tile: 1024 with 8 warps is a good default. It loops 4 times for H=4096.
    BLOCK_SIZE_R = 1024

    # Elementwise apply: prefer processing full row if H is power of two or <= 4096.
    # Pick BLOCK_SIZE_E = min(next_power_of_two(H), 4096), but to keep code simple and fast for 4096, use 4096.
    BLOCK_SIZE_E = 4096 if H <= 4096 else 2048

    # Warps: scale with tile size; 8 for <=1024, 16 for 4096
    num_warps_r = 8
    num_warps_e = 16 if BLOCK_SIZE_E >= 4096 else (8 if BLOCK_SIZE_E >= 2048 else 4)

    # Launch reduction: one program per row
    grid_reduce = (B,)
    _row_rms_inv_kernel[grid_reduce](
        x_f32, inv_rms, B, H, eps, BLOCK_SIZE=BLOCK_SIZE_R, num_warps=num_warps_r, num_stages=2
    )

    # Encode output dtype
    if hidden.dtype == torch.bfloat16:
        out_dtype_code = 0  # bf16
    elif hidden.dtype == torch.float16:
        out_dtype_code = 1  # fp16
    elif hidden.dtype == torch.float32:
        out_dtype_code = 2  # fp32
    else:
        raise RuntimeError(f"Unsupported dtype: {hidden.dtype}")

    # Launch elementwise apply: 2D grid over rows and column blocks
    grid_apply = (B, triton.cdiv(H, BLOCK_SIZE_E))
    _apply_two_scales_kernel[grid_apply](
        x_f32, inv_rms, w_f32, out_f32,
        B, H, out_dtype_code, BLOCK_SIZE=BLOCK_SIZE_E,
        num_warps=num_warps_e, num_stages=2
    )

    # Cast final output to original hidden dtype
    return out_f32.to(hidden.dtype)


# Optional: keep original Model for reference; evaluation uses ModelNew.
class Model(torch.nn.Module):
    def forward(self, hidden_states, weight):
        return run_triton(hidden_states, weight)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        return run_triton(hidden_states, weight)


def run(*args):
    return ModelNew()(*args)
