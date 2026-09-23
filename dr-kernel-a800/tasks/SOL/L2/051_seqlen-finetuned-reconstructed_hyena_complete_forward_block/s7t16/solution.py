import torch
import torch.nn as nn

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton LayerNorm forward kernel:
# Input: x_ptr [M*D] linearized, weight_ptr [D], bias_ptr [D]
# Output: out_ptr [M*D] linearized
# Each program handles one row (normalized across D).
@triton.jit
def layernorm_fwd_kernel(
    x_ptr,          # *f32, input pointer (linearized length M * D)
    w_ptr,          # *f32, gamma (weight), length D
    b_ptr,          # *f32, beta  (bias),   length D
    out_ptr,        # *f32, output pointer (linearized length M * D)
    M: tl.int32,    # number of rows
    D: tl.int32,    # number of columns (d_model), typically 256
    eps: tl.float32
):
    row = tl.program_id(0)  # one program per row
    row_start = row * D

    # Pass 1: compute sum and sum of squares for mean/var
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)
    for d in range(0, D):
        x = tl.load(x_ptr + row_start + d)
        sum_val += x
        sum_sq += x * x

    mean = sum_val / D
    var = sum_sq / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine, write to output
    for d in range(0, D):
        x = tl.load(x_ptr + row_start + d)
        norm = (x - mean) * inv_std
        gamma = tl.load(w_ptr + d)
        beta = tl.load(b_ptr + d)
        y = norm * gamma + beta
        tl.store(out_ptr + row_start + d, y)


# Second LayerNorm kernel identical to first one
@triton.jit
def layernorm_fwd_kernel_2(
    x_ptr,          # *f32, input pointer (linearized length M * D)
    w_ptr,          # *f32, gamma (weight), length D
    b_ptr,          # *f32, beta  (bias),   length D
    out_ptr,        # *f32, output pointer (linearized length M * D)
    M: tl.int32,
    D: tl.int32,
    eps: tl.float32
):
    row = tl.program_id(0)
    row_start = row * D

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)
    for d in range(0, D):
        x = tl.load(x_ptr + row_start + d)
        sum_val += x
        sum_sq += x * x

    mean = sum_val / D
    var = sum_sq / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    for d in range(0, D):
        x = tl.load(x_ptr + row_start + d)
        norm = (x - mean) * inv_std
        gamma = tl.load(w_ptr + d)
        beta = tl.load(b_ptr + d)
        y = norm * gamma + beta
        tl.store(out_ptr + row_start + d, y)


# Triton copy kernel: copy y2_flat (linearized) into out (3D tensor) using linearized indexing
@triton.jit
def copy_to_3d_kernel(
    src_ptr,        # *f32, source linearized (M * D)
    out_ptr,        # *f32, destination pointer for 3D tensor [B, S, D]
    B: tl.int32,    # batch size
    S: tl.int32,    # seq length
    D: tl.int32,    # d_model
    M: tl.int32      # M = B * S
):
    idx = tl.program_id(0)  # linear index 0..M*D-1
    # Compute row id (which row in [B, S]) and column in D
    row = idx // D
    col = idx % D
    # base for the destination: out[row, col]
    out_index = row * D + col
    # Note: out_ptr points to the linear memory of [B, S, D] as well; writing linearized works since out is contiguous.
    tl.store(out_ptr + out_index, tl.load(src_ptr + idx))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # hidden_states: [B, S, D], norm* are [D]
        B, S, D = hidden_states.shape
        M = B * S

        # Prepare output buffer for second LN result (linearized)
        y2_flat = torch.empty(M * D, dtype=torch.float32, device=hidden_states.device)

        # First LayerNorm + affine
        # We operate on hidden_states directly; Triton will read from it via pointer arithmetic (no .reshape or .to).
        grid1 = (M,)
        layernorm_fwd_kernel[grid1](
            hidden_states,                 # input pointer
            norm1_weight,
            norm1_bias,
            y2_flat,                       # output pointer
            M, D,
            1e-5,
            num_warps=4
        )

        # Second LayerNorm + affine (on y2_flat)
        grid2 = (M,)
        layernorm_fwd_kernel_2[grid2](
            y2_flat,                       # input pointer (already contains LN1+affine result)
            norm2_weight,
            norm2_bias,
            y2_flat,                       # output pointer (overwrite)
            M, D,
            1e-5,
            num_warps=4
        )

        # Prepare final output [B, S, D] and copy y2_flat into it using Triton (avoid host-side .view/.reshape)
        out = torch.empty((B, S, D), dtype=torch.float32, device=hidden_states.device)
        grid_copy = (M * D,)
        copy_to_3d_kernel[grid_copy](
            y2_flat,                       # source linearized
            out,                           # destination [B, S, D] (we pass its linear pointer; Triton will write linearly)
            B, S, D, M,
            num_warps=1
        )

        return out


def run(*args):
    return ModelNew()(*args)
