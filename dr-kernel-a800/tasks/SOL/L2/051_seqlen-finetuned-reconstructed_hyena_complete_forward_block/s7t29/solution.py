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
# Input: x_ptr [M*D] contiguous, weight_ptr [D], bias_ptr [D]
# Output: out_ptr [M*D] contiguous
# Each program handles one row (length D), normalizes across D, applies affine (gamma/beta).
@triton.jit
def layernorm_fwd_kernel(
    x_ptr,          # *f32, input flattened [M*D]
    w_ptr,          # *f32, gamma [D]
    b_ptr,          # *f32, beta  [D]
    out_ptr,        # *f32, output [M*D]
    M: tl.constexpr,
    D: tl.constexpr,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= M:
        return

    row_start = row_id * D

    # First pass: compute sum and sum of squares over D in tiles
    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, D, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    D_f = tl.full((), D, tl.float32)
    mean = sum_val / D_f
    var = sum_sq / D_f - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for off in range(0, D, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        gamma = tl.load(w_ptr + idx, mask=mask, other=1.0)
        beta = tl.load(b_ptr + idx, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        y = y * gamma + beta
        tl.store(out_ptr + row_start + idx, y, mask=mask)


# Triton kernel to fill a 1D output tensor with a scalar value
@triton.jit
def fill_kernel(
    out_ptr,        # *f32, output [N]
    val: tl.float32,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    v = val + tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    tl.store(out_ptr + offs, v, mask=mask)


# Triton kernel to fill a scalar (0D) output
@triton.jit
def fill_scalar_kernel(
    out_ptr,        # *f32, single element
    val: tl.float32,
):
    tl.store(out_ptr, val)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias):
        # hidden_states: [B, S, D], dtype float32
        # norm1_weight, norm1_bias, norm2_weight, norm2_bias: [D], dtype float32
        assert hidden_states.dim() == 3, "hidden_states must be [B, S, D]"
        assert norm1_weight.dim() == 1 and norm1_bias.dim() == 1, "norm1_weight/bias must be 1D [D]"
        assert norm2_weight.dim() == 1 and norm2_bias.dim() == 1, "norm2_weight/bias must be 1D [D]"

        B, S, D = hidden_states.shape
        M = B * S

        # Ensure contiguous and flatten for kernel
        hidden = hidden_states.contiguous().view(M * D)
        # Allocate outputs for LayerNorms
        y1 = torch.empty(M * D, dtype=torch.float32, device=hidden_states.device)
        y2 = torch.empty(M * D, dtype=torch.float32, device=hidden_states.device)

        # Launch LayerNorm 1
        grid = (M,)
        layernorm_fwd_kernel[grid](
            hidden, norm1_weight, norm1_bias, y1,
            M=M, D=D, eps=1e-5, BLOCK_SIZE=128,
            num_warps=4,
        )

        # Launch LayerNorm 2
        layernorm_fwd_kernel[grid](
            y1, norm2_weight, norm2_bias, y2,
            M=M, D=D, eps=1e-5, BLOCK_SIZE=128,
            num_warps=4,
        )

        # Reshape back to [B, S, D]
        output = y2.view(B, S, D)
        return output


def run(*args):
    return ModelNew()(*args)
