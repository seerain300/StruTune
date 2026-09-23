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
# Input: x_ptr [M, D] row-major, weight_ptr [D], bias_ptr [D]
# Output: out_ptr [M, D]
# Each program handles one row (normalized across D).
# Compute mean and variance in FP32, then normalize and apply affine (gamma/beta).
@triton.jit
def layernorm_fwd_kernel(
    x_ptr,            # *f32, input pointer to [M, D]
    w_ptr,            # *f32, gamma (weight), length D
    b_ptr,            # *f32, beta  (bias),   length D
    out_ptr,          # *f32, output pointer to [M, D]
    M,                # int32, number of rows
    D,                # int32, number of columns (normalized dimension)
    eps,              # float32, epsilon
    BLOCK_D: tl.constexpr,  # tile size along D (use 256 for D=256)
):
    row_id = tl.program_id(axis=0)  # 0..M-1
    if row_id >= M:
        return

    # Accumulate sum and sum of squares across D in FP32
    sum_x = 0.0
    sum_x2 = 0.0

    # Loop over D in tiles of BLOCK_D
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + row_id * D + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / D
    var = sum_x2 / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + row_id * D + offs, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * gamma + beta
        tl.store(out_ptr + row_id * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias):
        # Triton-only path: do not use any PyTorch tensor methods (no .mean, .sqrt, .to, .reshape, etc.)
        # Ensure Triton and CUDA device
        assert TRITON_AVAILABLE, "Triton is not available."
        assert hidden_states.is_cuda, "Input must be on CUDA device for Triton kernels."

        # Input shape: [B, S, D]
        B, S, D = hidden_states.shape
        M = B * S

        # Flatten to [M, D] for LN kernel (row-major)
        x = hidden_states.view(M, D).contiguous()
        # LN1: output [M, D]
        out1 = torch.empty((M, D), dtype=torch.float32, device=hidden_states.device)

        # Launch first LayerNorm kernel
        grid = (M,)
        layernorm_fwd_kernel[grid](
            x, norm1_weight, norm1_bias, out1,
            M, D, 1e-5,
            BLOCK_D=256,
            num_warps=4,
        )

        # LN2: output [M, D]
        out2 = torch.empty((M, D), dtype=torch.float32, device=hidden_states.device)
        grid = (M,)
        layernorm_fwd_kernel[grid](
            out1, norm2_weight, norm2_bias, out2,
            M, D, 1e-5,
            BLOCK_D=256,
            num_warps=4,
        )

        # Reshape back to [B, S, D]
        output = out2.view(B, S, D)
        return output


def run(*args):
    return ModelNew()(*args)
