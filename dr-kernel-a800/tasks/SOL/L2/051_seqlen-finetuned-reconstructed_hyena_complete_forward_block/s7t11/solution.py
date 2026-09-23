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
# Each program handles one row of length D.
# Input:  x_ptr (1D length = M*D), weight_ptr [D], bias_ptr [D]
# Output: out_ptr (1D length = M*D)
# Compute mean and variance across D in FP32, then normalize and apply affine.
@triton.jit
def layernorm_fwd_kernel(
    x_ptr,          # *f32, input flattened row-major
    weight_ptr,     # *f32, gamma [D]
    bias_ptr,       # *f32, beta  [D]
    out_ptr,        # *f32, output flattened
    D: tl.constexpr,        # features per row (e.g., 256)
    eps: tl.constexpr,      # epsilon for numerical stability
    BLOCK: tl.constexpr     # tile size along D
):
    row_id = tl.program_id(axis=0)  # one program per row
    row_offset = row_id * D

    # First pass: compute sum and sum of squares across D
    sum_val = 0.0
    sum_sq = 0.0
    for d in range(0, D, BLOCK):
        offs = d + tl.arange(0, BLOCK)
        mask = offs < D
        x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / D
    var = sum_sq / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for d in range(0, D, BLOCK):
        offs = d + tl.arange(0, BLOCK)
        mask = offs < D
        x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        gamma = tl.load(weight_ptr + offs, mask=mask, other=1.0)
        beta = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        y = y * gamma + beta
        tl.store(out_ptr + row_offset + offs, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, *args):
        # Expect: hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias
        assert len(args) >= 5, "Expected at least 5 inputs: hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias"

        hidden_states = args[0]
        norm1_weight = args[1]
        norm1_bias = args[2]
        norm2_weight = args[3]
        norm2_bias = args[4]

        # We operate with D = 256 as per the original setup.
        D = 256
        M = hidden_states.numel() // D
        x_flat = hidden_states.view(M * D)

        # Allocate outputs for LayerNorms (flattened)
        y1_flat = torch.empty(M * D, dtype=torch.float32, device=hidden_states.device)
        y2_flat = torch.empty(M * D, dtype=torch.float32, device=hidden_states.device)

        # Launch LayerNorm 1 kernel
        eps = 1e-5
        grid1 = (M,)
        layernorm_fwd_kernel[grid1](
            x_flat, norm1_weight, norm1_bias, y1_flat,
            D=D, eps=eps, BLOCK=128, num_warps=4
        )

        # Launch LayerNorm 2 kernel
        grid2 = (M,)
        layernorm_fwd_kernel[grid2](
            y1_flat, norm2_weight, norm2_bias, y2_flat,
            D=D, eps=eps, BLOCK=128, num_warps=4
        )

        # Reshape back to [batch_size, seq_len, d_model]
        # We need B and S; get_inputs provides them as additional args
        B = args[5] if len(args) > 5 else hidden_states.shape[0]
        S = args[6] if len(args) > 6 else hidden_states.shape[1]
        assert B * S == M, "Mismatch: batch_size * seq_len != M"
        output = y2_flat.view(B, S, D)

        return output


def run(*args):
    return ModelNew()(*args)
