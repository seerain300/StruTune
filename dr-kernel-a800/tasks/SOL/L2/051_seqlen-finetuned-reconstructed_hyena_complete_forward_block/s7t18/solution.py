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
# Each program handles one row of [M, D]. It performs:
# - compute mean and variance over D (unbiased=False)
# - normalize and apply affine (gamma, beta)
@triton.jit
def layernorm_fwd_kernel(
    in_ptr,      # *f32, input flattened to [M*D]
    w_ptr,       # *f32, gamma [D]
    b_ptr,       # *f32, beta  [D]
    out_ptr,     # *f32, output flattened [M*D]
    D: tl.constexpr,      # last-dim size
    eps,                  # epsilon (float)
    M,                    # number of rows
):
    row_id = tl.program_id(axis=0)
    # First pass: accumulate sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for i in range(0, D):
        x = tl.load(in_ptr + row_id * D + i)
        sum_val += x
        sum_sq += x * x
    mean = sum_val / D
    var = sum_sq / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    # Second pass: normalize and apply affine
    for i in range(0, D):
        x = tl.load(in_ptr + row_id * D + i)
        y = (x - mean) * inv_std
        gamma = tl.load(w_ptr + i)
        beta = tl.load(b_ptr + i)
        y = y * gamma + beta
        tl.store(out_ptr + row_id * D + i, y)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: hidden_states [B, S, D], norm1_weight [D], norm1_bias [D], norm2_weight [D], norm2_bias [D]
        hidden_states = args[0]
        norm1_weight = args[1]
        norm1_bias = args[2]
        norm2_weight = args[3]
        norm2_bias = args[4]

        B, S, D = hidden_states.shape
        M = B * S
        device = hidden_states.device

        # We will implement LayerNorm in Triton and return [B, S, D].
        # Reshape to [M, D] (PyTorch reshape is acceptable here; heavy compute remains in Triton).
        hs = hidden_states.reshape(M, D)

        # Allocate outputs
        y1 = torch.empty((M, D), dtype=torch.float32, device=device)
        y2 = torch.empty((M, D), dtype=torch.float32, device=device)

        # Launch first LayerNorm kernel
        eps = 1e-5
        grid1 = (M,)
        layernorm_fwd_kernel[grid1](
            hs, norm1_weight, norm1_bias, y1,
            D, eps, M,
            num_warps=4,
        )

        # Launch second LayerNorm kernel
        grid2 = (M,)
        layernorm_fwd_kernel[grid2](
            y1, norm2_weight, norm2_bias, y2,
            D, eps, M,
            num_warps=4,
        )

        # Return with original shape
        output = y2.view(B, S, D)
        return output


def run(*args):
    return ModelNew()(*args)
