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
# Each program handles one row (over D features). It computes mean/variance and normalizes.
@triton.jit
def layernorm_fwd_kernel(
    x_ptr,          # *f32, input pointer to flattened [M*D]
    w_ptr,          # *f32, gamma (weight) of length D
    b_ptr,          # *f32, beta  (bias)   of length D
    out_ptr,        # *f32, output pointer to flattened [M*D]
    M,              # int32, number of rows (unused but kept for signature symmetry)
    D,              # int32, features per row
    eps,            # f32, epsilon for var + eps
    BLOCK: tl.constexpr,  # tile size along D (e.g., 128 or 256)
):
    row = tl.program_id(axis=0)  # one program per row
    # First pass: compute sum and sum of squares (for mean and variance)
    sum_ = 0.0
    sumsq_ = 0.0
    offs = 0
    while offs < D:
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < D
        x = tl.load(x_ptr + row * D + idx, mask=mask, other=0.0)
        # reduce within the tile
        sum_ += tl.sum(x, axis=0)
        sumsq_ += tl.sum(x * x, axis=0)
        offs += BLOCK
    mean = sum_ / D
    var = sumsq_ / D - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    offs = 0
    while offs < D:
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < D
        x = tl.load(x_ptr + row * D + idx, mask=mask, other=0.0)
        gamma = tl.load(w_ptr + idx, mask=mask, other=1.0)
        beta = tl.load(b_ptr + idx, mask=mask, other=0.0)
        y = (x - mean) * rstd
        y = y * gamma + beta
        tl.store(out_ptr + row * D + idx, y, mask=mask)
        offs += BLOCK


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias):
        # hidden_states: [B, S, D], norm1_weight/bias: [D], norm2_weight/bias: [D]
        # We avoid any PyTorch tensor methods for computation. Only necessary layout handling.
        B = hidden_states.shape[0]
        S = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert D == 256, "d_model must be 256 as per get_inputs"
        M = B * S

        # Create row-major [M, D] view without PyTorch tensor methods
        # We rely on the incoming tensor being contiguous in the last dim; .view(M, D) is safe if contiguous.
        # The provided get_inputs constructs hidden_states as contiguous by default.
        x = hidden_states.view(M, D)

        # Allocate outputs (row-major [M, D])
        y1 = torch.empty((M, D), dtype=torch.float32, device=hidden_states.device)
        y2 = torch.empty((M, D), dtype=torch.float32, device=hidden_states.device)

        # Launch LayerNorm kernel twice
        grid = (M,)
        layernorm_fwd_kernel[grid](
            x.view(-1),                 # x_ptr: 1D of length M*D
            norm1_weight,               # gamma length D
            norm1_bias,                 # beta   length D
            y1.view(-1),                # out1
            M, D, 1e-5,                 # eps
            BLOCK=256,                  # tile over D
            num_warps=4,
        )

        layernorm_fwd_kernel[grid](
            y1.view(-1),                # x_ptr: 1D of length M*D
            norm2_weight,               # gamma length D
            norm2_bias,                 # beta   length D
            y2.view(-1),                # out2
            M, D, 1e-5,                 # eps
            BLOCK=256,                  # tile over D
            num_warps=4,
        )

        # Reshape back to [B, S, D]
        output = y2.view(B, S, D)
        return output


def run(*args):
    return ModelNew()(*args)
