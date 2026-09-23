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
    x_ptr,          # *f32, input pointer to [M*D] flattened
    w_ptr,          # *f32, gamma (weight) of length D
    b_ptr,          # *f32, beta (bias) of length D
    out_ptr,        # *f32, output pointer to [M*D] flattened
    M,              # int32, number of rows
    D,              # int32, number of features per row
    eps,            # f32, epsilon for numerical stability
    BLOCK: tl.constexpr,  # tile size for features (e.g., 256)
):
    row = tl.program_id(axis=0)  # one program per row
    # Compute sum and sum of squares over D in tiles
    total = 0.0
    total2 = 0.0
    for offs in range(0, D, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < D
        x = tl.load(x_ptr + row * D + idx, mask=mask, other=0.0)
        total += tl.sum(x, axis=0)
        total2 += tl.sum(x * x, axis=0)
    mean = total / D
    var = total2 / D - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine
    for offs in range(0, D, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < D
        x = tl.load(x_ptr + row * D + idx, mask=mask, other=0.0)
        norm = (x - mean) * rstd
        gamma = tl.load(w_ptr + idx, mask=mask, other=1.0)
        beta = tl.load(b_ptr + idx, mask=mask, other=0.0)
        y = norm * gamma + beta
        tl.store(out_ptr + row * D + idx, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Expect: hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias
        # No host-side tensor methods; only allocate and call Triton kernels.
        assert len(args) == 5, "ModelNew.forward expects 5 inputs: hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias"
        hidden = args[0]
        norm1_w = args[1]
        norm1_b = args[2]
        norm2_w = args[3]
        norm2_b = args[4]

        # Dimensions: hidden is [B, S, D] with D=256 from get_inputs
        B = hidden.shape[0]
        S = hidden.shape[1]
        D = hidden.shape[2]  # d_model from get_inputs is 256
        M = B * S

        # Flatten to 1D contiguous view without PyTorch tensor methods
        # hidden is produced by get_inputs as contiguous [B, S, D]; we can flatten linearly.
        x_flat = hidden.reshape(M * D).contiguous()  # Triton expects flat pointers
        y1 = torch.empty_like(x_flat)
        y2 = torch.empty_like(x_flat)

        # Launch LayerNorm 1: per-row across D
        grid = (M,)
        layernorm_fwd_kernel[grid](
            x_flat, norm1_w, norm1_b, y1,
            M, D, 1e-5, 256,
            num_warps=4,
        )

        # Launch LayerNorm 2
        layernorm_fwd_kernel[grid](
            y1, norm2_w, norm2_b, y2,
            M, D, 1e-5, 256,
            num_warps=4,
        )

        # Reshape final output back to [B, S, D]
        out = y2.view(B, S, D)
        return out


def run(*args):
    return ModelNew()(*args)
