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
# Each program handles a single row across features D.
# Compute mean and variance in FP32, then normalize and apply affine gamma/beta.
@triton.jit
def layernorm_fwd_kernel(
    x_ptr,       # *f32, input flattened over rows: length = M * D
    w_ptr,       # *f32, gamma (weight), length D
    b_ptr,       # *f32, beta  (bias),   length D
    out_ptr,     # *f32, output flattened over rows: length = M * D
    M,           # int32, number of rows
    D,           # int32, features per row (e.g., 256)
    eps,         # f32, epsilon
    BLOCK: tl.constexpr,  # tile size for feature dimension
):
    row = tl.program_id(axis=0)  # each program handles one row
    row_start = row * D

    # First pass: compute sum and sum of squares
    sum_x = 0.0
    sum_x2 = 0.0
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / D
    var = sum_x2 / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        gamma = tl.load(w_ptr + cols, mask=mask, other=1.0)
        beta = tl.load(b_ptr + cols, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        y = y * gamma + beta
        tl.store(out_ptr + row_start + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect: hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias
        assert len(args) == 5, "ModelNew.forward expects 5 inputs"
        hidden_states = args[0]  # [B, S, D], D=256
        norm1_weight = args[1]   # [D]
        norm1_bias = args[2]     # [D]
        norm2_weight = args[3]   # [D]
        norm2_bias = args[4]     # [D]

        B, S, D = hidden_states.shape
        assert D == 256, "Expected d_model=256"

        # Flatten to 1D view: total rows M = B * S * D
        M = B * S * D
        # Create a 1D view of hidden_states without using PyTorch tensor methods:
        # We cannot call .view here, but Triton will interpret the tensor based on row-major indexing.
        # For Triton kernels, we pass the tensor directly and compute offsets using row_start = row * D.
        # Output buffers (flat)
        y1_flat = torch.empty(M, dtype=torch.float32, device=hidden_states.device)
        y2_flat = torch.empty(M, dtype=torch.float32, device=hidden_states.device)

        # Choose BLOCK size; for D=256, 128 is fine
        BLOCK = 128
        eps = 1e-5

        # Launch LayerNorm 1
        grid_ln1 = (M,)
        layernorm_fwd_kernel[grid_ln1](
            hidden_states,          # input tensor; Triton will access via row_start
            norm1_weight,           # gamma
            norm1_bias,             # beta
            y1_flat,                # output
            M, D, eps, BLOCK
        )

        # Launch LayerNorm 2 on y1_flat (treated as [M, D] flattened)
        grid_ln2 = (M,)
        layernorm_fwd_kernel[grid_ln2](
            y1_flat,
            norm2_weight,
            norm2_bias,
            y2_flat,
            M, D, eps, BLOCK
        )

        # Reshape back to [B, S, D]
        output = y2_flat.view(B, S, D)
        return output


def run(*args):
    return ModelNew()(*args)
