import torch
import torch.nn as nn

# Triton imports and availability flag
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton LayerNorm forward kernel over 2D input [M, D]:
# Each program handles one row (length D). Computes mean/var across D, normalizes, applies affine gamma/beta.
@triton.jit
def layernorm_fwd_kernel_2d(
    x_ptr,          # *f32, input pointer to [M, D]
    w_ptr,          # *f32, gamma (weight) of length D
    b_ptr,          # *f32, beta  (bias)   of length D
    y_ptr,          # *f32, output pointer to [M, D]
    M,              # int32, number of rows
    D,              # int32, number of columns (d_model)
    eps,            # f32, epsilon for numerical stability
):
    row = tl.program_id(0)  # one program per row
    # Base pointers for this row
    x_row = x_ptr + row * D
    y_row = y_ptr + row * D
    # First pass: sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, D):
        mask_off = off < D  # redundant but keeps intent explicit
        x_val = tl.load(x_row + off, mask=mask_off, other=0.0)
        sum_val += x_val
        sum_sq += x_val * x_val
    mean = sum_val / D
    var = sum_sq / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    # Second pass: normalize + affine and store
    for off in range(0, D):
        x_val = tl.load(x_row + off, mask=off < D, other=0.0)
        gamma = tl.load(w_ptr + off)
        beta = tl.load(b_ptr + off)
        y_val = (x_val - mean) * inv_std
        y_val = y_val * gamma + beta
        tl.store(y_row + off, y_val, mask=off < D)


class ModelNew(nn.Module):
    def forward(self, hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias):
        # hidden_states: [B, S, D]
        B, S, D = hidden_states.shape
        M = B * S

        # Triton requires CUDA; guard availability
        if not TRITON_AVAILABLE or not hidden_states.is_cuda:
            # Minimal fallback: use PyTorch functional layer_norm for correctness if Triton/CUDA not available
            y1 = nn.functional.layer_norm(hidden_states, normalized_shape=(D,), weight=norm1_weight, bias=norm1_bias, eps=1e-5)
            y2 = nn.functional.layer_norm(y1, normalized_shape=(D,), weight=norm2_weight, bias=norm2_bias, eps=1e-5)
            return y2

        # We will operate directly on 2D [M, D] pointers; no host-side tensor methods.
        # Allocate output buffers as 2D [M, D]
        y1 = torch.empty((M, D), dtype=torch.float32, device=hidden_states.device)
        y2 = torch.empty((M, D), dtype=torch.float32, device=hidden_states.device)

        # First LayerNorm: input = hidden_states, gamma=norm1_weight, beta=norm1_bias
        grid = (M,)
        layernorm_fwd_kernel_2d[grid](
            hidden_states,             # x_ptr
            norm1_weight,              # gamma
            norm1_bias,                # beta
            y1,                        # out
            M,                         # rows
            D,                         # cols
            1e-5,                      # eps
            num_warps=4,
        )

        # Second LayerNorm: input = y1, gamma=norm2_weight, beta=norm2_bias
        layernorm_fwd_kernel_2d[grid](
            y1,                        # x_ptr
            norm2_weight,              # gamma
            norm2_bias,                # beta
            y2,                        # out
            M,                         # rows
            D,                         # cols
            1e-5,                      # eps
            num_warps=4,
        )

        # Return y2 with shape [B, S, D]
        return y2


def run(*args):
    return ModelNew()(*args)
