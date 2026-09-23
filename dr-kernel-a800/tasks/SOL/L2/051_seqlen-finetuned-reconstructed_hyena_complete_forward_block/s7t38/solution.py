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
# Normalize each row of length D of a linearized tensor of length M*D.
# We pass D as tl.constexpr to let Triton compile-time unroll the loop.
@triton.jit
def layernorm_fwd_kernel(
    x_ptr,          # *f32, pointer to input linearized [M*D]
    w_ptr,          # *f32, gamma (weight), length D
    b_ptr,          # *f32, beta  (bias),   length D
    y_ptr,          # *f32, pointer to output linearized [M*D]
    M,              # int32, number of rows
    D: tl.constexpr,# int32 constexpr, row length
    eps: tl.constexpr,  # float32 constexpr, epsilon for numerical stability
    x_stride_row,   # int32, stride for row in input (usually D)
    y_stride_row    # int32, stride for row in output (usually D)
):
    row = tl.program_id(0)  # each program handles one row
    # First pass: compute sum and sum of squares across D
    sum_val = 0.0
    sum_sq = 0.0
    for i in range(D):
        xi = tl.load(x_ptr + row * x_stride_row + i)
        sum_val += xi
        sum_sq += xi * xi
    mean = sum_val / D
    var = sum_sq / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, store to output
    for i in range(D):
        xi = tl.load(x_ptr + row * x_stride_row + i)
        norm = (xi - mean) * inv_std
        gamma = tl.load(w_ptr + i)
        beta = tl.load(b_ptr + i)
        yi = norm * gamma + beta
        tl.store(y_ptr + row * y_stride_row + i, yi)


def _run_layernorm(x, weight, bias, eps=1e-5):
    """
    Helper to run the Triton LayerNorm forward kernel. x is [B, S, D], weight/bias are [D].
    Returns y of same shape as x.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    # Shapes
    B, S, D = x.shape
    M = B * S
    # Flatten views (no tensor method usage)
    x_flat = x
    # We will treat x_flat as linearized [M, D] with row stride = D
    # Allocate output
    y = torch.empty_like(x)
    x_ptr = x_flat
    y_ptr = y
    # Prepare pointers for weight/bias (contiguous)
    w_ptr = weight
    b_ptr = bias
    # Launch kernel: grid over rows
    grid = (M,)
    # eps as constexpr float
    layernorm_fwd_kernel[grid](
        x_ptr, w_ptr, b_ptr, y_ptr,
        M, D, eps,
        D, D,  # x_stride_row = D, y_stride_row = D
        num_warps=4,
    )
    return y


class ModelNew(nn.Module):
    def forward(self, hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias):
        """
        hidden_states: [B, S, D] float32 tensor
        norm1_weight, norm1_bias: [D] float32 tensors
        norm2_weight, norm2_bias: [D] float32 tensors
        Returns: [B, S, D] tensor after applying LayerNorm with norm1, then LayerNorm with norm2.
        """
        # Ensure Triton/CUDA
        if not TRITON_AVAILABLE or not hidden_states.is_cuda:
            # Fallback to PyTorch LayerNorm to maintain correctness if Triton/CUDA not available
            y1 = torch.nn.functional.layer_norm(hidden_states, (hidden_states.shape[-1],), norm1_weight, norm1_bias, 1e-5)
            y2 = torch.nn.functional.layer_norm(y1, (y1.shape[-1],), norm2_weight, norm2_bias, 1e-5)
            return y2

        B, S, D = hidden_states.shape
        # First LayerNorm using Triton
        y1 = _run_layernorm(hidden_states, norm1_weight, norm1_bias, eps=1e-5)
        # Second LayerNorm using Triton
        y2 = _run_layernorm(y1, norm2_weight, norm2_bias, eps=1e-5)
        return y2


def run(*args):
    return ModelNew()(*args)
