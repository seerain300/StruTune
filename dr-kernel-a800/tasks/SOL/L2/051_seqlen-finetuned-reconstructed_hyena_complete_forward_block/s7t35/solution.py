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
# We normalize each row (length D) of an input tensor of shape [M, D] (M = batch_size * seq_len).
# We receive x as a 1D pointer of length M*D, but we use strides to access rows:
#   x_row_base = x_ptr + row * stride_row, and load across D with mask offsets < D.
# Each program handles one row (row = program_id(0)).
@triton.jit
def layernorm_fwd_kernel(
    x_ptr,          # *f32, linearized input pointer to [M*D]
    w_ptr,          # *f32, gamma (weight), length D
    b_ptr,          # *f32, beta  (bias),   length D
    y_ptr,          # *f32, output pointer to [M*D]
    M,              # int32, number of rows
    D,              # int32, row length (i.e., hidden size)
    eps,            # f32, epsilon for numerical stability
    stride_x_row,   # int32, stride between rows in x (elements), equals D for [B,S,D] flattened
    stride_y_row,   # int32, stride between rows in y (elements), equals D for [B,S,D] flattened
    BLOCK_SIZE: tl.constexpr,  # tile size along D
):
    row = tl.program_id(0)
    if row >= M:
        return

    # First pass: compute mean and variance over D
    sum_val = 0.0
    sum_sq = 0.0
    offs = 0
    while offs < D:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_row_base = x_ptr + row * stride_x_row
        x = tl.load(x_row_base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        offs += BLOCK_SIZE

    mean = sum_val / D
    var = sum_sq / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    offs = 0
    while offs < D:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_row_base = x_ptr + row * stride_x_row
        y_row_base = y_ptr + row * stride_y_row
        x = tl.load(x_row_base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        norm = (x - mean) * inv_std
        gamma = tl.load(w_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(b_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = norm * gamma + beta
        tl.store(y_row_base + idx, y, mask=mask)
        offs += BLOCK_SIZE


def _launch_layernorm(x, weight, bias, eps=1e-5):
    """
    x: [B, S, D] tensor, dtype float32, CUDA
    weight: [D] float32, CUDA
    bias: [D] float32, CUDA
    Returns y: [B, S, D] normalized tensor, allocated as empty_like(x)
    """
    assert x.is_cuda, "Input must be CUDA tensor for Triton."
    assert x.dtype == torch.float32, "x must be float32."
    B, S, D = x.shape
    M = B * S
    # Flatten to [M*D] linear; kernel will use strides to treat as [M, D]
    x_flat = x.view(M * D)
    # Preallocate output with same shape and flatten
    y_out = torch.empty_like(x)
    y_flat = y_out.view(M * D)
    # Grid: one program per row
    grid = (M,)
    BLOCK_SIZE = 256  # D=256 in the original problem; mask handles other sizes
    num_warps = 4
    # Strides in elements: for a [B,S,D] tensor flattened to [M*D], row stride is D
    stride_x_row = D
    stride_y_row = D
    layernorm_fwd_kernel[grid](
        x_flat, weight, bias, y_flat,
        M, D, eps, stride_x_row, stride_y_row,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y_out


class ModelNew(nn.Module):
    def forward(self, hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias):
        """
        hidden_states: tensor of shape [batch_size, seq_len, d_model], dtype float32, device CUDA
        norm1_weight, norm1_bias: [d_model], float32, CUDA
        norm2_weight, norm2_bias: [d_model], float32, CUDA
        Returns: tensor of shape [batch_size, seq_len, d_model], float32
        """
        # Ensure inputs are CUDA and float32; do not use PyTorch tensor methods for compute.
        # First LayerNorm: write into y1_out
        y1_out = _launch_layernorm(hidden_states, norm1_weight, norm1_bias, eps=1e-5)

        # Second LayerNorm: write into final output
        y2_out = _launch_layernorm(y1_out, norm2_weight, norm2_bias, eps=1e-5)

        return y2_out


def run(*args):
    return ModelNew()(*args)
