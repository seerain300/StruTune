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
# Operates on a flattened tensor of length N = M * D.
# Each program handles one row (pid = 0..M-1), normalizes across D, and applies affine.
@triton.jit
def layernorm_fwd_kernel(
    x_ptr,         # *f32, input pointer to [M*D] flattened
    w_ptr,         # *f32, gamma [D]
    b_ptr,         # *f32, beta  [D]
    out_ptr,       # *f32, output pointer to [M*D] flattened
    M,             # int32, number of rows
    D,             # int32, number of features per row
    eps,           # f32, epsilon for variance
    BLOCK: tl.constexpr,  # tile size over D
):
    pid = tl.program_id(axis=0)
    base = pid * D
    # Accumulate sum and sum of squares across D
    sum_val = 0.0
    sum_sq = 0.0
    for offs in range(0, D, BLOCK):
        cols = offs + tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / D
    var = sum_sq / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for offs in range(0, D, BLOCK):
        cols = offs + tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
        gamma = tl.load(w_ptr + cols, mask=mask, other=1.0)
        beta = tl.load(b_ptr + cols, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        y = y * gamma + beta
        tl.store(out_ptr + base + cols, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Forward expects:
        - hidden_states: [batch_size, seq_len, d_model] float32
        - norm1_weight: [d_model] float32
        - norm1_bias:   [d_model] float32
        - norm2_weight: [d_model] float32
        - norm2_bias:   [d_model] float32
        Returns: normalized tensor of shape [batch_size, seq_len, d_model]
        """
        # Extract dimensions
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        d_model = hidden_states.shape[2]

        # Compute number of rows M = B*S
        M = batch_size * seq_len

        # Allocate outputs [B, S, D] and get flat views for Triton
        y1 = torch.empty((batch_size, seq_len, d_model), dtype=hidden_states.dtype, device=hidden_states.device)
        y2 = torch.empty((batch_size, seq_len, d_model), dtype=hidden_states.dtype, device=hidden_states.device)

        in_flat = hidden_states.view(-1)  # length = M * d_model
        y1_flat = y1.view(-1)             # length = M * d_model
        y2_flat = y2.view(-1)             # length = M * d_model

        # Launch LayerNorm kernel for LN1
        grid = (M,)
        # Choose BLOCK=128, works for D=256; loops handle general D
        layernorm_fwd_kernel[grid](
            in_flat, norm1_weight, norm1_bias, y1_flat,
            M, d_model, 1e-5,
            BLOCK=128,
            num_warps=4,
        )

        # Launch LayerNorm kernel for LN2
        grid2 = (M,)
        layernorm_fwd_kernel[grid2](
            y1_flat, norm2_weight, norm2_bias, y2_flat,
            M, d_model, 1e-5,
            BLOCK=128,
            num_warps=4,
        )

        # Return y2 with shape [batch_size, seq_len, d_model]
        return y2


def run(*args):
    return ModelNew()(*args)
