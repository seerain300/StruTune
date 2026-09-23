import torch
import torch.nn as nn

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton LayerNorm forward kernel for 3D input [B, S, D], normalize along last dim (D).
# We view the input as [M, D] with M = B * S. Each program handles one row.
@triton.jit
def layernorm_fwd_kernel_3d(
    in_ptr,         # *f32, input pointer to tensor of shape [B, S, D] (we'll treat as [M, D])
    out_ptr,        # *f32, output pointer to tensor of shape [B, S, D]
    weight_ptr,     # *f32, gamma [D]
    bias_ptr,       # *f32, beta  [D]
    B,              # int, batch size
    S,              # int, seq_len
    D,              # int, d_model (last dim)
    eps,            # float, epsilon
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row of the [M, D] view
    row_id = tl.program_id(0)
    M = B * S
    # If row_id >= M, do nothing (grid will be M)
    # Compute sum and sum of squares across D
    sum_ = 0.0
    sum_sq = 0.0
    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        # row base offset in flattened [M, D] is row_id * D
        x = tl.load(in_ptr + row_id * D + offs, mask=mask, other=0.0)
        sum_ += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_ / D
    var = sum_sq / D - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(in_ptr + row_id * D + offs, mask=mask, other=0.0)
        gamma = tl.load(weight_ptr + offs, mask=mask, other=1.0)
        beta = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        y = (x - mean) * rstd
        y = y * gamma + beta
        tl.store(out_ptr + row_id * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, *args):
        """
        Expected args:
        0: hidden_states tensor of shape [batch_size, seq_len, d_model]
        1: norm1_weight [d_model]
        2: norm1_bias [d_model]
        3: norm2_weight [d_model]
        4: norm2_bias [d_model]
        """
        if len(args) < 5:
            raise RuntimeError("ModelNew.forward requires at least 5 arguments: hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias")

        hidden_states = args[0]
        norm1_weight = args[1]
        norm1_bias = args[2]
        norm2_weight = args[3]
        norm2_bias = args[4]

        # Ensure contiguous tensors; do not use .to(...) tensor methods
        hidden_states = hidden_states.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)
        d_model = hidden_states.size(2)

        # We will perform LayerNorm using Triton kernels. No host-side tensor ops.

        # First LayerNorm
        # Create output tensor y1
        y1 = torch.empty_like(hidden_states, dtype=torch.float32)
        # Grid: one program per row of [B, S, D] => M = B * S
        M = batch_size * seq_len
        # Choose BLOCK_SIZE as 256 (d_model=256). If D != 256, mask handles it, but here D=256.
        BLOCK_SIZE = 256
        grid = (M,)
        layernorm_fwd_kernel_3d[grid](
            hidden_states, y1, norm1_weight, norm1_bias,
            batch_size, seq_len, d_model, 1e-5,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        # Second LayerNorm
        y2 = torch.empty_like(hidden_states, dtype=torch.float32)
        M2 = batch_size * seq_len
        grid2 = (M2,)
        layernorm_fwd_kernel_3d[grid2](
            y1, y2, norm2_weight, norm2_bias,
            batch_size, seq_len, d_model, 1e-5,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        # Return final output (dtype float32 as original uses float32)
        return y2


def run(*args):
    return ModelNew()(*args)
