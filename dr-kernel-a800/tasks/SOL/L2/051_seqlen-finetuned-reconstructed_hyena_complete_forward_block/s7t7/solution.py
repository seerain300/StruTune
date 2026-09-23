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
# Operates on a 2D logical view [M, D] where each program handles one row.
# Computes mean and variance across D (last dim), then normalizes and applies affine (gamma, beta).
@triton.jit
def layernorm_fwd_kernel(
    x_ptr,            # *f32, input pointer to [M*D] contiguous
    weight_ptr,       # *f32, gamma (size D)
    bias_ptr,         # *f32, beta  (size D)
    out_ptr,          # *f32, output pointer to [M*D] contiguous
    M,                # int32, number of rows
    D,                # int32, normalized dimension
    eps,              # f32, epsilon
    BLOCK_D: tl.constexpr,  # tile size along D
):
    row_id = tl.program_id(axis=0)
    if row_id >= M:
        return

    # First pass: compute sum and sum of squares over D
    sum_x = 0.0
    sum_x2 = 0.0
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + row_id * D + offs, mask=mask, other=0.0)
        # accumulate in fp32
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    D_f = tl.float32(D)
    mean = sum_x / D_f
    var = sum_x2 / D_f - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + row_id * D + offs, mask=mask, other=0.0)
        gamma = tl.load(weight_ptr + offs, mask=mask, other=1.0)
        beta = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        y = y * gamma + beta
        tl.store(out_ptr + row_id * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Triton-only forward:
        - hidden_states: [batch_size, seq_len, d_model] (float32 on CUDA)
        - norm1_weight, norm1_bias: [d_model], float32 on CUDA
        - norm2_weight, norm2_bias: [d_model], float32 on CUDA
        Returns: output [batch_size, seq_len, d_model], computed via Triton LayerNorm kernels.
        """
        # Ensure tensors are on CUDA and dtype float32 for Triton
        device = hidden_states.device
        if hidden_states.dtype != torch.float32 or not hidden_states.is_cuda:
            hidden_states = hidden_states.to(torch.float32).contiguous()

        # Dimensions
        B, S, D = hidden_states.shape
        M = B * S

        # Flatten to [M, D] contiguous
        x2d = hidden_states.reshape(M, D).contiguous()

        # Allocate outputs
        y1 = torch.empty((M, D), dtype=torch.float32, device=device)
        y2 = torch.empty((M, D), dtype=torch.float32, device=device)

        # Kernel launch parameters
        BLOCK_D = 256  # matches d_model=256; general case works with masked loops
        num_warps = 4

        # First LayerNorm: y1 = LN1(x2d)
        layernorm_fwd_kernel[(M,)](
            x2d, norm1_weight, norm1_bias, y1, M, D, 1e-5, BLOCK_D=BLOCK_D, num_warps=num_warps
        )

        # Second LayerNorm: y2 = LN2(y1)
        layernorm_fwd_kernel[(M,)](
            y1, norm2_weight, norm2_bias, y2, M, D, 1e-5, BLOCK_D=BLOCK_D, num_warps=num_warps
        )

        # Reshape back to [B, S, D]
        output = y2.reshape(B, S, D)
        return output


def run(*args):
    return ModelNew()(*args)
