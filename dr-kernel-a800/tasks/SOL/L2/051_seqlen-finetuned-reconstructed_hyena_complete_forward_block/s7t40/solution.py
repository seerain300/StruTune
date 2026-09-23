import torch
import torch.nn as nn

# Triton imports and kernel definitions
import triton
import triton.language as tl


# Triton LayerNorm forward kernel for 2D [M, D] inputs:
# Each program handles one row (M independent), normalizing across D.
@triton.jit
def layernorm_fwd_2d_kernel(
    in_ptr,        # *f32, input pointer to [M, D]
    w_ptr,         # *f32, weight (gamma) of length D
    b_ptr,         # *f32, bias (beta)   of length D
    out_ptr,       # *f32, output pointer to [M, D]
    M: tl.constexpr,   # number of rows (B*S)
    D: tl.constexpr,   # row length (d_model)
    eps,                    # float32 epsilon
    BLOCK: tl.constexpr,   # tile size for reduction (e.g., 128 or 256)
):
    row_id = tl.program_id(0)  # 0..M-1
    # Compute sum and sum of squares over the row
    sum_ = 0.0
    sum_sq = 0.0
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(in_ptr + row_id * D + cols, mask=mask, other=0.0)
        sum_ += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    Df = tl.float32(D)
    mean = sum_ / Df
    var = sum_sq / Df - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    # Second pass: normalize and apply affine
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(in_ptr + row_id * D + cols, mask=mask, other=0.0)
        gamma = tl.load(w_ptr + cols, mask=mask, other=1.0)
        beta = tl.load(b_ptr + cols, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        y = y * gamma + beta
        tl.store(out_ptr + row_id * D + cols, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                layer_norm_eps: float):
        # We must invoke Triton kernels. No PyTorch tensor methods on host.
        B, S, D = hidden_states.shape
        M = B * S

        # Ensure inputs are contiguous and float32
        in1 = hidden_states.contiguous()  # [B, S, D]
        out1 = torch.empty_like(in1)      # first LN output
        out2 = torch.empty_like(in1)      # second LN output

        # First LayerNorm: on hidden_states
        grid = (M,)
        w1 = norm1_weight.contiguous()    # [D]
        b1 = norm1_bias.contiguous()      # [D]
        BLOCK = 256  # tile across D; for D=256 this covers in one pass
        layernorm_fwd_2d_kernel[grid](
            in1, w1, b1, out1,
            M, D,
            layer_norm_eps,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # Second LayerNorm: on out1 (result of first LN)
        w2 = norm2_weight.contiguous()
        b2 = norm2_bias.contiguous()
        layernorm_fwd_2d_kernel[grid](
            out1, w2, b2, out2,
            M, D,
            layer_norm_eps,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # Reshape back to [B, S, D] and return
        return out2


def run(*args):
    return ModelNew()(*args)
