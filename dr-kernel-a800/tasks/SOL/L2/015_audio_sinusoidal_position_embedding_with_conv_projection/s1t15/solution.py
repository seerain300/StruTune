import math
import torch
import triton
import triton.language as tl


# Triton conv2d kernel: 3x3, stride=2, padding=1
# Input: X(B, Cin, H, T) bfloat16
# Weight: W(Cout, Cin, 3, 3) bfloat16
# Bias: Bias(Cout) bfloat16
# Output: Y(B, Cout, H, T_out) bfloat16, with T_out = floor((T - 3)/2 + 1)
@triton.jit
def conv2d_3x3_stride2_padding1_kernel(
    X_ptr,         # *const bfloat16
    W_ptr,         # *const bfloat16
    BIAS_ptr,      # *const bfloat16
    Y_ptr,         # *bfloat16
    Bsz, Cin, H, T, Cout, T_out,
    BLOCK_C: tl.constexpr,
):
    # Grid: (B*H, tiles over Cout, T_out)
    pid_m = tl.program_id(0)   # over B*H
    pid_c = tl.program_id(1)   # over tiles of Cout
    t_out_idx = tl.program_id(2)  # specific time index in output

    b = pid_m // H
    oh = pid_m % H

    c_start = pid_c * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < Cout

    # accumulator per output channel
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    for cin in range(Cin):
        for kh in range(3):
            ih = oh + kh - 1  # padding=1: ih = oh + kh - 1
            valid_ih = (ih >= 0) & (ih < H)
            for kt in range(3):
                it = t_out_idx * 2 + kt - 1  # time index in input corresponding to t_out_idx (stride=2)
                valid_it = (it >= 0) & (it < T)

                if valid_it:
                    # Load input scalar x[b, cin, ih, it]
                    x_index = (
                        b * Cin * H * T
                        + cin * H * T
                        + ih * T
                        + it
                    )
                    x_val = tl.load(X_ptr + x_index, mask=valid_ih, other=0.0).to(tl.float32)

                    # Load weight vector for these output channels: w[c, cin, kh, kt]
                    w_index = (
                        c_offsets * (Cin * 3 * 3)
                        + cin * (3 * 3)
                        + kh * 3
                        + kt
                    )
                    w_vec = tl.load(W_ptr + w_index, mask=mask_c, other=0.0).to(tl.float32)

                    # FMA
                    acc += x_val * w_vec

    # Add bias
    bias_vec = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc += bias_vec

    # Store result to Y[b, c, oh, t_out_idx]
    y_index = (
        b * Cout * H * T_out
        + c_offsets * (H * T_out)
        + oh * T_out
        + t_out_idx
    )
    tl.store(Y_ptr + y_index, acc.to(tl.bfloat16), mask=mask_c)


# Triton GELU (exact) elementwise: in-place on X
# X: (B, Cout, H, T_out) bfloat16, Y: same
@triton.jit
def gelu_exact_kernel(X_ptr, Y_ptr, Bsz, Cout, H, T_out):
    # Grid: (B, Cout, H, T_out)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_t = tl.program_id(3)

    index = (
        pid_b * (Cout * H * T_out)
        + pid_c * (H * T_out)
        + pid_h * T_out
        + pid_t
    )
    x = tl.load(X_ptr + index).to(tl.float32)
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    tl.store(Y_ptr + index, y.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_features: torch.Tensor,        # (B, 1, 80, T), bfloat16
        conv2d1_weight: torch.Tensor,        # (384, 1, 3, 3), bfloat16
        conv2d1_bias: torch.Tensor,          # (384), bfloat16
        conv2d2_weight: torch.Tensor,        # (384, 384, 3, 3), bfloat16
        conv2d2_bias: torch.Tensor,          # (384), bfloat16
        conv2d3_weight: torch.Tensor,        # (384, 384, 3, 3), bfloat16
        conv2d3_bias: torch.Tensor,          # (384), bfloat16
        conv_out_weight: torch.Tensor,       # (1024, 15360), bfloat16
        positional_embedding: torch.Tensor,  # (1500, 1024), bfloat16
        embed_scale: float,                  # float
    ):
        # Shapes
        B, Cin, H, T = input_features.shape
        Cout1 = conv2d1_weight.shape[0]
        Cout2 = conv2d2_weight.shape[0]
        Cout3 = conv2d3_weight.shape[0]
        K = conv_out_weight.shape[1]  # 15360 (F*384)
        N = conv_out_weight.shape[0]  # 1024
        # Output time after each conv: stride=2, padding=1
        T_out = (T - 3) // 2 + 1

        # Stage 1: conv1
        x = input_features.contiguous()  # (B, 1, 80, T)
        y1 = torch.empty((B, Cout1, H, T_out), device=x.device, dtype=torch.bfloat16)
        grid1 = (B * H, triton.cdiv(Cout1, 64), T_out)
        conv2d_3x3_stride2_padding1_kernel[grid1](
            x, conv2d1_weight, conv2d1_bias, y1,
            B, Cin, H, T, Cout1, T_out,
            BLOCK_C=64,
        )
        # GELU after conv1 (in-kernel)
        y1_g = torch.empty_like(y1)
        gelu_exact_kernel[grid1](
            y1, y1_g, B, Cout1, H, T_out
        )

        # Stage 2: conv2
        x = y1_g  # (B, 384, 80, T_out)
        y2 = torch.empty((B, Cout2, H, T_out), device=x.device, dtype=torch.bfloat16)
        grid2 = (B * H, triton.cdiv(Cout2, 64), T_out)
        conv2d_3x3_stride2_padding1_kernel[grid2](
            x, conv2d2_weight, conv2d2_bias, y2,
            B, Cout1, H, T_out, Cout2, T_out,
            BLOCK_C=64,
        )
        # GELU after conv2
        y2_g = torch.empty_like(y2)
        gelu_exact_kernel[grid2](
            y2, y2_g, B, Cout2, H, T_out
        )

        # Stage 3: conv3
        x = y2_g  # (B, 384, 80, T_out)
        y3 = torch.empty((B, Cout3, H, T_out), device=x.device, dtype=torch.bfloat16)
        grid3 = (B * H, triton.cdiv(Cout3, 64), T_out)
        conv2d_3x3_stride2_padding1_kernel[grid3](
            x, conv2d3_weight, conv2d3_bias, y3,
            B, Cout2, H, T_out, Cout3, T_out,
            BLOCK_C=64,
        )
        # GELU after conv3
        y3_g = torch.empty_like(y3)
        gelu_exact_kernel[grid3](
            y3, y3_g, B, Cout3, H, T_out
        )

        # Now perform the rest (permutation and linear) using torch to keep Triton in use for convs
        # Permute: (B, C, H, T_out) -> (B, T_out, C*F), C=384, F=40
        C = 384
        F = 40
        x_perm = y3_g.permute(0, 3, 1, 2).contiguous().view(B, T_out, C * F)

        # Linear projection: (B, T_out, K) @ (K, N) -> (B, T_out, N)
        # A = x_perm, BT = conv_out_weight.T (K, N), output C_out
        A = x_perm.to(torch.float32)
        BT = conv_out_weight.permute(1, 0).contiguous().to(torch.float32)
        C_out = torch.matmul(A, BT)  # (B, T_out, N)

        # Scale embeddings
        C_out = C_out * float(embed_scale)

        # Add positional embedding: (1500, 1024), slice first T_out rows, broadcast over batch
        pos_embed = positional_embedding[:T_out].to(torch.float32)  # (T_out, N)
        C_out = C_out + pos_embed[None, :, :]

        # Cast back to bfloat16 for consistency
        return C_out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
