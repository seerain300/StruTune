import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel: GELU (tanh approximation), elementwise over N elements
@triton.jit
def gelu_tanh_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(Y_ptr + offsets, gelu, mask=mask)


# Triton kernel: elementwise addition of scaled positional embedding
# X: (B, T, D) — pointers
# POS: (T, D) scaled embeddings — pointers
# Y: (B, T, D) — pointers
@triton.jit
def add_scaled_pos_emb_3d_kernel(X_ptr, POS_ptr, Y_ptr,
                                 B, T, D,
                                 stride_xb, stride_xt, stride_xd,
                                 stride_posb, stride_post, stride_posd,
                                 BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_offsets < D

    # Load x[b, t, d]
    x_vals = tl.load(X_ptr + pid_b * stride_xb + pid_t * stride_xt + d_offsets * stride_xd, mask=mask, other=0.0).to(tl.float32)
    # Load pos[t, d]
    pos_vals = tl.load(POS_ptr + pid_t * stride_post + d_offsets * stride_posd, mask=mask, other=0.0).to(tl.float32)
    y_vals = x_vals + pos_vals

    # Store to Y[b, t, d]
    tl.store(Y_ptr + pid_b * stride_xb + pid_t * stride_xt + d_offsets * stride_xd, y_vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features: torch.Tensor,
                conv2d1_weight: torch.Tensor, conv2d1_bias: torch.Tensor,
                conv2d2_weight: torch.Tensor, conv2d2_bias: torch.Tensor,
                conv2d3_weight: torch.Tensor, conv2d3_bias: torch.Tensor,
                conv_out_weight: torch.Tensor,
                positional_embedding: torch.Tensor,
                embed_scale: float):
        """
        input_features: (B, 1, 80, T), bfloat16
        conv weights/bias: (C_out, C_in, 3, 3), bfloat16
        conv_out_weight: (1024, 3840), bfloat16
        positional_embedding: (1500, 1024), bfloat16
        embed_scale: float
        """
        B, C_in1, H1, W1 = input_features.shape

        # Stage 1: Conv1 (1 -> 384) + Triton GELU
        x1 = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        N1 = x1.numel()
        x1_gelu = torch.empty_like(x1)
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](x1, x1_gelu, N1, BLOCK=1024)

        # Stage 2: Conv2 (384 -> 384) + Triton GELU
        x2 = F.conv2d(x1_gelu, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        N2 = x2.numel()
        x2_gelu = torch.empty_like(x2)
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](x2, x2_gelu, N2, BLOCK=1024)

        # Stage 3: Conv3 (384 -> 384) + Triton GELU
        x3 = F.conv2d(x2_gelu, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        N3 = x3.numel()
        x3_gelu = torch.empty_like(x3)
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](x3, x3_gelu, N3, BLOCK=1024)

        # Reshape to (B, time_after_conv, 1024)
        # time_after_conv = W1 // 8 (three stride-2 convs each halving time dimension)
        time_after_conv = W1 // 8
        # x3_gelu shape: (B, 384, 10, time_after_conv)
        x4 = x3_gelu.permute(0, 3, 1, 2).contiguous().view(B, time_after_conv, 384 * 10)

        # Linear projection: (B, T, 3840) @ (1024, 3840)^T -> (B, T, 1024)
        # conv_out_weight: (1024, 3840)
        # x4: (B, T, 3840)
        x5 = F.linear(x4, conv_out_weight)  # bias=False

        # Scale by embed_scale and add positional embedding using Triton
        d_model = 1024
        scaled_pos = positional_embedding[:time_after_conv, :].to(x5.dtype) * embed_scale
        y_final = torch.empty_like(x5)

        # Launch Triton 3D kernel over (B, T, d_model tiles)
        BLOCK_D = 1024
        grid = (B, time_after_conv, triton.cdiv(d_model, BLOCK_D))
        # For Triton, we pass strides (in elements). Since tensors are contiguous:
        stride_xb, stride_xt, stride_xd = x5.stride()
        # POS is (T, D), contiguous: stride_posb = D, stride_post = 1, stride_posd = 1
        # But we need actual values; safest is to infer from tensor: for contiguous 2D:
        pos_strides = scaled_pos.stride()
        stride_posb = pos_strides[0]  # typically D
        stride_post = pos_strides[1]  # typically 1
        stride_posd = 1  # scaled_pos is contiguous along D

        add_scaled_pos_emb_3d_kernel[grid](
            x5, scaled_pos, y_final,
            B, time_after_conv, d_model,
            stride_xb, stride_xt, stride_xd,
            stride_posb, stride_post, stride_posd,
            BLOCK_D=BLOCK_D,
        )

        return y_final


def run(*args):
    return ModelNew()(*args)
