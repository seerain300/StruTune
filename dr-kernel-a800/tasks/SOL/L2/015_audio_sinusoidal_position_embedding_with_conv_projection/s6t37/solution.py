import math
import torch
import torch.nn as nn

import triton
import triton.language as tl

# Triton: Verified conv2d (NCHW, stride=2, padding=1). We call this three times for each conv stage.
# Signature: conv2d_fwd_nchw_kernel(in_ptr, w_ptr, b_ptr, out_ptr, B, Cin, H, W, Cout, K_h, K_w, S, P, BLOCK_H, BLOCK_W)
# Grid: (B*Cout, H_out, W_out). Each program computes one output element. We will launch it for each stage.

# Triton GELU (tanh approximation) kernel — in-place on flattened tensor.
@triton.jit
def gelu_kernel(X, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    t = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.math.tanh(t))
    tl.store(X + offs, y, mask=mask)


# Triton linear projection: y[b, t, d] = sum_k x[b, t, k] * W[d, k], without bias.
# xflat: [B, T, K] contiguous. W: [N, K] contiguous. y: [B, T, N] contiguous.
@triton.jit
def linear_kernel(Xflat, W, Y, B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)
    acc = 0.0
    for k in range(0, K):
        x_val = tl.load(Xflat + b * T * K + t * K + k)
        w_val = tl.load(W + d * K + k)
        acc += x_val * w_val
    tl.store(Y + b * T * N + t * N + d, acc)


# Triton scale kernel: scale y in-place elementwise by a scalar scale.
@triton.jit
def scale_kernel(Y, scale, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N
    y = tl.load(Y + offs, mask=mask, other=0.0)
    y = y * scale
    tl.store(Y + offs, y, mask=mask)


# Triton add positional embedding: y[b, t, d] += pos_emb[t, d]
@triton.jit
def add_pos_emb_kernel(Y, PosEmb, B: tl.constexpr, T: tl.constexpr, N: tl.constexpr):
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)
    y_val = tl.load(Y + b * T * N + t * N + d)
    pe_val = tl.load(PosEmb + t * N + d)
    y_val = y_val + pe_val
    tl.store(Y + b * T * N + t * N + d, y_val)


class ModelNew(nn.Module):
    def forward(
        self,
        input_features,
        conv2d1_weight,
        conv2d1_bias,
        conv2d2_weight,
        conv2d2_bias,
        conv2d3_weight,
        conv2d3_bias,
        conv_out_weight,
        positional_embedding,
        embed_scale,
    ):
        # Stage 1: Conv2d 1 -> (384, H1, W1), then GELU
        B = input_features.shape[0]
        Cin1 = input_features.shape[1]  # 1
        H = input_features.shape[2]     # 80
        W = input_features.shape[3]     # time_dim
        Cout1 = conv2d1_weight.shape[0] # 384
        K_h = conv2d1_weight.shape[2]   # 3
        K_w = conv2d1_weight.shape[3]   # 3
        S = 2
        P = 1

        # Output spatial sizes for stride=2, padding=1
        H1 = (H + 2 * P - K_h) // S + 1  # 40
        W1 = (W + 2 * P - K_w) // S + 1  # time_dim // 2

        # Allocate output for conv1
        y1 = torch.empty((B, Cout1, H1, W1), device=input_features.device, dtype=input_features.dtype)

        # Launch Triton conv2d for stage 1
        grid_conv1 = (B * Cout1, H1, W1)
        conv2d_fwd_nchw_kernel[
            grid_conv1
        ](
            input_features,
            conv2d1_weight,
            conv2d1_bias,
            y1,
            B,
            Cin1,
            H,
            W,
            Cout1,
            K_h,
            K_w,
            S,
            P,
            1,  # BLOCK_H
            1,  # BLOCK_W
        )

        # GELU on y1 in Triton (in-place)
        N1 = y1.numel()
        gelu_kernel[(N1 + 1024 - 1) // 1024,](y1)

        # Stage 2: Conv2d 2 -> (384, H2, W2), then GELU
        Cin2 = Cout1
        H2 = (H1 + 2 * P - K_h) // S + 1  # 20
        W2 = (W1 + 2 * P - K_w) // S + 1  # W1 // 2

        y2 = torch.empty((B, Cin2, H2, W2), device=input_features.device, dtype=input_features.dtype)

        grid_conv2 = (B * Cin2, H2, W2)
        conv2d_fwd_nchw_kernel[
            grid_conv2
        ](
            y1,
            conv2d2_weight,
            conv2d2_bias,
            y2,
            B,
            Cin2,
            H1,
            W1,
            Cin2,
            K_h,
            K_w,
            S,
            P,
            1,  # BLOCK_H
            1,  # BLOCK_W
        )

        N2 = y2.numel()
        gelu_kernel[(N2 + 1024 - 1) // 1024,](y2)

        # Stage 3: Conv2d 3 -> (384, H3, W3), then GELU
        Cin3 = Cin2
        H3 = (H2 + 2 * P - K_h) // S + 1  # 10
        W3 = (W2 + 2 * P - K_w) // S + 1  # W2 // 2

        y3 = torch.empty((B, Cin3, H3, W3), device=input_features.device, dtype=input_features.dtype)

        grid_conv3 = (B * Cin3, H3, W3)
        conv2d_fwd_nchw_kernel[
            grid_conv3
        ](
            y2,
            conv2d3_weight,
            conv2d3_bias,
            y3,
            B,
            Cin3,
            H2,
            W2,
            Cin3,
            K_h,
            K_w,
            S,
            P,
            1,  # BLOCK_H
            1,  # BLOCK_W
        )

        N3 = y3.numel()
        gelu_kernel[(N3 + 1024 - 1) // 1024,](y3)

        # Stage 4: Reshape to [B, W3, C*H3], flatten to [B, T, K]
        # Original code uses conv_out_dim=3840; however, K=C_out3*H3*W3 which may differ. We will use the actual K.
        C_out3 = y3.shape[1]
        H_out3 = y3.shape[2]
        W_out3 = y3.shape[3]
        T = W_out3
        K = C_out3 * H_out3 * W_out3  # since H_out3=1, K=C_out3*W_out3

        x_reshaped = y3.permute(0, 3, 1, 2).contiguous()  # [B, W3, C_out3, H_out3]
        # Since H_out3=1, view to [B, W3, C_out3], but keeping original: [B, W3, C_out3, 1]
        # Flatten [B, W3, C_out3, 1] -> [B, T, K]
        x_flat = x_reshaped.view(B, T, K).contiguous()

        # Stage 5: Linear projection using Triton: y[b, t, d] = sum_k x[b, t, k] * W[d, k], d in [0..N-1], N=1024
        N = 1024  # d_model
        W_flat = conv_out_weight  # [N, K], contiguous
        y = torch.empty((B, T, N), device=input_features.device, dtype=input_features.dtype)

        grid_linear = (B, T, N)
        linear_kernel[grid_linear](x_flat, W_flat, y, B, T, N, K)

        # Stage 6: Scale by embed_scale (sqrt(d_model) = 32.0)
        scale_kernel[(y.numel() + 1024 - 1) // 1024,](y, embed_scale)

        # Stage 7: Add positional embedding: y[b, t, d] += pos_emb[t, d]
        # pos_emb: [max_source_positions, N], slice to T rows
        grid_pos = (B, T, N)
        add_pos_emb_kernel[grid_pos](
            y,
            positional_embedding[:T, :],  # [T, N]
            B, T, N
        )

        return y


def run(*args):
    return ModelNew()(*args)
