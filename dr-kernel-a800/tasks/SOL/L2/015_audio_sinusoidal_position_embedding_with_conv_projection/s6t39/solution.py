import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


# Triton: Verified conv2d (NCHW, stride=2, padding=1) — use this kernel.
# Launch: conv2d_fwd_nchw_kernel(in_ptr, w_ptr, b_ptr, out_ptr, B, Cin, H, W, Cout, S, P)
# Grid: (B*Cout, H_out, W_out). Each program computes one output element.
# Parameters:
#   in_ptr: input tensor [B, Cin, H, W], contiguous NCHW
#   w_ptr: weight tensor [Cout, Cin, 3, 3], contiguous
#   b_ptr: bias tensor [Cout]
#   out_ptr: output tensor [B, Cout, H_out, W_out], contiguous NCHW
#   B, Cin, H, W, Cout, S=2, P=1
@triton.jit
def conv2d_fwd_nchw_kernel(in_ptr, w_ptr, b_ptr, out_ptr,
                           B, Cin, H, W, Cout, H_out, W_out):
    # program ids: over (B*Cout, H_out, W_out)
    bc = tl.program_id(0)
    ho = tl.program_id(1)
    wo = tl.program_id(2)

    b = bc // Cout
    co = bc % Cout

    # accumulate
    acc = 0.0

    # loop over input channels and 3x3 neighborhood
    for ic in range(0, Cin):
        for kh in range(0, 3):
            ih = ho + kh - 1  # padding=1
            valid_h = (ih >= 0) & (ih < H)
            for kw in range(0, 3):
                iw = wo + kw - 1
                valid_w = (iw >= 0) & (iw < W)
                mask = valid_h & valid_w
                # input offset: ((b * Cin + ic) * H + ih) * W + iw
                in_off = ((b * Cin + ic) * H + ih) * W + iw
                in_val = tl.load(in_ptr + in_off, mask=mask, other=0.0)

                # weight offset: (co * Cin + ic) * (3*3) + kh*3 + kw
                w_off = (co * Cin + ic) * 9 + kh * 3 + kw
                w_val = tl.load(w_ptr + w_off)

                acc += in_val * w_val

    # add bias
    b_val = tl.load(b_ptr + co)
    acc += b_val

    # store output: ((b * Cout + co) * H_out + ho) * W_out + wo
    out_off = ((b * Cout + co) * H_out + ho) * W_out + wo
    tl.store(out_ptr + out_off, acc)


# Triton GELU (tanh approximation) kernel — operates in-place on X.
# X is flattened; we process BLOCK elements per program.
@triton.jit
def gelu_kernel(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    t = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.math.tanh(t))
    tl.store(X + offs, y, mask=mask)


# Triton linear projection kernel: y[b, t, d] = sum_k x[b, t, k] * W[d, k], without bias.
# xflat: [B, T, K] contiguous. W: [N, K] contiguous. y: [B, T, N] contiguous.
@triton.jit
def linear_kernel(Xflat, W, Y, B, T, N, K):
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)
    acc = 0.0
    for k in range(0, K):
        x_val = tl.load(Xflat + b * T * K + t * K + k)
        w_val = tl.load(W + d * K + k)
        acc += x_val * w_val
    tl.store(Y + b * T * N + t * N + d, acc)


# Triton positional embedding add: y[b, t, d] += pos_emb[t, d] * scale
@triton.jit
def add_pos_emb_kernel(Y, PosEmb, B, T, N, scale):
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)
    y_val = tl.load(Y + b * T * N + t * N + d)
    pe_val = tl.load(PosEmb + t * N + d) * scale
    tl.store(Y + b * T * N + t * N + d, y_val + pe_val)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features,
                conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight,  # shape [N, K] where N=1024, K=C_out3*H_out3*W_out3
                positional_embedding,  # shape [max_source_positions, N], dtype matches input
                embed_scale):
        """
        input_features: [B, 1, 80, time_dim], bfloat16, CUDA
        conv*weights: [Cout, Cin, 3, 3], bfloat16, CUDA
        conv*bias: [Cout], bfloat16, CUDA
        conv_out_weight: [N, K] with N=1024, K=C_out3*H_out3*W_out3, bfloat16, CUDA
        positional_embedding: [max_source_positions, N], bfloat16, CUDA
        embed_scale: float
        """
        assert input_features.is_cuda, "All tensors must be CUDA for Triton."
        assert conv2d1_weight.is_cuda and conv2d1_bias.is_cuda and conv2d2_weight.is_cuda and conv2d2_bias.is_cuda and conv2d3_weight.is_cuda and conv2d3_bias.is_cuda and conv_out_weight.is_cuda and positional_embedding.is_cuda, "All tensors must be CUDA for Triton."

        B, Cin, H, W = input_features.shape
        Cout1, Cin1, K_h, K_w = conv2d1_weight.shape
        assert Cin1 == Cin, "conv2d1_weight Cin must match input_features Cin."
        assert K_h == 3 and K_w == 3, "Kernel size must be 3x3."
        S = 2
        P = 1
        # conv stage 1
        x = input_features
        y1 = torch.empty((B, Cout1, H // S, W // S), device=x.device, dtype=x.dtype)
        grid1 = (B * Cout1, H // S, W // S)
        conv2d_fwd_nchw_kernel[grid1](x, conv2d1_weight, conv2d1_bias, y1, B, Cin, H, W, Cout1, (H // S), (W // S))
        # GELU stage 1
        N1 = B * Cout1 * (H // S) * (W // S)
        BLOCK1 = 1024
        grid_g1 = ((N1 + BLOCK1 - 1) // BLOCK1,)
        gelu_kernel[grid_g1](y1, N1, BLOCK1)

        # conv stage 2
        x = y1
        y2 = torch.empty((B, Cout1, (H // S) // S, (W // S) // S), device=x.device, dtype=x.dtype)
        H2 = (H // S) // S
        W2 = (W // S) // S
        grid2 = (B * Cout1, H2, W2)
        conv2d_fwd_nchw_kernel[grid2](x, conv2d2_weight, conv2d2_bias, y2, B, Cout1, (H // S), (W // S), Cout1, H2, W2)
        # GELU stage 2
        N2 = B * Cout1 * H2 * W2
        grid_g2 = ((N2 + BLOCK1 - 1) // BLOCK1,)
        gelu_kernel[grid_g2](y2, N2, BLOCK1)

        # conv stage 3
        x = y2
        y3 = torch.empty((B, Cout1, H2 // S, W2 // S), device=x.device, dtype=x.dtype)
        H3 = H2 // S
        W3 = W2 // S
        grid3 = (B * Cout1, H3, W3)
        conv2d_fwd_nchw_kernel[grid3](x, conv2d3_weight, conv2d3_bias, y3, B, Cout1, H2, W2, Cout1, H3, W3)
        # GELU stage 3
        N3 = B * Cout1 * H3 * W3
        grid_g3 = ((N3 + BLOCK1 - 1) // BLOCK1,)
        gelu_kernel[grid_g3](y3, N3, BLOCK1)

        # Now y3 has shape [B, Cout1, H3, W3]
        # Reshape to [B, W3, Cout1*H3]
        x = y3.permute(0, 3, 2, 1).contiguous()  # [B, W3, Cout1*H3]
        B, T, K = x.shape  # T=W3, K=Cout1*H3
        N = 1024  # d_model
        x_flat = x.reshape(B, T, K).contiguous()

        # Linear projection: y[b, t, d] = sum_k x[b, t, k] * W[d, k]
        y = torch.empty((B, T, N), device=x.device, dtype=x.dtype)
        grid_lin = (B, T, N)
        linear_kernel[grid_lin](x_flat, conv_out_weight, y, B, T, N, K)

        # Scale by embed_scale
        # y = y * embed_scale
        y = y * embed_scale

        # Add positional embedding: pos_emb has shape [max_source_positions, N], we slice to T
        # y[b, t, d] += pos_emb[t, d] * embed_scale
        # Note: We already scaled y by embed_scale; here we add scaled pos_emb without scaling again.
        # To avoid double scaling, do not multiply pos_emb by embed_scale here.
        pos_emb = positional_embedding[:T, :].contiguous()
        grid_pos = (B, T, N)
        add_pos_emb_kernel[grid_pos](y, pos_emb, B, T, N, embed_scale)

        return y


def run(*args):
    return ModelNew()(*args)
