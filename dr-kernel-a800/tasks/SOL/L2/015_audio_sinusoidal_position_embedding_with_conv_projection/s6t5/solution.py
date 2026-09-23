import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: 2D convolution (NCHW) with 3x3 kernel, stride=2, padding=1,
# out-of-place. Accumulates over in_channels and spatial neighbors. Applies GELU and bias.
@triton.jit
def conv3x3_stride2_gelu_nchw_inplace(
    x_ptr,  # *float32 (input), we'll cast to bfloat16 at launch
    w_ptr,  # *float32 (weights), shape [C_out, C_in, 3, 3]
    b_ptr,  # *float32 (bias), shape [C_out]
    y_ptr,  # *float32 (output), we'll cast to bfloat16 at launch
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr,
    # sizes for output
    H_out: tl.constexpr, W_out: tl.constexpr,
    # kernel params
    K_H: tl.constexpr, K_W: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
):
    # Program ids: grid = (B, H_out, W_out, C_out)
    b = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)
    oc = tl.program_id(3)

    # Accumulator for output
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood
    for ic in range(0, C_in):
        for kh in range(0, K_H):
            for kw in range(0, K_W):
                # Compute input coordinates
                ih = oh * stride_h - pad_h + kh
                iw = ow * stride_w - pad_w + kw

                # Valid if within bounds
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                # Compute input offset (NCHW)
                # x index for (b, ic, ih, iw)
                x_offset = (((b * C_in + ic) * H + ih) * W + iw)
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                # Multiply by weight[oc, ic, kh, kw]
                # weight linearized as [C_out, C_in, 3, 3]
                w_offset = (((oc * C_in) + ic) * (K_H * K_W) + (kh * K_W + kw))
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    # Add bias
    b_val = tl.load(b_ptr + oc)
    acc += b_val

    # GELU activation (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    # Constants
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.math.tanh(c0 * (acc + c1 * x3)))

    # Store to y (NCHW), y index for (b, oc, oh, ow)
    y_offset = (((b * C_out) + oc) * (H_out * W_out) + (oh * W_out + ow))
    tl.store(y_ptr + y_offset, gelu)


# Triton kernel: Linear GEMM without bias, in-kernel. Computes y[b, t, d] = sum_k x[b, t, k] * W[d, k]
# Inputs:
#   x: [B, T, K] flattened pointer
#   W: [N, K] flattened pointer
#   y: [B, T, N] flattened pointer
@triton.jit
def linear_gemm_gelu_bf16(
    x_ptr, W_ptr, y_ptr,
    B: tl.constexpr, T: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Grid: (B, T, N) — one program per output element y[b, t, d]
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)

    # Accumulator in float32
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        # Load x[b, t, k_idx] — x is [B, T, K], contiguous along K
        x_off = ((b * T + t) * K) + k_idx
        x_vec = tl.load(x_ptr + x_off, mask=k_mask, other=0.0).to(tl.float32)  # cast to fp32 for compute

        # Load W[d, k_idx] — W is [N, K], contiguous along K
        w_off = (d * K) + k_idx
        w_vec = tl.load(W_ptr + w_off, mask=k_mask, other=0.0).to(tl.float32)

        # Dot product over this tile
        # acc += sum_j x_vec[j] * w_vec[j]
        acc += tl.sum(x_vec * w_vec, axis=0)

    # GELU after linear (tanh approximation)
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.math.tanh(c0 * (acc + c1 * x3)))

    # Store y[b, t, d] as bfloat16
    y_off = ((b * T + t) * N) + d
    tl.store(y_ptr + y_off, gelu.to(tl.bfloat16))


# Triton kernel: add positional embedding scaled by embed_scale
# y: [B, T, N] bfloat16, pos_emb: [T, N] bfloat16, embed_scale: float32
@triton.jit
def add_pos_embed_bf16(
    y_ptr, pos_ptr, embed_scale,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr
):
    # Grid: (B, T, N)
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)

    # Load y[b, t, d]
    y_off = ((b * T + t) * N) + d
    y_val = tl.load(y_ptr + y_off).to(tl.float32)

    # Load pos_emb[t, d]
    pos_off = (t * N) + d
    pos_val = tl.load(pos_ptr + pos_off).to(tl.float32)

    # Add scaled positional embedding
    y_val = y_val + embed_scale * pos_val

    # Store back
    tl.store(y_ptr + y_off, y_val.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: [B, 1, 80, T]
        conv1/2/3 weights: [C_out, C_in, 3, 3]
        biases: [C_out]
        conv_out_weight: [N, K_linear] = [d_model, conv_out_dim] in helper
        positional_embedding: [max_source_positions, d_model] (bf16)
        embed_scale: float
        """
        B, C_in, H, W = input_features.shape
        C_in = C_in  # always 1 per provided inputs
        assert C_in == 1, "This implementation expects input_features with in_channels=1."

        # Allocate outputs for conv stages (float32 for numerical stability in Triton)
        x1 = torch.empty((B, conv2d1_weight.shape[0], (H + 2 - 3) // 2 + 1, (W + 2 - 3) // 2 + 1),
                         dtype=torch.float32, device=input_features.device)
        C_out1 = conv2d1_weight.shape[0]
        H_out1 = (H + 2 - 3) // 2 + 1
        W_out1 = (W + 2 - 3) // 2 + 1

        # Launch conv1 + GELU
        grid1 = (B, H_out1, W_out1, C_out1)
        conv3x3_stride2_gelu_nchw_inplace[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, 1, H, W, C_out1, H_out1, W_out1, 3, 3, 2, 1, pad_h=1, pad_w=1
        )

        # Conv2: in_channels=C_out1, out_channels=C_out2
        C_in2 = C_out1
        C_out2 = conv2d2_weight.shape[0]
        H_out2 = (H_out1 + 2 - 3) // 2 + 1
        W_out2 = (W_out1 + 2 - 3) // 2 + 1

        x2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=torch.float32, device=input_features.device)

        grid2 = (B, H_out2, W_out2, C_out2)
        conv3x3_stride2_gelu_nchw_inplace[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, C_in2, H_out1, W_out1, C_out2, H_out2, W_out2, 3, 3, 2, 1, pad_h=1, pad_w=1
        )

        # Conv3: in_channels=C_out2, out_channels=C_out3
        C_in3 = C_out2
        C_out3 = conv2d3_weight.shape[0]
        H_out3 = (H_out2 + 2 - 3) // 2 + 1
        W_out3 = (W_out2 + 2 - 3) // 2 + 1

        x3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=torch.float32, device=input_features.device)

        grid3 = (B, H_out3, W_out3, C_out3)
        conv3x3_stride2_gelu_nchw_inplace[grid3](
            x2, conv2d3_weight, conv3_bias, x3,
            B, C_in3, H_out2, W_out2, C_out3, H_out3, W_out3, 3, 3, 2, 1, pad_h=1, pad_w=1
        )

        # Reshape: (B, C_out3, H_out3, W_out3) -> (B, W_out3, C_out3*H_out3) as in original code
        # We need [B, T, K] with T=W_out3, K=C_out3*H_out3
        T = W_out3
        K = C_out3 * H_out3 * W_out3  # number of features per (batch, time)

        # Flatten x3 to [B, T, K] by permute and view
        # x3 is [B, C_out3, H_out3, W_out3]
        x3_perm = x3.permute(0, 3, 2, 1).contiguous()  # [B, W_out3, H_out3, C_out3]
        x3_flat = x3_perm.view(B, T, K).contiguous()  # [B, W_out3, C_out3*H_out3*W_out3] -> [B, T, K]
        # Note: K may be large; this is expected in helper.

        # Linear projection in Triton: y[B, T, N] where N=d_model (1024)
        N = conv_out_weight.shape[0]  # d_model = 1024
        # Ensure conv_out_weight is [N, K_linear], here K_linear=K since helper sets conv_out_dim accordingly
        K_linear = K  # matches the actual features from conv3
        y = torch.empty((B, T, N), dtype=torch.bfloat16, device=input_features.device)

        # Launch linear GEMM + GELU
        # Choose BLOCK_K and BLOCK_N; modest values for stability
        BLOCK_K = 128
        BLOCK_N = 64
        grid_linear = (B, T, N)
        linear_gemm_gelu_bf16[grid_linear](
            x3_flat, conv_out_weight, y,
            B, T, K, N,
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N
        )

        # Scale by embed_scale
        y_fp32 = y.to(torch.float32)
        y_fp32 = y_fp32 * embed_scale

        # Add positional embedding (elementwise along last dim)
        # pos_emb: [T, N] (bf16), slice first T rows (already T) and all N columns
        pos_emb = positional_embedding[:T, :].to(torch.bfloat16)

        # Launch add kernel
        grid_add = (B, T, N)
        add_pos_embed_bf16[grid_add](y_fp32, pos_emb, embed_scale, B, T, N)

        # Cast back to bfloat16 (output in bf16 as original)
        y_out = y_fp32.to(torch.bfloat16)

        return y_out


def run(*args):
    return ModelNew()(*args)
