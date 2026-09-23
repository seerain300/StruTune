import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    # Elementwise GELU tanh approximation over N elements
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    tanh_arg = c * (x + 0.044715 * x3)
    tanh_val = tl.math.tanh(tanh_arg)
    y = 0.5 * x * (1.0 + tanh_val)
    tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def add_pos_embed_kernel(Y_ptr, Pos_ptr, scale, B, T, N):
    # Grid: (B, T, N)
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)

    # Compute linear offsets
    y_off = (b * T + t) * N + d
    pos_off = t * N + d

    y_val = tl.load(Y_ptr + y_off)
    pos_val = tl.load(Pos_ptr + pos_off)
    new_val = y_val + pos_val * scale
    tl.store(Y_ptr + y_off, new_val)


class ModelNew(torch.nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # input_features: [B, 1, 80, time_dim], bfloat16
        device = input_features.device
        B, C_in, H, W = input_features.shape

        # Conv1: in_channels=1 -> 384
        x1 = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        # Apply GELU via Triton elementwise kernel
        x1_fp32 = x1.contiguous().to(torch.float32)
        N1 = x1_fp32.numel()
        BLOCK = 1024
        grid_gelu1 = (triton.cdiv(N1, BLOCK),)
        gelu_kernel[grid_gelu1](x1_fp32, x1_fp32, N1, BLOCK, num_warps=4)
        x1 = x1_fp32.to(x1.dtype)  # ensure dtype matches original conv output dtype

        # Conv2: 384 -> 384
        x2 = F.conv2d(x1, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x2_fp32 = x2.contiguous().to(torch.float32)
        N2 = x2_fp32.numel()
        grid_gelu2 = (triton.cdiv(N2, BLOCK),)
        gelu_kernel[grid_gelu2](x2_fp32, x2_fp32, N2, BLOCK, num_warps=4)
        x2 = x2_fp32.to(x2.dtype)

        # Conv3: 384 -> 384
        x3 = F.conv2d(x2, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x3_fp32 = x3.contiguous().to(torch.float32)
        N3 = x3_fp32.numel()
        grid_gelu3 = (triton.cdiv(N3, BLOCK),)
        gelu_kernel[grid_gelu3](x3_fp32, x3_fp32, N3, BLOCK, num_warps=4)
        x3 = x3_fp32.to(x3.dtype)

        # Reshape: (B, channels, H, W) -> (B, W, channels*H)
        # After 3 convs: H_out3 = floor(H/8), W_out3 = floor(W/8). In original helper, time_after_conv is set.
        # We follow the helper logic: final T = time_after_conv provided by axes.
        # Compute H_out3 and W_out3 for clarity (not used for T since axes provides it):
        H_out3 = H // 8 if H % 8 == 0 else (H // 8)
        W_out3 = W // 8 if W % 8 == 0 else (W // 8)
        T = x3.shape[3]  # final width after last conv, which equals axes['time_after_conv']
        K = conv2d3_weight.shape[0] * H_out3 * W_out3
        X = x3.permute(0, 3, 1, 2).contiguous().view(B, T, K)

        # Linear projection: [B, T, K] @ [N, K] -> [B, T, N], N=d_model=1024
        # conv_out_weight is [N, K] (provided by helper). Compute in fp32 for stability.
        X_fp32 = X.to(torch.float32)
        W_fp32 = conv_out_weight.to(torch.float32)  # [N, K]
        Bvec, Tvec, Kvec = X_fp32.shape
        N = W_fp32.shape[0]
        Y = torch.nn.functional.linear(X_fp32, W_fp32)  # [B, T, N], fp32

        # Scale by embed_scale
        Y_scaled = Y * float(embed_scale)

        # Add scaled positional embedding: shape [max_source_positions, N], slice to T
        # Triton kernel adds pos_emb[t, :] * scale to Y[b, t, :]
        pos_emb = positional_embedding.to(torch.float32)  # [max_positions, N]
        # Slice to T
        pos_emb = pos_emb[:T, :]

        # Launch Triton kernel to add positional embedding
        grid_pos = (B, T, N)
        add_pos_embed_kernel[grid_pos](
            Y_scaled, pos_emb, float(embed_scale), B, T, N, num_warps=4
        )

        # Return in bfloat16
        return Y_scaled.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
