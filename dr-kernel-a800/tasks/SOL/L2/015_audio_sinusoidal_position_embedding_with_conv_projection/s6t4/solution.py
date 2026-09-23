import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


# Triton kernel: per (b, oh, ow, oc), compute a 3x3 conv over C_in, accumulate into output vector, add bias, and apply GELU (tanh approx).
# This is a naive implementation that loops over channels and spatial neighbors; it ensures Triton executes the conv + GELU.
@triton.jit
def conv_gelu_per_pixel_bf16(
    x_ptr,                  # *const bfloat16, input [B, C_in, H, W]
    w_ptr,                  # *const bfloat16, weights [C_out, C_in, 3, 3]
    bias_ptr,               # *const bfloat16, bias [C_out]
    out_ptr,                # *bfloat16, output [B, C_out, H_out, W_out]
    B, H, W, C_in, C_out, H_out, W_out,                    # int32 runtime
    kernel_h: tl.constexpr, kernel_w: tl.constexpr,       # constexpr = 3
    stride_h: tl.constexpr, stride_w: tl.constexpr,       # constexpr = 2
    pad_h: tl.constexpr, pad_w: tl.constexpr,             # constexpr = 1
    BLOCK_CIN: tl.constexpr,                               # compile-time tiling over C_in (we'll use 1)
    BLOCK_OC: tl.constexpr,                               # compile-time tiling over output channels (we'll use 1)
):
    b = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)
    oc = tl.program_id(3)

    # Accumulator for a single output channel oc
    acc = tl.zeros((), dtype=tl.bfloat16)

    # Loop over input channels and 3x3 neighborhood
    # Using naive tiling with BLOCK_CIN=1, BLOCK_OC=1 to ensure actual computation.
    for cin in range(0, C_in):
        for kh in range(0, kernel_h):
            for kw in range(0, kernel_w):
                # Compute input coordinates
                ih = oh * stride_h - pad_h + kh
                iw = ow * stride_w - pad_w + kw
                # Only accumulate if coordinates are in bounds
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                # Address for x[b, cin, ih, iw]
                x_offset = (b * C_in + cin) * H * W + ih * W + iw
                # Load x with mask; if out-of-bounds, use 0
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=tl.zeros((), dtype=tl.bfloat16))
                # Load weight for (oc, cin, kh, kw)
                w_offset = oc * (C_in * kernel_h * kernel_w) + cin * (kernel_h * kernel_w) + kh * kernel_w + kw
                w_val = tl.load(w_ptr + w_offset, mask=True, other=tl.zeros((), dtype=tl.bfloat16))
                acc += x_val * w_val

    # Add bias
    b_val = tl.load(bias_ptr + oc)
    acc = acc + b_val

    # GELU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c0 = 0.5
    c1 = 0.7978845608028654  # sqrt(2/pi)
    c2 = 0.044715
    acc_cubed = acc * acc * acc
    inner = c1 * (acc + c2 * acc_cubed)
    tanh_inner = tl.libdevice.tanh(inner)  # Triton provides tanh via libdevice
    gelu = c0 * acc * (1.0 + tanh_inner)

    # Store to out[b, oc, oh, ow]
    out_offset = (b * C_out + oc) * (H_out * W_out) + oh * W_out + ow
    tl.store(out_ptr + out_offset, gelu)


# Triton kernel: batched GEMM with fused GELU over K dimension for each (b, t), producing [N]
# y[b, t, d] = sum_k GELU(x[b, t, k]) * W[d, k]
# We pass x as [B, T, K] and W as [N, K]; output is [B, T, N].
@triton.jit
def linear_gemm_gelu_bf16(
    x_ptr,  # *const bfloat16, input [B, T, K]
    w_ptr,  # *const bfloat16, weights [N, K]
    y_ptr,  # *bfloat16, output [B, T, N]
    B, T, K, N,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    # Reduce over K in chunks
    for n_start in range(0, N, BLOCK_N):
        for k_start in range(0, K, BLOCK_K):
            # Accumulator for this (b, t)
            acc_vec = tl.zeros((BLOCK_N,), dtype=tl.bfloat16)
            # Load x_vec for this (b, t): [BLOCK_K]
            # Compute offsets for x[b, t, k_start : k_start+BLOCK_K]
            x_offsets = k_start + tl.arange(0, BLOCK_K)
            mask_x = x_offsets < K
            x_vec = tl.load(x_ptr + (b * T + t) * K + x_offsets, mask=mask_x, other=tl.zeros((), dtype=tl.bfloat16))
            # GELU on x_vec (tanh approximation)
            c0 = 0.5
            c1 = 0.7978845608028654  # sqrt(2/pi)
            c2 = 0.044715
            x_cubed = x_vec * x_vec * x_vec
            inner_x = c1 * (x_vec + c2 * x_cubed)
            tanh_x = tl.libdevice.tanh(inner_x)
            gelu_x = c0 * x_vec * (1.0 + tanh_x)
            # Load weight block W[n_start:n_start+BLOCK_N, k_start:k_start+BLOCK_K] -> [BLOCK_N, BLOCK_K]
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            mask_n = n_offsets < N
            mask_k = k_offsets < K
            w_block = tl.load(w_ptr + n_offsets[:, None] * K + k_offsets[None, :], mask=mask_n[:, None] & mask_k[None, :], other=tl.zeros((), dtype=tl.bfloat16))
            # Accumulate: acc_vec += sum over k of w_block[:, k] * gelu_x[k]
            # This is a simple outer-product reduce: acc_vec += dot(w_block, gelu_x)
            # We'll do a manual reduce over BLOCK_K:
            for kk in range(0, BLOCK_K):
                # w_col = w_block[:, kk], gelu_k = gelu_x[kk]
                w_col = w_block[:, kk]
                gelu_k = gelu_x[kk]
                acc_vec += w_col * gelu_k
        # Store acc_vec to y[b, t, n_start : n_start+BLOCK_N]
        y_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_y = y_offsets < N
        tl.store(y_ptr + (b * T + t) * N + y_offsets, acc_vec, mask=mask_y)


# Triton kernel: add positional embedding to y along the last dimension.
# y: [B, T, N], pos_emb: [T, N] bfloat16
@triton.jit
def add_pos_embed_bf16(
    y_ptr,           # *bfloat16, output [B, T, N]
    pos_ptr,         # *bfloat16, positional embedding [T, N]
    B, T, N,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    n = tl.program_id(2)
    # Load y[b, t, n] and pos[t, n], add, and store back
    y_offset = (b * T + t) * N + n
    pos_offset = t * N + n
    y_val = tl.load(y_ptr + y_offset)
    pos_val = tl.load(pos_ptr + pos_offset)
    y_val = y_val + pos_val
    tl.store(y_ptr + y_offset, y_val)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args are: input_features, conv2d1_weight, conv2d1_bias,
        #            conv2d2_weight, conv2d2_bias,
        #            conv2d3_weight, conv2d3_bias,
        #            conv_out_weight, positional_embedding, embed_scale
        # Note: we will not use torch.conv2d or F.linear here; everything is Triton.

        # Extract inputs
        input_features = args[0]          # [B, 1, 80, T]
        conv2d1_weight = args[1]          # [384, 1, 3, 3]
        conv2d1_bias = args[2]            # [384]
        conv2d2_weight = args[3]          # [384, 384, 3, 3]
        conv2d2_bias = args[4]            # [384]
        conv2d3_weight = args[5]          # [384, 384, 3, 3]
        conv3_bias = args[6]              # [384]
        conv_out_weight = args[7]         # [N=1024, K=3840]
        positional_embedding = args[8]    # [max_source_positions, N], bfloat16
        embed_scale = args[9]             # float

        # Ensure dtype is bfloat16
        if input_features.dtype != torch.bfloat16:
            input_features = input_features.to(torch.bfloat16)
        if conv2d1_weight.dtype != torch.bfloat16:
            conv2d1_weight = conv2d1_weight.to(torch.bfloat16)
        if conv2d1_bias.dtype != torch.bfloat16:
            conv2d1_bias = conv2d1_bias.to(torch.bfloat16)
        if conv2d2_weight.dtype != torch.bfloat16:
            conv2d2_weight = conv2d2_weight.to(torch.bfloat16)
        if conv2d2_bias.dtype != torch.bfloat16:
            conv2d2_bias = conv2d2_bias.to(torch.bfloat16)
        if conv2d3_weight.dtype != torch.bfloat16:
            conv2d3_weight = conv2d3_weight.to(torch.bfloat16)
        if conv3_bias.dtype != torch.bfloat16:
            conv3_bias = conv3_bias.to(torch.bfloat16)
        if conv_out_weight.dtype != torch.bfloat16:
            conv_out_weight = conv_out_weight.to(torch.bfloat16)
        if positional_embedding.dtype != torch.bfloat16:
            positional_embedding = positional_embedding.to(torch.bfloat16)

        B, C_in, H, W = input_features.shape
        kernel_h, kernel_w = 3, 3
        stride_h, stride_w = 2, 2
        pad_h, pad_w = 1, 1

        # First convolution: in_channels=1 -> out_channels=384
        C_out1 = conv2d1_weight.shape[0]  # 384
        H_out1 = (H + 2 * pad_h - kernel_h) // stride_h + 1  # (80 - 3)//2 + 1 = 39
        W_out1 = (W + 2 * pad_w - kernel_w) // stride_w + 1  # varies with W; we will use actual W

        x1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=torch.bfloat16, device=input_features.device)
        grid1 = (B, H_out1, W_out1, C_out1)
        conv_gelu_per_pixel_bf16[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, H, W, 1, C_out1, H_out1, W_out1,
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            BLOCK_CIN=1, BLOCK_OC=1,
        )

        # Second convolution: in_channels=384 -> out_channels=384
        C_in2 = C_out1
        C_out2 = conv2d2_weight.shape[0]  # 384
        H_out2 = (H_out1 + 2 * pad_h - kernel_h) // stride_h + 1
        W_out2 = (W_out1 + 2 * pad_w - kernel_w) // stride_w + 1

        x2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=torch.bfloat16, device=input_features.device)
        grid2 = (B, H_out2, W_out2, C_out2)
        conv_gelu_per_pixel_bf16[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, H_out1, W_out1, C_in2, C_out2, H_out2, W_out2,
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            BLOCK_CIN=1, BLOCK_OC=1,
        )

        # Third convolution: in_channels=384 -> out_channels=384
        C_in3 = C_out2
        C_out3 = conv2d3_weight.shape[0]  # 384
        H_out3 = (H_out2 + 2 * pad_h - kernel_h) // stride_h + 1
        W_out3 = (W_out2 + 2 * pad_w - kernel_w) // stride_w + 1

        x3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=torch.bfloat16, device=input_features.device)
        grid3 = (B, H_out3, W_out3, C_out3)
        conv_gelu_per_pixel_bf16[grid3](
            x2, conv2d3_weight, conv3_bias, x3,
            B, H_out2, W_out2, C_in3, C_out3, H_out3, W_out3,
            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
            BLOCK_CIN=1, BLOCK_OC=1,
        )

        # Reshape to [B, T, K] where T=W_out3, K=C_out3
        T = W_out3
        K = C_out3
        x3_perm = x3.permute(0, 2, 3, 1).contiguous()  # [B, H_out3, W_out3, C_out3]
        x3_reshaped = x3_perm.view(B, W_out3, C_out3).contiguous()  # [B, T, K]

        # Linear projection to N = d_model = 1024
        N = conv_out_weight.shape[0]  # 1024
        y = torch.empty((B, T, N), dtype=torch.bfloat16, device=input_features.device)

        # Launch Triton GEMM + GELU
        # Grid: (B, T, N), but Triton grid dims are integers, so we set a large grid; kernel loops over N and K.
        grid_gemm = (B, T, 1)  # the kernel will internally loop over N and K
        # Note: Triton requires a tuple of integers for grid; we set the third dim to cover N via loops inside the kernel.
        # To be explicit, we can set grid=(B, T, 1), as the kernel handles N via BLOCK_N tiling.
        # We pick BLOCK_N=64, BLOCK_K=64 for reasonable vectorization.
        BLOCK_N = 64
        BLOCK_K = 64
        linear_gemm_gelu_bf16[grid_gemm](
            x3_reshaped, conv_out_weight, y,
            B, T, K, N,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # Scale embeddings
        y = y * embed_scale  # scalar multiply, done in PyTorch (simple and fast)

        # Add positional embedding: positional_embedding is [max_source_positions, N], we only need first T rows.
        # Ensure shape: [T, N]
        pos_slice = positional_embedding[:T, :].contiguous()  # [T, N]
        # Launch Triton add kernel
        grid_add = (B, T, N)
        add_pos_embed_bf16[grid_add](
            y, pos_slice,
            B, T, N,
        )

        return y


def run(*args):
    return ModelNew()(*args)
