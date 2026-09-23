import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


# Triton kernel: batched GEMV for y[b, t, d] = sum_k x[b, t, k] * W[d, k]
# x is [B, T, K] (row-major), W is [N, K], y is [B, T, N].
@triton.jit
def linear_gemv_kernel(
    x_ptr,          # *const float32, input [B, T, K]
    w_ptr,          # *const float32, weight [N, K]
    y_ptr,          # *float32, output [B, T, N]
    B: tl.int32, T: tl.int32, K: tl.int32, N: tl.int32,
    x_stride_b: tl.int32, x_stride_t: tl.int32, x_stride_k: tl.int32,
    y_stride_b: tl.int32, y_stride_t: tl.int32, y_stride_n: tl.int32,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)

    # accumulator for N outputs (float32)
    acc = tl.zeros((N,), dtype=tl.float32)

    # Loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load x[b, t, k_offsets] as a vector (row of length K)
        x_off = b * x_stride_b + t * x_stride_t + k_offsets * x_stride_k
        x_vals = tl.load(x_ptr + x_off, mask=k_mask, other=0.0)  # [BLOCK_K], float32

        # Load W[:, k_offsets] -> for each d in [0..N-1], multiply by x_vals and accumulate
        for d in range(0, N):
            w_row = tl.load(w_ptr + d * K + k_offsets, mask=k_mask, other=0.0)  # [BLOCK_K]
            acc[d] += tl.sum(w_row * x_vals, axis=0)

    # Store y[b, t, d] for all d
    for d in range(0, N):
        y_off = b * y_stride_b + t * y_stride_t + d * y_stride_n
        tl.store(y_ptr + y_off, acc[d])


# Triton kernel: add positional embedding to y, where y is [B, T, N] and pos_emb is 1D [T*N]
# pos_emb is flattened [T*N], and we index via t*N + n in the kernel.
@triton.jit
def add_pos_emb_kernel(
    y_ptr,          # *float32, input/output [B, T, N]
    pos_ptr,        # *const float32, positional embedding flattened [T*N]
    B: tl.int32, T: tl.int32, N: tl.int32,
    y_stride_b: tl.int32, y_stride_t: tl.int32, y_stride_n: tl.int32,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)

    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        # Load pos_emb[t, n_offsets] as a vector
        pos_off = t * N + n_offsets
        pos_vals = tl.load(pos_ptr + pos_off, mask=n_mask, other=0.0)

        # Load y[b, t, n_offsets]
        y_off = b * y_stride_b + t * y_stride_t + n_offsets * y_stride_n
        y_vals = tl.load(y_ptr + y_off, mask=n_mask, other=0.0)

        # Compute y_vals + pos_vals
        y_vals = y_vals + pos_vals

        # Store back
        tl.store(y_ptr + y_off, y_vals)


class ModelNew(nn.Module):
    def __init__(self, batch_size: int, time_dim: int, d_model: int = 1024, conv_out_dim: int = 3840):
        super().__init__()
        self.batch_size = batch_size
        self.time_dim = time_dim
        self.d_model = d_model
        self.conv_out_dim = conv_out_dim

    def forward(self, *args):
        # args provided by get_inputs:
        # input_features: [B, 1, 80, time_dim], dtype=bfloat16
        # conv weights and biases for 3 convs
        # conv_out_weight: [d_model, conv_out_dim]
        # positional_embedding: [max_source_positions, d_model], dtype=bfloat16
        # embed_scale: float
        (input_features, conv2d1_weight, conv2d1_bias,
         conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
         conv_out_weight, positional_embedding, embed_scale) = args

        # Use PyTorch for convs and GELU to ensure correctness and avoid Triton conv crashes
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = F.gelu(x)

        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = F.gelu(x)

        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Reshape to [B, T, K] where T=W_out and K=C_out*H_out*W_out
        b, c, f, t = x.size()  # x is [B, C_out, H_out, W_out]
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Linear projection to d_model (no bias); conv_out_weight is [d_model, conv_out_dim]
        # Note: The original helper uses conv_out_dim=3840. To match that behavior, we will use only the first conv_out_dim columns of x.
        # In the given helper, conv_out_dim == c*f, but here we will restrict by conv_out_dim to match code behavior.
        K = x.shape[2]
        N = self.d_model  # 1024
        # Ensure conv_out_weight second dim matches our K (the original helper sets it to 3840, which equals K).
        # If not, we could slice, but for safety, assume the helper sets it correctly. Cast to float32 for Triton kernels.
        x_3d = x.contiguous().float()  # [B, T, K]
        W = conv_out_weight.float()    # [N, conv_out_dim], but we will treat conv_out_dim == K in this setup.

        # Allocate output y_f32 [B, T, N]
        y_f32 = torch.empty((x_3d.shape[0], x_3d.shape[1], N), dtype=torch.float32, device=x_3d.device)

        # Launch GEMV Triton kernel
        B = x_3d.shape[0]
        T = x_3d.shape[1]
        K = x_3d.shape[2]

        x_stride_b = x_3d.stride(0)
        x_stride_t = x_3d.stride(1)
        x_stride_k = x_3d.stride(2)

        y_stride_b = y_f32.stride(0)
        y_stride_t = y_f32.stride(1)
        y_stride_n = y_f32.stride(2)

        BLOCK_K = 128
        grid = (B, T)
        linear_gemv_kernel[grid](
            x_3d, W, y_f32,
            B, T, K, N,
            x_stride_b, x_stride_t, x_stride_k,
            y_stride_b, y_stride_t, y_stride_n,
            BLOCK_K=BLOCK_K,
        )

        # Scale embeddings
        scale = float(embed_scale)  # sqrt(d_model) = 32.0
        y_scaled = y_f32 * scale

        # Prepare positional embedding: positional_embedding is [max_source_positions, d_model] bfloat16
        # We need to slice to first T rows and broadcast across batch. Flatten to [T, N].
        pos_emb = positional_embedding.to(torch.float32)  # cast to float32 for math
        pos_emb = pos_emb[:T, :].view(T, self.d_model)   # [T, N]
        pos_emb_1d = pos_emb.view(T * self.d_model)      # [T*N] flattened

        # Allocate final output y_final [B, T, N] in float32
        y_final_f32 = torch.empty((B, T, self.d_model), dtype=torch.float32, device=y_scaled.device)

        # Launch add positional embedding Triton kernel
        BLOCK_N = 128
        grid_add = (B, T)
        add_pos_emb_kernel[grid_add](
            y_scaled, pos_emb_1d, y_final_f32,
            B, T, self.d_model,
            y_final_f32.stride(0), y_final_f32.stride(1), y_final_f32.stride(2),
            BLOCK_N=BLOCK_N,
        )

        # Cast back to bfloat16 to match original
        return y_final_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
