import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


# GELU (exact) elementwise kernel
# input: x_ptr [M], output: y_ptr [M], M = numel of tensor
@triton.jit
def gelu_kernel(x_ptr, y_ptr, M, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475  # 1/sqrt(2)
    y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    tl.store(y_ptr + offsets, y, mask=mask)


# Matmul-like kernel for y[m, n] = sum_k x[m, k] * w[k, n]
# x: [M, K] (row-major), w: [K, N] (row-major), y: [M, N] (row-major)
# We launch grid over (M, N tiles). Each program computes one row m for a block of N outputs.
@triton.jit
def matmul_block_kernel(x_ptr, w_ptr, y_ptr,
                         M, K, N,
                         x_stride_m, x_stride_k,
                         w_stride_k, w_stride_n,
                         y_stride_m, y_stride_n,
                         BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    m = tl.program_id(0)
    n_block = tl.program_id(1)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # Load x[m, k_offsets] as vector
        x_vec = tl.load(x_ptr + m * x_stride_m + k_offsets * x_stride_k,
                        mask=(k_offsets < K), other=0.0)  # [BLOCK_K], float32
        # Load w[k_offsets, n_offsets] as [BLOCK_K, BLOCK_N]
        w_block = tl.load(
            w_ptr + k_offsets[:, None] * w_stride_k + n_offsets[None, :] * w_stride_n,
            mask=((k_offsets[:, None] < K) & (n_offsets[None, :] < N)),
            other=0.0
        )
        # Accumulate: acc[n] += sum_k x_vec[k] * w_block[k, n]
        # w_block is [BLOCK_K, BLOCK_N]; x_vec is [BLOCK_K]; sum over axis=0
        acc += tl.sum(w_block * x_vec[:, None], axis=0)
    # Store acc to y[m, n_offsets]
    y_ptrs = y_ptr + m * y_stride_m + n_offsets * y_stride_n
    store_mask = n_offsets < N
    tl.store(y_ptrs, acc, mask=store_mask)


# Scale elementwise: y *= scale
@triton.jit
def scale_kernel(y_ptr, M, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y = y * scale
    tl.store(y_ptr + offsets, y, mask=mask)


# Add positional embedding: y += pos (pos is [S, N], broadcast over batch)
@triton.jit
def add_pos_emb_kernel(y_ptr, pos_ptr, M, N, S, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    # M = B*S*N, map linear index to (b, s, n)
    b = offsets // (S * N)
    rem = offsets % (S * N)
    s = rem // N
    n = rem % N
    # y index: b*S*N + s*N + n = offsets
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    pos_vals = tl.load(pos_ptr + s * N + n, mask=mask, other=0.0)
    y_vals = y_vals + pos_vals
    tl.store(y_ptr + offsets, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Ensure dtype is bfloat16 (as provided in get_inputs)
        device = input_features.device
        dtype = torch.bfloat16

        # 1) conv1: (1, 80, T) -> (384, 40, T//2)
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1, bias=True)
        # GELU in Triton (exact erf-based)
        M1 = x.numel()
        x_gelu = torch.empty_like(x, dtype=torch.bfloat16)
        gelu_kernel[(triton.cdiv(M1, 4096),)](x, x_gelu, M1, 1.0, BLOCK_SIZE=4096, num_warps=4, num_stages=2)

        # 2) conv2: (384, 40, T//4) -> (384, 20, T//8)
        x = F.conv2d(x_gelu, conv2d2_weight, conv2d2_bias, stride=2, padding=1, bias=True)
        M2 = x.numel()
        x_gelu = torch.empty_like(x, dtype=torch.bfloat16)
        gelu_kernel[(triton.cdiv(M2, 4096),)](x, x_gelu, M2, 1.0, BLOCK_SIZE=4096, num_warps=4, num_stages=2)

        # 3) conv3: (384, 20, T//8) -> (384, 10, T//16)
        x = F.conv2d(x_gelu, conv2d3_weight, conv2d3_bias, stride=2, padding=1, bias=True)
        M3 = x.numel()
        x_gelu = torch.empty_like(x, dtype=torch.bfloat16)
        gelu_kernel[(triton.cdiv(M3, 4096),)](x, x_gelu, M3, 1.0, BLOCK_SIZE=4096, num_warps=4, num_stages=2)

        # Reshape to [B, S, K] where S = time_after_conv, K = 384*10 = 3840
        B = input_features.shape[0]
        S = x_gelu.shape[-1]  # time_after_conv, for given axes, equals T//16
        K = x_gelu.shape[1] * x_gelu.shape[2]  # 384*10 = 3840

        x_reshaped = x_gelu.permute(0, 3, 1, 2).contiguous().view(B, S, K)  # [B, S, K], bfloat16

        # Prepare W for Triton matmul: we need W[k, n] where original weight is [n, k] -> transpose
        W = conv_out_weight.transpose(0, 1).contiguous()  # [K, N], N=1024
        M_total = B * S
        N_out = W.shape[1]  # 1024

        # Allocate output [M_total, N_out] as bfloat16
        y = torch.empty((M_total, N_out), device=device, dtype=torch.bfloat16)

        # Launch Triton matmul: grid over (M_total rows, N tiles)
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (M_total, triton.cdiv(N_out, BLOCK_N))
        matmul_block_kernel[grid](
            x_reshaped, W, y,
            M_total, K, N_out,
            x_reshaped.stride(0), x_reshaped.stride(1),
            W.stride(0), W.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Scale by embed_scale (sqrt(1024)=32)
        y_flat = y.view(-1)
        N_total = y_flat.numel()
        scale = float(embed_scale)
        scale_kernel[(triton.cdiv(N_total, 4096),)](y_flat, N_total, scale, BLOCK_SIZE=4096, num_warps=4, num_stages=2)

        # Reshape back to [B, S, N_out]
        y = y_flat.view(B, S, N_out)

        # Add positional embedding [S, N_out] broadcast over batch
        pos_emb = positional_embedding[:S, :].to(torch.bfloat16).contiguous()  # [S, N_out]
        add_pos_emb_kernel[(triton.cdiv(S * N_out, 4096),)](
            y.view(-1), pos_emb.view(-1), S * N_out, N_out, S, BLOCK_SIZE=4096, num_warps=4, num_stages=2
        )

        return y


def run(*args):
    return ModelNew()(*args)
