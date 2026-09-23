import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def linear_matmul_kernel(
    x_ptr,          # *const float (we'll pass bf16, Triton treats as float internally)
    w_ptr,          # *const float (weight [N, K], bf16)
    y_ptr,          # *float (output [B*S, N], bf16)
    B, S, K, N,
    stride_x_row, stride_x_k,     # strides for x: x_ptr[i, k] = x_ptr + i*stride_x_row + k*stride_x_k
    stride_w_n, stride_w_k,       # strides for w: w_ptr[n, k] = w_ptr + n*stride_w_n + k*stride_w_k
    stride_y_row, stride_y_n,     # strides for y: y_ptr[i, n] = y_ptr + i*stride_y_row + n*stride_y_n
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one row i in [0, B*S) and a tile of N columns
    i = tl.program_id(0)  # row index in [0, B*S)
    n_block = tl.program_id(1)  # tile index along N

    n_start = n_block * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Accumulator for this row and N-tile
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load x[i, k_offsets] as vector (bf16 -> float32)
        x_vals = tl.load(
            x_ptr + i * stride_x_row + k_offsets * stride_x_k,
            mask=k_mask,
            other=0.0
        ).to(tl.float32)  # [BLOCK_K]

        # Load w[n_offsets, k_offsets] as matrix (bf16 -> float32)
        w_vals = tl.load(
            w_ptr + n_offsets[:, None] * stride_w_n + k_offsets[None, :] * stride_w_k,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0
        ).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # Accumulate: acc[n] += sum_k w[n, k] * x[k]
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Store results back to y[i, n_offsets] in bfloat16
    tl.store(y_ptr + i * stride_y_row + n_offsets * stride_y_n, acc.to(tl.bfloat16), mask=n_mask)


@triton.jit
def scale_elementwise_kernel(y_ptr, scale, N_elems: tl.constexpr):
    idx = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    mask = idx < N_elems
    y = tl.load(y_ptr + idx, mask=mask, other=0.0)
    y = y * scale
    tl.store(y_ptr + idx, y, mask=mask)


@triton.jit
def add_pos_emb_kernel(y_ptr, pos_ptr, N_elems: tl.constexpr):
    idx = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    mask = idx < N_elems
    y = tl.load(y_ptr + idx, mask=mask, other=0.0)
    pos = tl.load(pos_ptr + idx, mask=mask, other=0.0)  # pos_ptr is [N] flattened
    y = y + pos
    tl.store(y_ptr + idx, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Compute convolutions with PyTorch for correctness and speed
        # Conv1: [B, 1, 80, T] -> [B, 384, 40, T//2]
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = F.gelu(x)  # GELU

        # Conv2: [B, 384, 40, T//2] -> [B, 384, 20, T//4]
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Conv3: [B, 384, 20, T//4] -> [B, 384, 10, T//8]
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Reshape to [B, time_after_conv, 384*10] = [B, S, K]
        B, C, H, W = x.size()
        S = W  # time_after_conv
        K = C * H  # conv_out_dim = 384 * 10 = 3840

        # Permute to [B, S, K]
        x_reshaped = x.permute(0, 3, 1, 2).contiguous().view(B, S, K)

        # Output tensor [B, S, N] where N = d_model = 1024
        N = 1024
        y = torch.empty((B, S, N), device=x_reshaped.device, dtype=torch.bfloat16)

        # Launch GEMM Triton kernel
        # x_rowwise: treat as (B*S, K), w: (N, K), y: (B*S, N)
        # Make sure inputs are in bfloat16
        x_rowwise = x_reshaped  # already bfloat16
        w = conv_out_weight  # [N, K], bfloat16
        # Strides
        stride_x_row = x_rowwise.stride(0)  # K
        stride_x_k = x_rowwise.stride(2)    # 1
        stride_w_n = w.stride(0)            # K
        stride_w_k = w.stride(1)            # 1
        stride_y_row = y.stride(0)          # N
        stride_y_n = y.stride(2)            # 1

        BLOCK_N = 64
        BLOCK_K = 64
        grid = (B * S, triton.cdiv(N, BLOCK_N))
        linear_matmul_kernel[grid](
            x_rowwise, w, y,
            B, S, K, N,
            stride_x_row, stride_x_k,
            stride_w_n, stride_w_k,
            stride_y_row, stride_y_n,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Scale by embed_scale (float32 constant)
        y_flat = y.view(-1)
        N_elems = y_flat.numel()
        grid_scale = (triton.cdiv(N_elems, 1024),)
        scale_elementwise_kernel[grid_scale](
            y_flat, float(embed_scale), N_elems=N_elems
        )

        # Add positional embedding [S, N], broadcast over batch
        pos_emb = positional_embedding[:S, :].to(torch.bfloat16).contiguous()  # [S, N]
        grid_add = (triton.cdiv(N_elems, 1024),)
        add_pos_emb_kernel[grid_add](
            y_flat, pos_emb.view(-1), N_elems=N_elems
        )

        return y


def run(*args):
    return ModelNew()(*args)
