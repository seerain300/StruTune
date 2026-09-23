import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def linear_matmul_kernel(
    X,          # *ptr* input [B*S, K], row-major, bfloat16
    W,          # *ptr* weight [N, K], row-major, bfloat16
    Y,          # *ptr* output [B*S, N], row-major, bfloat16
    B, S, K, N, # problem sizes
    stride_x_row, stride_x_k,      # strides for X: row (S), K
    stride_w_n, stride_w_k,        # strides for W: N, K
    stride_y_row, stride_y_n,      # strides for Y: row (S), N
    BLOCK_N: tl.constexpr,         # output channels tile
    BLOCK_K: tl.constexpr,         # K tile
):
    row = tl.program_id(0)  # which row of [B*S]
    n_block = tl.program_id(1)  # which tile of N

    # Map row to batch and time index if needed
    b = row // S
    s = row % S

    # Base pointer for this row in X and Y
    x_row_ptr = X + row * stride_x_row
    y_row_ptr = Y + row * stride_y_row

    # Output channel indices for this tile
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Accumulator for this row across N tile
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks
    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load X row segment: [BLOCK_K]
        x_vals = tl.load(x_row_ptr + k_offsets * stride_x_k, mask=k_mask, other=0.0)
        x_vals = x_vals.to(tl.float32)

        # Load W tile: [BLOCK_N, BLOCK_K]
        w_ptrs = W + n_offsets[:, None] * stride_w_n + k_offsets[None, :] * stride_w_k
        w_mask = n_mask[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
        w_vals = w_vals.to(tl.float32)

        # Accumulate: dot per output channel
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        k0 += BLOCK_K

    # Store results for this row's N tile
    y_ptrs = y_row_ptr + n_offsets * stride_y_n
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=n_mask)


@triton.jit
def scale_elementwise_kernel(
    Y,          # *ptr* input/output [total elements], bfloat16
    scale,      # float32 scalar
    N_elems: tl.constexpr,  # total elements
):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N_elems
    y = tl.load(Y + offs, mask=mask, other=0.0)
    y = y * scale
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def add_pos_emb_kernel(
    Y,          # *ptr* input/output [total elements], bfloat16
    pos_ptr,    # *ptr* positional embedding [S*N] flattened, bfloat16
    N_elems: tl.constexpr,  # total elements
):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N_elems
    y = tl.load(Y + offs, mask=mask, other=0.0)
    p = tl.load(pos_ptr + offs, mask=mask, other=0.0)
    y = y + p
    tl.store(Y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Three convs and GELU in PyTorch for correctness
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = F.gelu(x)
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = F.gelu(x)
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Reshape to [B, S, K] where K = channels*freq
        B, C, F, T8 = x.size()
        K = C * F
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T8, C, F]
        x = x.view(B, T8, K)  # [B, S, K] with K=3840

        # Triton linear projection: (B*S, K) @ (N, K)^T -> (B*S, N)
        B2, S, K = x.shape
        N = conv_out_weight.shape[0]  # 1024
        x_row = x.reshape(B2 * S, K).contiguous()  # [B*S, K]
        W = conv_out_weight.contiguous()           # [N, K]
        y = torch.empty((B2, S, N), device=x.device, dtype=torch.bfloat16)  # output in bfloat16

        # Launch Triton GEMM
        grid_linear = (B2 * S, triton.cdiv(N, 128))
        linear_matmul_kernel[grid_linear](
            x_row, W, y,
            B2, S, K, N,
            x_row.stride(0), x_row.stride(1),
            W.stride(0), W.stride(1),
            y.stride(0), y.stride(2),
            BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # Scale by embed_scale (32)
        y_flat = y.view(-1)
        N_elems = y_flat.numel()
        grid_scale = (triton.cdiv(N_elems, 1024),)
        scale_elementwise_kernel[grid_scale](y_flat, float(embed_scale), N_elems=N_elems, num_warps=4, num_stages=2)

        # Add positional embedding [S, N], broadcast over batch
        pos_emb = positional_embedding[:S, :].to(torch.bfloat16).contiguous()  # [S, N]
        grid_add = (triton.cdiv(N_elems, 1024),)
        add_pos_emb_kernel[grid_add](y_flat, pos_emb.view(-1), N_elems=N_elems, num_warps=4, num_stages=2)

        return y


def run(*args):
    return ModelNew()(*args)
