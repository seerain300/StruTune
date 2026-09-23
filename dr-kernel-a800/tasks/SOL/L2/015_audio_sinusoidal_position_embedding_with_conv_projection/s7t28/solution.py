import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def triton_linear_proj(x_row: tl.pointer, w: tl.pointer, y: tl.pointer,
                        B: tl.int32, S: tl.int32, K: tl.int32, N: tl.int32,
                        stride_x_row: tl.int32, stride_x_k: tl.int32,
                        stride_w_n: tl.int32, stride_w_k: tl.int32,
                        stride_y_row: tl.int32, stride_y_n: tl.int32,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute y = x_row @ w^T, where:
      x_row: [B*S, K] (row-major), bfloat16
      w:     [N, K],   bfloat16
      y:     [B*S, N], bfloat16 (output)
    """
    pid_row = tl.program_id(0)  # index over B*S
    pid_n = tl.program_id(1)    # tile over N

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load x_row[pid_row, offs_k]
        x_ptr = x_row + pid_row * stride_x_row + offs_k * stride_x_k  # [BLOCK_K]
        x_vals = tl.load(x_ptr, mask=mask_k, other=0.0)

        # Load w[offs_n, offs_k]
        w_ptr = w + offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k  # [BLOCK_N, BLOCK_K]
        w_vals = tl.load(w_ptr, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        # Accumulate
        # x_vals: [BLOCK_K], w_vals: [BLOCK_N, BLOCK_K]
        # Multiply w_vals by x_vals, reduce over K
        prod = w_vals * x_vals[None, :]
        acc += tl.sum(prod, axis=1)

    # Store result
    y_ptr = y + pid_row * stride_y_row + offs_n * stride_y_n
    tl.store(y_ptr, acc.to(tl.bfloat16), mask=mask_n)


@triton.jit
def scale_elementwise(y_flat: tl.pointer, scale: tl.float32, N: tl.int32, BLOCK: tl.constexpr):
    """
    Scale y_flat elementwise by 'scale'. y_flat is [B*S*N] flattened and bfloat16.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    y_ptr = y_flat + offs
    y_vals = tl.load(y_ptr, mask=mask, other=0.0)
    y_vals = y_vals * scale
    tl.store(y_ptr, y_vals, mask=mask)


@triton.jit
def add_pos_emb(y_flat: tl.pointer, pos_flat: tl.pointer, N: tl.int32, BLOCK: tl.constexpr):
    """
    Add positional embedding pos_flat to y_flat. y_flat: [B*S*N], pos_flat: [S*N], broadcast over batch.
    We rely on N being S*N here, so each pos element is mapped to its corresponding y position.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    y_ptr = y_flat + offs
    pos_ptr = pos_flat + offs
    y_vals = tl.load(y_ptr, mask=mask, other=0.0)
    pos_vals = tl.load(pos_ptr, mask=mask, other=0.0)
    y_vals = y_vals + pos_vals
    tl.store(y_ptr, y_vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Expect the same 10 args as Model.forward:
        # (input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale)
        input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale = args

        # Ensure dtype and device
        device = input_features.device
        B, _, _, T = input_features.shape

        # Conv1: (1, 80, T) -> (384, 40, T//2)
        x1 = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x1 = F.gelu(x1)

        # Conv2: (384, 40, T//2) -> (384, 20, T//4)
        x2 = F.conv2d(x1, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x2 = F.gelu(x2)

        # Conv3: (384, 20, T//4) -> (384, 10, T//8)
        x3 = F.conv2d(x2, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x3 = F.gelu(x3)

        # Reshape: (B, 384, 10, T//8) -> (B, T//8, 384*10) for later linear
        B, C, H, W = x3.shape  # C=384, H=10, W=T//8
        S = W  # time_after_conv
        K = C * H  # 384*10 = 3840

        x_row = x3.view(B, S, K).contiguous()  # [B, S, K], bfloat16

        N = conv_out_weight.shape[0]  # 1024
        K_w = conv_out_weight.shape[1]  # 3840
        assert K_w == K, "Weight K must match input K dimension"

        # Triton linear projection: y = x_row @ conv_out_weight (conv_out_weight is [N, K])
        y = torch.empty((B * S, N), device=device, dtype=torch.bfloat16)

        BLOCK_N = 128
        BLOCK_K = 64
        grid = (B * S, triton.cdiv(N, BLOCK_N))
        triton_linear_proj[grid](
            x_row, conv_out_weight, y,
            B, S, K, N,
            x_row.stride(0), x_row.stride(2),  # stride_x_row = K, stride_x_k = 1
            conv_out_weight.stride(0), conv_out_weight.stride(1),  # stride_w_n = K, stride_w_k = 1
            y.stride(0), y.stride(1),  # y is [B*S, N] contiguous: row stride=N, col stride=1
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Scale by embed_scale
        y_flat = y.view(-1)
        N_elems = y_flat.numel()
        scale = float(embed_scale)
        grid_scale = (triton.cdiv(N_elems, 1024),)
        scale_elementwise_kernel = scale_elementwise  # define function above
        scale_elementwise_kernel[grid_scale](y_flat, scale, N_elems, num_warps=4, num_stages=2)

        # Add positional embedding [S, N], broadcast over batch
        pos_emb = positional_embedding[:S, :].to(torch.bfloat16).contiguous()  # [S, N]
        # We need to add pos_emb across the flattened B*S rows. y_flat has length B*S*N.
        # To align, we can recompute add on y view:
        # Create a temporary y2 to add pos_emb; safer to add directly to y_flat in-place
        # But we already scaled in-place above. We'll perform addition in-place now.
        grid_add = (triton.cdiv(N_elems, 1024),)
        add_pos_emb_kernel = add_pos_emb  # define function above
        add_pos_emb_kernel[grid_add](y_flat, pos_emb.view(-1), N_elems, num_warps=4, num_stages=2)

        # Reshape to [B, S, N]
        y = y.view(B, S, N)

        return y


def run(*args):
    return ModelNew()(*args)
