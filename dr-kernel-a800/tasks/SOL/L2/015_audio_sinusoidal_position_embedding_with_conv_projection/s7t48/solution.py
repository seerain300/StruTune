import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_pad1_bias_gelu_kernel(
    x_ptr,          # *bf16, input: [B, C_in, H, W]
    w_ptr,          # *bf16, weight: [C_out, C_in, 3, 3]
    b_ptr,          # *bf16, bias: [C_out]
    y_ptr,          # *bf16, output: [B, C_out, H_out, W_out]
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,   # strides for x
    w_s0, w_s1, w_s2, w_s3,   # strides for w
    y_s0, y_s1, y_s2, y_s3,   # strides for y
    embed_scale,              # float32 scale (unused in conv but kept for future)
    BLOCK_C: tl.constexpr, BLOCK_HO: tl.constexpr, BLOCK_WO: tl.constexpr,
):
    # program ids
    b_id = tl.program_id(0)
    co_block = tl.program_id(1)
    ho_block = tl.program_id(2)
    wo_block = tl.program_id(3)

    # ranges
    co = co_block * BLOCK_C + tl.arange(0, BLOCK_C)
    ho = ho_block * BLOCK_HO + tl.arange(0, BLOCK_HO)
    wo = wo_block * BLOCK_WO + tl.arange(0, BLOCK_WO)

    # masks for outputs
    co_mask = co < C_out
    ho_mask = ho < H_out
    wo_mask = wo < W_out

    # initialize accumulator for output channels
    acc = tl.zeros((BLOCK_C, BLOCK_HO, BLOCK_WO), dtype=tl.float32)

    # loop over input channels and 3x3 kernel
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                # compute input coordinates for this (kh, kw)
                hi = 2 * ho + 1 - kh
                wi = 2 * wo + 1 - kw
                # bounds check for input
                hi_in = (hi >= 0) & (hi < H)
                wi_in = (wi >= 0) & (wi < W)
                in_mask = hi_in[:, None] & wi_in[None, :] & ho_mask[:, None] & wo_mask[None, :]

                # load input tile [BLOCK_HO, BLOCK_WO] for channel ci
                x_ptrs = x_ptr + b_id * x_s0 + ci * x_s1 + hi[:, None] * x_s2 + wi[None, :] * x_s3
                x_vals = tl.load(x_ptrs, mask=in_mask, other=0.0)  # bf16 -> cast to fp32 for compute
                x_vals = x_vals.to(tl.float32)

                # load weights for these output channels
                w_ptrs = w_ptr + co * w_s0 + ci * w_s1 + kh * w_s2 + kw * w_s3
                w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0)  # bf16 -> fp32
                w_vals = w_vals.to(tl.float32)

                # outer product accumulate: (BLOCK_C, BLOCK_HO, BLOCK_WO)
                # w_vals[:, None, None] broadcasts over BLOCK_HO x BLOCK_WO
                acc += w_vals[:, None, None] * x_vals[None, :, :]

    # add bias per output channel
    b_vals = tl.load(b_ptr + co, mask=co_mask, other=0.0).to(tl.float32)
    acc += b_vals[:, None, None]

    # GELU in-kernel (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    # constants
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715
    x3 = acc * acc * acc
    inner = acc + coeff * x3
    gelu = 0.5 * acc * (1.0 + tl.tanh(sqrt_2_over_pi * inner))

    # store result to y
    y_ptrs = y_ptr + b_id * y_s0 + co[:, None, None] * y_s1 + ho[None, :, None] * y_s2 + wo[None, None, :] * y_s3
    out_mask = co_mask[:, None, None] & ho_mask[None, :, None] & wo_mask[None, None, :]
    tl.store(y_ptrs, gelu.to(tl.bfloat16), mask=out_mask)


@triton.jit
def linear_matmul_kernel(
    x_ptr,            # *bf16, input [M, K] row-major view
    w_ptr,            # *bf16, weight [N, K]
    y_ptr,            # *bf16, output [M, N]
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    x_s0, x_s1,       # strides for x (row, col)
    w_s0, w_s1,       # strides for w (row, col)
    y_s0, y_s1,       # strides for y (row, col)
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid is (M, ceil_div(N, BLOCK_N))
    row = tl.program_id(0)
    n_block = tl.program_id(1)

    # output columns for this block
    n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < N

    # accumulator for [BLOCK_N]
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K

        # load x_row [BLOCK_K] (row is fixed)
        x_row_ptrs = x_ptr + row * x_s0 + k * x_s1
        x_row = tl.load(x_row_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_K]

        # load weight block [BLOCK_N, BLOCK_K]
        w_ptrs = w_ptr + n[:, None] * w_s0 + k[None, :] * w_s1
        w_block = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # FMA: acc[n] += sum_k w_block[n,k] * x_row[k]
        acc += tl.sum(w_block * x_row[None, :], axis=1)

    # store result row
    y_row_ptrs = y_ptr + row * y_s0 + n * y_s1
    tl.store(y_row_ptrs, acc.to(tl.bfloat16), mask=n_mask)


@triton.jit
def scale_elementwise_kernel(
    x_ptr,         # *bf16, input (we will modify in-place)
    scale,         # float32
    N_elems,       # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_elems
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = (x.to(tl.float32) * scale).to(tl.bfloat16)
    tl.store(x_ptr + offs, x, mask=mask)


@triton.jit
def add_pos_emb_kernel(
    x_ptr,          # *bf16, input (we will modify in-place): y after scaling
    pos_ptr,        # *bf16, positional embedding: [time_after_conv, d_model]
    N_elems,        # total number of elements in x
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_elems

    # Compute corresponding (row, col) for each linear index
    S = tl.program_id(1)  # time_after_conv passed as grid dim
    N = tl.program_id(2)  # d_model passed as grid dim

    # For each element index, map to (s, n)
    s = offs // N
    n = offs % N

    # Load positional embedding value at (s, n)
    pos_val = tl.load(pos_ptr + s * N + n, mask=mask, other=0.0).to(tl.bfloat16)
    x_val = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = x_val + pos_val
    tl.store(x_ptr + offs, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding):
        """
        All computation is done in Triton kernels. No torch conv/gelu in forward.
        """
        device = input_features.device
        dtype = torch.bfloat16

        # Ensure tensors are contiguous and dtype
        input_features = input_features.contiguous().to(dtype)
        conv2d1_weight = conv2d1_weight.contiguous().to(dtype)
        conv2d1_bias = conv2d1_bias.contiguous().to(dtype)
        conv2d2_weight = conv2d2_weight.contiguous().to(dtype)
        conv2d2_bias = conv2d2_bias.contiguous().to(dtype)
        conv2d3_weight = conv2d3_weight.contiguous().to(dtype)
        conv2d3_bias = conv2d3_bias.contiguous().to(dtype)
        conv_out_weight = conv_out_weight.contiguous().to(dtype)
        positional_embedding = positional_embedding.contiguous().to(dtype)

        B, C_in, H, W = input_features.shape  # B=, C_in=1, H=80, W=T
        # Stage 1: conv1 (1->384) + GELU, stride=2, pad=1
        C_out1 = conv2d1_weight.shape[0]
        H_out1 = (H + 2*1 - 3)//2 + 1
        W_out1 = (W + 2*1 - 3)//2 + 1
        x1 = torch.empty((B, C_out1, H_out1, W_out1), device=device, dtype=dtype)
        grid1 = (B, triton.cdiv(C_out1, 64), triton.cdiv(H_out1, 8), triton.cdiv(W_out1, 8))
        conv2d_stride2_pad1_bias_gelu_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, C_in, H, W, C_out1, H_out1, W_out1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            float(math.sqrt(1024.0)),  # embed_scale placeholder
            BLOCK_C=64, BLOCK_HO=8, BLOCK_WO=8,
            num_warps=4, num_stages=2
        )

        # Stage 2: conv2 (384->384) + GELU, stride=2, pad=1
        C_in2 = C_out1
        H2 = H_out1
        W2 = W_out1
        C_out2 = conv2d2_weight.shape[0]
        H_out2 = (H2 + 2*1 - 3)//2 + 1
        W_out2 = (W2 + 2*1 - 3)//2 + 1
        x2 = torch.empty((B, C_out2, H_out2, W_out2), device=device, dtype=dtype)
        grid2 = (B, triton.cdiv(C_out2, 64), triton.cdiv(H_out2, 8), triton.cdiv(W_out2, 8))
        conv2d_stride2_pad1_bias_gelu_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, C_in2, H2, W2, C_out2, H_out2, W_out2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            float(math.sqrt(1024.0)),
            BLOCK_C=64, BLOCK_HO=8, BLOCK_WO=8,
            num_warps=4, num_stages=2
        )

        # Stage 3: conv3 (384->384) + GELU, stride=2, pad=1
        C_in3 = C_out2
        H3 = H_out2
        W3 = W_out2
        C_out3 = conv2d3_weight.shape[0]
        H_out3 = (H3 + 2*1 - 3)//2 + 1
        W_out3 = (W3 + 2*1 - 3)//2 + 1
        x3 = torch.empty((B, C_out3, H_out3, W_out3), device=device, dtype=dtype)
        grid3 = (B, triton.cdiv(C_out3, 64), triton.cdiv(H_out3, 8), triton.cdiv(W_out3, 8))
        conv2d_stride2_pad1_bias_gelu_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, C_in3, H3, W3, C_out3, H_out3, W_out3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            float(math.sqrt(1024.0)),
            BLOCK_C=64, BLOCK_HO=8, BLOCK_WO=8,
            num_warps=4, num_stages=2
        )

        # Reshape: (B, 384, 10, T//8) -> (B, T//8, 384*10)
        B2, C4, H4, W4 = x3.shape
        S = W4  # time_after_conv
        x3_view = x3.permute(0, 3, 1, 2).contiguous().view(B2, S, C4 * H4)  # (B, S, 3840)

        # Linear projection: (B*S, 3840) @ (1024, 3840)^T -> (B*S, 1024)
        M = B2 * S
        K = 3840
        N = conv_out_weight.shape[0]  # 1024
        # Make x_row [M, K] contiguous row-major view (we take x3_view as [B, S, K] and reshape)
        x_row = x3_view.reshape(M, K).contiguous()
        y = torch.empty((M, N), device=device, dtype=dtype)

        grid_linear = (M, triton.cdiv(N, 128))
        linear_matmul_kernel[grid_linear](
            x_row, conv_out_weight, y,
            M, N, K,
            x_row.stride(0), x_row.stride(1),
            conv_out_weight.stride(0), conv_out_weight.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # Scale by embed_scale
        y_flat = y.view(-1)  # [B*S*N]
        N_elems = y_flat.numel()
        grid_scale = (triton.cdiv(N_elems, 1024),)
        # embed_scale is sqrt(1024) = 32.0
        scale_elementwise_kernel[grid_scale](y_flat, float(32.0), N_elems, BLOCK_SIZE=1024, num_warps=4, num_stages=2)

        # Add positional embedding [S, N], broadcast over batch
        pos_emb = positional_embedding[:S, :].contiguous()  # [S, N]
        grid_add = (triton.cdiv(N_elems, 1024),)
        # We need to launch with 3D grid: (ceil_div(N_elems, 1024), S, N)
        grid_add_3d = (triton.cdiv(N_elems, 1024), S, N)
        add_pos_emb_kernel[grid_add_3d](
            y_flat, pos_emb.view(-1), N_elems,
            BLOCK_SIZE=1024,
            num_warps=4, num_stages=2
        )

        # Reshape back to (B, S, N)
        y_final = y.view(B2, S, N)
        return y_final


def run(*args):
    return ModelNew()(*args)
