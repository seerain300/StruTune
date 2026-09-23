import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: conv2d 3x3 stride=2 padding=1, fused bias and GELU (tanh approx)
@triton.jit
def conv2d_stride2_pad1_bias_gelu_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    B, C_in, H, W, C_out, H_out, W_out,
    # x strides
    stride_x_b, stride_x_ci, stride_x_h, stride_x_w,
    # w strides: w is [C_out, C_in, 3, 3]
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    # y strides: y is [B, C_out, H_out, W_out]
    stride_y_b, stride_y_co, stride_y_h, stride_y_w,
    scale,  # used after GELU, pass embed_scale here to scale y (optional, can be 1.0)
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    b = tl.program_id(0)
    co = tl.program_id(1)
    oh_block = tl.program_id(2)
    ow_block = tl.program_id(3)

    oh = oh_block * BLOCK_H + tl.arange(0, BLOCK_H)  # [BH]
    ow = ow_block * BLOCK_W + tl.arange(0, BLOCK_W)  # [BW]

    # Valid output mask
    valid_oh = oh < H_out
    valid_ow = ow < W_out
    valid = valid_oh[:, None] & valid_ow[None, :]

    # Initialize accumulator for this (b, co)
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.bfloat16)

    # Loop over input channels and 3x3 kernel
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                # Compute input coordinates
                ih = oh * 2 + (1 - kh)  # stride=2, padding=1
                iw = ow * 2 + (1 - kw)
                # Valid if within input bounds and output is valid
                in_bounds = (ih[:, None] >= 0) & (ih[:, None] < H) & (iw[None, :] >= 0) & (iw[None, :] < W)
                mask = valid & in_bounds

                # Compute pointers
                x_off = b * stride_x_b + ci * stride_x_ci + ih[:, None] * stride_x_h + iw[None, :] * stride_x_w
                w_off = co * stride_w_co + ci * stride_w_ci + kh * stride_w_kh + kw * stride_w_kw

                x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0)  # bfloat16
                w_val = tl.load(w_ptr + w_off)  # scalar bfloat16
                acc += x_val * w_val

    # Add bias
    bias_val = tl.load(bias_ptr + co)  # scalar bfloat16
    acc = acc + bias_val

    # GELU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + 0.044715 * x3)
    gelu = 0.5 * acc * (1.0 + tl.math.tanh(inner))

    # Optional scale after GELU (we'll pass embed_scale to match original: x * embed_scale)
    gelu = gelu * scale

    # Store to y
    y_off = b * stride_y_b + co * stride_y_co + oh[:, None] * stride_y_h + ow[None, :] * stride_y_w
    tl.store(y_ptr + y_off, gelu, mask=valid)


# Triton GEMM: X [B*S, K], W [N, K] -> Y [B*S, N]
@triton.jit
def linear_matmul_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, S, K, N,
    stride_x_row, stride_x_k,
    stride_w_n, stride_w_k,
    stride_y_row, stride_y_n,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_row = tl.program_id(0)  # over B*S
    pid_n = tl.program_id(1)    # over N tiles

    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.bfloat16)

    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)  # [BK]
        mask_k = kk < K

        # Load X row slice for current (b,s)
        x_off = pid_row * stride_x_row + kk * stride_x_k
        x_vec = tl.load(X_ptr + x_off, mask=mask_k, other=0.0)  # [BK], bfloat16

        # Load W block [BN, BK]
        w_off = n[:, None] * stride_w_n + kk[None, :] * stride_w_k
        w_mat = tl.load(W_ptr + w_off, mask=(n[:, None] < N) & mask_k[None, :], other=0.0)  # [BN, BK], bfloat16

        # Accumulate: acc[n] += sum_{kk} x_vec[kk] * w_mat[n,kk]
        # Manual FMA across BK
        for j in range(0, BLOCK_K):
            acc += x_vec[j] * w_mat[:, j]

    # Store result
    y_off = pid_row * stride_y_row + n * stride_y_n
    tl.store(Y_ptr + y_off, acc, mask=(n < N))


# Triton elementwise scale: y = y * scale
@triton.jit
def scale_elementwise_kernel(y_flat_ptr, scale, N_elems: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elems
    y = tl.load(y_flat_ptr + offs, mask=mask, other=0.0)
    y = y * scale
    tl.store(y_flat_ptr + offs, y, mask=mask)


# Triton elementwise add pos_emb: y = y + pos_emb (pos_emb [S, N])
@triton.jit
def add_pos_emb_kernel(y_flat_ptr, pos_flat_ptr, N_elems: tl.constexpr, S: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_elems
    # Compute row index for y: row = offs // N, col = offs % N
    row = offs // N
    col = offs % N
    # pos_emb[row, col] = pos_flat_ptr[row * N + col]
    pos_offs = row * N + col
    y_val = tl.load(y_flat_ptr + offs, mask=mask, other=0.0)
    pos_val = tl.load(pos_flat_ptr + pos_offs, mask=mask, other=0.0)
    y_val = y_val + pos_val
    tl.store(y_flat_ptr + offs, y_val, mask=mask)


class ModelNew(nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Ensure tensors are on CUDA and contiguous
        device = input_features.device
        assert device.type == 'cuda', "ModelNew.forward requires CUDA tensors"

        # Conv1: (B, 1, 80, T) -> (B, 384, 40, T//2)
        B = input_features.shape[0]
        C_in1 = 1
        H1 = 80
        W1 = input_features.shape[-1]
        C_out1 = conv2d1_weight.shape[0]
        H_out1 = (H1 + 2*1 - 3)//2 + 1
        W_out1 = (W1 + 2*1 - 3)//2 + 1
        x1 = torch.empty((B, C_out1, H_out1, W_out1), device=device, dtype=torch.bfloat16)
        grid1 = (B, C_out1, triton.cdiv(H_out1, 8), triton.cdiv(W_out1, 8))
        conv2d_stride2_pad1_bias_gelu_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, C_in1, H1, W1, C_out1, H_out1, W_out1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            float(embed_scale),
            BLOCK_H=8, BLOCK_W=8, num_warps=4, num_stages=2
        )

        # Conv2: (B, 384, 40, T//2) -> (B, 384, 20, T//4)
        C_in2 = C_out1
        H2 = H_out1
        W2 = W_out1
        C_out2 = conv2d2_weight.shape[0]
        H_out2 = (H2 + 2*1 - 3)//2 + 1
        W_out2 = (W2 + 2*1 - 3)//2 + 1
        x2 = torch.empty((B, C_out2, H_out2, W_out2), device=device, dtype=torch.bfloat16)
        grid2 = (B, C_out2, triton.cdiv(H_out2, 8), triton.cdiv(W_out2, 8))
        conv2d_stride2_pad1_bias_gelu_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, C_in2, H2, W2, C_out2, H_out2, W_out2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            float(embed_scale),
            BLOCK_H=8, BLOCK_W=8, num_warps=4, num_stages=2
        )

        # Conv3: (B, 384, 20, T//4) -> (B, 384, 10, T//8)
        C_in3 = C_out2
        H3 = H_out2
        W3 = W_out2
        C_out3 = conv2d3_weight.shape[0]
        H_out3 = (H3 + 2*1 - 3)//2 + 1
        W_out3 = (W3 + 2*1 - 3)//2 + 1
        x3 = torch.empty((B, C_out3, H_out3, W_out3), device=device, dtype=torch.bfloat16)
        grid3 = (B, C_out3, triton.cdiv(H_out3, 8), triton.cdiv(W_out3, 8))
        conv2d_stride2_pad1_bias_gelu_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, C_in3, H3, W3, C_out3, H_out3, W_out3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            float(embed_scale),
            BLOCK_H=8, BLOCK_W=8, num_warps=4, num_stages=2
        )

        # Reshape: (B, 10, T//8) -> (B, T//8, 384*10)
        T8 = (input_features.shape[-1] // 8)
        assert H_out3 == 10 and W_out3 == T8, "Conv3 must produce (10, T//8)"
        x3_reshaped = x3.view(B, T8, C_out3 * H_out3)

        # Linear projection: X_rowwise [B*T8, K] @ W [N, K]^T -> Y [B*T8, N]
        B_S = B * T8
        K = x3_reshaped.shape[-1]  # 384*10
        N = conv_out_weight.shape[0]  # 1024
        X_row = x3_reshaped.reshape(B_S, K).contiguous()
        W = conv_out_weight.contiguous()  # [N, K]
        Y_flat = torch.empty((B_S, N), device=device, dtype=torch.bfloat16)

        grid_lin = (B_S, triton.cdiv(N, 128))
        linear_matmul_kernel[grid_lin](
            X_row, W, Y_flat,
            B, T8, K, N,
            X_row.stride(0), X_row.stride(1),
            W.stride(0), W.stride(1),
            Y_flat.stride(0), Y_flat.stride(1),
            BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2
        )

        Y = Y_flat.view(B, T8, N)

        # Scale by embed_scale
        Y_flat = Y.view(-1)
        N_elems_scale = Y_flat.numel()
        grid_scale = (triton.cdiv(N_elems_scale, 1024),)
        scale_elementwise_kernel[grid_scale](Y_flat, float(embed_scale), N_elems=N_elems_scale, BLOCK=1024)

        # Add positional embedding [T8, N] broadcast over batch
        pos_emb = positional_embedding[:T8, :].contiguous()  # [T8, N]
        N_elems_add = T8 * N
        grid_add = (triton.cdiv(N_elems_add, 1024),)
        add_pos_emb_kernel[grid_add](Y_flat, pos_emb.view(-1), N_elems=N_elems_add, S=T8, N=N, BLOCK=1024)

        Y = Y_flat.view(B, T8, N)
        return Y


def run(*args):
    return ModelNew()(*args)
