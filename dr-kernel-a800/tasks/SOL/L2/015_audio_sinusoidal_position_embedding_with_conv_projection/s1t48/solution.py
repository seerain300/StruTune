import math
import torch
import triton
import triton.language as tl


# Triton conv2d kernel: 3x3, stride=2, padding=1
# Input X: (B, Cin, H, T) bfloat16
# Weight W: (Cout, Cin, 3, 3) bfloat16
# Bias: (Cout) bfloat16
# Output Y: (B, Cout, H_out, T_out) bfloat16, where H_out = floor((H - 3)/2 + 1), T_out = floor((T - 3)/2 + 1)
@triton.jit
def conv2d_3x3_stride2_padding1_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B: tl.int32, Cin: tl.int32, H: tl.int32, T: tl.int32, Cout: tl.int32, T_out: tl.int32,
    stride_xb: tl.int32, stride_xc: tl.int32, stride_xh: tl.int32, stride_xt: tl.int32,
    stride_wco: tl.int32, stride_wci: tl.int32, stride_wkh: tl.int32, stride_wkt: tl.int32,
    stride_yb: tl.int32, stride_yc: tl.int32, stride_yh: tl.int32, stride_yt: tl.int32,
    BLOCK_C: tl.constexpr,
):
    # Grid dims: (B*H, tiles over Cout, T_out)
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    # Decode b and oh
    oh = pid_m % H
    b = pid_m // H

    # Output time index
    t_out_idx = pid_t

    # Tile of output channels
    c_start = pid_c * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < Cout

    # Accumulator for tile
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    for cin_idx in range(Cin):
        # For each (kh, kt) in 3x3
        for kh in range(3):
            for kt in range(3):
                ih = oh + kh - 1
                it = t_out_idx + kt - 1
                # Valid if ih in [0, H-1], it in [0, T-1]
                valid = (ih >= 0) & (ih < H) & (it >= 0) & (it < T)
                # Load X[b, cin, ih, it]
                x_ptr = X_ptr + b * stride_xb + cin_idx * stride_xc + ih * stride_xh + it * stride_xt
                x_val = tl.load(x_ptr, mask=valid, other=0.0).to(tl.float32)
                # Load W[c_offsets, cin_idx, kh, kt]
                w_ptr_base = W_ptr + c_offsets * stride_wco + cin_idx * stride_wci
                w_ptr = w_ptr_base + kh * stride_wkh + kt * stride_wkt
                w_val = tl.load(w_ptr, mask=mask_c, other=0.0).to(tl.float32)
                # FMA
                acc += x_val * w_val

    # Add bias
    bias = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc += bias

    # Apply GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    # Store Y[b, c_offsets, oh, t_out_idx] as bfloat16
    y_ptr = Y_ptr + b * stride_yb + c_offsets * stride_yc + oh * stride_yh + t_out_idx * stride_yt
    tl.store(y_ptr, gelu.to(tl.bfloat16), mask=mask_c)


# Triton GEMM + add positional embedding:
# X: (B, T, K) bfloat16
# WT: (K, N) bfloat16 (conv_out_weight.T)
# POS: (T, N) bfloat16 (positional_embedding slice)
# Y: (B, T, N) bfloat16
@triton.jit
def gemm_add_pos_kernel(
    X_ptr, WT_ptr, POS_ptr, Y_ptr,
    B: tl.int32, T: tl.int32, K: tl.int32, N: tl.int32,
    stride_xb: tl.int32, stride_xt: tl.int32, stride_xk: tl.int32,
    stride_wtk: tl.int32, stride_wtn: tl.int32,
    stride_pos_t: tl.int32, stride_pos_n: tl.int32,
    stride_yb: tl.int32, stride_yt: tl.int32, stride_yn: tl.int32,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B*T, tiles over N)
    pid_m = tl.program_id(0)  # over B*T
    pid_n = tl.program_id(1)  # tiles over N

    # Decode b and t
    b = pid_m // T
    t = pid_m % T

    # N tile
    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # Accumulator for N tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_N):
        k_offsets = k0 + tl.arange(0, BLOCK_N)
        mask_k = k_offsets < K

        # Load X[b, t, k_offsets]
        x_ptr = X_ptr + b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptr, mask=mask_k, other=0.0).to(tl.float32)  # (BLOCK_N,)

        # Load WT[k_offsets, n_offsets] as (BLOCK_N, BLOCK_N)
        wt_ptr = WT_ptr + k_offsets[:, None] * stride_wtk + n_offsets[None, :] * stride_wtn
        wt_block = tl.load(wt_ptr, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.sum(wt_block * x_vec[:, None], axis=0)

    # Add positional embedding POS[t, n_offsets]
    pos_ptr = POS_ptr + t * stride_pos_t + n_offsets * stride_pos_n
    pos_vec = tl.load(pos_ptr, mask=mask_n, other=0.0).to(tl.float32)
    acc += pos_vec

    # Store Y[b, t, n_offsets]
    y_ptr = Y_ptr + b * stride_yb + t * stride_yt + n_offsets * stride_yn
    tl.store(y_ptr, acc.to(tl.bfloat16), mask=mask_n)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features,
                conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Ensure CUDA and dtype
        device = input_features.device
        assert device.type == 'cuda', "ModelNew requires CUDA device for Triton kernels"

        # Conv1: (B, 1, 80, T) -> (B, 384, 40, T1)
        B, Cin1, H, T = input_features.shape
        Cin, Cout, _, _ = conv2d1_weight.shape
        assert Cin == 1, "Conv2d1 expected input channels=1"
        T1 = (T - 3) // 2 + 1
        H_out1 = (H - 3) // 2 + 1

        X1 = input_features
        W1 = conv2d1_weight
        B1 = conv2d1_bias

        Y1 = torch.empty((B, Cout, H_out1, T1), dtype=torch.bfloat16, device=device)
        grid1 = (B * H_out1, triton.cdiv(Cout, 64), T1)
        conv2d_3x3_stride2_padding1_kernel[grid1](
            X1, W1, B1, Y1,
            B, Cin, H, T, Cout, T1,
            X1.stride(0), X1.stride(1), X1.stride(2), X1.stride(3),
            W1.stride(0), W1.stride(1), W1.stride(2), W1.stride(3),
            Y1.stride(0), Y1.stride(1), Y1.stride(2), Y1.stride(3),
            BLOCK_C=64,
        )

        # Conv2: (B, 384, 40, T1) -> (B, 384, 20, T2)
        Cin2, Cout2, _, _ = conv2d2_weight.shape
        assert Cin2 == Cout, "Conv2d2 input channels must equal Conv2d1 output channels"
        T2 = (T1 - 3) // 2 + 1
        H_out2 = (H_out1 - 3) // 2 + 1

        X2 = Y1
        W2 = conv2d2_weight
        B2 = conv2d2_bias

        Y2 = torch.empty((B, Cout2, H_out2, T2), dtype=torch.bfloat16, device=device)
        grid2 = (B * H_out2, triton.cdiv(Cout2, 64), T2)
        conv2d_3x3_stride2_padding1_kernel[grid2](
            X2, W2, B2, Y2,
            B, Cin2, H_out1, T1, Cout2, T2,
            X2.stride(0), X2.stride(1), X2.stride(2), X2.stride(3),
            W2.stride(0), W2.stride(1), W2.stride(2), W2.stride(3),
            Y2.stride(0), Y2.stride(1), Y2.stride(2), Y2.stride(3),
            BLOCK_C=64,
        )

        # Conv3: (B, 384, 20, T2) -> (B, 384, 10, T3)
        Cin3, Cout3, _, _ = conv2d3_weight.shape
        assert Cin3 == Cout2, "Conv2d3 input channels must equal Conv2d2 output channels"
        T3 = (T2 - 3) // 2 + 1
        H_out3 = (H_out2 - 3) // 2 + 1

        X3 = Y2
        W3 = conv2d3_weight
        B3 = conv2d3_bias

        Y3 = torch.empty((B, Cout3, H_out3, T3), dtype=torch.bfloat16, device=device)
        grid3 = (B * H_out3, triton.cdiv(Cout3, 64), T3)
        conv2d_3x3_stride2_padding1_kernel[grid3](
            X3, W3, B3, Y3,
            B, Cin3, H_out2, T2, Cout3, T3,
            X3.stride(0), X3.stride(1), X3.stride(2), X3.stride(3),
            W3.stride(0), W3.stride(1), W3.stride(2), W3.stride(3),
            Y3.stride(0), Y3.stride(1), Y3.stride(2), Y3.stride(3),
            BLOCK_C=64,
        )

        # Permute to (B, T3, 384*10)
        K = Cout3 * 10  # 384 * 10
        x = Y3.permute(0, 3, 1, 2).contiguous().view(B, T3, K)  # (B, T3, 3840)

        # GEMM + add scaled positional embedding
        # conv_out_weight shape: (N=1024, K=15360), we pass WT = weight.T of shape (K, N)
        N = conv_out_weight.shape[0]  # d_model = 1024
        WT = conv_out_weight.t()  # (K, N)
        # Scale positional embedding by embed_scale
        pos = positional_embedding.to(torch.bfloat16)
        pos = pos * embed_scale  # embed_scale = sqrt(N) = 32.0

        # Only first T3 rows are needed; ensure slice is contiguous
        pos_slice = pos[:T3, :].contiguous()

        Y = torch.empty((B, T3, N), dtype=torch.bfloat16, device=device)

        # Launch GEMM + add POS
        grid_gemm = (B * T3, triton.cdiv(N, 128))
        gemm_add_pos_kernel[grid_gemm](
            x, WT, pos_slice, Y,
            B, T3, K, N,
            x.stride(0), x.stride(1), x.stride(2),
            WT.stride(0), WT.stride(1),
            pos_slice.stride(0), pos_slice.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_N=128,
        )

        return Y


def run(*args):
    return ModelNew()(*args)
