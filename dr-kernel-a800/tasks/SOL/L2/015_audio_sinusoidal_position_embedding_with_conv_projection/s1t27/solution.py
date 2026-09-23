import math
import torch
import triton
import triton.language as tl


# Triton conv2d kernel: 3x3, stride=2, padding=1
# Input: X(B, Cin, H, T) bfloat16, shape: (B, Cin, H, T)
# Weight: W(Cout, Cin, 3, 3) bfloat16, shape: (Cout, Cin, 3, 3)
# Bias: Bias(Cout) bfloat16, shape: (Cout,)
# Output: Y(B, Cout, H, T_out) bfloat16, with T_out = floor((T - 3)/2 + 1)
@triton.jit
def conv2d_3x3_stride2_padding1_kernel(
    X_ptr,         # *const bfloat16
    W_ptr,         # *const bfloat16
    BIAS_ptr,      # *const bfloat16
    Y_ptr,         # *bfloat16
    Bsz, Cin, H, T, Cout, T_out,
    BLOCK_C: tl.constexpr,
):
    # Grid: (B*H, tiles over Cout, T_out)
    pid_m = tl.program_id(0)   # over B*H
    pid_c = tl.program_id(1)   # over tiles of Cout
    t_out_idx = tl.program_id(2)  # specific time index in output

    b = pid_m // H
    oh = pid_m % H

    c_start = pid_c * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < Cout

    # accumulator per output channel
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel with stride=2, padding=1
    for cin in range(Cin):
        for kh in range(3):
            ih = oh + kh - 1
            valid_ih = (ih >= 0) & (ih < H)
            for kt in range(3):
                it = t_out_idx + kt - 1
                valid_it = (it >= 0) & (it < T)
                if valid_ih and valid_it:
                    # Load X[b, cin, ih, it]
                    x_idx = b * (Cin * H * T) + cin * (H * T) + ih * T + it
                    x_val = tl.load(X_ptr + x_idx, mask=True, other=0.0).to(tl.float32)
                else:
                    x_val = 0.0
                # Accumulate over output channels in tile
                for c_idx in range(BLOCK_C):
                    c = c_offsets[c_idx]
                    if c < Cout:
                        w_idx = c * (Cin * 3 * 3) + cin * (3 * 3) + kh * 3 + kt
                        w_val = tl.load(W_ptr + w_idx, mask=True, other=0.0).to(tl.float32)
                        acc[c_idx] += x_val * w_val

    # Add bias
    bias = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc += bias

    # Store to Y[b, c_offsets, oh, t_out_idx] as bfloat16
    y_idx = b * (Cout * H * T_out) + c_offsets * (H * T_out) + oh * T_out + t_out_idx
    tl.store(Y_ptr + y_idx, acc.to(tl.bfloat16), mask=mask_c)


# Triton GELU (exact, erf-based) applied to a tensor Y
# Input: Y flattened pointer (N elements), output overwritten in same buffer
@triton.jit
def gelu_erf_kernel(Y_ptr, N: tl.int32):
    pid = tl.program_id(0)
    if pid < N:
        val = tl.load(Y_ptr + pid, mask=True, other=0.0).to(tl.float32)
        inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
        gelu = 0.5 * val * (1.0 + tl.math.erf(val * inv_sqrt2))
        tl.store(Y_ptr + pid, gelu.to(tl.bfloat16), mask=True)


# Triton GEMM + positional embedding add for final linear projection
# Inputs:
#   X: (B, T, K) bfloat16 (we pass as float32 for compute), strides (stride_Xb, stride_Xt, stride_Xk)
#   WT: (K, N) bfloat16 (conv_out_weight.T), strides (stride_WTk, stride_WTn)
#   POS: (T, N) bfloat16 positional embedding slice, strides (stride_PosT, stride_PosN)
# Output:
#   Y: (B, T, N) bfloat16
@triton.jit
def gemm_add_pos_kernel(
    X_ptr, WT_ptr, POS_ptr, Y_ptr,
    B, T, K, N,
    stride_Xb, stride_Xt, stride_Xk,
    stride_WTk, stride_WTn,
    stride_PosT, stride_PosN,
    scale: tl.float32,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B*T, tiles over N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    b = pid_m // T
    t = pid_m % T

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load X[b, t, k_offsets] as float32 -> shape (BLOCK_K,)
        x_tile = tl.load(
            X_ptr + b * stride_Xb + t * stride_Xt + k_offsets * stride_Xk,
            mask=mask_k,
            other=0.0
        ).to(tl.float32)

        # Load WT[k_offsets, n_offsets] -> shape (BLOCK_K, BLOCK_N)
        wt_tile = tl.load(
            WT_ptr + k_offsets[:, None] * stride_WTk + n_offsets[None, :] * stride_WTn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        ).to(tl.float32)

        # Accumulate outer product: acc += sum_k x_tile[k] * wt_tile[k, :]
        # Equivalent to: acc += tl.sum(wt_tile * x_tile[:, None], axis=0)
        acc += tl.sum(wt_tile * x_tile[:, None], axis=0)

    # Add scaled positional embedding row
    pos_row = tl.load(
        POS_ptr + t * stride_PosT + n_offsets * stride_PosN,
        mask=mask_n,
        other=0.0
    ).to(tl.float32)
    acc = acc + scale * pos_row

    # Store to Y[b, t, n_offsets]
    y_ptrs = Y_ptr + b * (T * N) + t * N + n_offsets
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_n)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_features: torch.Tensor,
        conv2d1_weight: torch.Tensor,
        conv2d1_bias: torch.Tensor,
        conv2d2_weight: torch.Tensor,
        conv2d2_bias: torch.Tensor,
        conv2d3_weight: torch.Tensor,
        conv2d3_bias: torch.Tensor,
        conv_out_weight: torch.Tensor,  # (1024, 15360)
        positional_embedding: torch.Tensor,  # (1500, 1024)
        embed_scale: float,
    ):
        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        B, Cin, H, T = input_features.shape
        Cout1 = conv2d1_weight.shape[0]
        T1_out = (T - 3) // 2 + 1

        x1 = torch.empty((B, Cout1, H, T1_out), dtype=torch.bfloat16, device=input_features.device)
        grid_conv1 = (
            B * H,
            triton.cdiv(Cout1, 64),
            T1_out,
        )
        conv2d_3x3_stride2_padding1_kernel[grid_conv1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, Cin, H, T, Cout1, T1_out,
            BLOCK_C=64,
        )
        # GELU
        x1_flat = x1.flatten()
        gelu_erf_kernel[(x1_flat.numel(),)](x1_flat)
        x1 = x1_flat.view(B, Cout1, H, T1_out)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        Cout2 = conv2d2_weight.shape[0]
        T2_out = (T1_out - 3) // 2 + 1
        x2 = torch.empty((B, Cout2, H, T2_out), dtype=torch.bfloat16, device=input_features.device)
        grid_conv2 = (
            B * H,
            triton.cdiv(Cout2, 64),
            T2_out,
        )
        conv2d_3x3_stride2_padding1_kernel[grid_conv2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, Cout1, H, T1_out, Cout2, T2_out,
            BLOCK_C=64,
        )
        x2_flat = x2.flatten()
        gelu_erf_kernel[(x2_flat.numel(),)](x2_flat)
        x2 = x2_flat.view(B, Cout2, H, T2_out)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        Cout3 = conv2d3_weight.shape[0]
        T3_out = (T2_out - 3) // 2 + 1
        x3 = torch.empty((B, Cout3, H, T3_out), dtype=torch.bfloat16, device=input_features.device)
        grid_conv3 = (
            B * H,
            triton.cdiv(Cout3, 64),
            T3_out,
        )
        conv2d_3x3_stride2_padding1_kernel[grid_conv3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, Cout2, H, T2_out, Cout3, T3_out,
            BLOCK_C=64,
        )
        x3_flat = x3.flatten()
        gelu_erf_kernel[(x3_flat.numel(),)](x3_flat)
        x3 = x3_flat.view(B, Cout3, H, T3_out)

        # Reshape: (B, channels, freq, time) -> (B, time, channels*freq)
        # Given H=freq=40, C=Cout3=384: channels*freq = 384*40 = 15360
        Bsz, C, F, t = x3.size()
        x = x3.permute(0, 3, 1, 2).contiguous().view(Bsz, t, C * F)

        # Final linear projection: y = x @ conv_out_weight, no bias, then scale and add positional embedding
        K = x.shape[2]  # 15360
        N = conv_out_weight.shape[0]  # 1024

        # Ensure bfloat16 I/O; compute in float32 in kernel
        x_bf = x.to(torch.bfloat16)
        wt = conv_out_weight.transpose(0, 1).to(torch.bfloat16)  # (K, N)
        # Slice positional embedding to first t rows
        pos_slice = positional_embedding[:t, :].to(torch.bfloat16)

        y = torch.empty((Bsz, t, N), dtype=torch.bfloat16, device=input_features.device)

        # Prepare strides for Triton
        stride_Xb = x_bf.stride(0)
        stride_Xt = x_bf.stride(1)
        stride_Xk = x_bf.stride(2)

        stride_WTk = wt.stride(0)
        stride_WTn = wt.stride(1)

        stride_PosT = pos_slice.stride(0)
        stride_PosN = pos_slice.stride(1)

        # Launch GEMM + add positional embedding
        BLOCK_N = 128
        BLOCK_K = 1024
        grid = (Bsz * t, triton.cdiv(N, BLOCK_N))
        gemm_add_pos_kernel[grid](
            x_bf, wt, pos_slice, y,
            Bsz, t, K, N,
            stride_Xb, stride_Xt, stride_Xk,
            stride_WTk, stride_WTn,
            stride_PosT, stride_PosN,
            float(embed_scale),
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        return y


def run(*args):
    return ModelNew()(*args)
