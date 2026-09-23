import math
import torch
import triton
import triton.language as tl


# Triton conv2d kernel: 3x3, stride=2, padding=1, with GELU after
# Input: X(B, Cin, H, T) bfloat16
# Weight: W(Cout, Cin, 3, 3) bfloat16
# Bias: Bias(Cout) bfloat16
# Output: Y(B, Cout, H, T_out) bfloat16, with T_out = floor((T - 3)/2 + 1)
@triton.jit
def conv2d_3x3_stride2_padding1_kernel(
    X_ptr,         # *const bfloat16
    W_ptr,         # *const bfloat16
    BIAS_ptr,      # *const bfloat16
    Y_ptr,         # *bfloat16
    B, Cin, H, T, Cout, T_out,
    BLOCK_C: tl.constexpr,
):
    # Grid: (B*H, tiles over Cout, T_out)
    pid_bh = tl.program_id(0)
    pid_ct = tl.program_id(1)
    pid_t  = tl.program_id(2)

    # Decode b, oh from pid_bh
    b = pid_bh // H
    oh = pid_bh % H

    # Tile of output channels
    c_offsets = pid_ct * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < Cout

    # Output time index for this program
    t_out_idx = pid_t

    # Accumulator for this (b, oh, t_out) across Cin and 3x3
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    for cin in range(Cin):
        for kh in range(3):
            ih = oh + kh - 1
            in_row_valid = (ih >= 0) and (ih < H)
            for kt in range(3):
                it = t_out_idx + kt - 1
                in_col_valid = (it >= 0) and (it < T)

                # Linear index into X: b*(Cin*H*T) + cin*(H*T) + ih*T + it
                x_idx = b * (Cin * H * T) + cin * (H * T) + ih * T + it
                x_val = tl.load(X_ptr + x_idx, mask=(in_row_valid and in_col_valid), other=0.0).to(tl.float32)

                # Load weights for this (cin, kh, kt) and all c_offsets
                # W layout: [Cout, Cin, 3, 3] contiguous
                for c_idx in range(BLOCK_C):
                    c = c_offsets[c_idx]
                    w_idx = c * (Cin * 3 * 3) + cin * (3 * 3) + kh * 3 + kt
                    w_val = tl.load(W_ptr + w_idx, mask=mask_c[c_idx], other=0.0).to(tl.float32)
                    acc[c_idx] += x_val * w_val

    # Add bias
    bias = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc += bias

    # Apply GELU (exact, erf-based) in-kernel
    inv_sqrt2 = 0.7071067811865476
    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    # Store Y[b, c_offsets, oh, t_out_idx] as bfloat16
    y_idx = b * (Cout * H * T_out) + (c_offsets * (H * T_out)) + (oh * T_out) + t_out_idx
    tl.store(Y_ptr + y_idx, gelu.to(tl.bfloat16), mask=mask_c)


# Triton GELU (exact, erf-based) over a flattened tensor
# Input: Y flattened pointer, size N
@triton.jit
def gelu_erf_kernel(Y_ptr, N: tl.int32):
    pid = tl.program_id(0)
    if pid < N:
        val = tl.load(Y_ptr + pid).to(tl.float32)
        inv_sqrt2 = 0.7071067811865476
        gelu = 0.5 * val * (1.0 + tl.math.erf(val * inv_sqrt2))
        tl.store(Y_ptr + pid, gelu.to(tl.bfloat16))


# Triton GEMM + add scaled positional embedding (unscaled pos as original adds pos, not scaled)
# Inputs:
#   X: (B, T, K) bfloat16 (we load as float32 for compute), strides (stride_Xb, stride_Xt, stride_Xk)
#   WT: (K, N) bfloat16, conv_out_weight.T, contiguous
#   POS: (T, N) bfloat16 positional embedding slice, contiguous
# Output:
#   Y: (B, T, N) bfloat16
@triton.jit
def gemm_add_pos_kernel(
    X_ptr, WT_ptr, POS_ptr, Y_ptr,
    B: tl.int32, T: tl.int32, K: tl.int32, N: tl.int32,
    stride_Xb: tl.int32, stride_Xt: tl.int32, stride_Xk: tl.int32,
    stride_WTk: tl.int32, stride_WTn: tl.int32,
    stride_PosT: tl.int32, stride_PosN: tl.int32,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles over N, T)
    pid_b = tl.program_id(0)
    pid_nt = tl.program_id(1)
    pid_t  = tl.program_id(2)

    t_idx = pid_t
    n_offsets = pid_nt * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # Accumulator for this (b, t) and n tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load X[b, t, k_offsets]
        x_ptrs = X_ptr + pid_b * stride_Xb + t_idx * stride_Xt + k_offsets * stride_Xk
        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load WT[k_offsets, n_offsets] => [BLOCK_K, BLOCK_N]
        wt_ptrs = WT_ptr + k_offsets[:, None] * stride_WTk + n_offsets[None, :] * stride_WTn
        wt_vals = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        # Accumulate outer-product
        for kk in range(BLOCK_K):
            if mask_k[kk]:
                acc += x_vals[kk] * wt_vals[kk, :]

    # Add unscaled positional embedding: POS[t_idx, n_offsets]
    pos_ptrs = POS_ptr + t_idx * stride_PosT + n_offsets * stride_PosN
    pos_vals = tl.load(pos_ptrs, mask=mask_n, other=0.0).to(tl.float32)
    acc += pos_vals

    # Store Y[b, t_idx, n_offsets] as bfloat16
    y_ptrs = Y_ptr + pid_b * (T * N) + t_idx * N + n_offsets
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_n)


class ModelNew(torch.nn.Module):
    def __init__(self, block_c: int = 64, block_n: int = 64, block_k: int = 128):
        super().__init__()
        self.block_c = block_c
        self.block_n = block_n
        self.block_k


def run(*args):
    return ModelNew()(*args)
