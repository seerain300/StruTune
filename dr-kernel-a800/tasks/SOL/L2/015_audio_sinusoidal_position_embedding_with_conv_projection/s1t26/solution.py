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

    # accumulator per output channel (float32)
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel with stride=2, padding=1
    # For each output position (b, oh, t_out_idx), we have input positions:
    # ih = oh + kh - 1, it = t_out_idx + kt - 1, with bounds checks
    for cin in range(Cin):
        for kh in range(3):
            ih = oh + kh - 1
            valid_ih = (ih >= 0) & (ih < H)
            for kt in range(3):
                it = t_out_idx + kt - 1
                valid_it = (it >= 0) & (it < T)
                # If both are valid, load X[b, cin, ih, it]; else 0
                x_val = tl.zeros((), dtype=tl.float32)
                if valid_ih and valid_it:
                    # Linear index for X: ((b*Cin+cin)*H + ih)*T + it
                    idx = (b * Cin + cin) * H * T + ih * T + it
                    x_val = tl.load(X_ptr + idx, mask=True, other=0.0).to(tl.float32)

                # Accumulate over weights for all channels in tile
                # W layout: (Cout, Cin, 3, 3); linear index for W[c, cin, kh, kt]
                for c_idx in range(BLOCK_C):
                    c = c_offsets[c_idx]
                    if c < Cout:
                        # Linear index for W: c*(Cin*3*3) + cin*(3*3) + kh*3 + kt
                        w_idx = c * (Cin * 3 * 3) + cin * (3 * 3) + kh * 3 + kt
                        w_val = tl.load(W_ptr + w_idx, mask=True, other=0.0).to(tl.float32)
                        acc[c_idx] += x_val * w_val

    # Add bias
    bias = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc += bias

    # Store to Y[b, c_offsets, oh, t_out_idx] as bfloat16
    y_base = b * (Cout * H * T_out)
    for c_idx in range(BLOCK_C):
        c = c_offsets[c_idx]
        if c < Cout:
            y_idx = y_base + c * (H * T_out) + oh * T_out + t_out_idx
            tl.store(Y_ptr + y_idx, acc[c_idx].to(tl.bfloat16), mask=True)


# Triton GELU (exact, erf-based) applied to a tensor Y
# Input: Y flattened pointer (B*Cout*H*T_out), size N, output overwritten
@triton.jit
def gelu_erf_kernel(Y_ptr, N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(Y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    gelu = 0.5 * vals * (1.0 + tl.math.erf(vals * inv_sqrt2))
    tl.store(Y_ptr + offs, gelu.to(tl.bfloat16), mask=mask)


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
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)  # tile over T
    pid_n = tl.program_id(2)  # tile over N

    b = pid_b
    t_start = pid_t * BLOCK_N
    n_start = pid_n * BLOCK_K

    n_offsets = n_start + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    # Initialize accumulator for [BLOCK_N] channels
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets_chunk = k0 + k_offsets
        mask_k = k_offsets_chunk < K

        # Load X[b, t, k_offsets_chunk]: shape (1, BLOCK_K) -> element vector
        x_ptr = X_ptr + b * stride_Xb + pid_t * stride_Xt
        x_vals = tl.load(x_ptr + k_offsets_chunk * stride_Xk, mask=mask_k, other=0.0).to(tl.float32)

        # Load WT[k_offsets_chunk, n_offsets]: shape (BLOCK_K, BLOCK_N)
        wt_ptr = WT_ptr + k_offsets_chunk[:, None] * stride_WTk + n_offsets[None, :] * stride_WTn
        wt_vals = tl.load(wt_ptr, mask=mask_k[:, None], other=0.0).to(tl.float32)

        # Accumulate: (1, BLOCK_K) dot (BLOCK_K, BLOCK_N) -> (BLOCK_N,)
        acc += tl.sum(x_vals[:, None] * wt_vals, axis=0)

    # Add scaled positional embedding: POS[t, n_offsets]
    pos_ptr = POS_ptr + pid_t * stride_PosT + n_offsets * stride_PosN
    pos_vals = tl.load(pos_ptr, mask=(n_offsets < N), other=0.0).to(tl.float32)
    acc = acc + pos_vals * scale

    # Store Y[b, t, n_offsets]
    y_ptr = Y_ptr + b * (T * N) + pid_t * N + n_offsets
    tl.store(y_ptr, acc.to(tl.bfloat16), mask=(n_offsets < N))


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
        conv_out_weight: torch.Tensor,  # (1024, 15360) -> after.T is (15360, 1024)
        positional_embedding: torch.Tensor,  # (1500, 1024), bfloat16
        embed_scale: float,
    ):
        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        B, Cin, H, T = input_features.shape
        Cout1 = conv2d1_weight.shape[0]
        T_out1 = (T - 3) // 2 + 1
        X1 = torch.empty((B, Cout1, H, T_out1), dtype=torch.bfloat16, device=input_features.device)
        grid1 = (B * H, triton.cdiv(Cout1, 64), T_out1)
        conv2d_3x3_stride2_padding1_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, X1,
            B, Cin, H, T, Cout1, T_out1,
            BLOCK_C=64,
        )
        # GELU
        N1 = B * Cout1 * H * T_out1
        gelu_erf_kernel[(N1 + 1024 - 1) // 1024](X1, N1, BLOCK=1024)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        Bsz2, Cin2, H2, T2 = X1.shape  # B, Cout1, H, T_out1
        Cout2 = conv2d2_weight.shape[0]
        T_out2 = (T2 - 3) // 2 + 1
        X2 = torch.empty((Bsz2, Cout2, H2, T_out2), dtype=torch.bfloat16, device=X1.device)
        grid2 = (Bsz2 * H2, triton.cdiv(Cout2, 64), T_out2)
        conv2d_3x3_stride2_padding1_kernel[grid2](
            X1, conv2d2_weight, conv2d2_bias, X2,
            Bsz2, Cin2, H2, T2, Cout2, T_out2,
            BLOCK_C=64,
        )
        N2 = Bsz2 * Cout2 * H2 * T_out2
        gelu_erf_kernel[(N2 + 1024 - 1) // 1024](X2, N2, BLOCK=1024)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        Bsz3, Cin3, H3, T3 = X2.shape
        Cout3 = conv2d3_weight.shape[0]
        T_out3 = (T3 - 3) // 2 + 1
        X3 = torch.empty((Bsz3, Cout3, H3, T_out3), dtype=torch.bfloat16, device=X2.device)
        grid3 = (Bsz3 * H3, triton.cdiv(Cout3, 64), T_out3)
        conv2d_3x3_stride2_padding1_kernel[grid3](
            X2, conv2d3_weight, conv2d3_bias, X3,
            Bsz3, Cin3, H3, T3, Cout3, T_out3,
            BLOCK_C=64,
        )
        N3 = Bsz3 * Cout3 * H3 * T_out3
        gelu_erf_kernel[(N3 + 1024 - 1) // 1024](X3, N3, BLOCK=1024)

        # Reshape: (B, channels, freq, time) -> (B, time, channels*freq)
        b, c, f, t = X3.shape
        # We need to fuse reshape and projection. Instead, we explicitly materialize (B, T_out3, c*f)
        T_total = t
        Ctot = c * f
        X_flat = X3.reshape(b, T_total, Ctot)

        # Final linear projection (no bias) to d_model = 1024: y = X_flat @ conv_out_weight.T
        # We implement GEMM + add positional embedding in Triton.
        # conv_out_weight.T: (K=15360, N=1024)
        K = conv_out_weight.shape[1]  # 15360
        N = 1024
        Y = torch.empty((b, T_total, N), dtype=torch.bfloat16, device=X_flat.device)

        # We need to pass X_flat to Triton. Triton kernels generally read linear memory; to keep it simple,
        # we compute linear strides for X_flat (B, T, K) and pass as pointers with strides.
        # First, ensure X_flat is contiguous.
        X_flat_contig = X_flat.contiguous()  # (B, T, K) contiguous
        # Prepare strides
        Bsz, Tsz, Ksz = X_flat_contig.shape
        # For (B, T, K), strides:
        stride_Xb = X_flat_contig.stride(0)
        stride_Xt = X_flat_contig.stride(1)
        stride_Xk = X_flat_contig.stride(2)
        WT = conv_out_weight.t().contiguous()  # (K, N) bfloat16
        WT_bf = WT  # already bfloat16
        stride_WTk = WT_bf.stride(0)
        stride_WTn = WT_bf.stride(1)
        # positional_embedding: (T, N)
        POS = positional_embedding[:T_total, :].contiguous()
        stride_PosT = POS.stride(0)
        stride_PosN = POS.stride(1)

        # Launch GEMM + add pos
        BLOCK_N = 128
        BLOCK_K = 384
        grid = (Bsz, triton.cdiv(Tsz, BLOCK_N), triton.cdiv(N, BLOCK_N))
        gemm_add_pos_kernel[grid](
            X_flat_contig, WT_bf, POS, Y,
            Bsz, Tsz, K, N,
            stride_Xb, stride_Xt, stride_Xk,
            stride_WTk, stride_WTn,
            stride_PosT, stride_PosN,
            scale=embed_scale,  # float
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        return Y


def run(*args):
    return ModelNew()(*args)
