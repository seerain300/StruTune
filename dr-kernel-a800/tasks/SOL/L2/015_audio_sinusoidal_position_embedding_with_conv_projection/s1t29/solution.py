import math
import torch
import triton
import triton.language as tl


# Triton conv2d kernel: 3x3, stride=2, padding=1
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
    Bsz, Cin, H, T, Cout, T_out,
    BLOCK_C: tl.constexpr,
):
    # Grid: (B*H, tiles over Cout, T_out)
    pid_m = tl.program_id(0)   # over B*H
    pid_c = tl.program_id(1)   # over tiles of Cout
    t_out_idx = tl.program_id(2)  # specific output time index

    b = pid_m // H
    oh = pid_m % H

    c_start = pid_c * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < Cout

    # accumulator per output channel
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    # stride=2, padding=1 -> for output (oh, t_out_idx), input positions are (oh + kh - 1, t_out_idx + kt - 1)
    for cin in range(Cin):
        for kh in range(3):
            ih = oh + kh - 1
            valid_ih = (ih >= 0) & (ih < H)
            for kt in range(3):
                it = t_out_idx + kt - 1
                valid_it = (it >= 0) & (it < T)
                if valid_ih & valid_it:
                    # Load input scalar x[b, cin, ih, it] as bfloat16 then float32
                    x_val = tl.load(
                        X_ptr + b * (Cin * H * T) + cin * (H * T) + ih * T + it,
                        mask=True,
                        other=0.0
                    ).to(tl.float32)
                    # Load weights for all c in tile: W[c, cin, kh, kt], then multiply and accumulate
                    for c_idx in range(BLOCK_C):
                        c = c_offsets[c_idx]
                        if c < Cout:
                            w_val = tl.load(
                                W_ptr + c * (Cin * 3 * 3) + cin * (3 * 3) + kh * 3 + kt,
                                mask=True,
                                other=0.0
                            ).to(tl.float32)
                            acc[c_idx] += x_val * w_val

    # Add bias
    bias = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc += bias

    # Store to Y[b, c_offsets, oh, t_out_idx] as bfloat16
    y_ptr = Y_ptr + b * (Cout * H * T_out) + (c_offsets * (H * T_out)) + (oh * T_out) + t_out_idx
    tl.store(y_ptr, acc.to(tl.bfloat16), mask=mask_c)


# Triton GELU (exact, erf-based): Y is flattened (N elements), in/out bfloat16
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
#   X: (B, T, K) bfloat16, we'll load as float32 and compute in float32
#   WT: (K, N) bfloat16 (conv_out_weight.T), strides (stride_WTk, stride_WTn)
#   POS: (T, N) bfloat16 positional embedding slice, strides (stride_PosT, stride_PosN)
# Output:
#   Y: (B, T, N) bfloat16
@triton.jit
def gemm_add_pos_kernel(
    X_ptr, WT_ptr, POS_ptr, Y_ptr,
    Bsz, T, K, N,
    stride_Xb, stride_Xt, stride_Xk,
    stride_WTk, stride_WTn,
    stride_PosT, stride_PosN,
    scale,  # float32 scale (embed_scale)
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, T, tiles of N)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_n = tl.program_id(2)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # accumulator per output channel tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: X[pid_b, pid_t, k_offsets] -> shape (BLOCK_K,)
        A_tile = tl.load(
            X_ptr + pid_b * stride_Xb + pid_t * stride_Xt + k_offsets * stride_Xk,
            mask=mask_k,
            other=0.0
        ).to(tl.float32)

        # Load WT tile: WT[k_offsets, n_offsets] -> shape (BLOCK_K, BLOCK_N)
        WT_tile = tl.load(
            WT_ptr + k_offsets[:, None] * stride_WTk + n_offsets[None, :] * stride_WTn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        ).to(tl.float32)

        # Accumulate dot product over K chunk
        acc += tl.sum(WT_tile * A_tile[:, None], axis=0)

    # Add scaled positional embedding row
    pos_row = tl.load(
        POS_ptr + pid_t * stride_PosT + n_offsets * stride_PosN,
        mask=mask_n,
        other=0.0
    ).to(tl.float32)
    acc = acc + pos_row * scale

    # Store result
    y_ptr = Y_ptr + pid_b * (T * N) + pid_t * N + n_offsets
    tl.store(y_ptr, acc.to(tl.bfloat16), mask=mask_n)


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
        conv_out_weight: torch.Tensor,  # (d_model, conv_out_dim) = (1024, 15360)
        positional_embedding: torch.Tensor,  # (max_source_positions, d_model), dtype bfloat16
        embed_scale: float,  # sqrt(1024) = 32.0
    ):
        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        x = input_features  # (B, 1, 80, T)
        B, Cin, H, T = x.shape
        Cout = conv2d1_weight.shape[0]
        T_out = (T - 3) // 2 + 1  # int, should equal time_after_conv

        x_conv1 = torch.empty((B, Cout, H, T_out), dtype=torch.bfloat16, device=x.device)
        BLOCK_C = 64
        grid_conv1 = (B * H, triton.cdiv(Cout, BLOCK_C), T_out)
        conv2d_3x3_stride2_padding1_kernel[grid_conv1](
            x, conv2d1_weight, conv2d1_bias, x_conv1,
            B, Cin, H, T, Cout, T_out,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        # GELU
        x_conv1_flat = x_conv1.view(-1)  # flatten to apply elementwise
        N1 = x_conv1_flat.numel()
        gelu_erf_kernel[(N1,)](x_conv1_flat, N1, num_warps=1)
        x_conv1 = x_conv1_flat.view(B, Cout, H, T_out)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        Cin2 = Cout
        H2 = H
        T2 = T_out
        Cout2 = conv2d2_weight.shape[0]
        T_out2 = (T2 - 3) // 2 + 1

        x_conv2 = torch.empty((B, Cout2, H2, T_out2), dtype=torch.bfloat16, device=x_conv1.device)
        BLOCK_C2 = 64
        grid_conv2 = (B * H2, triton.cdiv(Cout2, BLOCK_C2), T_out2)
        conv2d_3x3_stride2_padding1_kernel[grid_conv2](
            x_conv1, conv2d2_weight, conv2d2_bias, x_conv2,
            B, Cin2, H2, T2, Cout2, T_out2,
            BLOCK_C=BLOCK_C2,
            num_warps=4,
        )
        x_conv2_flat = x_conv2.view(-1)
        N2 = x_conv2_flat.numel()
        gelu_erf_kernel[(N2,)](x_conv2_flat, N2, num_warps=1)
        x_conv2 = x_conv2_flat.view(B, Cout2, H2, T_out2)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        Cin3 = Cout2
        H3 = H2
        T3 = T_out2
        Cout3 = conv2d3_weight.shape[0]
        T_out3 = (T3 - 3) // 2 + 1

        x_conv3 = torch.empty((B, Cout3, H3, T_out3), dtype=torch.bfloat16, device=x_conv2.device)
        BLOCK_C3 = 64
        grid_conv3 = (B * H3, triton.cdiv(Cout3, BLOCK_C3), T_out3)
        conv2d_3x3_stride2_padding1_kernel[grid_conv3](
            x_conv2, conv2d3_weight, conv2d3_bias, x_conv3,
            B, Cin3, H3, T3, Cout3, T_out3,
            BLOCK_C=BLOCK_C3,
            num_warps=4,
        )
        x_conv3_flat = x_conv3.view(-1)
        N3 = x_conv3_flat.numel()
        gelu_erf_kernel[(N3,)](x_conv3_flat, N3, num_warps=1)
        x_conv3 = x_conv3_flat.view(B, Cout3, H3, T_out3)

        # Prepare for final linear projection: (B, t, C*F) where C=384, F=40
        Bsz, _, Hf, T3_out = x_conv3.shape
        C = 384
        F = 40
        t = T3_out
        x_proj = x_conv3.permute(0, 3, 1, 2).contiguous().view(Bsz, t, C * F)  # (B, t, 15360)

        # Final GEMM: (B, t, 15360) @ (15360, 1024) -> (B, t, 1024), add scaled positional embedding
        # conv_out_weight is (d_model, conv_out_dim) = (1024, 15360); we need WT = (conv_out_dim, d_model)
        WT = conv_out_weight.transpose(0, 1).contiguous()  # (15360, 1024), bfloat16
        y = torch.empty((Bsz, t, 1024), dtype=torch.bfloat16, device=x_proj.device)

        # Slice positional embedding to (t, 1024)
        pos_slice = positional_embedding[:t, :].contiguous()  # (t, 1024), bfloat16

        # Launch Triton GEMM + add pos
        BLOCK_N = 128
        BLOCK_K = 256
        grid_gemm = (Bsz, t, triton.cdiv(1024, BLOCK_N))
        gemm_add_pos_kernel[grid_gemm](
            x_proj, WT, pos_slice, y,
            Bsz, t, 15360, 1024,
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            WT.stride(0), WT.stride(1),
            pos_slice.stride(0), pos_slice.stride(1),
            float(embed_scale),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8,
        )

        return y


def run(*args):
    return ModelNew()(*args)
