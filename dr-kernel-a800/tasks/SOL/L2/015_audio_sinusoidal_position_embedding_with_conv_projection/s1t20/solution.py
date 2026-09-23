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
    t_out_idx = tl.program_id(2)  # specific time index in output

    b = pid_m // H
    oh = pid_m % H

    c_start = pid_c * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < Cout

    # accumulator per output channel (vector of BLOCK_C)
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel with stride=2, padding=1
    # ih = oh + kh - 1, it = t_out_idx + kt - 1
    for cin in range(Cin):
        for kh in range(3):
            ih = oh + kh - 1
            valid_ih = (ih >= 0) & (ih < H)
            for kt in range(3):
                it = t_out_idx + kt - 1
                valid_it = (it >= 0) & (it < T)
                if valid_ih & valid_it:
                    # Load x[b, cin, ih, it] as scalar
                    x_val = tl.load(
                        X_ptr + b * (Cin * H * T) + cin * (H * T) + ih * T + it,
                        mask=valid_ih & valid_it,
                        other=0.0
                    ).to(tl.float32)
                    # Load weights for current cin, kh, kt across c_offsets
                    # W layout: (Cout, Cin, 3, 3) => contiguous
                    w_base = (c_start * Cin * 9) + (cin * 9) + (kh * 3 + kt)
                    w_vec = tl.load(W_ptr + w_base + tl.arange(0, BLOCK_C), mask=mask_c, other=0.0).to(tl.float32)
                    acc += w_vec * x_val

    # Add bias
    bias_vec = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc = acc + bias_vec

    # Apply exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    # Store result: Y[b, c, oh, t_out_idx]
    out_ptr = Y_ptr + b * (Cout * H * T_out) + c_offsets * (H * T_out) + oh * T_out + t_out_idx
    tl.store(out_ptr, gelu.to(tl.bfloat16), mask=mask_c)


# Triton GEMM + add scaled positional embedding
# A: (B, M, K) where M = B * t * (C*F)
# BT: (N, K) where N = 1024, BT = conv_out_weight.T (1024, 15360)
# POS: (M, N), scale: float32
# Output C: (B, M, N)
@triton.jit
def gemm_pos_kernel(
    A_ptr,        # *const bfloat16, shape (B, M, K)
    BT_ptr,       # *const bfloat16, shape (N, K) where K=15360
    POS_ptr,      # *const bfloat16, shape (M, N)
    SCALE,        # float32
    C_ptr,        # *bfloat16, shape (B, M, N)
    Bsz, M, N, K,
    BLOCK_M: tl.constexpr,  # tile over M
    BLOCK_N: tl.constexpr,  # tile over N
    BLOCK_K: tl.constexpr,  # tile over K
):
    pid_b = tl.program_id(0)  # over batch
    pid_m = tl.program_id(1)  # over tiles of M
    pid_n = tl.program_id(2)  # over tiles of N

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: (BLOCK_M, BLOCK_K)
        A_tile_ptr = A_ptr + pid_b * (M * K) + m_offsets[:, None] * K + k_offsets[None, :]
        A_tile = tl.load(
            A_tile_ptr,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        ).to(tl.float32)

        # Load BT tile: (BLOCK_K, BLOCK_N)
        BT_tile_ptr = BT_ptr + n_offsets[None, :] * K + k_offsets[:, None]
        BT_tile = tl.load(
            BT_tile_ptr,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        ).to(tl.float32)

        acc += tl.dot(A_tile, BT_tile)

    # Add scaled positional embedding: POS is (M, N)
    POS_tile_ptr = POS_ptr + m_offsets[:, None] * N + n_offsets[None, :]
    POS_tile = tl.load(
        POS_tile_ptr,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0
    ).to(tl.float32)

    # Scale
    acc = acc * SCALE + POS_tile

    # Store
    C_out_ptr = C_ptr + pid_b * (M * N) + m_offsets[:, None] * N + n_offsets[None, :]
    tl.store(
        C_out_ptr,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :]
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_features: torch.Tensor,         # (B, 1, 80, T)
        conv2d1_weight: torch.Tensor,         # (384, 1, 3, 3)
        conv2d1_bias: torch.Tensor,           # (384)
        conv2d2_weight: torch.Tensor,         # (384, 384, 3, 3)
        conv2d2_bias: torch.Tensor,           # (384)
        conv2d3_weight: torch.Tensor,         # (384, 384, 3, 3)
        conv2d3_bias: torch.Tensor,           # (384)
        conv_out_weight: torch.Tensor,        # (1024, 15360)
        positional_embedding: torch.Tensor,   # (1500, 1024)
        embed_scale: float,                   # 32.0
    ):
        device = input_features.device
        assert device.type == "cuda", "This implementation requires CUDA device"

        B = input_features.shape[0]
        Cin = 1
        H = 80
        T = input_features.shape[-1]

        # First conv: (B, 1, 80, T) -> (B, 384, 80, T_out1)
        T_out1 = (T - 3) // 2 + 1
        y1 = torch.empty((B, 384, H, T_out1), dtype=torch.bfloat16, device=device)
        grid1 = (B * H, triton.cdiv(384, 64), T_out1)
        conv2d_3x3_stride2_padding1_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            B, Cin, H, T, 384, T_out1,
            BLOCK_C=64,
        )

        # Second conv: (B, 384, 80, T_out1) -> (B, 384, 40, T_out2)
        T_out2 = (T_out1 - 3) // 2 + 1
        y2 = torch.empty((B, 384, 40, T_out2), dtype=torch.bfloat16, device=device)
        grid2 = (B * 40, triton.cdiv(384, 64), T_out2)
        conv2d_3x3_stride2_padding1_kernel[grid2](
            y1, conv2d2_weight, conv2d2_bias, y2,
            B, 384, 40, T_out1, 384, T_out2,
            BLOCK_C=64,
        )

        # Third conv: (B, 384, 40, T_out2) -> (B, 384, 20, T_out3)
        T_out3 = (T_out2 - 3) // 2 + 1
        y3 = torch.empty((B, 384, 20, T_out3), dtype=torch.bfloat16, device=device)
        grid3 = (B * 20, triton.cdiv(384, 64), T_out3)
        conv2d_3x3_stride2_padding1_kernel[grid3](
            y2, conv2d3_weight, conv2d3_bias, y3,
            B, 384, 20, T_out2, 384, T_out3,
            BLOCK_C=64,
        )

        # Reshape: (B, t, 15360)
        t = T_out3  # time_after_conv from inputs
        x = y3.permute(0, 3, 1, 2).contiguous().view(B, t, 384 * 40)  # (B, t, 15360)

        # Final GEMM and add scaled positional embedding in Triton
        K = 384 * 40  # 15360
        N = conv_out_weight.shape[0]  # 1024
        M = B * t  # batch * time

        # Prepare BT = conv_out_weight.T contiguous
        BT = conv_out_weight.transpose(0, 1).contiguous()  # (15360, 1024), bfloat16

        # Prepare A: (B, M, K), copy x (B, t, 15360) into A
        A = x.contiguous().to(torch.bfloat16)  # shape (B, t, 15360)

        # Prepare POS: (M, N) slice positional_embedding[:t, :]
        pos = positional_embedding[:t, :].contiguous().to(torch.bfloat16)  # (t, 1024)

        # Output C: (B, M, N)
        C_out = torch.empty((B, M, N), dtype=torch.bfloat16, device=device)

        # Triton grid
        grid_gemm = (B, triton.cdiv(M, 64), triton.cdiv(N, 64))
        gemm_pos_kernel[grid_gemm](
            A, BT, pos, embed_scale, C_out,
            B, M, N, K,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        # Reshape back to (B, t, 1024)
        y = C_out.view(B, t, 1024)
        return y


def run(*args):
    return ModelNew()(*args)
