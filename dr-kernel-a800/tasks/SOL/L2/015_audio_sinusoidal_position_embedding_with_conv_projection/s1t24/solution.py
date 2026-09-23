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
    B, Cin, H, T, Cout, T_out,
    BLOCK_C: tl.constexpr,
):
    # Grid: (B*H, tiles over Cout, T_out)
    pid_m = tl.program_id(0)   # over B*H
    pid_c = tl.program_id(1)   # over tiles of Cout
    t_out_idx = tl.program_id(2)  # specific time index in output

    b = pid_m // H
    oh = pid_m % H

    c_start = pid_c * BLOCK_C
    co_offsets = c_start + tl.arange(0, BLOCK_C)
    mask_c = co_offsets < Cout

    # accumulator per output channel
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    for cin in range(Cin):
        # kh, kt in [0, 3)
        for kh in range(3):
            ih = oh + kh - 1
            in_row_valid = (ih >= 0) & (ih < H)
            for kt in range(3):
                it = t_out_idx + kt - 1
                in_col_valid = (it >= 0) & (it < T)

                # base offset for X[b, cin, ih, it]
                base = b * (Cin * H * T) + cin * (H * T) + ih * T + it

                # Load input vector for this (cin, kh, kt) across co tile
                # Pointer arithmetic for W: w_offset = co * (Cin*9) + cin*(3*3) + kh*3 + kt
                for j in range(BLOCK_C):
                    co = c_start + j
                    w_offset = co * (Cin * 9) + cin * 9 + kh * 3 + kt
                    w_val = tl.load(W_ptr + w_offset, mask=mask_c[j], other=0.0).to(tl.float32)
                    x_val = tl.load(X_ptr + base, mask=in_row_valid & in_col_valid & mask_c[j], other=0.0).to(tl.float32)
                    acc[j] += w_val * x_val

    # Add bias
    for j in range(BLOCK_C):
        co = c_start + j
        bval = tl.load(BIAS_ptr + co, mask=mask_c[j], other=0.0).to(tl.float32)
        acc[j] += bval

    # Apply exact GELU: gelu(z) = 0.5 * z * (1 + erf(z / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475  # 1 / sqrt(2)
    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    # Store to Y[b, co, oh, t_out]
    y_base = b * (Cout * H * T_out) + oh * T_out + t_out_idx
    for j in range(BLOCK_C):
        co = c_start + j
        y_offset = y_base * Cout + co  # y is laid out as [B, Cout, H, T_out]
        tl.store(Y_ptr + y_offset, gelu[j].to(tl.bfloat16), mask=mask_c[j])


# Triton GEMM + add scaled positional embedding:
# A: (B, M, K) bfloat16, BT: (K, N) bfloat16, POS: (M, N) bfloat16
# Output C: (B, M, N) bfloat16
@triton.jit
def gemm_add_pos_kernel(
    A_ptr, BT_ptr, POS_ptr, SCALE, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile over M (batch * time)
    pid_n = tl.program_id(1)  # tile over N (output channels)
    pid_k = tl.program_id(2)  # tile over K (input features)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = m_offsets < M
    mask_n = n_offsets < N
    mask_k = k_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        # Compute current k_offsets
        cur_k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = cur_k_offsets < K

        # Load A tile: shape (BLOCK_M, BLOCK_K), A is (B, M, K) flattened
        # We need to recover (b, m, k) from linear offsets. Since A is contiguous with strides:
        # A[b, m, k] linear offset = b*(M*K) + m*K + k. However, we only have M,N,K without b split.
        # In our use case, M = B * t, and A is (B, t, K) flattened to (M, K). We recover b and t via:
        # b = m_offsets // t, but here we flatten (B, t, K) to (M, K), so we pass pre-flattened A.
        # The caller ensures A is laid out accordingly. Triton sees A_ptr as flat memory, we load via:
        A_tile = tl.load(
            A_ptr + m_offsets[:, None] * K + cur_k_offsets[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        ).to(tl.float32)

        # Load BT tile: shape (BLOCK_K, BLOCK_N)
        BT_tile = tl.load(
            BT_ptr + cur_k_offsets[:, None] * N + n_offsets[None, :],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        ).to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, BT_tile)

    # Add scaled positional embedding: POS is (M, N)
    pos_tile = tl.load(
        POS_ptr + m_offsets[:, None] * N + n_offsets[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0
    ).to(tl.float32)
    acc = acc + pos_tile * SCALE

    # Store
    tl.store(
        C_ptr + m_offsets[:, None] * N + n_offsets[None, :],
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :]
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_features: torch.Tensor,
        conv2d1_weight: torch.Tensor, conv2d1_bias: torch.Tensor,
        conv2d2_weight: torch.Tensor, conv2d2_bias: torch.Tensor,
        conv2d3_weight: torch.Tensor, conv2d3_bias: torch.Tensor,
        conv_out_weight: torch.Tensor,  # shape (d_model, K) where K=15360 (from provided inputs)
        positional_embedding: torch.Tensor,  # shape (max_source_positions, d_model), dtype bfloat16
        embed_scale: float,
    ):
        # Ensure everything is on CUDA and contiguous
        device = input_features.device
        assert device.type == "cuda", "Input must be on CUDA device for Triton."

        B, Cin, H, T = input_features.shape
        Cin1 = Cin  # 1
        Cout1 = conv2d1_weight.shape[0]  # 384
        H1 = H
        T1 = T
        # conv1: (B, 1, 80, T) -> (B, 384, 40, T_out1)
        T_out1 = (T1 - 3) // 2 + 1
        y1 = torch.empty((B, Cout1, H1, T_out1), dtype=torch.bfloat16, device=device)
        grid1 = (B * H1, triton.cdiv(Cout1, 64), T_out1)
        conv2d_3x3_stride2_padding1_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            B, Cin1, H1, T1, Cout1, T_out1,
            BLOCK_C=64,
        )

        # conv2: (B, 384, 40, T_out1) -> (B, 384, 20, T_out2)
        Cout2 = conv2d2_weight.shape[0]  # 384
        H2 = T_out1  # 40
        T_out2 = (H2 - 3) // 2 + 1  # 20
        y2 = torch.empty((B, Cout2, H2, T_out2), dtype=torch.bfloat16, device=device)
        grid2 = (B * H2, triton.cdiv(Cout2, 64), T_out2)
        conv2d_3x3_stride2_padding1_kernel[grid2](
            y1, conv2d2_weight, conv2d2_bias, y2,
            B, Cout1, H2, T_out1, Cout2, T_out2,
            BLOCK_C=64,
        )

        # conv3: (B, 384, 20, T_out2) -> (B, 384, 10, T_out3)
        Cout3 = conv2d3_weight.shape[0]  # 384
        H3 = T_out2  # 20
        T_out3 = (H3 - 3) // 2 + 1  # 10
        y3 = torch.empty((B, Cout3, H3, T_out3), dtype=torch.bfloat16, device=device)
        grid3 = (B * H3, triton.cdiv(Cout3, 64), T_out3)
        conv2d_3x3_stride2_padding1_kernel[grid3](
            y2, conv2d3_weight, conv2d3_bias, y3,
            B, Cout2, H3, T_out2, Cout3, T_out3,
            BLOCK_C=64,
        )

        # Reshape to (B, t, C*F): with C=384, F=40 -> (B, 10, 15360)
        # However, conv_out_weight has shape (d_model, K) where d_model=1024 and K=15360.
        # We will use the original pipeline: (B, t, C*F) where C=384, F=40, so K=15360.
        t = T_out3  # time_after_conv for final conv
        x = y3.permute(0, 3, 1, 2).contiguous().view(B, t, 384 * 40)  # (B, t, 15360), bfloat16

        # Final GEMM and add scaled positional embedding using Triton
        # conv_out_weight shape: (d_model, K) = (1024, 15360)
        d_model = conv_out_weight.shape[0]  # 1024
        K = conv_out_weight.shape[1]  # 15360
        M = B * t

        # A is (B, t, K); flatten to (M, K) for kernel
        A = x  # bfloat16
        BT = conv_out_weight.transpose(0, 1).contiguous()  # (K, d_model), bfloat16
        # POS is (M, d_model): slice positional_embedding[:t, :] and broadcast scale
        pos = positional_embedding[:t, :].contiguous().to(torch.bfloat16)  # (t, d_model)

        # Output C: (B, M, d_model)
        C_out = torch.empty((B, M, d_model), dtype=torch.bfloat16, device=device)

        grid_gemm = (triton.cdiv(M, 64), triton.cdiv(d_model, 64), triton.cdiv(K, 64))
        gemm_add_pos_kernel[grid_gemm](
            A, BT, pos, embed_scale, C_out,
            M, d_model, K,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        # Reshape back to (B, t, d_model)
        y = C_out.view(B, t, d_model)
        return y


def run(*args):
    return ModelNew()(*args)
