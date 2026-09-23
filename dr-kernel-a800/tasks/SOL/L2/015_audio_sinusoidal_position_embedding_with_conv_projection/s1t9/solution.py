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

    # accumulator per output channel
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    # padding=1: ih = oh + kh - 1
    # time padding: it = t_out_idx + kt - 1
    for cin in range(Cin):
        for kh in range(3):
            ih = oh + kh - 1
            valid_ih = (ih >= 0) & (ih < H)
            for kt in range(3):
                it = t_out_idx + kt - 1
                valid_it = (it >= 0) & (it < T)

                # Load input X[b, cin, ih, it] if valid; otherwise 0
                x_val = tl.load(
                    X_ptr + b * (Cin * H * T) + cin * (H * T) + ih * T + it,
                    mask=valid_ih & valid_it,
                    other=0.0
                ).to(tl.float32)

                # Load weight W[c_offsets, cin, kh, kt]
                w_vals = tl.load(
                    W_ptr + c_offsets * (Cin * 3 * 3) + cin * (3 * 3) + kh * 3 + kt,
                    mask=mask_c,
                    other=0.0
                ).to(tl.float32)

                # Fused multiply-accumulate: acc += x_val * w_vals
                acc += x_val * w_vals

    # Add bias
    bias_vals = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc += bias_vals

    # Store output Y[b, c_offsets, oh, t_out_idx]
    y_off = b * (Cout * H * T_out) + c_offsets * (H * T_out) + oh * T_out + t_out_idx
    tl.store(Y_ptr + y_off, acc.to(tl.bfloat16), mask=mask_c)


# Triton GELU kernel (exact, erf-based)
@triton.jit
def gelu_exact_kernel(
    X_ptr,          # *const bfloat16
    Y_ptr,          # *bfloat16
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    tl.store(Y_ptr + offsets, y.to(tl.bfloat16), mask=mask)


# Triton GEMM: A is (M, K_in) = x, BT is (K_out, N) = conv_out_weight.T (15360, 1024), C is (M, N) = y
# Here, x is (B, T_out3, 15360) -> flatten to M = B * T_out3, K_in = 15360, N = 1024.
@triton.jit
def gemm_mul_add_pos_kernel(
    A_ptr,            # *const bfloat16, A: (M, K_in)
    BT_ptr,           # *const bfloat16, BT: (K_out, N) where K_out=15360, N=1024
    POS_ptr,          # *const bfloat16, positional embedding: (M, N) where M=B * T_out3
    C_ptr,            # *bfloat16, output: (M, N)
    M: tl.constexpr,  # total rows in A (B * T_out3)
    K_in: tl.constexpr,  # input feature dim (15360)
    N: tl.constexpr,     # output channels (1024)
    stride_Am, stride_Ak,
    stride_BTk, stride_BTn,
    stride_Cm, stride_Cn,
    scale: tl.constexpr,  # float32 scaling factor, e.g., sqrt(d_model) = 32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile over M
    pid_n = tl.program_id(1)  # tile over N

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K_in, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K_in

        # Load A tile: shape (BLOCK_M, BLOCK_K)
        A_tile = tl.load(
            A_ptr + m_offsets[:, None] * stride_Am + k_offsets[None, :] * stride_Ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        ).to(tl.float32)

        # Load BT tile: shape (BLOCK_K, BLOCK_N)
        BT_tile = tl.load(
            BT_ptr + k_offsets[:, None] * stride_BTk + n_offsets[None, :] * stride_BTn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        ).to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, BT_tile)

    # Scale
    acc = acc * scale

    # Add scaled positional embedding: POS is (M, N)
    pos_tile = tl.load(
        POS_ptr + m_offsets[:, None] * stride_Cm + n_offsets[None, :] * stride_Cn,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0
    ).to(tl.float32)
    acc = acc + pos_tile  # broadcasting over BLOCK_M, BLOCK_N

    # Store
    tl.store(
        C_ptr + m_offsets[:, None] * stride_Cm + n_offsets[None, :] * stride_Cn,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :]
    )


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
        positional_embedding: torch.Tensor,  # (1500, 1024), bfloat16
        embed_scale: float,
    ):
        # Ensure all tensors are CUDA and bfloat16 I/O, compute in fp32 inside kernels.
        device = input_features.device
        dtype_in = input_features.dtype  # bfloat16
        Bsz, Cin, H, T = input_features.shape
        assert Cin == 1, "This implementation expects Cin=1 for conv1."

        # Conv1: (B, 1, 80, T) -> (B, 384, 80, T_out1)
        T_out1 = (T - 3) // 2 + 1
        x1 = torch.empty((Bsz, 384, H, T_out1), dtype=torch.bfloat16, device=device)
        grid_conv1 = (Bsz * H, triton.cdiv(384, 64), T_out1)
        conv2d_3x3_stride2_padding1_kernel[grid_conv1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            Bsz, 1, H, T, 384, T_out1,
            BLOCK_C=64,
        )

        # GELU after conv1 (exact)
        x1_gelu = torch.empty_like(x1, dtype=torch.bfloat16, device=device)
        N1 = x1_gelu.numel()
        gelu_exact_kernel[(N1 + 1024 - 1) // 1024,](x1, x1_gelu, N1, 1024)

        # Conv2: (B, 384, 80, T_out1) -> (B, 384, 40, T_out2)
        T_out2 = (T_out1 - 3) // 2 + 1
        x2 = torch.empty((Bsz, 384, 40, T_out2), dtype=torch.bfloat16, device=device)
        grid_conv2 = (Bsz * 40, triton.cdiv(384, 64), T_out2)
        conv2d_3x3_stride2_padding1_kernel[grid_conv2](
            x1_gelu, conv2d2_weight, conv2d2_bias, x2,
            Bsz, 384, 40, T_out1, 384, T_out2,
            BLOCK_C=64,
        )

        # GELU after conv2
        x2_gelu = torch.empty_like(x2, dtype=torch.bfloat16, device=device)
        N2 = x2_gelu.numel()
        gelu_exact_kernel[(N2 + 1024 - 1) // 1024,](x2_gelu, x2_gelu, N2, 1024)

        # Conv3: (B, 384, 40, T_out2) -> (B, 384, 20, T_out3)
        T_out3 = (T_out2 - 3) // 2 + 1
        x3 = torch.empty((Bsz, 384, 20, T_out3), dtype=torch.bfloat16, device=device)
        grid_conv3 = (Bsz * 20, triton.cdiv(384, 64), T_out3)
        conv2d_3x3_stride2_padding1_kernel[grid_conv3](
            x2_gelu, conv2d3_weight, conv2d3_bias, x3,
            Bsz, 384, 20, T_out2, 384, T_out3,
            BLOCK_C=64,
        )

        # GELU after conv3
        x3_gelu = torch.empty_like(x3, dtype=torch.bfloat16, device=device)
        N3 = x3_gelu.numel()
        gelu_exact_kernel[(N3 + 1024 - 1) // 1024,](x3_gelu, x3_gelu, N3, 1024)

        # Reshape: (B, 20, 384) -> (B, T_out3, 384*20) = (B, T_out3, 7680)
        Bsz, C, F, T_out3 = x3_gelu.shape
        x_flat = x3_gelu.permute(0, 3, 1, 2).contiguous().view(Bsz, T_out3, C * F)

        # Final GEMM and add scaled positional embedding: x_flat (B, T_out3, 15360) @ conv_out_weight.T (15360, 1024) -> y (B, T_out3, 1024)
        M = Bsz * T_out3
        K_in = 15360
        N = 1024
        A = x_flat  # bfloat16
        BT = conv_out_weight  # (1024, 15360) bfloat16
        # Slice positional embedding to first M rows: (M, N)
        POS = positional_embedding[:M, :].to(torch.bfloat16).contiguous()
        C_out = torch.empty((M, N), dtype=torch.bfloat16, device=device)

        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        gemm_mul_add_pos_kernel[grid_gemm](
            A, BT, POS, C_out,
            M, K_in, N,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C_out.stride(0), C_out.stride(1),
            scale=float(embed_scale),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # Reshape back to (B, T_out3, 1024)
        y = C_out.view(Bsz, T_out3, N)
        return y


def run(*args):
    return ModelNew()(*args)
