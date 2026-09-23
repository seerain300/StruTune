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
                if valid_ih and valid_it:
                    # compute input indices: X[b, cin, ih, it]
                    x_off = b * (Cin * H * T) + cin * (H * T) + ih * T + it
                    x_val = tl.load(X_ptr + x_off).to(tl.float32)
                    # load corresponding weights for this cout tile: W[c_offsets, cin, kh, kt]
                    w_off = c_offsets * (Cin * 3 * 3) + cin * (3 * 3) + kh * 3 + kt
                    w_vec = tl.load(W_ptr + w_off, mask=mask_c, other=0.0).to(tl.float32)
                    acc += x_val * w_vec

    # add bias
    bias_vec = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc += bias_vec

    # store result to Y[b, c_offsets, oh, t_out_idx]
    for j in range(BLOCK_C):
        cout_j = c_start + j
        if cout_j < Cout:
            y_off = b * (Cout * H * T_out) + cout_j * (H * T_out) + oh * T_out + t_out_idx
            tl.store(Y_ptr + y_off, acc[j].to(tl.bfloat16))


# Triton GELU kernel (exact, erf-based): apply to a 1D buffer of length N
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
    inv_sqrt2 = 0.7071067811865476
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    tl.store(Y_ptr + offsets, y.to(tl.bfloat16), mask=mask)


# Triton GEMM + add scaled positional embedding
# A: (M, K_in) = flattened x of shape (B * T_out3, 15360), bfloat16
# BT: (K_out, N) = conv_out_weight.T of shape (15360, 1024), bfloat16
# POS: (M, N) = positional embedding sliced to first M rows, bfloat16
# C: (M, N) = output y, bfloat16
@triton.jit
def gemm_add_pos_kernel(
    A_ptr,            # *const bfloat16
    BT_ptr,           # *const bfloat16
    POS_ptr,          # *const bfloat16
    C_ptr,            # *bfloat16
    M: tl.constexpr,  # total rows in A
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K_in, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K_in

        A_tile = tl.load(
            A_ptr + m_offsets[:, None] * stride_Am + k_offsets[None, :] * stride_Ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        ).to(tl.float32)

        BT_tile = tl.load(
            BT_ptr + k_offsets[:, None] * stride_BTk + n_offsets[None, :] * stride_BTn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        ).to(tl.float32)

        acc += tl.dot(A_tile, BT_tile)

    # scale
    acc = acc * scale

    # add scaled positional embedding
    pos_tile = tl.load(
        POS_ptr + m_offsets[:, None] * stride_Cm + n_offsets[None, :] * stride_Cn,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0
    ).to(tl.float32)
    acc = acc + pos_tile  # broadcasting

    # store
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
        embed_scale: float,  # sqrt(d_model) = 32.0
    ):
        # Ensure CUDA tensors
        assert input_features.is_cuda and conv2d1_weight.is_cuda and conv2d2_weight.is_cuda and conv2d3_weight.is_cuda \
               and conv_out_weight.is_cuda and positional_embedding.is_cuda, "All tensors must be on CUDA"

        Bsz, Cin, H, T = input_features.shape
        # First conv: (B,1,80,T) -> (B,384,80,T_out1)
        T_out1 = (T - 3) // 2 + 1
        x1 = torch.empty((Bsz, conv2d1_weight.shape[0], H, T_out1), dtype=torch.bfloat16, device=input_features.device)
        # Launch conv kernel: grid = (B*H, tiles over Cout, T_out1)
        BLOCK_C = 64
        grid1 = (Bsz * H, triton.cdiv(conv2d1_weight.shape[0], BLOCK_C), T_out1)
        conv2d_3x3_stride2_padding1_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            Bsz, Cin, H, T, conv2d1_weight.shape[0], T_out1,
            BLOCK_C,
        )
        # GELU after conv1
        x1 = x1.contiguous()
        x1_gelu = torch.empty_like(x1)
        N1 = x1.numel()
        BLOCK_GELU = 1024
        gelu_exact_kernel[(N1 + BLOCK_GELU - 1) // BLOCK_GELU,](
            x1, x1_gelu, N1, BLOCK_GELU
        )

        # Second conv: (B,384,80,T_out1) -> (B,384,80,T_out2)
        T_out2 = (T_out1 - 3) // 2 + 1
        x2 = torch.empty((Bsz, conv2d2_weight.shape[0], H, T_out2), dtype=torch.bfloat16, device=input_features.device)
        grid2 = (Bsz * H, triton.cdiv(conv2d2_weight.shape[0], BLOCK_C), T_out2)
        conv2d_3x3_stride2_padding1_kernel[grid2](
            x1_gelu, conv2d2_weight, conv2d2_bias, x2,
            Bsz, conv2d2_weight.shape[0], H, T_out1, conv2d2_weight.shape[0], T_out2,
            BLOCK_C,
        )
        # GELU after conv2
        x2 = x2.contiguous()
        x2_gelu = torch.empty_like(x2)
        N2 = x2.numel()
        gelu_exact_kernel[(N2 + BLOCK_GELU - 1) // BLOCK_GELU,](x2, x2_gelu, N2, BLOCK_GELU)

        # Third conv: (B,384,80,T_out2) -> (B,384,80,T_out3)
        T_out3 = (T_out2 - 3) // 2 + 1
        x3 = torch.empty((Bsz, conv2d3_weight.shape[0], H, T_out3), dtype=torch.bfloat16, device=input_features.device)
        grid3 = (Bsz * H, triton.cdiv(conv2d3_weight.shape[0], BLOCK_C), T_out3)
        conv2d_3x3_stride2_padding1_kernel[grid3](
            x2_gelu, conv2d3_weight, conv2d3_bias, x3,
            Bsz, conv2d3_weight.shape[0], H, T_out2, conv2d3_weight.shape[0], T_out3,
            BLOCK_C,
        )
        # GELU after conv3
        x3 = x3.contiguous()
        x3_gelu = torch.empty_like(x3)
        N3 = x3.numel()
        gelu_exact_kernel[(N3 + BLOCK_GELU - 1) // BLOCK_GELU,](x3, x3_gelu, N3, BLOCK_GELU)

        # Reshape to (B, t, C*F) where C=384, F=40
        # x3_gelu: (B, 384, 80, T_out3) -> (B, T_out3, 384*40) = (B, T_out3, 15360)
        Bsz, Cout, H3, T_out3 = x3_gelu.shape
        x_flat = x3_gelu.permute(0, 3, 1, 2).contiguous().view(Bsz, T_out3, Cout * H3)

        # Final GEMM: x_flat (B, T_out3, 15360) @ conv_out_weight.T (15360, 1024) -> y (B, T_out3, 1024)
        # A: (M, K_in) with M = B * T_out3, K_in = 15360
        M = Bsz * T_out3
        K_in = conv_out_weight.shape[1]  # 15360
        N = conv_out_weight.shape[0]     # 1024
        A = x_flat  # bfloat16
        BT = conv_out_weight.transpose(0, 1).contiguous()  # (15360, 1024), bfloat16

        # POS: (M, N) = slice positional_embedding to first M rows
        POS = positional_embedding[:M, :].contiguous().to(torch.bfloat16)

        C_out = torch.empty((M, N), dtype=torch.bfloat16, device=input_features.device)

        # Strides for A, BT, C
        stride_Am = K_in
        stride_Ak = 1
        stride_BTk = N
        stride_BTn = 1
        stride_Cm = N
        stride_Cn = 1

        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        gemm_add_pos_kernel[grid_gemm](
            A, BT, POS, C_out,
            M, K_in, N,
            stride_Am, stride_Ak,
            stride_BTk, stride_BTn,
            stride_Cm, stride_Cn,
            float(embed_scale),  # scale in fp32
            BLOCK_M, BLOCK_N, BLOCK_K,
        )

        # Reshape back to (B, t, N)
        y = C_out.view(Bsz, T_out3, N)

        return y


def run(*args):
    return ModelNew()(*args)
