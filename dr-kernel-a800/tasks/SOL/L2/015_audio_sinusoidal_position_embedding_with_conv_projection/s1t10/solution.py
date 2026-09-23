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
                if valid_ih & valid_it:
                    # Load input vector for this (b, cin, ih, it): shape (BLOCK_C,)
                    # Input layout: X[b, cin, h, t] -> index = b*(Cin*H*T) + cin*(H*T) + h*T + t
                    x_ptr = X_ptr + b * (Cin * H * T) + cin * (H * T) + ih * T + it
                    x_vals = tl.load(
                        x_ptr + c_offsets,
                        mask=mask_c,
                        other=0.0
                    ).to(tl.float32)  # (BLOCK_C,)

                    # Load weight vector for these c_offsets: shape (BLOCK_C,)
                    # Weight layout: W[c, cin, kh, kw] -> index = c*(Cin*3*3) + cin*(3*3) + kh*3 + kw
                    w_ptr = W_ptr + c_offsets * (Cin * 3 * 3) + cin * (3 * 3) + kh * 3 + kt
                    w_vals = tl.load(
                        w_ptr,
                        mask=mask_c,
                        other=0.0
                    ).to(tl.float32)

                    # Fused multiply-add
                    acc += x_vals * w_vals

    # Add bias
    bias = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc += bias

    # Store output
    y_ptr = Y_ptr + b * (Cout * H * T_out) + c_offsets * (H * T_out) + oh * T_out + t_out_idx
    tl.store(y_ptr, acc.to(tl.bfloat16), mask=mask_c)


# Triton GELU (exact) kernel: elementwise over flat tensor
@triton.jit
def gelu_erf_kernel_flat(
    X_ptr, Y_ptr, SIZE,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.70710678118654752440  # 1/sqrt(2)
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    tl.store(Y_ptr + offsets, y.to(tl.bfloat16), mask=mask)


# Triton GEMM + positional add kernel:
# A: (B, t, K), BT: (K, N) where BT[k, n] = conv_out_weight[n, k] (so conv_out_weight is (N, K) original, we pass as (K, N) by using BT = conv_out_weight)
# Compute C[b, t, n] = sum_k A[b, t, k] * BT[k, n], then add POS[t, n] * scale
@triton.jit
def gemm_pos_add_kernel(
    A_ptr, BT_ptr, POS_ptr, C_ptr,
    Bsz, t, K, N,
    scale,  # float
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program computes one (b, t, tile of N)
    pid_b = tl.program_id(0)  # over B
    pid_n = tl.program_id(1)  # over tiles of N

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # Initialize accumulator for this (b, t, tile_n)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A row slice: shape (BLOCK_K,)
        a_row = tl.load(
            A_ptr + pid_b * (t * K) + k_offsets,
            mask=mask_k,
            other=0.0
        ).to(tl.float32)

        # Load BT tile: shape (BLOCK_K, BLOCK_N)
        bt_tile = tl.load(
            BT_ptr + k_offsets[:, None] * N + n_offsets[None, :],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        ).to(tl.float32)

        # Accumulate dot product across K chunk
        acc += tl.sum(a_row[:, None] * bt_tile, axis=0)

    # Add scaled positional embedding: POS is (t, N), scale is float
    pos_row = tl.load(
        POS_ptr + pid_b * N + n_offsets,
        mask=mask_n,
        other=0.0
    ).to(tl.float32) * scale
    acc += pos_row

    # Store
    tl.store(
        C_ptr + pid_b * (t * N) + n_offsets,
        acc.to(tl.bfloat16),
        mask=mask_n
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
        # Ensure contiguity and dtype
        input_features = input_features.contiguous()
        conv2d1_weight = conv2d1_weight.contiguous()
        conv2d1_bias = conv2d1_bias.contiguous()
        conv2d2_weight = conv2d2_weight.contiguous()
        conv2d2_bias = conv2d2_bias.contiguous()
        conv2d3_weight = conv2d3_weight.contiguous()
        conv2d3_bias = conv2d3_bias.contiguous()
        conv_out_weight = conv_out_weight.contiguous()
        positional_embedding = positional_embedding.contiguous()

        # Shapes
        B, Cin_in, H, T = input_features.shape
        # conv1: in_channels=Cin_in, out_channels=Cout1=384
        Cout1 = conv2d1_weight.shape[0]
        # conv2: in_channels=Cout1, out_channels=Cout2=384
        Cout2 = conv2d2_weight.shape[0]
        # conv3: in_channels=Cout2, out_channels=Cout3=384
        Cout3 = conv2d3_weight.shape[0]

        # T_out = floor((T - 3)/2 + 1) for each conv
        def T_out_from_T(T_in):
            return (T_in - 3) // 2 + 1

        T1_out = T_out_from_T(T)     # after conv1
        T2_out = T_out_from_T(T1_out)  # after conv2
        T3_out = T_out_from_T(T2_out)  # after conv3

        # Output tensors (bfloat16)
        y1 = torch.empty((B, Cout1, H, T1_out), device=input_features.device, dtype=torch.bfloat16)
        y2 = torch.empty((B, Cout2, H, T2_out), device=input_features.device, dtype=torch.bfloat16)
        y3 = torch.empty((B, Cout3, H, T3_out), device=input_features.device, dtype=torch.bfloat16)

        # Launch conv1
        grid1 = (B * H, triton.cdiv(Cout1, 64), T1_out)
        conv2d_3x3_stride2_padding1_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            B, Cin_in, H, T, Cout1, T1_out,
            BLOCK_C=64,
            num_warps=4, num_stages=2
        )

        # GELU conv1 in Triton (flat over all elements)
        y1_flat = y1.reshape(-1)


def run(*args):
    return ModelNew()(*args)
