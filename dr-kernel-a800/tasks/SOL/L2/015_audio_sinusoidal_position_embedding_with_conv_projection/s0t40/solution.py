import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: Conv2d 3x3, stride=2, padding=1, generic IC -> OC
# X: [B, IC, F_in, T_in], W: [OC, IC, 3, 3], bias: [OC], Y: [B, OC, F_out, T_out]
@triton.jit
def conv3x3_s2_p1_gelu(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, IC, F_in, T_in, OC, F_out, T_out,
    x_sN, x_sC, x_sF, x_sT,
    w_sOC, w_sIC, w_sKH, w_sKW,
    y_sN, y_sOC, y_sF, y_sT,
    BLOCK_F: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # Launch grid: 1D over (b, oc, f_block, t_block)
    total_blocks = B * OC * ((F_out + BLOCK_F - 1) // BLOCK_F) * ((T_out + BLOCK_T - 1) // BLOCK_T)
    pid = tl.program_id(0)

    # Decode pid into (b, oc, f_block, t_block)
    grid_f_blocks = (F_out + BLOCK_F - 1) // BLOCK_F
    grid_t_blocks = (T_out + BLOCK_T - 1) // BLOCK_T
    b = pid // (OC * grid_f_blocks * grid_t_blocks)
    rem = pid % (OC * grid_f_blocks * grid_t_blocks)
    oc = rem // (grid_f_blocks * grid_t_blocks)
    f_block = rem % (grid_f_blocks * grid_t_blocks) // 1
    t_block = rem % grid_t_blocks

    # Tile indices
    f_out_start = f_block * BLOCK_F
    t_out_start = t_block * BLOCK_T
    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)[:, None]  # [BF, 1]
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)[None, :]  # [1, BT]
    out_mask = (f_out_idx < F_out) & (t_out_idx < T_out)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    for ic in range(0, IC):
        for kh in range(0, 3):
            for kw in range(0, 3):
                # stride=2, padding=1: input index = (out + 1) * 2 - 1 + kh
                f_in_idx = f_out_idx * 2 + 1 - 1 + kh  # [BF, 1]
                t_in_idx = t_out_idx * 2 + 1 - 1 + kw  # [1, BT]
                in_mask = (f_in_idx >= 0) & (f_in_idx < F_in) & (t_in_idx >= 0) & (t_in_idx < T_in)
                # Load X[b, ic, f_in, t_in]
                x_ptrs = X_ptr + b * x_sN + ic * x_sC + f_in_idx * x_sF + t_in_idx * x_sT
                x_vals = tl.load(x_ptrs, mask=in_mask, other=0.0)  # [BF, BT]
                # Load W[oc, ic, kh, kw]
                w_ptrs = W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW
                w_val = tl.load(w_ptrs)  # scalar
                # Accumulate
                acc += x_vals * w_val

    # Add bias
    b_ptr = BIAS_ptr + oc
    bias_val = tl.load(b_ptr)
    acc += bias_val

    # GELU: exact tanh-based approximation to match PyTorch default
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = acc * acc * acc
    inner = c0 * (acc + c1 * x3)
    gelu = 0.5 * acc * (1.0 + tl.math.tanh(inner))

    # Store
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out_idx * y_sF + t_out_idx * y_sT
    tl.store(y_ptrs, gelu, mask=out_mask)


# Triton matmul: A [M, K] @ B [K, N] -> C [M, N]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    a_sM, a_sK,
    b_sK, b_sN,
    c_sM, c_sN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + offs_m[:, None] * a_sM + offs_k[None, :] * a_sK,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        b = tl.load(
            B_ptr + offs_k[:, None] * b_sK + offs_n[None, :] * b_sN,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    tl.store(
        C_ptr + offs_m[:, None] * c_sM + offs_n[None, :] * c_sN,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


# Triton elementwise: scale and add positional embedding
@triton.jit
def scale_add_pos_embed(
    X_ptr, POS_ptr, Y_ptr, SCALE, N, D,
    x_sN, x_sT, x_sD,
    y_sN, y_sT, y_sD,
    BLOCK: tl.constexpr,
):
    total = N * D
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    n = offs // D
    d = offs % D

    x_ptrs = X_ptr + n * x_sN + d * x_sD
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0) * SCALE

    pos_ptrs = POS_ptr + d * POS_ptr.stride(1)  # POS is [N, D], use strides along D
    pos_vals = tl.load(pos_ptrs, mask=mask, other=0.0)

    y_vals = x_vals + pos_vals

    y_ptrs = Y_ptr + n * y_sN + d * y_sD
    tl.store(y_ptrs, y_vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args correspond to: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale

        # Ensure Triton is available
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")

        # Extract tensors
        input_features = args[0]  # [B, 1, 80, time_dim]
        conv2d1_weight = args[1]  # [OC, IC, 3, 3] = [384, 1, 3, 3]
        conv2d1_bias = args[2]    # [OC]
        conv2d2_weight = args[3]  # [384, 384, 3, 3]
        conv2d2_bias = args[4]    # [384]
        conv2d3_weight = args[5]  # [384, 384, 3,


def run(*args):
    return ModelNew()(*args)
