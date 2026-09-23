import math
import torch
import triton
import triton.language as tl


# =========================
# Triton kernels
# =========================

@triton.jit
def conv1d_per_channel_kernel(
    u_ptr,                # *const float, input [B, C, S]
    w_ptr,                # *const float, weight [C, 1, F]
    y_ptr,                # *float, output [B, C, L_out], L_out = S - F + 1
    B, C, S, F,           # int
    u_stride_b, u_stride_c, u_stride_s,
    w_stride_c, w_stride_f,
    y_stride_b, y_stride_c, y_stride_l,
    L_out,                # int, output length
    BLOCK_L: tl.constexpr
):
    b = tl.program_id(0)  # batch index
    c = tl.program_id(1)  # channel index
    pid_l = tl.program_id(2)  # tile index along output length
    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = l_offsets < L_out

    acc = tl.zeros([BLOCK_L], dtype=tl.float32)
    # Sum over filter taps f in [0, F)
    for f in range(0, F):
        src_s = l_offsets + f  # corresponds to u[b, c, l+f]
        mask_s = src_s < S
        # Load input element u[b, c, src_s]
        u_offset = b * u_stride_b + c * u_stride_c + src_s * u_stride_s
        u_val = tl.load(u_ptr + u_offset, mask=mask_s, other=0.0)
        # Load weight w[c, 0, f]
        w_val = tl.load(w_ptr + c * w_stride_c + f * w_stride_f)
        acc += u_val * w_val

    # Store to output y[b, c, l_offsets]
    y_offset = b * y_stride_b + c * y_stride_c + l_offsets * y_stride_l
    tl.store(y_ptr + y_offset, acc, mask=mask_l)


@triton.jit
def ln_forward_kernel(
    x_ptr,                # *const float, input [M, D]
    weight_ptr,           # *const float, [D]
    bias_ptr,             # *const float, [D]
    y_ptr,                # *float, output [M, D]
    M, D, eps,            # int, float
    stride_xm, stride_xd,
    stride_ym, stride_yd,
    BLOCK_SIZE: tl.constexpr
):
    m = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D
    x = tl.load(x_ptr + m * stride_xm + cols * stride_xd, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / D
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)
    y = x_centered * inv_std
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    y = y * w + b
    tl.store(y_ptr + m * stride_ym + cols * stride_yd, y, mask=mask)


@triton.jit
def matmul_bias_kernel(
    A_ptr,                # *const float, [M, K]
    Bt_ptr,               # *const float, [K, N] (transposed weight)
    bias_ptr,             # *const float, [N]
    C_ptr,                # *float, [M, N]
    M, K, N, beta,        # int, float
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < M
    mask_n = rn < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mask_k = rk < K
        a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b_ptrs = Bt_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(bias_ptr + rn, mask=mask_n, other=0.0)
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc + bias[None, :], mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def exp_mod_kernel(
    v_ptr,                # *const float, input [H, D] flattened rows
    delta_ptr,            # *const float, [D]
    shift,                # float
    H, D,
    stride_vh, stride_vd,
    BLOCK_SIZE: tl.constexpr
):
    h = tl.program_id(0)  # row index
    d_group = tl.program_id(1)  # tile index over columns
    d_start = d_group * BLOCK_SIZE
    d_offsets = d_start + tl.arange(0, BLOCK_SIZE)
    mask_d = d_offsets < D
    v_row_ptr = v_ptr + h * stride_vh + d_offsets * stride_vd
    v = tl.load(v_row_ptr, mask=mask_d, other=0.0)
    delta = tl.load(delta_ptr + d_offsets, mask=mask_d, other=0.0)
    t = h / (H - 1) if H > 1 else 0.0
    mod = tl.exp(-t * tl.abs(delta)) + shift
    v = v * mod
    tl.store(v_row_ptr, v, mask=mask_d)


# =========================
# ModelNew forward
# =========================

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Args order follows original run function. We treat hidden_states first, then parameters.
        hidden_states = args[0]
        norm1_weight = args[1]
        norm1_bias = args[2]
        norm2_weight = args[3]
        norm2_bias = args[4]
        in_proj_weight = args[5]        # [inner_width, d_model]
        in_proj_bias = args[6]          # [inner_width]
        short_conv_weight = args[7]     # [C, 1, F] where C=inner_width, F=3
        short_conv_bias = args[8]       # unused
        filter_linear1_weight = args[9] # not used
        filter_linear1_bias = args[10]  # not used
        sin_freq = args[11]             # not used
        filter_linear2_weight = args[12]  # not used
        filter_linear2_bias = args[13]   # not used
        filter_linear3_weight = args[14]  # not used
        filter_linear3_bias = args[15]   # not used
        filter_linear_final_weight = args[16]  # not used
        filter_bias = args[17]          # not used
        exp_mod_deltas = args[18]       # [1, 1, d_model]
        out_proj_weight = args[19]      # [d_model, d_model]
        out_proj_bias = args[20]        # [d_model]
        mlp_fc1_weight = args[21]       # not used
        mlp_fc1_bias = args[22]         # not used
        mlp_fc2_weight = args[23]       # not used
        mlp_fc2_bias = args[24]         # not used
        layer_norm_eps = float(args[25]) # 1e-5


def run(*args):
    return ModelNew()(*args)
