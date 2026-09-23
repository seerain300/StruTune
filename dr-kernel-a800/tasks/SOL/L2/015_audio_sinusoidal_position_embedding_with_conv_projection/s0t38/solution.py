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
    f_block = rem // grid_t_blocks % grid_f_blocks
    t_block = rem % grid_t_blocks

    # Compute output indices for this tile
    f_out_start = f_block * BLOCK_F
    t_out_start = t_block * BLOCK_T
    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)[:, None]  # [BLOCK_F, 1]
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)[None, :]  # [1, BLOCK_T]
    out_mask = (f_out_idx < F_out) & (t_out_idx < T_out)

    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    for ic in range(IC):
        for kh in range(3):
            for kw in range(3):
                # Input coordinates with padding=1, stride=2
                f_in = f_out_idx * 2 + kh - 1  # [BLOCK_F, 1]
                t_in = t_out_idx * 2 + kw - 1  # [1, BLOCK_T]
                valid = (f_in >= 0) & (f_in < F_in) & (t_in >= 0) & (t_in < T_in)
                # Compute input pointers: (b, ic, f_in, t_in)
                x_ptrs = X_ptr + b * x_sN + ic * x_sC + f_in * x_sF + t_in * x_sT
                x_vals = tl.load(x_ptrs, mask=valid & out_mask, other=0.0)
                # Weight scalar: (oc, ic, kh, kw)
                w_ptrs = W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW
                w_val = tl.load(w_ptrs)
                acc += x_vals * w_val

    # Add bias
    b_val = tl.load(BIAS_ptr + oc)
    acc += b_val

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    gelu = acc * 0.5 * (1.0 + tl.math.erf(acc / inv_sqrt2))

    # Store output: (b, oc, f_out, t_out)
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out_idx * y_sF + t_out_idx * y_sT
    tl.store(y_ptrs, gelu, mask=out_mask)


# Triton matmul kernel: Y[M, N] = A[M, K] @ B[K, N]
@triton.jit
def matmul(
    A_ptr, B_ptr, Y_ptr,
    M, N, K,
    a_sM, a_sK,
    b_sK, b_sN,
    y_sM, y_sN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid // triton.cdiv(N, BLOCK_N) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = (pid % triton.cdiv(N, BLOCK_N)) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:, None] * a_sM + offs_k[None, :] * a_sK
        b_ptrs = B_ptr + offs_k[:, None] * b_sK + offs_n[None, :] * b_sN
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    y_ptrs = Y_ptr + offs_m[:, None] * y_sM + offs_n[None, :] * y_sN
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton elementwise kernel: Y[:] = X[:] * scale
@triton.jit
def scale_vec(
    X_ptr, Y_ptr, N,
    scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = x * scale
    tl.store(Y_ptr + offs, y, mask=mask)


# Triton elementwise kernel: Y[m, n] = X[m, n] + P[n]
@triton.jit
def add_pos_emb_vec(
    X_ptr, P_ptr, Y_ptr,
    M, N,
    x_sM, x_sN,
    y_sM, y_sN,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    total = M * N
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    n_idx = offs % N
    m_idx = offs // N
    x_ptrs = X_ptr + m_idx * x_sM + n_idx * x_sN
    p_ptrs = P_ptr + n_idx
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
    p_vals = tl.load(p_ptrs, mask=mask, other=0.0)
    y_vals = x_vals + p_vals
    y_ptrs = Y_ptr + m_idx * y_sM + n_idx * y_sN
    tl.store(y_ptrs, y_vals, mask=mask)


class ModelNew(nn.Module):
    def forward(self, *args):
        # args: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        device = args[0].device
        input_features = args[0].contiguous().float()  # [B, 1, 80, T_in]
        conv2d1_weight = args[1].contiguous().float()  # [OC, IC, 3, 3] IC=1
        conv2d1_bias = args[2].contiguous().float()    # [OC]
        conv2d2_weight = args[3].contiguous().float()  # [OC, OC, 3, 3]
        conv2d2_bias = args[4].contiguous().float()    # [OC]
        conv2d3_weight = args[5].contiguous().float()  # [OC, OC, 3, 3]
        conv2d3_bias = args[6].contiguous().float()    # [OC]
        conv_out_weight = args[7].contiguous().float() # [d_model, conv_out_dim] = [1024, 3840]
        positional_embedding = args[8].contiguous().float() # [max_source_positions, d_model]
        embed_scale = float(args[9])  # python float

        B, IC_in, F_in, T_in = input_features.shape

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        OC1 = conv2d1_weight.shape[0]
        x1 = input_features
        w1 = conv2d1_weight
        b1 = conv2d1_bias
        F_out1 = (F_in + 2 * 1 - 3) // 2 + 1
        T_out1 = (T_in + 2 * 1 - 3) // 2 + 1
        y1 = torch.empty((B, OC1, F_out1, T_out1), device=device, dtype=torch.float32)
        grid1 = (B * OC1 * triton.cdiv(F_out1, 32) * triton.cdiv(T_out1, 32),)
        conv3x3_s2_p1_gelu[grid1](
            x1, w1, b1, y1,
            B, 1, F_in, T_in, OC1, F_out1, T_out1,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_F=32, BLOCK_T=32
        )

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        x2 = y1
        OC2 = conv2d2_weight.shape[0]
        w2 = conv2d2_weight
        b2 = conv2d2_bias
        F_in2 = F_out1
        T_in2 = T_out1
        F_out2 = (F_in2 + 2 * 1 - 3) // 2 + 1
        T_out2 = (T_in2 + 2 * 1 - 3) // 2 + 1
        y2 = torch.empty((B, OC2, F_out2, T_out2), device=device, dtype=torch.float32)
        grid2 = (B * OC2 * triton.cdiv(F_out2, 32) * triton.cdiv(T_out2, 32),)
        conv3x3_s2_p1_gelu[grid2](
            x2, w2, b2, y2,
            B, OC2, F_in2, T_in2, OC2, F_out2, T_out2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_F=32, BLOCK_T=32
        )

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        x3 = y2
        OC3


def run(*args):
    return ModelNew()(*args)
