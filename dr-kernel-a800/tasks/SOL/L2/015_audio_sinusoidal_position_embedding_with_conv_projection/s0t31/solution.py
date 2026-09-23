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
    f_block = rem % (grid_f_blocks * grid_t_blocks) // grid_t_blocks
    t_block = rem % (grid_f_blocks * grid_t_blocks) % grid_t_blocks

    # Output tile indices
    f_out_start = f_block * BLOCK_F
    t_out_start = t_block * BLOCK_T
    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)[:, None]  # [BF, 1]
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)[None, :]  # [1, BT]
    out_mask = (f_out_idx < F_out) & (t_out_idx < T_out)

    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Compute receptive field for padding=1 and stride=2
    # Input indices for each (kh, kw): f_in = f_out*2 + (1 - kh), t_in = t_out*2 + (1 - kw)
    for ic in range(0, IC):
        for kh in range(3):
            f_in_idx = f_out_idx * 2 + (1 - kh)  # [BF, 1]
            for kw in range(3):
                t_in_idx = t_out_idx * 2 + (1 - kw)  # [1, BT]
                # Bounds check for input
                in_mask = (f_in_idx >= 0) & (f_in_idx < F_in) & (t_in_idx >= 0) & (t_in_idx < T_in)
                # Compute input pointers
                x_ptrs = X_ptr + b * x_sN + ic * x_sC + f_in_idx * x_sF + t_in_idx * x_sT
                x_tile = tl.load(x_ptrs, mask=out_mask & in_mask, other=0.0)  # [BF, BT]

                # Load weights for this (oc, ic, kh, kw)
                w_ptrs = W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW
                w_val = tl.load(w_ptrs)  # scalar

                # Accumulate: acc += x_tile * w_val
                acc += x_tile * w_val

    # Add bias
    b_val = tl.load(BIAS_ptr + oc)
    acc = acc + b_val

    # GELU activation: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    # Store to output
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out_idx * y_sF + t_out_idx * y_sT
    tl.store(y_ptrs, gelu, mask=out_mask)


# Triton GEMM for linear projection: X: [M, K], W: [K, N] -> Y: [M, N]
@triton.jit
def matmul_gemv(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    x_sM, x_sK,
    w_sK, w_sN,
    y_sM, y_sN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
    m_mask = m_offsets < M
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BK]
        k_mask = k_offsets < K

        # Load X tile: [BM, BK]
        x_ptrs = X_ptr + m_offsets[:, None] * x_sM + k_offsets[None, :] * x_sK
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load W tile: [BK, BN]
        w_ptrs = W_ptr + k_offsets[:, None] * w_sK + n_offsets[None, :] * w_sN
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Accumulate: acc += x_tile @ w_tile
        acc += tl.dot(x_tile, w_tile)

    # Store result
    y_ptrs = Y_ptr + m_offsets[:, None] * y_sM + n_offsets[None, :] * y_sN
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


class ModelNew(nn.Module):
    def forward(self, *args):
        # Extract tensors
        input_features = args[0]   # [B, 1, 80, time_dim], bfloat16
        conv2d1_weight = args[1]   # [384, 1, 3, 3], bfloat16
        conv2d1_bias = args[2]     # [384], bfloat16
        conv2d2_weight = args[3]   # [384, 384, 3, 3], bfloat16
        conv2d2_bias = args[4]     # [384], bfloat16
        conv2d3_weight = args[5]   # [384, 384, 3, 3], bfloat16
        conv2d3_bias = args[6]     # [384], bfloat16
        conv_out_weight = args[7]  # [1024, 3840], bfloat16
        positional_embedding = args[8]  # [max_source_positions, 1024], bfloat16
        embed_scale = float(args[9])     # python float

        # Ensure float32 for Triton kernels
        B, C_in, F_in, T_in = input_features.shape
        x = input_features.contiguous().float()  # [B, 1, 80, T_in]

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        OC1 = conv2d1_weight.shape[0]
        w1 = conv2d1_weight.contiguous().float()   # [384, 1, 3, 3]
        b1 = conv2d1_bias.contiguous().float()     # [384]
        F_out1 = (F_in + 2 * 1 - 3) // 2 + 1
        T_out1 = (T_in + 2 * 1 - 3) // 2 + 1
        y1 = torch.empty((B, OC1, F_out1, T_out1), device=x.device, dtype=torch.float32)
        grid1 = (B * OC1 * triton.cdiv(F_out1, 32) * triton.cdiv(T_out1, 32),)
        conv3x3_s2_p1_gelu[grid1](
            x, w1, b1, y1,
            B, 1, F_in, T_in, OC1, F_out1, T_out1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_F=32, BLOCK_T=32
        )

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        x2 = y1
        OC2 = conv2d2_weight.shape[0]
        w2 = conv2d2_weight.contiguous().float()   # [384, 384, 3, 3]
        b2 = conv2d2_bias.contiguous().float()     # [384]
        F_in2 = F_out1
        T_in2 = T_out1
        F_out2 = (F_in2 + 2 * 1 - 3) // 2 + 1
        T_out2 = (T_in2 + 2 * 1 - 3) // 2 + 1
        y2 = torch.empty((B, OC2, F_out2, T_out2), device=x.device, dtype=torch.float32)
        grid2 = (B * OC2 * triton.cdiv(F_out2, 32) * triton.cdiv(T_out2, 32),)
        conv3x3_s2_p1_gelu[grid2](
            x2, w2, b2, y2,
            B, OC1, F_in2, T_in2, OC2, F_out2, T_out2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_F=32, BLOCK_T=32
        )

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        x3 = y2
        OC3 = conv2d3_weight.shape[0]
        w3 = conv2d3_weight.contiguous().float()   # [384, 384, 3, 3]
        b3 = conv2d3_bias.contiguous().float()     # [384]
        F_in3 = F_out2
        T_in3 = T_out2
        F_out3 = (F_in3 + 2 * 1 - 3) // 2 + 1
        T_out3 = (T_in3 + 2 * 1 - 3) // 2 + 1
        y3 = torch.empty((B, OC3, F_out3, T_out3), device=x.device, dtype=torch.float32)
        grid3 = (B * OC3 * triton.cdiv(F_out3, 32) * triton.cdiv(T_out3, 32),)
        conv3x3_s2_p1_gelu[grid3](
            x3, w3, b3, y3,
            B, OC2, F_in3, T_in3, OC3, F_out3, T_out3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            w3.stride(0), w3.stride(1), w3.stride(2), w3.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_F=32, BLOCK_T=32
        )

        # Reshape: (batch, channels, F, T) -> (batch, T, channels*F)
        # y3: [B, 384, F_out3, T_out3]
        b, c, f, t = y3.shape
        y3_perm = y3.permute(0, 3, 1, 2).contiguous()  # [B, T_out3, 384, F_out3]
        D = 1024
        C = c  # 384
        F_out3 = f
        T_out3 = t
        x_lin = y3_perm.view(b, t, C * f)  # [B, T_out3, 384*F_out3]

        # Linear projection to d_model
        # X_lin: [M, K] where M = B * T_out3, K = 384*F_out3
        M = x_lin.numel() // D
        K = C * f
        assert M * D == x


def run(*args):
    return ModelNew()(*args)
