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

    # Compute output indices for this tile
    f_out_start = f_block * BLOCK_F
    t_out_start = t_block * BLOCK_T
    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)[:, None]  # [BF, 1]
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)[None, :]  # [1, BT]
    out_mask = (f_out_idx < F_out) & (t_out_idx < T_out)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    # padding=1, stride=2
    for ic in range(0, IC):
        for kh in range(3):
            for kw in range(3):
                # Input coordinates with padding=1 and stride=2
                f_in_idx = f_out_idx * 2 + (1 - kh)  # [BF, 1]
                t_in_idx = t_out_idx * 2 + (1 - kw)  # [1, BT]
                in_mask = (f_in_idx >= 0) & (f_in_idx < F_in) & (t_in_idx >= 0) & (t_in_idx < T_in)
                x_ptrs = X_ptr + b * x_sN + ic * x_sC + f_in_idx * x_sF + t_in_idx * x_sT
                x_tile = tl.load(x_ptrs, mask=out_mask & in_mask, other=0.0)  # [BF, BT]

                # Load corresponding weights (scalar per (oc, ic, kh, kw))
                w_ptr = W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW
                w_val = tl.load(w_ptr)  # scalar

                # Accumulate
                acc += x_tile * w_val

    # Add bias
    b_ptr = BIAS_ptr + oc
    b_val = tl.load(b_ptr)
    acc += b_val

    # GELU
    # Use exact GELU: 0.5*x*(1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)
    x = acc
    acc = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    # Store result
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out_idx * y_sF + t_out_idx * y_sT
    tl.store(y_ptrs, acc, mask=out_mask)


# Triton matmul kernel: X[M, K] @ W[N, K] -> Y[M, N], no bias
# X_ptr: [M*K], W_ptr: [N*K], Y_ptr: [M*N]
@triton.jit
def matmul_gemv(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    x_stride, w_stride, y_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_idx = m_start + tl.arange(0, BLOCK_M)[:, None]  # [BM, 1]
    n_idx = n_start + tl.arange(0, BLOCK_N)[None, :]  # [1, BN]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [BK]

        # Load X tiles: shape (BM, BK) and (BK, BN)
        x_ptrs = X_ptr + m_idx * x_stride[:, None] + k_idx[None, :]
        w_ptrs = W_ptr + n_idx * w_stride[None, :] + k_idx[:, None]

        x_mask = (m_idx[:, 0] < M) & (k_idx < K)
        w_mask = (n_idx[0, :] < N) & (k_idx < K)

        x_tile = tl.load(x_ptrs, mask=x_mask[:, None], other=0.0)  # [BM, BK]
        w_tile = tl.load(w_ptrs, mask=w_mask[None, :], other=0.0)  # [BK, BN]

        acc += tl.dot(x_tile, w_tile)  # [BM, BN]

    # Store results
    y_ptrs = Y_ptr + m_idx * y_stride[:, None] + n_idx[None, :]
    y_mask = (m_idx[:, 0] < M) & (n_idx[0, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


# Triton elementwise kernel: scale + add positional embedding
# Y is [M*N], P is [T3*N, N] (i.e., row-major per position along dim0, channels along dim1)
@triton.jit
def scale_add_pos_embed_1d(
    Y_ptr, P_ptr, NUMEL, D, scale, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUMEL

    # Load Y
    y = tl.load(Y_ptr + offsets, mask=mask, other=0.0)

    # Compute seq_pos and channel for each offset
    seq_pos = offsets // D
    channel = offsets % D

    # Flatten seq_pos to index rows of P: row = seq_pos * D + channel
    p_rows = seq_pos * D + channel
    # For valid offsets, seq_pos < T3 and channel < D -> p_rows in [0, T3*D)
    p_ptrs = P_ptr + p_rows * D + channel  # each row is of length D

    p_vals = tl.load(P_ptrs, mask=mask, other=0.0)

    y = y + p_vals
    y = y * scale

    tl.store(Y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Ensure Triton is available
        if not TRITON_AVAILABLE:
            # Fallback: compute with PyTorch (not used in evaluation which requires Triton)
            raise RuntimeError("Triton not available")

        # We must run everything through Triton. input_features shape is [B, 1, 80, time_dim]
        B, IC_in, F_in, T_in = input_features.shape
        x = input_features.contiguous().float()

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        OC1 = conv2d1_weight.shape[0]
        w1 = conv2d1_weight.contiguous().float()  # [OC1, 1, 3, 3]
        b1 = conv2d1_bias.contiguous().float()    # [OC1]
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
        x2 = y1  # [B, 384, F_out1, T_out1]
        OC2 = conv2d2_weight.shape[0]
        w2 = conv2d2_weight.contiguous().float()   # [OC2, 384, 3, 3] = [384, 384, 3, 3]
        b2 = conv2d2_bias.contiguous().float()     # [384]
        F_in2 = F_out1
        T_in2 = T_out1
        F_out2 = (F_in2 + 2 * 1 - 3) // 2 + 1
        T_out2 = (T_in2 + 2 * 1 - 3) // 2 + 1
        y2 = torch.empty((B, OC2, F_out2, T_out2), device=x.device, dtype=torch.float32)
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
        x3 = y2  # [B, 384, F_out2, T_out2]
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
            B, OC3, F_in3, T_in3, OC3, F_out3, T_out3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            w3.stride(0), w3.stride(1), w3.stride(2), w3.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_F=32, BLOCK_T=32
        )

        # Reshape: (batch, channels, F, T) -> (batch, T, channels*F)
        b, c, f, t = y3.shape
        y3_perm = y3.permute(0, 3, 1, 2).contiguous()  # [B, t, c, f]
        x_lin = y3_perm.view(b, t, c * f)  # [B, t, 384 * f]

        # Flatten for matmul
        M = x_lin.shape[0] * x_lin.shape[1]  # B * t
        C = conv_out_weight.shape[0]  # 1024
        K = x_lin.shape[2]             # 384 * f
        # Make inputs for Triton matmul: X[M, K], W[C, K]
        X = x_lin.reshape(M, K).contiguous()  # [M, K]
        W = conv_out_weight.t().contiguous()  # [K, C] -> need [C, K] again? We have conv_out_weight: [C, K], so keep as is
        # W must be [N, K] where N=C, K=K. conv_out_weight is already [C, K]
        # For matmul_gemv, W_ptr points to [N, K], which is [C, K]
        Y = torch.empty((M, C), device=X.device, dtype=torch.float32)

        # Launch matmul
        # We can set BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
        grid_mat = (triton.cdiv(M, 128), triton.cdiv(C, 128))
        matmul_gemv[grid_mat](
            X.flatten(), W.flatten(), Y.flatten(),
            M, C, K,
            X.stride(0), W.stride(0), Y.stride(0),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        )

        # Now Y is [M, C] with M = B * t. We need to apply scale and positional embedding
        # Scale = sqrt(d_model) = 32.0
        scale = float(embed_scale)
        M_ = Y.numel()
        C_ = C
        D = C_  # channels dimension for pos embedding
        # positional_embedding is [max_source_positions, D], dtype may be bfloat16; we need float32
        P = positional_embedding.float()
        # Reshape Y to 1D
        Y_flat = Y.reshape(-1)  # [M_*C_]

        # Launch scale + add positional embedding
        BLOCK_E = 4096
        grid_e = (triton.cdiv(M_, BLOCK_E),)
        scale_add_pos_embed_1d[grid_e](
            Y_flat, P.reshape(-1), M_, D, scale, BLOCK=BLOCK_E,
        )

        # Reshape back to [B, t, C]
        out = Y_flat.reshape(B, T_out3, C)

        return out


def run(*args):
    return ModelNew()(*args)
