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
    t_block = rem % grid_t_blocks

    # Output tile indices
    f_out_start = f_block * BLOCK_F
    t_out_start = t_block * BLOCK_T
    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)[:, None]  # [BF, 1]
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)[None, :]  # [1, BT]
    out_mask = (f_out_idx < F_out) & (t_out_idx < T_out)

    # Accumulator
    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Compute input indices for 3x3 kernel
    h_base = f_out_idx * 2 + 1  # output index * stride + padding
    w_base = t_out_idx * 2 + 1

    # Loop over input channels and 3x3 kernel
    for ic in range(IC):
        for kh in range(3):
            h = h_base + kh - 1  # -1 for kernel offset
            h_in_range = (h >= 0) & (h < F_in)
            h_in = h.to(tl.int32)
            for kw in range(3):
                w = w_base + kw - 1
                w_in_range = (w >= 0) & (w < T_in)
                w_in = w.to(tl.int32)
                in_mask = out_mask & h_in_range & w_in_range

                # Load input tile [BF, BT] for current ic
                x_ptrs = X_ptr + b * x_sN + ic * x_sC + h_in * x_sF + w_in * x_sT
                x_vals = tl.load(x_ptrs, mask=in_mask, other=0.0)

                # Load corresponding weight [OC] for this (ic, kh, kw)
                # W layout: [OC, IC, 3, 3]
                w_ptrs = W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW
                w_vals = tl.load(w_ptrs)  # scalar

                # Outer product and accumulate
                acc += x_vals * w_vals

    # Add bias
    bias_vals = tl.load(BIAS_ptr + oc)
    acc += bias_vals

    # GELU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    acc_cubed = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * (acc + c1 * acc_cubed)))

    # Store result
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out_idx * y_sF + t_out_idx * y_sT
    tl.store(y_ptrs, gelu, mask=out_mask)


# Triton kernel: elementwise scale and add positional embedding
# x: [B*T, OC] (contiguous), pos: [T, OC], scale: float, out: [B*T, OC]
@triton.jit
def scale_add_pos_emb(
    X_ptr, POS_ptr, SCALE, Y_ptr,
    TOTAL, OC,
    x_sB, x_sO,  # strides for X: X is [B*T, OC], strides (B stride, OC stride)
    y_sB, y_sO,  # strides for Y: same layout
):
    pid = tl.program_id(0)
    BLOCK = 128
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    # Decode batch and channel for each element
    b_idx = offs // OC
    o_idx = offs % OC

    # Load x
    x_ptrs = X_ptr + b_idx * x_sB + o_idx * x_sO
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

    # Load positional embedding at time index b_idx[o_idx], channel o_idx
    # Note: POS is [T, OC], we use b_idx as time index (since we operate on [B*T, OC] flattened).
    pos_ptrs = POS_ptr + b_idx * POS_ptr.stride(0) + o_idx * POS_ptr.stride(1)
    pos_vals = tl.load(pos_ptrs, mask=mask, other=0.0)

    # Scale and add
    y_vals = x_vals * SCALE + pos_vals

    # Store
    y_ptrs = Y_ptr + b_idx * y_sB + o_idx * y_sO
    tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: matmul [M, K] x [K, N] -> [M, N]
@triton.jit
def matmul_gemm(
    A_ptr, B_ptr, C_ptr,
    M, K, N,
    a_sM, a_sK,
    b_sK, b_sN,
    c_sM, c_sN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    g = grid_m * grid_n
    mp = pid // grid_n
    np = pid % grid_n

    m0 = mp * BLOCK_M
    n0 = np * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(
            A_ptr + m0 * a_sM + (k0 + tl.arange(0, BLOCK_K)) * a_sK,
            mask=(m0 + tl.arange(0, BLOCK_M)[:, None] < M) & (k0 + tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0,
        )  # [BM, BK]
        b = tl.load(
            B_ptr + (k0 + tl.arange(0, BLOCK_K)) * b_sK + n0 * b_sN,
            mask=(k0 + tl.arange(0, BLOCK_K)[:, None] < K) & (n0 + tl.arange(0, BLOCK_N)[None, :] < N),
            other=0.0,
        )  # [BK, BN]
        acc += tl.dot(a, b)  # [BM, BN]

    tl.store(
        C_ptr + m0 * c_sM + n0 * c_sN,
        acc,
        mask=(m0 + tl.arange(0, BLOCK_M)[:, None] < M) & (n0 + tl.arange(0, BLOCK_N)[None, :] < N),
    )


class ModelNew(nn.Module):
    def forward(self, *args):
        # Args order as provided by get_inputs():
        # 0: input_features [B, 1, 80, time_dim]
        # 1: conv2d1_weight [384, 1, 3, 3]
        # 2: conv2d1_bias [384]
        # 3: conv2d2_weight [384, 384, 3, 3]
        # 4: conv2d2_bias [384]
        # 5: conv2d3_weight [384, 384, 3, 3]
        # 6: conv2d3_bias [384]
        # 7: conv_out_weight [d_model=1024, conv_out_dim=3840]
        # 8: positional_embedding [max_source_positions, d_model]
        # 9: embed_scale (float) = sqrt(1024) = 32.0

        # Ensure float32 for Triton
        device = args[0].device
        B, IC_in, F_in, T_in = args[0].shape

        x = args[0].contiguous().float()  # [B, 1, 80, T_in]

        # Stage 1: Triton conv (1 -> 384 channels)
        OC1 = args[1].shape[0]  # 384
        w1 = args[1].contiguous().float()  # [384, 1, 3, 3]
        b1 = args[2].contiguous().float()  # [384]
        F_out1 = (F_in + 2 * 1 - 3) // 2 + 1
        T_out1 = (T_in + 2 * 1 - 3) // 2 + 1
        y1 = torch.empty((B, OC1, F_out1, T_out1), device=device, dtype=torch.float32)

        grid1 = (B, OC1, triton.cdiv(F_out1, 32), triton.cdiv(T_out1, 32))
        conv3x3_s2_p1_gelu[grid1](
            x, w1, b1, y1,
            B, 1, F_in, T_in, OC1, F_out1, T_out1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_F=32, BLOCK_T=32
        )

        # Stage 2: Triton conv (384 -> 384 channels)
        x2 = y1
        OC2 = args[3].shape[0]  # 384
        w2 = args[3].contiguous().float()  # [384, 384, 3, 3]
        b2 = args[4].contiguous().float()  # [384]
        F_in2 = F_out1
        T_in2 = T_out1
        F_out2 = (F_in2 + 2 * 1 - 3) // 2 + 1
        T_out2 = (T_in2 + 2 * 1 - 3) // 2 + 1
        y2 = torch.empty((B, OC2, F_out2, T_out2), device=device, dtype=torch.float32)

        grid2 = (B, OC2, triton.cdiv(F_out2, 32), triton.cdiv(T_out2, 32))
        conv3x3_s2_p1_gelu[grid2](
            x2, w2, b2, y2,
            B, OC1, F_in2, T_in2, OC2, F_out2, T_out2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_F=32, BLOCK_T=32
        )

        # Stage 3: Triton conv (384 -> 384 channels)
        x3 = y2
        OC3 = args[5].shape[0]  # 384
        w3 = args[5].contiguous().float()  # [384, 384, 3, 3]
        b3 = args[6].contiguous().float()  # [384]
        F_in3 = F_out2
        T_in3 = T_out2
        F_out3 = (F_in3 + 2 * 1 - 3) // 2 + 1
        T_out3 = (T_in3 + 2 * 1 - 3) // 2 + 1
        y3 = torch.empty((B, OC3, F_out3, T_out3), device=device, dtype=torch.float32)

        grid3 = (B, OC3, triton.cdiv(F_out3, 32), triton.cdiv(T_out3, 32))
        conv3x3_s2_p1_gelu[grid3](
            x3, w3, b3, y3,
            B, OC2, F_in3, T_in3, OC3, F_out3, T_out3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            w3.stride(0), w3.stride(1), w3.stride(2), w3.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_F=32, BLOCK_T=32
        )

        # Reshape: (B, channels, F, T) -> (B, T, channels*F)
        # After conv3, shape is (B, 384, F_out3, T_out3)
        B, OC, F_out, T_out = y3.shape
        channels = OC  # 384
        freq = F_out   # 40 for input 80 with stride 2, 3x conv
        # freq should equal args[0].shape[2] // (2^3) = 80 // 8 = 10
        # Here convs reduce spatial by 8: 80 -> 10
        T_after = T_out  # time after conv3, e.g., 211
        # View to (B, T_after, channels*freq)
        x_lin = y3.permute(0, 3, 1, 2).contiguous().view(B, T_after, channels * freq)

        # Linear projection: [B*T_after, 3840] x [3840, 1024] -> [B*T_after, 1024]
        B_lin, T_lin, K = x_lin.shape  # B*T_after, 3840
        w_lin = args[7].contiguous().float()  # [1024, 3840]
        X_lin_flat = x_lin.reshape(B_lin * T_lin, K).contiguous()  # [M, K] = [B*T_after, 3840]
        Y_proj = torch.empty((B_lin * T_lin, w_lin.shape[0]), device=device, dtype=torch.float32)

        # Launch Triton matmul
        M = B_lin * T_lin
        K_mat = w_lin.shape[1]  # 3840
        N = w_lin.shape[0]      # 1024

        grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
        matmul_gemm[grid](
            X_lin_flat, w_lin, Y_proj,
            M, K_mat, N,
            X_lin_flat.stride(0), X_lin_flat.stride(1),
            w_lin.stride(1), w_lin.stride(0),
            Y_proj.stride(0), Y_proj.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
        )

        # Reshape back to [B, T_after, 1024]
        x_scaled = Y_proj.view(B, T_after, 1024)

        # Scale and add positional embedding via Triton elementwise
        pos_emb = args[8].contiguous().float()  # [max_source_positions, 1024]
        # x_scaled is [B, T_after, 1024]; pos_emb is [T_after, 1024] if we slice
        # But args[8] has length max_source_positions; in provided inputs, max_source_positions >= T_after.
        T3 = T_after
        pos_slice = pos_emb[:T3, :].contiguous()  # [T3, 1024]
        # Flatten x_scaled to [B*T3, 1024]
        x_flat = x_scaled.reshape(B * T3, 1024).contiguous()
        y_final = torch.empty_like(x_flat)

        scale = float(args[9])  # 32.0
        total = B * T3 * 1024
        grid_scale = (triton.cdiv(total, 128),)
        scale_add_pos_emb[grid_scale](
            x_flat, pos_slice, scale, y_final,
            total, 1024,
            x_flat.stride(0), x_flat.stride(1),
            y_final.stride(0), y_final.stride(1),
        )

        # Reshape back to [B, T_after, 1024]
        out = y_final.view(B, T_after, 1024)

        return out


def run(*args):
    return ModelNew()(*args)
