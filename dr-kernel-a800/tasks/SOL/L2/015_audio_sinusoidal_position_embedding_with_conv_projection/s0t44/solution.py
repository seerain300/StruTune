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

    # Compute output tile indices
    f_out_start = f_block * BLOCK_F
    t_out_start = t_block * BLOCK_T
    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)[:, None]  # [BF, 1]
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)[None, :]  # [1, BT]
    out_mask = (f_out_idx < F_out) & (t_out_idx < T_out)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Compute input ranges for padding=1
    # for kh in [0,1,2], kw in [0,1,2]
    # input index = f_out_idx + 1 - kh, t_out_idx + 1 - kw
    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            f_in_idx = f_out_idx + 1 - kh
            t_in_idx = t_out_idx + 1 - kw
            in_mask = (f_in_idx >= 0) & (f_in_idx < F_in) & (t_in_idx >= 0) & (t_in_idx < T_in) & out_mask

            # Loop over input channels
            for ic in tl.static_range(IC):
                # pointers for X[b, ic, f_in, t_in]
                x_ptrs = X_ptr + b * x_sN + ic * x_sC + f_in_idx * x_sF + t_in_idx * x_sT
                x_vals = tl.load(x_ptrs, mask=in_mask, other=0.0)  # [BF, BT]

                # pointers for W[oc, ic, kh, kw]
                w_ptrs = W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW
                w_val = tl.load(w_ptrs)  # scalar

                acc += x_vals * w_val

    # Add bias
    bias_val = tl.load(BIAS_ptr + oc)
    acc += bias_val

    # Apply GELU
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    acc_cub = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.math.tanh(c * (acc + 0.044715 * acc_cub)))

    # Store output
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out_idx * y_sF + t_out_idx * y_sT
    tl.store(y_ptrs, gelu, mask=out_mask)


# Triton matmul kernel: A[M, K] * B[K, N] -> C[M, N]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    a_sM, a_sK, b_sK, b_sN, c_sM, c_sN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * a_sM + offs_k[None, :] * a_sK
    b_ptrs = B_ptr + offs_k[:, None] * b_sK + offs_n[None, :] * b_sN

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * a_sK
        b_ptrs += BLOCK_K * b_sK

    c_ptrs = C_ptr + offs_m[:, None] * c_sM + offs_n[None, :] * c_sN
    tl.store(
        c_ptrs,
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

    pos_ptrs = POS_ptr + d * POS_ptr.stride(1)  # POS is [N, D]
    pos_vals = tl.load(pos_ptrs, mask=mask, other=0.0)

    y_vals = x_vals + pos_vals

    y_ptrs = Y_ptr + n * y_sN + d * y_sD
    tl.store(y_ptrs, y_vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # TRITON-ONLY: ensure Triton is available
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")

        # args correspond to: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale

        # Extract tensors
        input_features = args[0]  # [B, 1, 80, time_dim]
        conv2d1_weight = args[1]  # [OC, IC, 3, 3] = [384, 1, 3, 3]
        conv2d1_bias = args[2]    # [OC]
        conv2d2_weight = args[3]  # [384, 384, 3, 3]
        conv2d2_bias = args[4]    # [384]
        conv2d3_weight = args[5]  # [384, 384, 3, 3]
        conv2d3_bias = args[6]    # [384]
        conv_out_weight = args[7] # [d_model, conv_out_dim] = [1024, 3840]
        positional_embedding = args[8]  # [max_source_positions, d_model]
        embed_scale = float(args[9])     # python float

        device = input_features.device

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        B, IC_in, F_in, T_in = input_features.shape
        OC1 = conv2d1_weight.shape[0]
        w1 = conv2d1_weight.contiguous().float()     # [OC1, 1, 3, 3]
        b1 = conv2d1_bias.contiguous().float()       # [OC1]
        x1 = input_features.contiguous().float()     # [B, 1, 80, T_in]
        F_out1 = (F_in - 3) // 2 + 1
        T_out1 = (T_in - 3) // 2 + 1
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
        OC2 = conv2d2_weight.shape[0]
        x2 = y1
        w2 = conv2d2_weight.contiguous().float()     # [OC2, OC1, 3, 3]
        b2 = conv2d2_bias.contiguous().float()       # [OC2]
        F_in2, T_in2 = F_out1, T_out1
        F_out2 = (F_in2 - 3) // 2 + 1
        T_out2 = (T_in2 - 3) // 2 + 1
        y2 = torch.empty((B, OC2, F_out2, T_out2), device=device, dtype=torch.float32)
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
        OC3 = conv2d3_weight.shape[0]
        x3 = y2
        w3 = conv2d3_weight.contiguous().float()     # [OC3, OC2, 3, 3]
        b3 = conv2d3_bias.contiguous().float()       # [OC3]
        F_in3, T_in3 = F_out2, T_out2
        F_out3 = (F_in3 - 3) // 2 + 1
        T_out3 = (T_in3 - 3) // 2 + 1
        y3 = torch.empty((B, OC3, F_out3, T_out3), device=device, dtype=torch.float32)
        grid3 = (B * OC3 * triton.cdiv(F_out3, 32) * triton.cdiv(T_out3, 32),)
        conv3x3_s2_p1_gelu[grid3](
            x3, w3, b3, y3,
            B, OC2, F_in3, T_in3, OC3, F_out3, T_out3,
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            w3.stride(0), w3.stride(1), w3.stride(2), w3.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_F=32, BLOCK_T=32
        )

        # Reshape: (B, channels, F, T) -> (B, T, channels*F)
        # After Stage 3: y3 shape is (B, 384, F_out3, T_out3)
        # We need T_total = T_out3
        B_y, OC3_y, F_out3_y, T_out3_y = y3.shape
        # The reference code uses time_after_conv in the original run, but our conv outputs depend on input time_dim.
        # To match the original logic, we use T_out3 as T_total.
        T_total = T_out3_y
        channels = OC3_y
        F_total = F_out3_y
        x_proj = y3.permute(0, 3, 1, 2).contiguous().view(B_y, T_total, channels * F_total)

        # Linear projection to d_model (no bias) using Triton matmul
        # x_proj: [B*T, 3840], conv_out_weight.T: [3840, 1024] (since conv_out_weight is [1024, 3840])
        B_T, input_dim = x_proj.shape  # B*T, 3840
        W_T = conv_out_weight.transpose(0, 1).contiguous().float()  # [3840, 1024]
        out_dim = 1024
        # Initialize output
        out = torch.empty((B_T, out_dim), device=device, dtype=torch.float32)
        # Launch grid: (cdiv(B_T, 64), cdiv(out_dim, 128))
        grid = (triton.cdiv(B_T, 64), triton.cdiv(out_dim, 128))
        matmul_kernel[grid](
            x_proj, W_T, out,
            B_T, out_dim, input_dim,
            x_proj.stride(0), input_dim, W_T.stride(0), W_T.stride(1), out.stride(0), out.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        # Scale by embed_scale and add positional embedding
        # embed_scale = sqrt(d_model) = sqrt(1024) = 32.0
        scaled = out * embed_scale  # [B*T, 1024]
        # positional_embedding is [max_source_positions, 1024], dtype likely bfloat16
        pos = positional_embedding.to(torch.float32)  # [max_source_positions, 1024]
        # We need only the first T_total rows
        pos = pos[:T_total, :]  # [T_total, 1024]
        # Reshape scaled to [B, T_total, 1024]
        scaled = scaled.view(B, T_total, out_dim)
        # Prepare elementwise output tensor
        final = torch.empty((B, T_total, out_dim), device=device, dtype=torch.float32)
        # Launch elementwise kernel
        grid_e = (triton.cdiv(T_total * out_dim, 1024),)
        scale_add_pos_embed[grid_e](
            scaled, pos, final, embed_scale, T_total, out_dim,
            scaled.stride(0), scaled.stride(1), scaled.stride(2),
            final.stride(0), final.stride(1), final.stride(2),
            BLOCK=1024
        )

        return final


def run(*args):
    return ModelNew()(*args)
