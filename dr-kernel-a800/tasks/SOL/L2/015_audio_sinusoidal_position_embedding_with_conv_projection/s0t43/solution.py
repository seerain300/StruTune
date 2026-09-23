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

    # Tile indices
    f_out_start = f_block * BLOCK_F
    t_out_start = t_block * BLOCK_T
    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)[:, None]  # [BF, 1]
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)[None, :]  # [1, BT]
    out_mask = (f_out_idx < F_out) & (t_out_idx < T_out)

    # Accumulator
    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    for ic in range(IC):
        # Unroll 3x3 taps
        for kh in range(3):
            for kw in range(3):
                # Input index for (kh, kw): f_in = f_out*2 - 1 + kh, t_in = t_out*2 - 1 + kw
                f_in_idx = f_out_idx * 2 - 1 + kh  # [BF, 1]
                t_in_idx = t_out_idx * 2 - 1 + kw  # [1, BT]

                # Mask for valid input
                in_mask = (f_in_idx >= 0) & (f_in_idx < F_in) & (t_in_idx >= 0) & (t_in_idx < T_in) & out_mask

                # Compute input pointers
                x_ptrs = X_ptr + b * x_sN + ic * x_sC + f_in_idx * x_sF + t_in_idx * x_sT
                x_vals = tl.load(x_ptrs, mask=in_mask, other=0.0)

                # Load weight vector for this (oc, ic, kh, kw)
                w_ptrs = W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW
                w_val = tl.load(w_ptrs)  # scalar

                # Accumulate
                acc += x_vals * w_val

    # Add bias
    bptr = BIAS_ptr + oc
    bias_val = tl.load(bptr)
    acc += bias_val

    # GELU activation
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x = acc
    x3 = x * x * x
    gelu = 0.5 * x * (1.0 + tl.math.tanh(c * (x + 0.044715 * x3)))

    # Store output
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out_idx * y_sF + t_out_idx * y_sT
    tl.store(y_ptrs, gelu, mask=out_mask)


# Triton matmul: A[M,K] x B[K,N] -> C[M,N]
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

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
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

    # POS is [N, D]
    pos_ptrs = POS_ptr + n * POS_ptr.stride(0) + d * POS_ptr.stride(1)
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

        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")

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

        # Ensure device and dtype: Triton expects float32 for compute
        device = input_features.device
        B, IC_in, F_in, T_in = input_features.shape

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        OC1 = conv2d1_weight.shape[0]
        w1 = conv2d1_weight.contiguous().float()     # [OC1, 1, 3, 3]
        b1 = conv2d1_bias.contiguous().float()       # [OC1]
        x1 = input_features.contiguous().float()     # [B, 1, 80, T_in]
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
        w2 = conv2d2_weight.contiguous().float()     # [OC2, OC1, 3, 3]
        b2 = conv2d2_bias.contiguous().float()       # [OC2]
        F_in2 = F_out1
        T_in2 = T_out1
        F_out2 = (F_in2 + 2 * 1 - 3) // 2 + 1
        T_out2 = (T_in2 + 2 * 1 - 3) // 2 + 1
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
        x3 = y2
        OC3 = conv2d3_weight.shape[0]
        w3 = conv2d3_weight.contiguous().float()     # [OC3, OC2, 3, 3]
        b3 = conv2d3_bias.contiguous().float()       # [OC3]
        F_in3 = F_out2
        T_in3 = T_out2
        F_out3 = (F_in3 + 2 * 1 - 3) // 2 + 1
        T_out3 = (T_in3 + 2 * 1 - 3) // 2 + 1
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

        # Reshape: (batch, channels, F, T) -> (batch, T, channels*F)
        # y3: [B, 384, F_out3, T_out3]
        B, C, F, T = y3.shape
        T3 = T
        # For this specific model, F_out3 == 1 and T_out3 == time_after_conv, so we can use the provided T3
        # but keep generic logic for correctness: we need channels*F, here C=384 and F=1, so C*F=384.
        # However, to be robust, we assume reshape based on original code logic: (B, T, C*F)
        # Since F_out3 is 1, reshape uses T_out3. We will use T3 provided externally (time_after_conv).
        out = y3.permute(0, 3, 1, 2).contiguous().view(B, T3, C * 1)  # T3 is time_after_conv from inputs

        # Linear projection: [B*T3, 3840] x [3840, 1024]
        # out: [B, T3, 384]
        # Convert to [M, K] where M=B*T3, K=3840
        M = B * T3
        K = conv_out_weight.shape[1]  # 3840
        N_proj = conv_out_weight.shape[0]  # 1024
        A = out.reshape(M, K).contiguous().float()  # [M, K]
        B_proj = conv_out_weight.contiguous().float()  # [N, K] = [1024, 3840]
        C_out = torch.empty((M, N_proj), device=device, dtype=torch.float32)
        grid_mm = (triton.cdiv(M, 128), triton.cdiv(N_proj, 128))
        matmul_kernel[grid_mm](
            A, B_proj, C_out,
            M, N_proj, K,
            A.stride(0), A.stride(1),
            B_proj.stride(0), B_proj.stride(1),
            C_out.stride(0), C_out.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
        )
        x = C_out.view(B, T3, N_proj)  # [B, T3, 1024]

        # Scale by embed_scale
        # embed_scale = sqrt(d_model) = sqrt(1024) = 32.0
        x = x * embed_scale

        # Add positional embedding: [max_source_positions, d_model], here [T3, 1024]
        # Convert x to [T3, d_model] by permuting to (T, N) then flatten (but we already have shape [B, T3, N])
        # Instead, we can use a temporary reshape: x: [B, T3, 1024] -> (B*T3, 1024)
        x_flat = x.reshape(M, N_proj).contiguous().float()
        pos_flat = positional_embedding[:T3, :].contiguous().float().reshape(T3, N_proj)
        y_flat = torch.empty((M, N_proj), device=device, dtype=torch.float32)
        grid_scale = (triton.cdiv(M * N_proj, 256),)
        scale_add_pos_embed[grid_scale](
            x_flat, pos_flat, y_flat, embed_scale, M, N_proj,
            x_flat.stride(0), x_flat.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK=256
        )
        y = y_flat.view(B, T3, N_proj)

        return y


def run(*args):
    return ModelNew()(*args)
