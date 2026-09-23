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
    oc_blocks = OC

    b = pid // (oc_blocks * grid_f_blocks * grid_t_blocks)
    rem = pid % (oc_blocks * grid_f_blocks * grid_t_blocks)
    oc = rem // (grid_f_blocks * grid_t_blocks)
    rem2 = rem % (grid_f_blocks * grid_t_blocks)
    f_block = rem2 // grid_t_blocks
    t_block = rem2 % grid_t_blocks

    # Compute output indices for this tile
    f_out_start = f_block * BLOCK_F
    t_out_start = t_block * BLOCK_T
    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)[:, None]  # [BF, 1]
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)[None, :]  # [1, BT]
    out_mask = (f_out_idx < F_out) & (t_out_idx < T_out)

    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    for ic in range(IC):
        for kh in range(3):
            for kw in range(3):
                # Map output (f_out, t_out) -> input (f_in, t_in) with padding=1
                f_in = f_out_idx + 1 - kh  # [BF, 1]
                t_in = t_out_idx + 1 - kw  # [1, BT]
                # Bounds check
                in_bounds = (f_in >= 0) & (f_in < F_in) & (t_in >= 0) & (t_in < T_in) & out_mask

                # Load input X[b, ic, f_in, t_in]
                x_ptrs = X_ptr + b * x_sN + ic * x_sC + f_in * x_sF + t_in * x_sT
                x_val = tl.load(x_ptrs, mask=in_bounds, other=0.0)  # [BF, BT], float32

                # Load weight W[oc, ic, kh, kw] (scalar per (oc, ic, kh, kw))
                w_ptrs = W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW
                w_val = tl.load(w_ptrs)  # scalar

                # FMA accumulate
                acc += x_val * w_val

    # Add bias
    bias_val = tl.load(BIAS_ptr + oc)  # scalar
    acc = acc + bias_val

    # GELU (erf-based approximation): y = 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    gelu_in = acc
    gelu = 0.5 * gelu_in * (1.0 + tl.math.erf(gelu_in * inv_sqrt2))

    # Store to Y[b, oc, f_out, t_out]
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out_idx * y_sF + t_out_idx * y_sT
    tl.store(y_ptrs, gelu, mask=out_mask)


# Triton matmul kernel: A[M, K] x B[K, N] -> C[M, N]
# We will use it to compute the linear projection: A = x.view(B*T3, 3840), B = conv_out_weight.T ([N=1024, K=3840])
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)  # [BM, BK]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)  # [BK, BN]

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton elementwise kernel: y = x * scale + pos
@triton.jit
def scale_add_kernel(X_ptr, Y_ptr, POS_ptr, NUMEL: tl.int32, SCALE: tl.float32):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < NUMEL
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    pos = tl.load(POS_ptr + offs, mask=mask, other=0.0)
    y = x * SCALE + pos
    tl.store(Y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, *args):
        # args: (input_features, conv2d1_weight, conv2d1_bias,
        #        conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        #        conv_out_weight, positional_embedding, embed_scale)
        input_features = args[0]
        conv2d1_weight = args[1]
        conv2d1_bias = args[2]
        conv2d2_weight = args[3]
        conv2d2_bias = args[4]
        conv2d3_weight = args[5]
        conv2d3_bias = args[6]
        conv_out_weight = args[7]  # [1024, 3840]
        positional_embedding = args[8]  # [max_source_positions, 1024], bfloat16
        embed_scale = args[9]  # float

        # Work in float32 for Triton kernels
        x = input_features.to(torch.float32)
        w1 = conv2d1_weight.to(torch.float32)
        b1 = conv2d1_bias.to(torch.float32)

        B, IC, F_in, T_in = x.shape
        OC = w1.shape[0]
        F_out1 = (F_in + 2 * 1 - 3) // 2 + 1
        T_out1 = (T_in + 2 * 1 - 3) // 2 + 1

        y1 = torch.empty((B, OC, F_out1, T_out1), device=x.device, dtype=torch.float32)
        BLOCK_F = 16
        BLOCK_T = 32
        grid = (B * OC * ((F_out1 + BLOCK_F - 1) // BLOCK_F) * ((T_out1 + BLOCK_T - 1) // BLOCK_T),)
        conv3x3_s2_p1_gelu[grid](
            x, w1, b1, y1,
            B, 1, F_in, T_in, OC, F_out1, T_out1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_F=BLOCK_F, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2,
        )

        # Stage 2 conv: y1 -> 384 channels
        C2 = 384
        F_in2, T_in2 = y1.shape[2], y1.shape[3]
        F_out2 = (F_in2 + 2 * 1 - 3) // 2 + 1
        T_out2 = (T_in2 + 2 * 1 - 3) // 2 + 1
        y2 = torch.empty((B, C2, F_out2, T_out2), device=x.device, dtype=torch.float32)
        w2 = conv2d2_weight.to(torch.float32)
        b2 = conv2d2_bias.to(torch.float32)
        grid2 = (B * C2 * ((F_out2 + BLOCK_F - 1) // BLOCK_F) * ((T_out2 + BLOCK_T - 1) // BLOCK_T),)
        conv3x3_s2_p1_gelu[grid2](
            y1, w2, b2, y2,
            B, C2, F_in2, T_in2, C2, F_out2, T_out2,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_F=BLOCK_F, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2,
        )

        # Stage 3 conv: y2 -> 384 channels
        C3 = 384
        F_in3, T_in3 = y2.shape[2], y2.shape[3]
        F_out3 = (F_in3 + 2 * 1 - 3) // 2 + 1
        T_out3 = (T_in3 + 2 * 1 - 3) // 2 + 1
        y3 = torch.empty((B, C3, F_out3, T_out3), device=x.device, dtype=torch.float32)
        w3 = conv2d3_weight.to(torch.float32)
        b3 = conv2d3_bias.to(torch.float32)
        grid3 = (B * C3 * ((F_out3 + BLOCK_F - 1) // BLOCK_F) * ((T_out3 + BLOCK_T - 1) // BLOCK_T),)
        conv3x3_s2_p1_gelu[grid3](
            y2, w3, b3, y3,
            B, C3, F_in3, T_in3, C3, F_out3, T_out3,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            w3.stride(0), w3.stride(1), w3.stride(2), w3.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_F=BLOCK_F, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2,
        )

        # Reshape: (B, C3=384, F_out3, T_out3) -> (B, T_out3, C3*F_out3)
        # Note: In original run function, after 3 convs, shape is (B, C=384, F=10, T=T_out3). We reshape to (B, T_out3, 384*10).
        # Here we keep general; for these inputs, F_out3 should be 10 (matching original post-conv3 shapes).
        F_out3, T_out3 = y3.shape[2], y3.shape[3]
        x_reshaped = y3.permute(0, 3, 1, 2).contiguous().view(B, T_out3, C3 * F_out3)

        # Linear projection using Triton matmul
        # A: [B*T_out3, 3840], B: [1024, 3840] (conv_out_weight), output C: [B*T_out3, 1024]
        M = B * T_out3
        K = 3840
        N = 1024
        A = x_reshaped.view(M, K).to(torch.float32)  # [M, K]
        BT = conv_out_weight.to(torch.float32)       # [N=1024, K=3840]
        C = torch.empty((M, N), device=x.device, dtype=torch.float32)

        grid_mm = (triton.cdiv(M, 128), triton.cdiv(N, 64))
        matmul_kernel[grid_mm](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=3,
        )

        # Scale by embed_scale using Triton elementwise kernel
        numel = M * N
        Y_scaled = torch.empty_like(C)
        grid_scale = (triton.cdiv(numel, 1024),)
        scale_add_kernel[grid_scale](
            C, Y_scaled, torch.zeros(numel, dtype=C.dtype, device=C.device), numel, float(embed_scale)
        )

        # Add positional embedding: [T_out3, 1024] first rows
        pos_emb = positional_embedding[:T_out3, :].to(torch.float32)  # [T_out3, 1024]
        Y_out = torch.empty_like(Y_scaled)
        grid_add = (triton.cdiv(numel, 1024),)
        scale_add_kernel[grid_add](
            Y_scaled, Y_out, pos_emb.reshape(-1), numel, 1.0  # adding pos_emb, scale=1.0
        )

        # Reshape to [B, T_out3, 1024]
        out = Y_out.view(B, T_out3, N)
        return out


def run(*args):
    return ModelNew()(*args)
