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

    # Use integer division and modulo to decode pid. We assume grid size equals B * OC * grid_f_blocks * grid_t_blocks.
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
                f_in = f_out_idx + 1 - kh  # [BF, 1]
                t_in = t_out_idx + 1 - kw  # [1, BT]
                in_bounds = (f_in >= 0) & (f_in < F_in) & (t_in >= 0) & (t_in < T_in) & out_mask

                # Compute pointers for X[b, ic, f_in, t_in]
                x_ptrs = X_ptr + b * x_sN + ic * x_sC + f_in * x_sF + t_in * x_sT
                x_val = tl.load(x_ptrs, mask=in_bounds, other=0.0)  # [BF, BT]

                # Compute pointers for W[oc, ic, kh, kw]
                w_val = tl.load(W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW)  # scalar
                acc += x_val * w_val  # broadcast scalar across tile

    # Add bias if provided
    bias_val = tl.load(BIAS_ptr + oc)
    acc = acc + bias_val

    # GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * (acc + c1 * x3)))

    # Store to Y[b, oc, f_out, t_out]
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out_idx * y_sF + t_out_idx * y_sT
    tl.store(y_ptrs, gelu, mask=out_mask)


# Triton matmul kernel: A[M, K] x B[K, N] -> C[M, N]
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
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton elementwise kernel: scale and add (for scaling only)
@triton.jit
def scale_kernel(X_ptr, Y_ptr, NUMEL: tl.int32, SCALE: tl.float32):
    pid = tl.program_id(0)
    offs = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
    mask = offs < NUMEL
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = x * SCALE
    tl.store(Y_ptr + offs, y, mask=mask)


# Triton elementwise kernel: add positional embedding (vector addition)
@triton.jit
def add_pos_kernel(X_ptr, POS_ptr, Y_ptr, NUMEL: tl.int32):
    pid = tl.program_id(0)
    offs = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
    mask = offs < NUMEL
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    pos = tl.load(POS_ptr + offs, mask=mask, other=0.0)
    y = x + pos
    tl.store(Y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, *args):
        # Args order: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale
        input_features = args[0]
        conv2d1_weight = args[1]
        conv2d1_bias = args[2]
        conv2d2_weight = args[3]
        conv2d2_bias = args[4]
        conv2d3_weight = args[5]
        conv2d3_bias = args[6]
        conv_out_weight = args[7]  # [1024, 3840]
        positional_embedding = args[8]  # [max_source_positions, 1024], dtype bfloat16
        embed_scale = args[9]

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        B = input_features.shape[0]
        IC = 1
        F_in1 = input_features.shape[2]
        T_in1 = input_features.shape[3]
        OC = 384
        F_out1 = (F_in1 + 2*1 - 3)//2 + 1
        T_out1 = (T_in1 + 2*1 - 3)//2 + 1

        y1 = torch.empty((B, OC, F_out1, T_out1), device=input_features.device, dtype=torch.float32)

        # Ensure inputs are float32 for Triton
        X1 = input_features.contiguous().to(torch.float32)
        W1 = conv2d1_weight.contiguous().to(torch.float32)
        BIAS1 = conv2d1_bias.contiguous().to(torch.float32)

        BLOCK_F = 8
        BLOCK_T = 32
        grid = ((F_out1 + BLOCK_F - 1) // BLOCK_F, (T_out1 + BLOCK_T - 1) // BLOCK_T)
        conv3x3_s2_p1_gelu[(B * OC * grid[0] * grid[1],)](
            X1, W1, BIAS1, y1,
            B, IC, F_in1, T_in1, OC, F_out1, T_out1,
            X1.stride(0), X1.stride(1), X1.stride(2), X1.stride(3),
            W1.stride(0), W1.stride(1), W1.stride(2), W1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_F=BLOCK_F, BLOCK_T=BLOCK_T,
            num_warps=4, num_stages=2,
        )

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        C2 = 384
        F_in2 = F_out1
        T_in2 = T_out1
        F_out2 = (F_in2 + 2*1 - 3)//2 + 1
        T_out2 = (T_in2 + 2*1 - 3)//2 + 1

        y2 = torch.empty((B, C2, F_out2, T_out2), device=input_features.device, dtype=torch.float32)

        X2 = y1.contiguous().to(torch.float32)
        W2 = conv2d2_weight.contiguous().to(torch.float32)
        BIAS2 = conv2d2_bias.contiguous().to(torch.float32)

        BLOCK_F = 8
        BLOCK_T = 32
        grid = ((F_out2 + BLOCK_F - 1) // BLOCK_F, (T_out2 + BLOCK_T - 1) // BLOCK_T)
        conv3x3_s2_p1_gelu[(B * C2 * grid[0] * grid[1],)](
            X2, W2, BIAS2, y2,
            B, C2, F_in2, T_in2, C2, F_out2, T_out2,
            X2.stride(0), X2.stride(1), X2.stride(2), X2.stride(3),
            W2.stride(0), W2.stride(1), W2.stride(2), W2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_F=BLOCK_F, BLOCK_T=BLOCK_T,
            num_warps=4, num_stages=2,
        )

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        F_in3 = F_out2
        T_in3 = T_out2
        F_out3 = (F_in3 + 2*1 - 3)//2 + 1
        T_out3 = (T_in3 + 2*1 - 3)//2 + 1

        y3 = torch.empty((B, C2, F_out3, T_out3), device=input_features.device, dtype=torch.float32)

        X3 = y2.contiguous().to(torch.float32)
        W3 = conv2d3_weight.contiguous().to(torch.float32)
        BIAS3 = conv2d3_bias.contiguous().to(torch.float32)

        BLOCK_F = 8
        BLOCK_T = 32
        grid = ((F_out3 + BLOCK_F - 1) // BLOCK_F, (T_out3 + BLOCK_T - 1) // BLOCK_T)
        conv3x3_s2_p1_gelu[(B * C2 * grid[0] * grid[1],)](
            X3, W3, BIAS3, y3,
            B, C2, F_in3, T_in3, C2, F_out3, T_out3,
            X3.stride(0), X3.stride(1), X3.stride(2), X3.stride(3),
            W3.stride(0), W3.stride(1), W3.stride(2), W3.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_F=BLOCK_F, BLOCK_T=BLOCK_T,
            num_warps=4, num_stages=2,
        )

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        y3 = y3.permute(0, 3, 1, 2).contiguous()  # [B, T_out3, C2, F_out3]
        B, T3, C2, F_out3 = y3.shape
        x_proj = y3.view(B, T3, C2 * F_out3)  # [B, T3, 384*10] = [B, T3, 3840]

        # Linear projection: [B*T3, 3840] @ conv_out_weight.T (conv_out_weight is [1024, 3840], so we pass BT as [3840, 1024] = conv_out_weight.T)
        BT = conv_out_weight.t().contiguous().to(torch.float32)  # [3840, 1024]
        A = x_proj.view(B * T3, 3840).contiguous().to(torch.float32)  # [M, K]
        C = torch.empty((A.shape[0], BT.shape[1]), device=input_features.device, dtype=torch.float32)  # [M, N=1024]

        grid_mm = (triton.cdiv(A.shape[0], 128), triton.cdiv(BT.shape[1], 64))
        matmul_kernel[grid_mm](
            A, BT, C,
            A.shape[0], BT.shape[1], BT.shape[0],  # M, N, K
            A.stride(0), A.stride(1),  # strides for A
            BT.stride(0), BT.stride(1),  # strides for BT
            C.stride(0), C.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=3,
        )

        # Scale by embed_scale using Triton elementwise kernel
        numel = C.numel()
        Y_scaled = torch.empty_like(C)
        grid_scale = (triton.cdiv(numel, 1024),)
        scale_kernel[grid_scale](C, Y_scaled, NUMEL=numel, SCALE=float(embed_scale))

        # Add positional embedding: [T3, 1024] first rows
        pos_emb = positional_embedding[:T3, :].to(torch.float32)  # [T3, 1024]
        Y_out = torch.empty_like(Y_scaled)
        grid_add = (triton.cdiv(numel, 1024),)
        add_pos_kernel[grid_add](Y_scaled, pos_emb.reshape(-1), Y_out, NUMEL=numel)

        # Reshape to [B, T3, 1024]
        out = Y_out.view(B, T3, 1024)
        return out


def run(*args):
    return ModelNew()(*args)
