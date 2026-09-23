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
    # Flatten launch grid: one program per (b, oc, f_block, t_block)
    total_blocks = B * OC * ((F_out + BLOCK_F - 1) // BLOCK_F) * ((T_out + BLOCK_T - 1) // BLOCK_T)
    pid = tl.program_id(0)
    grid_f_blocks = (F_out + BLOCK_F - 1) // BLOCK_F
    grid_t_blocks = (T_out + BLOCK_T - 1) // BLOCK_T

    # Decode pid -> (b, oc, f_block, t_block)
    oc = pid // (grid_f_blocks * grid_t_blocks)
    rem = pid % (grid_f_blocks * grid_t_blocks)
    f_block = rem // grid_t_blocks
    t_block = rem % grid_t_blocks

    b = B  # pid doesn't encode batch directly; grid0 size is B * ... so we need to recover b from total blocks
    # We cannot directly recover b from pid without second dimension; instead, relaunch conv per b inside Python.
    # But here we assume grid0 is constructed as B * (OC * grid_f_blocks * grid_t_blocks), which we do in Python.

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

                x_ptrs = X_ptr + b * x_sN + ic * x_sC + f_in * x_sF + t_in * x_sT
                x_val = tl.load(x_ptrs, mask=in_bounds, other=0.0)  # [BF, BT], float32

                # Load weight for (oc, ic, kh, kw): W shape [OC, IC, 3, 3]
                w_ptrs = W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW
                w_val = tl.load(w_ptrs)  # scalar

                acc += x_val * w_val

    # Add bias
    bias_val = tl.load(BIAS_ptr + oc)
    acc += bias_val

    # GELU (erf-based): 0.5*x*(1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    # Store to Y
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


# Triton elementwise kernel: Y_out = Y * SCALE + POS (add positional embedding)
@triton.jit
def scale_add_kernel(
    X_ptr, POS_ptr, Y_ptr,
    NUMEL: tl.constexpr, SCALE: tl.float32,
):
    pid = tl.program_id(0)
    offs = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
    mask = offs < NUMEL
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    pos = tl.load(POS_ptr + offs, mask=mask, other=0.0)
    y = x * SCALE + pos
    tl.store(Y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, *args):
        # args correspond to: (input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale)
        input_features = args[0]  # [B, 1, 80, time_dim]
        conv2d1_weight = args[1]  # [384, 1, 3, 3]
        conv2d1_bias = args[2]    # [384]
        conv2d2_weight = args[3]  # [384, 384, 3, 3]
        conv2d2_bias = args[4]    # [384]
        conv2d3_weight = args[5]  # [384, 384, 3, 3]
        conv2d3_bias = args[6]    # [384]
        conv_out_weight = args[7] # [1024, 3840]
        positional_embedding = args[8] # [max_source_positions, 1024], dtype can be bf16/fp16
        embed_scale = args[9]     # float

        B = input_features.shape[0]
        IC = input_features.shape[1]
        F_in = input_features.shape[2]
        T_in = input_features.shape[3]

        # Ensure dtypes are float32 for Triton (Triton kernels here assume fp32)
        X = input_features.contiguous().to(torch.float32)
        W1 = conv2d1_weight.contiguous().to(torch.float32)
        bias1 = conv2d1_bias.contiguous().to(torch.float32)
        W2 = conv2d2_weight.contiguous().to(torch.float32)
        bias2 = conv2d2_bias.contiguous().to(torch.float32)
        W3 = conv2d3_weight.contiguous().to(torch.float32)
        bias3 = conv2d3_bias.contiguous().to(torch.float32)

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        F_out1 = (F_in + 2*1 - 3)//2 + 1
        T_out1 = (T_in + 2*1 - 3)//2 + 1
        y1 = torch.empty((B, 384, F_out1, T_out1), device=X.device, dtype=torch.float32)

        grid0 = (B * 384 * triton.cdiv(F_out1, 8) * triton.cdiv(T_out1, 32),)
        conv3x3_s2_p1_gelu[grid0](
            X, W1, bias1, y1,
            B, IC, F_in, T_in, 384, F_out1, T_out1,
            X.stride(0), X.stride(1), X.stride(2), X.stride(3),
            W1.stride(0), W1.stride(1), W1.stride(2), W1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_F=8, BLOCK_T=32,
            num_warps=4, num_stages=2,
        )

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        F_in2 = F_out1
        T_in2 = T_out1
        F_out2 = (F_in2 + 2*1 - 3)//2 + 1
        T_out2 = (T_in2 + 2*1 - 3)//2 + 1
        y2 = torch.empty((B, 384, F_out2, T_out2), device=X.device, dtype=torch.float32)

        grid1 = (B * 384 * triton.cdiv(F_out2, 8) * triton.cdiv(T_out2, 32),)
        conv3x3_s2_p1_gelu[grid1](
            y1, W2, bias2, y2,
            B, 384, F_in2, T_in2, 384, F_out2, T_out2,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            W2.stride(0), W2.stride(1), W2.stride(2), W2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_F=8, BLOCK_T=32,
            num_warps=4, num_stages=2,
        )

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        F_in3 = F_out2
        T_in3 = T_out2
        F_out3 = (F_in3 + 2*1 - 3)//2 + 1
        T_out3 = (T_in3 + 2*1 - 3)//2 + 1
        y3 = torch.empty((B, 384, F_out3, T_out3), device=X.device, dtype=torch.float32)

        grid2 = (B * 384 * triton.cdiv(F_out3, 8) * triton.cdiv(T_out3, 32),)
        conv3x3_s2_p1_gelu[grid2](
            y2, W3, bias3, y3,
            B, 384, F_in3, T_in3, 384, F_out3, T_out3,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            W3.stride(0), W3.stride(1), W3.stride(2), W3.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_F=8, BLOCK_T=32,
            num_warps=4, num_stages=2,
        )

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        # We need channels*freq = 384 * (F_out3 * T_out3) is not 10. The original code does (B, T3, 1024).
        # However, the provided code returns (B, time_dim, 1024) after linear projection (which is conv_out_dim).
        # So, we directly compute the linear projection next.

        # Flatten for linear projection: x.view(B*T3, 3840) where T3 is the final time after conv3.
        # Compute T3
        T3 = T_out3

        # Linear projection: y3.view(B*T3, 384*10=3840) @ conv_out_weight.T
        # conv_out_weight is [1024, 3840], so we compute A[M= B*T3, K=3840] x B[K, N=1024]
        A = y3.reshape(B * T3, 3840).contiguous()
        BT = conv_out_weight.transpose(0, 1).contiguous()  # [3840, 1024]
        C = torch.empty((B * T3, 1024), device=X.device, dtype=torch.float32)

        grid_mm = (triton.cdiv(B * T3, 128), triton.cdiv(1024, 64))
        matmul_kernel[grid_mm](
            A, BT, C,
            B * T3, 1024, 3840,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=3,
        )

        # Reshape to [B, T3, 1024]
        out = C.view(B, T3, 1024)

        # Scale by embed_scale (sqrt(1024) = 32.0)
        Y = torch.empty_like(out)
        numel = B * T3 * 1024
        scale_add_kernel[(triton.cdiv(numel, 1024),)](
            out, torch.zeros(numel, dtype=out.dtype, device=out.device), Y,
            NUMEL=numel, SCALE=float(embed_scale),
        )

        # Add positional embedding: positional_embedding[:T3, :] of shape [T3, 1024]
        pos_emb = positional_embedding[:T3, :].to(torch.float32).contiguous()
        Y = torch.empty_like(Y)
        scale_add_kernel[(triton.cdiv(numel, 1024),)](
            Y, pos_emb.reshape(-1), Y,
            NUMEL=numel, SCALE=float(1.0),  # add, not scale
        )

        return Y


def run(*args):
    return ModelNew()(*args)
