import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton Conv2d: 3x3, stride=2, padding=1, for IC=1 input channel
@triton.jit
def conv2d_k3_s2_p1_ic1(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, F_in, T_in, OC, F_out, T_out,
    x_sN, x_sC, x_sF, x_sT,
    w_sOC, w_sIC, w_sKH, w_sKW,
    y_sN, y_sOC, y_sF, y_sT,
    BLOCK_F: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # Each program computes a tile of output over (f_out, t_out) for one batch b
    pid0 = tl.program_id(0)  # over B * T_out
    pid_f = tl.program_id(1)
    pid_t = tl.program_id(2)

    b = pid0 // T_out
    t_out = pid0 % T_out

    f_out_start = pid_f * BLOCK_F
    t_out_start = pid_t * BLOCK_T

    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)

    f_out = f_out_idx[:, None]  # shape [BF, 1]
    t_out_vec = t_out_idx[None, :]  # shape [1, BT]

    out_mask = (f_out < F_out) & (t_out_vec < T_out)

    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Sum over 3x3 kernel
    for kh in range(3):
        for kw in range(3):
            f_in = f_out + 1 - kh  # 1-based due to padding
            t_in = t_out_vec + 1 - kw

            in_bounds = (f_in >= 0) & (f_in < F_in) & (t_in >= 0) & (t_in < T_in) & out_mask

            # X is [B, 1, F_in, T_in], load x[b, 0, f_in, t_in]
            x_ptr = X_ptr + b * x_sN + 0 * x_sC + f_in * x_sF + t_in * x_sT
            x_val = tl.load(x_ptr, mask=in_bounds, other=0.0)

            # Weight vector for oc in [0..OC)
            # W is [OC, 1, 3, 3], ic fixed to 0
            w_vec = tl.load(W_ptr + tl.arange(0, OC) * w_sOC + 0 * w_sIC + kh * w_sKH + kw * w_sKW)

            # Outer product accumulate across oc
            for oc in range(0, OC):
                w_scalar = tl.load(W_ptr + oc * w_sOC + 0 * w_sIC + kh * w_sKH + kw * w_sKW)
                acc += x_val * w_scalar

    # Add bias
    for oc in range(0, OC):
        bias_val = tl.load(BIAS_ptr + oc)
        acc += bias_val

    # GELU via erf: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    x = acc
    gelu = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))

    # Store to Y
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out * y_sF + t_out_vec * y_sT
    tl.store(y_ptrs, gelu, mask=out_mask)


# Triton Conv2d: 3x3, stride=2, padding=1, for general input channel (IC >= 1)
@triton.jit
def conv2d_k3_s2_p1_ic(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, IC, F_in, T_in, OC, F_out, T_out,
    x_sN, x_sC, x_sF, x_sT,
    w_sOC, w_sIC, w_sKH, w_sKW,
    y_sN, y_sOC, y_sF, y_sT,
    BLOCK_F: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid0 = tl.program_id(0)  # over B * T_out
    pid_f = tl.program_id(1)
    pid_t = tl.program_id(2)

    b = pid0 // T_out
    t_out = pid0 % T_out

    f_out_start = pid_f * BLOCK_F
    t_out_start = pid_t * BLOCK_T

    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)

    f_out = f_out_idx[:, None]
    t_out_vec = t_out_idx[None, :]

    out_mask = (f_out < F_out) & (t_out_vec < T_out)

    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            f_in = f_out + 1 - kh
            t_in = t_out_vec + 1 - kw

            in_bounds = (f_in >= 0) & (f_in < F_in) & (t_in >= 0) & (t_in < T_in) & out_mask

            for ic in range(0, IC):
                x_ptr = X_ptr + b * x_sN + ic * x_sC + f_in * x_sF + t_in * x_sT
                x_val = tl.load(x_ptr, mask=in_bounds, other=0.0)

                # W is [OC, IC, 3, 3]
                w_vec = tl.load(W_ptr + tl.arange(0, OC) * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW)

                for oc in range(0, OC):
                    w_scalar = tl.load(W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW)
                    acc += x_val * w_scalar

    # Add bias
    for oc in range(0, OC):
        bias_val = tl.load(BIAS_ptr + oc)
        acc += bias_val

    # GELU
    inv_sqrt2 = 0.7071067811865476
    x = acc
    gelu = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))

    # Store to Y
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out * y_sF + t_out_vec * y_sT
    tl.store(y_ptrs, gelu, mask=out_mask)


# Triton matmul: A[M, K] x B[K, N] -> C[M, N]
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

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton elementwise scaling
@triton.jit
def scale_kernel(X_ptr, Y_ptr, NUMEL: tl.int32, SCALE: tl.float32):
    pid = tl.program_id(0)
    offs = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
    mask = offs < NUMEL
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = x * SCALE
    tl.store(Y_ptr + offs, y, mask=mask)


# Triton elementwise add with positional embedding
@triton.jit
def add_pos_emb_kernel(X_ptr, POS_ptr, Y_ptr, NUMEL: tl.int32):
    pid = tl.program_id(0)
    offs = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
    mask = offs < NUMEL
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    pos = tl.load(POS_ptr + offs, mask=mask, other=0.0)
    y = x + pos
    tl.store(Y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, *args):
        # Unpack inputs (same as original signature)
        input_features = args[0]             # [B, 1, 80, time_dim], bf16
        conv2d1_weight = args[1]             # [384, 1, 3, 3], bf16
        conv2d1_bias = args[2]               # [384], bf16
        conv2d2_weight = args[3]             # [384, 384, 3, 3], bf16
        conv2d2_bias = args[4]               # [384], bf16
        conv2d3_weight = args[5]             # [384, 384, 3, 3], bf16
        conv2d3_bias = args[6]               # [384], bf16
        conv_out_weight = args[7]            # [1024, 3840], bf16 (note: we use weight.T for matmul)
        positional_embedding = args[8]       # [max_source_positions, 1024], bf16
        embed_scale = args[9]                # float

        device = input_features.device
        dtype = input_features.dtype

        # Ensure tensors are on the same device and dtype
        conv2d1_weight = conv2d1_weight.to(device=device, dtype=dtype)
        conv2d1_bias = conv2d1_bias.to(device=device, dtype=dtype)
        conv2d2_weight = conv2d2_weight.to(device=device, dtype=dtype)
        conv2d2_bias = conv2d2_bias.to(device=device, dtype=dtype)
        conv2d3_weight = conv2d3_weight.to(device=device, dtype=dtype)
        conv2d3_bias = conv2d3_bias.to(device=device, dtype=dtype)
        conv_out_weight = conv_out_weight.to(device=device, dtype=dtype)
        positional_embedding = positional_embedding.to(device=device, dtype=dtype)

        # Constants
        B = input_features.shape[0]
        F_in = input_features.shape[2]
        T_in = input_features.shape[3]
        OC1 = conv2d1_weight.shape[0]  # 384
        F1_out = (F_in - 3) // 2 + 1    # 80 - 3 -> 77 // 2 + 1 = 39
        T1 = T_in // 2 - 1 + 1          # (time_dim - 3)//2 + 1
        OC2 = conv2d2_weight.shape[0]   # 384
        F2_out = (F1_out - 3) // 2 + 1  # 39 - 3 -> 36 // 2 + 1 = 19
        T2 = T1 // 2 - 1 + 1
        OC3 = conv2d3_weight.shape[0]   # 384
        F3_out = (F2_out - 3) // 2 + 1  # 19 - 3 -> 16 // 2 + 1 = 9
        T3 = T2 // 2 - 1 + 1

        # Allocate output for conv1
        y1 = torch.empty((B, OC1, F1_out, T1), device=device, dtype=dtype)

        # Launch conv2d_k3_s2_p1_ic1 for conv1
        BLOCK_F = 8
        BLOCK_T = 16
        grid1 = (B * T1, triton.cdiv(F1_out, BLOCK_F), triton.cdiv(T1, BLOCK_T))
        conv2d_k3_s2_p1_ic1[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            B, F_in, T_in, OC1, F1_out, T1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_F=BLOCK_F, BLOCK_T=BLOCK_T,
        )

        # GELU already applied in kernel

        # Allocate output for conv2
        y2 = torch.empty((B, OC2, F2_out, T2), device=device, dtype=dtype)

        # Launch conv2d_k3_s2_p1_ic for conv2
        BLOCK_F2 = 8
        BLOCK_T2 = 16
        grid2 = (B * T2, triton.cdiv(F2_out, BLOCK_F2), triton.cdiv(T2, BLOCK_T2))
        conv2d_k3_s2_p1_ic[grid2](
            y1, conv2d2_weight, conv2d2_bias, y2,
            B, OC1, F1_out, T1, OC2, F2_out, T2,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_F=BLOCK_F2, BLOCK_T=BLOCK_T2,
        )

        # Allocate output for conv3
        y3 = torch.empty((B, OC3, F3_out, T3), device=device, dtype=dtype)

        # Launch conv2d_k3_s2_p1_ic for conv3
        BLOCK_F3 = 8
        BLOCK_T3 = 16
        grid3 = (B * T3, triton.cdiv(F3_out, BLOCK_F3), triton.cdiv(T3, BLOCK_T3))
        conv2d_k3_s2_p1_ic[grid3](
            y2, conv2d3_weight, conv2d3_bias, y3,
            B, OC2, F2_out, T2, OC3, F3_out, T3,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_F=BLOCK_F3, BLOCK_T=BLOCK_T3,
        )

        # Reshape: [B, 384, 10, T3] -> [B, T3, 384*10] -> [B, T3, 3840]
        x = y3.view(B, T3, OC3 * F3_out)

        # Linear projection: A[M, K] = x.view(B*T3, 3840), B[K, N] = conv_out_weight.T (1024, 3840)
        M = B * T3
        K = 3840
        N = 1024
        A = x.view(M, K).to(torch.float32)  # compute in fp32
        BT = conv_out_weight.t().contiguous().to(torch.float32)  # [K, N]
        C = torch.empty((M, N), device=device, dtype=torch.float32)

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
        Y = torch.empty_like(C)
        numel = M * N
        grid_scale = (triton.cdiv(numel, 1024),)
        # Prepare a scalar tensor for SCALE
        scale_tensor = torch.tensor(float(embed_scale), dtype=C.dtype, device=C.device)
        # Triton expects scalar directly; here we pass float
        # Launch scale kernel
        scale_kernel[grid_scale](C, Y, NUMEL=numel, SCALE=float(embed_scale))

        # Add positional embedding: [T3, 1024] first rows
        pos_emb = positional_embedding[:T3, :].to(torch.float32)  # [T3, 1024]
        Y_out = torch.empty_like(Y)
        grid_add = (triton.cdiv(numel, 1024),)
        add_pos_emb_kernel[grid_add](Y, pos_emb.reshape(-1), Y_out, NUMEL=numel)

        # Reshape to [B, T3, 1024]
        out = Y_out.view(B, T3, N)
        return out


def run(*args):
    return ModelNew()(*args)
