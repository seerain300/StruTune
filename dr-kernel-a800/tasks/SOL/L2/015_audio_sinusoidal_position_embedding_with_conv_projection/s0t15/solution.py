import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: Conv2d for input channels = 1, 3x3, stride=2, padding=1, with GELU
@triton.jit
def conv2d_in1_c3s2p1_gelu(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, F_in, T_in, OC, F_out, T_out,
    x_sN, x_sC, x_sF, x_sT,
    w_sOC, w_sIC, w_sKH, w_sKW,
    y_sN, y_sOC, y_sF, y_sT,
    BLOCK_F: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid0 = tl.program_id(0)  # over B * T_out
    pid_f = tl.program_id(1)  # over F_out tiles
    pid_t = tl.program_id(2)  # over T_out tiles

    b = pid0 // T_out
    t_out = pid0 % T_out

    f_out_start = pid_f * BLOCK_F
    t_out_start = pid_t * BLOCK_T

    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)

    f_out = f_out_idx[:, None]  # [BF, 1]
    t_out_vec = t_out_idx[None, :]  # [1, BT]

    mask_f = f_out < F_out
    mask_t = t_out_vec < T_out
    out_mask = mask_f & mask_t

    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Accumulate over 3x3 kernel, single input channel
    for kh in range(3):
        for kw in range(3):
            f_in = f_out + 1 - kh
            t_in = t_out_vec + 1 - kw

            in_bounds = (f_in >= 0) & (f_in < F_in) & (t_in >= 0) & (t_in < T_in) & out_mask

            x_ptr = X_ptr + b * x_sN + 0 * x_sC + f_in * x_sF + t_in * x_sT
            x_val = tl.load(x_ptr, mask=in_bounds, other=0.0)

            # Load weight vector for all output channels at (kh, kw)
            for oc in range(0, OC):
                w_ptr = W_ptr + oc * w_sOC + 0 * w_sIC + kh * w_sKH + kw * w_sKW
                w_val = tl.load(w_ptr)  # scalar
                acc += x_val * w_val

    # Add bias
    for oc in range(0, OC):
        bias_val = tl.load(BIAS_ptr + oc)
        acc += bias_val

    # GELU: erf-based
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)
    x = acc
    x_scaled = x * inv_sqrt2
    # tl.math.erf is available in Triton
    erf_x = tl.math.erf(x_scaled)
    gelu = 0.5 * x * (1.0 + erf_x)

    # Store
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out * y_sF + t_out_vec * y_sT
    # Broadcast gelu over output channels: write for each oc
    for oc in range(0, OC):
        tl.store(y_ptrs + oc * y_sOC, gelu, mask=out_mask)


# Triton kernel: Conv2d for input channels >= 1, 3x3, stride=2, padding=1, with GELU
@triton.jit
def conv2d_in3_c3s2p1_gelu(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, IC, F_in, T_in, OC, F_out, T_out,
    x_sN, x_sC, x_sF, x_sT,
    w_sOC, w_sIC, w_sKH, w_sKW,
    y_sN, y_sOC, y_sF, y_sT,
    BLOCK_F: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid0 = tl.program_id(0)  # over B * T_out
    pid_f = tl.program_id(1)  # over F_out tiles
    pid_t = tl.program_id(2)  # over T_out tiles

    b = pid0 // T_out
    t_out = pid0 % T_out

    f_out_start = pid_f * BLOCK_F
    t_out_start = pid_t * BLOCK_T

    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)

    f_out = f_out_idx[:, None]
    t_out_vec = t_out_idx[None, :]

    mask_f = f_out < F_out
    mask_t = t_out_vec < T_out
    out_mask = mask_f & mask_t

    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            f_in = f_out + 1 - kh
            t_in = t_out_vec + 1 - kw

            in_bounds = (f_in >= 0) & (f_in < F_in) & (t_in >= 0) & (t_in < T_in) & out_mask

            for ic in range(0, IC):
                x_ptr = X_ptr + b * x_sN + ic * x_sC + f_in * x_sF + t_in * x_sT
                x_val = tl.load(x_ptr, mask=in_bounds, other=0.0)

                for oc in range(0, OC):
                    w_ptr = W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW
                    w_val = tl.load(w_ptr)
                    acc += x_val * w_val

    # Add bias
    for oc in range(0, OC):
        bias_val = tl.load(BIAS_ptr + oc)
        acc += bias_val

    # GELU
    inv_sqrt2 = 0.7071067811865476
    x = acc
    x_scaled = x * inv_sqrt2
    erf_x = tl.math.erf(x_scaled)
    gelu = 0.5 * x * (1.0 + erf_x)

    # Store
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out * y_sF + t_out_vec * y_sT
    for oc in range(0, OC):
        tl.store(y_ptrs + oc * y_sOC, gelu, mask=out_mask)


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


# Triton elementwise kernel: Y = X * SCALE + POS
@triton.jit
def scale_add_kernel(
    X_ptr, POS_ptr, Y_ptr,
    NUMEL: tl.int32, SCALE: tl.float32,
):
    pid = tl.program_id(0)
    offs = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
    mask = offs < NUMEL
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    pos = tl.load(POS_ptr + offs, mask=mask, other=0.0)
    y = x * SCALE + pos
    tl.store(Y_ptr + offs, y, mask=mask)


@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv2d1_weight: torch.Tensor,
    conv2d1_bias: torch.Tensor,
    conv2d2_weight: torch.Tensor,
    conv2d2_bias: torch.Tensor,
    conv2d3_weight: torch.Tensor,
    conv2d3_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    positional_embedding: torch.Tensor,
    embed_scale: float,
):
    # Dimensions
    B = input_features.shape[0]
    F_in = input_features.shape[2]  # 80
    T_in = input_features.shape[3]

    # Convolution 1: 1 -> 384, 3x3, stride=2, padding=1, GELU
    F1 = (F_in - 3) // 2 + 1  # 40
    T1 = (T_in - 3) // 2 + 1  # varies per workload
    x1 = torch.empty((B, 384, F1, T1), device=input_features.device, dtype=torch.float32)
    grid_conv1 = (B * T1, triton.cdiv(F1, 8), triton.cdiv(T1, 8))
    conv2d_in1_c3s2p1_gelu[grid_conv1](
        input_features, conv2d1_weight, conv2d1_bias, x1,
        B, F_in, T_in, 384, F1, T1,
        input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
        conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
        x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        BLOCK_F=8, BLOCK_T=8,
        num_warps=4, num_stages=2,
    )

    # GELU applied in kernel already

    # Convolution 2: 384 -> 384, 3x3, stride=2, padding=1, GELU
    F2 = (F1 - 3) // 2 + 1  # 20
    T2 = (T1 - 3) // 2 + 1
    x2 = torch.empty((B, 384, F2, T2), device=input_features.device, dtype=torch.float32)
    grid_conv2 = (B * T2, triton.cdiv(F2, 8), triton.cdiv(T2, 8))
    conv2d_in3_c3s2p1_gelu[grid_conv2](
        x1, conv2d2_weight, conv2d2_bias, x2,
        B, 384, F1, T1, 384, F2, T2,
        x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
        x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        BLOCK_F=8, BLOCK_T=8,
        num_warps=4, num_stages=2,
    )

    # Convolution 3: 384 -> 384, 3x3, stride=2, padding=1, GELU
    F3 = (F2 - 3) // 2 + 1  # 10
    T3 = (T2 - 3) // 2 + 1
    x3 = torch.empty((B, 384, F3, T3), device=input_features.device, dtype=torch.float32)
    grid_conv3 = (B * T3, triton.cdiv(F3, 8), triton.cdiv(T3, 8))
    conv2d_in3_c3s2p1_gelu[grid_conv3](
        x2, conv2d3_weight, conv2d3_bias, x3,
        B, 384, F2, T2, 384, F3, T3,
        x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
        x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        BLOCK_F=8, BLOCK_T=8,
        num_warps=4, num_stages=2,
    )

    # Reshape: (B, 384, F3, T3) -> (B, T3, 384*F3) = (B, T3, 3840)
    x_view = x3.permute(0, 3, 1, 2).contiguous().view(B, T3, 384 * F3)

    # Linear projection: [B*T3, 3840] @ [3840, 1024] -> [B*T3, 1024]
    M = B * T3
    K = 384 * F3
    N = 1024

    A = x_view.view(M, K).to(torch.float32)
    # conv_out_weight is [1024, 3840] -> B = [N=1024, K=3840]
    BT = conv_out_weight  # already [N, K]
    C = torch.empty((M, N), device=input_features.device, dtype=torch.float32)

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
    scale_add_kernel[grid_scale](
        C, torch.zeros(numel, dtype=C.dtype, device=C.device), Y,
        NUMEL=numel, SCALE=float(embed_scale),
    )

    # Add positional embedding: [T3, 1024] first rows
    pos_emb = positional_embedding[:T3, :].to(torch.float32)  # [T3, 1024]
    # Triton elementwise add kernel
    Y_out = torch.empty_like(Y)
    grid_add = (triton.cdiv(numel, 1024),)
    # We need to pass POS_ptr to scale_add_kernel as pos_emb flattened
    # However, scale_add_kernel was defined for scaling only. We will use it to add by setting SCALE=1.0 and passing pos_emb.
    # To avoid confusion, define a proper add kernel inline here:
    @triton.jit
    def add_kernel(X_ptr, Y_ptr, POS_ptr, NUMEL: tl.int32):
        pid = tl.program_id(0)
        offs = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
        mask = offs < NUMEL
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        pos = tl.load(POS_ptr + offs, mask=mask, other=0.0)
        y = x + pos
        tl.store(Y_ptr + offs, y, mask=mask)

    add_kernel[grid_add](Y, Y_out, pos_emb.reshape(-1), NUMEL=numel)

    # Reshape to [B, T3, 1024]
    out = Y_out.view(B, T3, N)
    return out


class ModelNew(nn.Module):
    def forward(self, *args):
        # Extract arguments (same as original Model.forward)
        input_features = args[0]
        conv2d1_weight = args[1]
        conv2d1_bias = args[2]


def run(*args):
    return ModelNew()(*args)
