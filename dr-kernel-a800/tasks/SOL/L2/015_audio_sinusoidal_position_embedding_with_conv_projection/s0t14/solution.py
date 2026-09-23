import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: Conv2d 3x3, stride=2, padding=1 for 1 input channel + GELU (erf-based)
@triton.jit
def conv2d_k3_s2_p1_in1_gelu(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, F_in, T_in, OC, F_out, T_out,
    x_sN, x_sC, x_sF, x_sT,
    w_sOC, w_sKH, w_sKW,
    y_sN, y_sOC, y_sF, y_sT,
    BLOCK_F: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    # Decompose pid into (b, oc, f_out, t_out)
    grid_f = tl.cdiv(F_out, BLOCK_F)
    grid_t = tl.cdiv(T_out, BLOCK_T)
    b = pid // (grid_t * grid_f * OC)
    rem = pid % (grid_t * grid_f * OC)
    oc = rem // (grid_t * grid_f)
    rem2 = rem % (grid_t * grid_f)
    t_out = rem2 // grid_t
    f_out = rem2 % grid_t

    f_out_start = f_out * BLOCK_F + tl.arange(0, BLOCK_F)
    t_out_start = t_out * BLOCK_T + tl.arange(0, BLOCK_T)

    f_out_idx = f_out_start[:, None]  # [BF, 1]
    t_out_idx = t_out_start[None, :]  # [1, BT]

    mask_f = f_out_idx < F_out
    mask_t = t_out_idx < T_out
    out_mask = mask_f & mask_t

    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Sum over 3x3 kernel, loop over input channel is implicit (IC=1 here)
    for kh in range(3):
        for kw in range(3):
            f_in = f_out_idx + 1 - kh  # [BF, 1]
            t_in = t_out_idx + 1 - kw  # [1, BT]
            in_bounds = (f_in >= 0) & (f_in < F_in) & (t_in >= 0) & (t_in < T_in) & out_mask

            x_ptrs = X_ptr + b * x_sN + 0 * x_sC + f_in * x_sF + t_in * x_sT
            x_val = tl.load(x_ptrs, mask=in_bounds, other=0.0)  # [BF, BT]

            # Load weight for this (oc, kh, kw). W has shape [OC, 1, 3, 3]
            w_val = tl.load(W_ptr + oc * w_sOC + kh * w_sKH + kw * w_sKW)  # scalar

            acc += x_val * w_val

    # Add bias
    bias_val = tl.load(BIAS_ptr + oc)
    acc += bias_val

    # GELU (erf-based): 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    u = acc * inv_sqrt2
    erf_u = tl.libdevice.erf(u)
    acc = 0.5 * acc * (1.0 + erf_u)

    # Store
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out_idx * y_sF + t_out_idx * y_sT
    tl.store(y_ptrs, acc, mask=out_mask)


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


# Triton elementwise kernel: Y = X * SCALE
@triton.jit
def scale_kernel(
    X_ptr, Y_ptr, NUMEL,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * SCALE + tl.arange(0, SCALE)
    mask = offs < NUMEL
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = x * SCALE
    tl.store(Y_ptr + offs, y, mask=mask)


# Triton elementwise kernel: Y = X + POS (add first T rows of positional_embedding)
@triton.jit
def add_pos_emb_kernel(
    X_ptr, POS_ptr, Y_ptr,
    B, T, N,
    x_sB, x_sT, x_sN,
    y_sB, y_sT, y_sN,
    pos_sT, pos_sN,
    T_rows,  # T_rows <= T
):
    pid = tl.program_id(0)
    # 1D launch over B*T*N
    idx = pid
    if idx >= B * T * N:
        return

    b = idx // (T * N)
    rem = idx % (T * N)
    t = rem // N
    n = rem % N

    # For t >= T_rows, set pos = 0
    t_is_valid = t < T_rows
    x_val = tl.load(X_ptr + b * x_sB + t * x_sT + n * x_sN)

    # Load corresponding positional embedding value at row t
    pos_val = tl.load(POS_ptr + t * pos_sT + n * pos_sN) if t_is_valid else 0.0

    y_val = x_val + pos_val
    tl.store(Y_ptr + b * y_sB + t * y_sT + n * y_sN, y_val)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Ensure CUDA tensors if Triton available; else fallback (not expected here)
        device = input_features.device
        assert device.type == 'cuda', "Input must be on CUDA device for Triton kernels."

        # Convolution 1: [B, 1, 80, T_in] -> [B, 384, 40, T1]
        B, IC, F_in, T_in = input_features.shape
        OC = conv2d1_weight.shape[0]
        F_out = (F_in + 2*1 - 3) // 2 + 1  # padding=1, stride=2
        T1 = (T_in + 2*1 - 3) // 2 + 1
        x1 = torch.empty((B, OC, F_out, T1), device=device, dtype=conv2d1_weight.dtype)
        grid = (B * tl.cdiv(F_out, 8) * tl.cdiv(T1, 32) * OC,)
        conv2d_k3_s2_p1_in1_gelu[grid](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, F_in, T_in, OC, F_out, T1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            BLOCK_F=8, BLOCK_T=32,
        )
        x1 = x1.to(torch.float32)

        # Convolution 2: [B, 384, 40, T1] -> [B, 384, 20, T2]
        F_in2, T_in2 = x1.shape[-2], x1.shape[-1]
        OC2 = conv2d2_weight.shape[0]
        assert OC2 == 384, "Expected conv2d2 output channels = 384"
        F_out2 = (F_in2 + 2*1 - 3) // 2 + 1
        T2 = (T_in2 + 2*1 - 3) // 2 + 1
        x2 = torch.empty((B, OC2, F_out2, T2), device=device, dtype=conv2d2_weight.dtype)
        grid2 = (B * tl.cdiv(F_out2, 8) * tl.cdiv(T2, 32) * OC2,)
        conv2d_k3_s2_p1_in1_gelu[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, F_in2, T_in2, OC2, F_out2, T2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            BLOCK_F=8, BLOCK_T=32,
        )
        x2 = x2.to(torch.float32)

        # Convolution 3: [B, 384, 20, T2] -> [B, 384, 10, T3]
        F_in3, T_in3 = x2.shape[-2], x2.shape[-1]
        OC3 = conv2d3_weight.shape[0]
        assert OC3 == 384, "Expected conv2d3 output channels = 384"
        F_out3 = (F_in3 + 2*1 - 3) // 2 + 1
        T3 = (T_in3 + 2*1 - 3) // 2 + 1
        x3 = torch.empty((B, OC3, F_out3, T3), device=device, dtype=conv2d3_weight.dtype)
        grid3 = (B * tl.cdiv(F_out3, 8) * tl.cdiv(T3, 32) * OC3,)
        conv2d_k3_s2_p1_in1_gelu[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, F_in3, T_in3, OC3, F_out3, T3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            BLOCK_F=8, BLOCK_T=32,
        )
        x3 = x3.to(torch.float32)

        # Reshape to [B, T3, 384*10]
        C = 384
        F_t = 10
        x_flat = x3.view(B, T3, C * F_t)  # [B, T3, 3840]

        # Linear projection: [B*T3, 3840] x [3840, 1024] -> [B*T3, 1024]
        M = B * T3
        K = C * F_t  # 3840
        N = 1024
        A = x_flat.to(torch.float32)  # [B*T3, 3840]
        # conv_out_weight is [N, K] in the original, we pass as [K, N] by transposing for the kernel
        BT = conv_out_weight.t().contiguous()  # [3840, 1024]
        Cmat = torch.empty((M, N), device=device, dtype=torch.float32)

        grid_mm = (triton.cdiv(M, 128), triton.cdiv(N, 64))
        matmul_kernel[grid_mm](
            A, BT, Cmat,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            Cmat.stride(0), Cmat.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
        )

        # Scale by embed_scale (sqrt(1024) = 32.0)
        Yscale = torch.empty_like(Cmat)
        num_elems = M * N
        scale_kernel[(num_elems // 1024 + (num_elems % 1024 > 0)) * 1024](  # grid size heuristic
            Cmat, Yscale, num_elems, SCALE=32.0
        )

        # Reshape to [B, T3, 1024]
        out = Yscale.view(B, T3, N)

        # Add positional embedding: first T3 rows of positional_embedding (shape [1500, 1024])
        # Convert to float32 and add
        pos_emb = positional_embedding[:T3, :].to(torch.float32)
        # Ensure all tensors have consistent strides; out is float32
        out_add = torch.empty_like(out)
        add_pos_emb_kernel[(B * T3 * N)](
            out, pos_emb, out_add,
            B, T3, N,
            out.stride(0), out.stride(1), out.stride(2),
            out_add.stride(0), out_add.stride(1), out_add.stride(2),
            pos_emb.stride(0), pos_emb.stride(1),
            T3,
        )

        return out_add


def run(*args):
    return ModelNew()(*args)
