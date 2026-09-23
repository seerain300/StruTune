import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Helper: erf approximation (tanh-based GELU approximation)
# Note: Triton does not provide tl.erf, so we implement GELU via erf approximation using torch in host.
# However, since we need Triton-only compute, we implement GELU inside Triton using a tanh-based approximation.
# GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
# This is standard fast GELU approximation.
# We will implement this in Triton kernels for conv post-ops.
# We will not use torch.nn.functional.gelu; instead, apply GELU in Triton.

# Constants for GELU tanh approximation
SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/pi)
GELU_COEFF = 0.044715

# Triton kernel: 2D Conv with 3x3, stride=2, padding=1, input channels=1
# Input: X[B, 1, F_in, T_in], W[OC, 1, 3, 3], Bias[OC], Output Y[B, OC, F_out, T_out]
@triton.jit
def conv2d_k3_s2_p1_in1_out(  # specific to input channels=1
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, F_in, T_in, OC, F_out, T_out,
    # strides
    x_sN, x_sC, x_sF, x_sT,
    w_sOC, w_sIC, w_sKH, w_sKW,
    y_sN, y_sOC, y_sF, y_sT,
    # tiling
    BLOCK_F: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # Grid: (B * T_out, ceil(F_out/BLOCK_F), ceil(T_out/BLOCK_T))
    pid0 = tl.program_id(0)
    pid_f = tl.program_id(1)
    pid_t = tl.program_id(2)

    b = pid0 // T_out
    t_out = pid0 % T_out

    f_out_start = pid_f * BLOCK_F
    t_out_start = pid_t * BLOCK_T

    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)

    # Create 2D grid for output tile
    f_out = f_out_idx[:, None]  # shape [BLOCK_F, 1]
    t_out_vec = t_out_idx[None, :]  # shape [1, BLOCK_T]
    mask_f = f_out < F_out
    mask_t = t_out_vec < T_out
    out_mask = mask_f & mask_t

    # Accumulator
    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Loop over output channels
    for oc in range(0, OC):
        # Sum over 3x3 kernel
        for kh in range(3):
            for kw in range(3):
                f_in = f_out + 1 - kh  # since padding=1, and output index maps to input as f_in = f_out*stride - padding + kh
                t_in = t_out_vec + 1 - kw

                # Mask for valid input
                in_bounds = (f_in >= 0) & (f_in < F_in) & (t_in >= 0) & (t_in < T_in) & out_mask

                # Load input X[b, 0, f_in, t_in]
                x_ptr = X_ptr + b * x_sN + 0 * x_sC + f_in * x_sF + t_in * x_sT
                x_val = tl.load(x_ptr, mask=in_bounds, other=0.0)

                # Load weight W[oc, 0, kh, kw]
                w_ptr = W_ptr + oc * w_sOC + 0 * w_sIC + kh * w_sKH + kw * w_sKW
                w_val = tl.load(w_ptr)  # scalar per oc, kh, kw

                acc += x_val * w_val

        # Add bias
        bias_val = tl.load(BIAS_ptr + oc)
        acc += bias_val

        # GELU via tanh approximation
        # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
        x = acc
        x3 = x * x * x
        inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3)
        tanh_inner = tl.tanh(inner)
        gelu = 0.5 * x * (1.0 + tanh_inner)
        acc = gelu

    # Store output Y[b, oc, f_out, t_out] for all oc -> we accumulated per oc loop above
    # We need to store for each oc separately, since Y is [B, OC, F_out, T_out]
    # But kernel is per (b, f_out, t_out), loop over oc:
    for oc in range(0, OC):
        y_ptr = Y_ptr + b * y_sN + oc * y_sOC + f_out * y_sF + t_out_vec * y_sT
        tl.store(y_ptr, acc, mask=out_mask)


# Triton kernel: Linear projection (matmul) A[M, K] x B[K, N] -> C[M, N]
# A: x[B*T, 3840], B: conv_out_weight[1024, 3840], C: [B*T, 1024]
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

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
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


# Triton kernel: Elementwise scale and add positional embedding
# X[B*T, N], POS[B*T, N], SCALE scalar, Y[B*T, N]
@triton.jit
def scale_add_pos_emb_kernel(
    X_ptr, POS_ptr, Y_ptr,
    L, N, SCALE,
    stride_xm, stride_xn,
    stride_pm, stride_pn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < L
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    pos_ptrs = POS_ptr + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    pos = tl.load(pos_ptrs, mask=mask, other=0.0)
    y = x * SCALE + pos
    tl.store(y_ptrs, y, mask=mask)


def _launch_conv2d_in1(B, F_in, T_in, OC, F_out, T_out, X, W, BIAS, Y):
    # Grid config
    BLOCK_F = 16
    BLOCK_T = 16
    grid = (B * T_out, triton.cdiv(F_out, BLOCK_F), triton.cdiv(T_out, BLOCK_T))
    # Launch Triton kernel
    conv2d_k3_s2_p1_in1_out[grid](
        X, W, BIAS, Y,
        B, F_in, T_in, OC, F_out, T_out,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        W.stride(0), W.stride(1), W.stride(2), W.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        BLOCK_F=BLOCK_F, BLOCK_T=BLOCK_T,
        num_warps=4, num_stages=2,
    )


def _launch_linear_matmul(A, B, C):
    # A: [M, K] = [B*T, 3840], B: [K, N] = [3840, 1024], C: [M, N]
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    BLOCK_M = 1
    BLOCK_N = 64
    BLOCK_K = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C


def _launch_scale_add_pos_emb(L, N, X, POS, Y, SCALE):
    # X: [L, N], POS: [L, N], Y: [L, N], all float32
    BLOCK_M = 32
    BLOCK_N = 64
    grid = (triton.cdiv(L, BLOCK_M), triton.cdiv(N, BLOCK_N))
    scale_add_pos_emb_kernel[grid](
        X, POS, Y,
        L, N, SCALE,
        X.stride(0), X.stride(1),
        POS.stride(0), POS.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; everything is computed via Triton kernels

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Ensure all tensors are on CUDA device and Triton is available
        assert TRITON_AVAILABLE, "Triton is not available"
        assert input_features.is_cuda, "Input must be on CUDA"
        assert conv2d1_weight.is_cuda and conv2d1_bias.is_cuda and conv2d2_weight.is_cuda and conv2d2_bias.is_cuda and conv2d3_weight.is_cuda and conv2d3_bias.is_cuda and conv_out_weight.is_cuda and positional_embedding.is_cuda, "All tensors must be on CUDA"

        # 1) Conv2d1: 1 -> 384, k=3x3, stride=2, padding=1
        B, C_in, F_in, T_in = input_features.shape  # C_in = 1
        OC1 = conv2d1_weight.shape[0]  # 384
        F_out1 = (F_in + 2*1 - 3)//2 + 1  # padding=1, stride=2
        T1 = (T_in + 2*1 - 3)//2 + 1
        x1 = torch.empty((B, OC1, F_out1, T1), dtype=torch.float32, device=input_features.device)
        _launch_conv2d_in1(B, F_in, T_in, OC1, F_out1, T1, input_features, conv2d1_weight, conv2d1_bias, x1)

        # GELU applied in-kernel already

        # 2) Conv2d2: 384 -> 384
        OC2 = conv2d2_weight.shape[0]  # 384
        F_in2 = F_out1
        T_in2 = T1
        F_out2 = (F_in2 + 2*1 - 3)//2 + 1
        T2 = (T_in2 + 2*1 - 3)//2 + 1
        x2 = torch.empty((B, OC2, F_out2, T2), dtype=torch.float32, device=input_features.device)
        _launch_conv2d_in1(B, F_in2, T_in2, OC2, F_out2, T2, x1, conv2d2_weight, conv2d2_bias, x2)

        # GELU applied in-kernel already

        # 3) Conv2d3: 384 -> 384
        OC3 = conv2d3_weight.shape[0]  # 384
        F_in3 = F_out2
        T_in3 = T2
        F_out3 = (F_in3 + 2*1 - 3)//2 + 1
        T3 = (T_in3 + 2*1 - 3)//2 + 1  # should be 10
        x3 = torch.empty((B, OC3, F_out3, T3), dtype=torch.float32, device=input_features.device)
        _launch_conv2d_in1(B, F_in3, T_in3, OC3, F_out3, T3, x2, conv2d3_weight, conv2d3_bias, x3)

        # GELU applied in-kernel already

        # Reshape to [B, T3, 384*10] -> [B, 10, 3840]
        x3_flat = x3.view(B, T3, OC3 * F_out3)  # OC3 == 384, F_out3 == 10
        x3_2d = x3_flat.reshape(B * T3, OC3 * F_out3)  # [M, 3840], M = B*T3
        N_out = 1024
        W_out = conv_out_weight  # [N_out, 3840] -> [1024, 3840]
        # 4) Linear projection via Triton matmul
        C_mat = _launch_linear_matmul(x3_2d, W_out, None)  # returns [M, N_out]
        C = C_mat.view(B, T3, N_out)  # [B, 10, 1024]

        # 5) Scale by embed_scale and add positional embedding
        # Ensure positions embedding is contiguous float32 and slice first T3 rows
        POS = positional_embedding[:T3, :].to(torch.float32).contiguous()  # [T3, 1024]
        L = B * T3
        N = N_out
        C_scaled = torch.empty((L, N), dtype=torch.float32, device=C.device)
        # We need to flatten C for elementwise ops
        C_flat = C.view(L, N).contiguous()
        # Launch Triton kernel to compute Y = C_flat * embed_scale + POS
        _launch_scale_add_pos_emb(L, N, C_flat, POS, C_scaled, float(embed_scale))
        # Reshape back to [B, T3, 1024]
        Y = C_scaled.view(B, T3, N)

        return Y


# The rest (get_inputs, run) can remain the same interface; but we only need ModelNew here.


def run(*args):
    return ModelNew()(*args)
