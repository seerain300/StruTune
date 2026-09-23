import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _matmul_linear_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # C = A @ B + Bias, A: (M, K), B: (K, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if OUT_DTYPE == 0:
        out_val = acc
    elif OUT_DTYPE == 1:
        out_val = acc.to(tl.float16)
    elif OUT_DTYPE == 2:
        out_val = acc.to(tl.bfloat16)
    else:
        out_val = acc

    tl.store(c_ptrs, out_val, mask=c_mask)


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # C = A * B, A: (M, N), B: (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)
    c = a * b

    if OUT_DTYPE == 0:
        out_val = c
    elif OUT_DTYPE == 1:
        out_val = c.to(tl.float16)
    elif OUT_DTYPE == 2:
        out_val = c.to(tl.bfloat16)
    else:
        out_val = c

    tl.store(c_ptrs, out_val, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, Bias_ptr, Out_ptr,
    B, H, S,
    stride_xb, stride_xh, stride_xs,
    stride_w_h, stride_w_k,
    stride_ob, stride_oh, stride_os,
    OUT_DTYPE: tl.constexpr,
):
    # Each program computes one output element: conv_out[b, h, s] for b in [0,B), h in [0,H), s in [0,S)
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    # kernel_size=4, causal padding implies shifting input by s+3 on the left
    for k in range(4):
        idx = s + 3 - k
        in_range = (idx >= 0) & (idx < S)
        # load x[b, h, idx] if in_range else 0
        x_val = tl.load(X_ptr + b * stride_xb + h * stride_xh + idx * stride_xs, mask=in_range, other=0.0).to(tl.float32)
        w_val = tl.load(W_ptr + h * stride_w_h + k * stride_w_k).to(tl.float32)
        acc += x_val * w_val

    if Bias_ptr is not None:
        bias_val = tl.load(Bias_ptr + h).to(tl.float32)
        acc += bias_val

    if OUT_DTYPE == 0:
        out_val = acc
    elif OUT_DTYPE == 1:
        out_val = acc.to(tl.float16)
    elif OUT_DTYPE == 2:
        out_val = acc.to(tl.bfloat16)
    else:
        out_val = acc

    tl.store(Out_ptr + b * stride_ob + h * stride_oh + s * stride_os, out_val)


@triton.jit
def _out_proj_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # C = A @ B + Bias, A: (M, K), B: (K, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    if Bias_ptr is not None:
        bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if OUT_DTYPE == 0:
        out_val = acc
    elif OUT_DTYPE == 1:
        out_val = acc.to(tl.float16)
    elif OUT_DTYPE == 2:
        out_val = acc.to(tl.bfloat16)
    else:
        out_val = acc

    tl.store(c_ptrs, out_val, mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only forward:
        - in_proj: BCx = x @ in_proj_weight^T + in_proj_bias (Triton matmul + bias)
        - Split BCx into B, C, x_proj (each (B, S, H))
        - Elementwise gate: Bx = B * x_proj (Triton)
        - Grouped causal conv (kernel_size=4, groups=H) with conv_weight = in_proj_weight[:, -4:] reshaped (H, 4)
          Implemented directly in Triton with masked indexing (no F.pad).
        - Output gating: y = C * conv_out (conv_out is (B, H, S); transpose to (B, S, H) via elementwise multiply on transposed view)
        - Final out-proj: output = y @ out_proj_weight^T + out_proj_bias (Triton matmul + bias)
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        assert x.ndim == 3, "x must be (B, S, H)"
        B, S, H = x.shape

        # Determine output dtype from x (to match original behavior)
        if x.dtype == torch.float32:
            out_dtype_code = 0
        elif x.dtype == torch.float16:
            out_dtype_code = 1
        elif x.dtype == torch.bfloat16:
            out_dtype_code = 2
        else:
            out_dtype_code = 0

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias (A: (B*S, H), B: (H, 3H))
        K = H
        N = 3 * H
        M = B * S

        # Flatten input and weight for matmul
        A = x.contiguous().view(M, K)
        B_w = in_proj_weight.t().contiguous().view(K, N)
        Bias_in = in_proj_bias.contiguous().view(N)

        # Allocate output tensor and let Triton write it (we still need a torch tensor; Triton cannot allocate)
        BCx = torch.empty((B, S, 3 * H), dtype=x.dtype, device=x.device)
        C_BCx = BCx.view(M, N)

        BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_linear_kernel[grid](
            A, B_w, Bias_in, C_BCx,
            M, N, K,
            A.stride(0), A.stride(1),
            B_w.stride(0), B_w.stride(1),
            C_BCx.stride(0), C_BCx.stride(1),
            out_dtype_code,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # Split BCx into B, C, x_proj (avoid .view/.transpose; directly index)
        # BCx is (B, S, 3H), split by last dim
        B_mat = BCx[:, :, :H]            # (B, S, H)
        C_mat = BCx[:, :, H:2*H]         #


def run(*args):
    return ModelNew()(*args)
