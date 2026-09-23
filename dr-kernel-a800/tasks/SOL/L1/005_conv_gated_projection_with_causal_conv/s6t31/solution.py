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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute C = A @ B + Bias, A: (M,K), B: (K,N), Bias: (N,)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols in N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # along K
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Add bias: broadcast bias over rows
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # C = A * B, A and B: (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(a_ptrs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptrs, mask=mask, other=1.0).to(tl.float32)
    c = a * b

    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, Bias_ptr, Out_ptr,
    B, H, S, P,  # P = kernel_size - 1 = 3
    stride_xb, stride_xh, stride_xs,
    stride_wh, stride_wk,
    stride_outb, stride_outh, stride_outs,
    BLOCK_S: tl.constexpr,
):
    """
    Grouped causal 1D conv with groups=H and kernel_size=4 (P=3 padding).
    Input X is already padded along sequence to length S+P, shape (B, H, S+P).
    Output Out: (B, H, S).
    For each (b,h,s): Out[b,h,s] = sum_{k=0..3} X[b,h,s+P-k] * W[h,k] + Bias[h]
    """
    pid = tl.program_id(0)
    # One program per (b, h) pair
    b = pid // H
    h = pid % H

    offs_s = tl.arange(0, BLOCK_S)
    s = offs_s + tl.zeros((), dtype=tl.int32)  # scalar block
    mask_s = (s < S)

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Loop over kernel positions k=0..3
    for k in range(4):
        idx = s + P - k  # padded index
        x_ptrs = X_ptr + (b * stride_xb + h * stride_xh + idx * stride_xs)
        # load as vector over BLOCK_S, mask by validity of idx (idx in [0, S+P-1])
        x = tl.load(x_ptrs, mask=(idx >= 0) & (idx < (S + P)) & mask_s, other=0.0)
        w = tl.load(W_ptr + (h * stride_wh + k * stride_wk), mask=True, other=0.0)
        acc += x * w

    # Add bias[h]
    bias = tl.load(Bias_ptr + h, mask=True, other=0.0).to(tl.float32)
    acc += bias

    out_ptrs = Out_ptr + (b * stride_outb + h * stride_outh + s * stride_outs)
    tl.store(out_ptrs, acc, mask=mask_s)


@triton.jit
def _matmul_linear_kernel_out_proj(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, K, N,  # M = B*S, K = H, N = H
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B + Bias
    A: (M, K), float32
    B: (K, N), float32
    Bias: (N,), float32
    C: (M, N), float32
    """
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
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute BCx = x @ in_proj_weight^T + in_proj_bias.
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns BCx: (B, S, 3H), float32. Triton kernel is used.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    A = x.contiguous().view(M, K).to(torch.float32)
    B_w = in_proj_weight.t().contiguous().view(K, N).to(torch.float32)
    Bias = in_proj_bias.contiguous().view(N).to(torch.float32)

    C = torch.empty((M, N), dtype=torch.float32, device=x.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        A, B_w, Bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C.view(B, S, N)


def _triton_elementwise_mul_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiply A and B as 2D tensors, using Triton.
    a, b: (M, N), return (M, N)
    """
    assert a.shape == b.shape, "A and B must have the same shape"
    M, N = a.shape
    C = torch.empty((M, N), dtype=torch.float32, device=a.device)
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _elementwise_mul_2d_kernel[grid](
        a, b, C,
        M, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return C


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Grouped causal 1D convolution:
    - Input Bx: (B, H, S) contiguous
    - conv_weight: (H, 4), conv_bias: (H,)
    - Output: (B, H, S)
    We pre-padded Bx to (B, H, S+3) in Python code before calling this kernel.
    """
    B, H, S = Bx.shape
    # Ensure dtype and contiguity
    Bx = Bx.to(torch.float32).contiguous()                     # (B, H, S+P)
    conv_weight = conv_weight.to(torch.float32).contiguous()   # (H, 4)
    conv_bias = conv_bias.to(torch.float32).contiguous()       # (H,)

    Out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)

    # Launch one program per (b, h)
    grid = (B * H,)
    BLOCK_S = 128
    _grouped_causal_conv1d_kernel[grid](
        Bx, conv_weight, conv_bias, Out,
        B, H, S, 3,  # P=3
        Bx.stride(0), Bx.stride(1), Bx.stride(2),
        conv_weight.stride(0), conv_weight.stride(1),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_S=BLOCK_S,
    )
    return Out


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute output = y @ out_proj_weight^T + out_proj_bias
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns output: (B, S, H), float32. Triton kernel is used.
    """
    B, S, H = y.shape
    M = B * S
    K = H
    N = H

    A = y.contiguous().view(M, K).to(torch.float32)
    B_w = out_proj_weight.t().contiguous().view(K, N).to(torch.float32)
    Bias = out_proj_bias.contiguous().view(N).to(torch.float32)

    C = torch.empty((M, N), dtype=torch.float32, device=y.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel_out_proj[grid](
        A, B_w, Bias, C,
        M, K, N,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C.view(B, S, N)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only forward. No PyTorch functional calls.
        Implements:
        1) BCx = in_proj(x) using Triton matmul.
        2) Split BCx into B, C, x_proj; gate Bx = B * x_proj using Triton.
        3) Pre-pad Bx to (B, H, S+3), conv with grouped causal kernel (kernel_size=4), Triton.
        4) Gate y = C * conv_out (after transposing conv_out to (B, S, H)), Triton.
        5) Final out-proj using Triton matmul.
        All outputs are float32 tensors.
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        B, S, H = x.shape

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias, output (B, S, 3H)
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H), float32

        # Split BCx into B, C, x_proj
        Bv = BCx[:, :, :H]               # (B, S, H)
        Cv = BCx[:, :, H:2*H]            # (B, S, H)
        x_proj = BCx[:, :, 2*H:]         # (B, S, H)

        # 2) Elementwise gate: Bx = B * x_proj
        Bx = _triton_elementwise_mul_2d(Bv, x_proj)           # (B, S, H), float32

        # 3) Grouped causal 1D conv: conv_weight derived from in_proj_weight's last 4 columns
        conv_w = in_proj_weight[:, -4:].contiguous()          # (H, 4)
        conv_out = _triton_grouped_causal_conv1d(
            Bx, conv_w, conv_bias                              # conv_out: (B, H, S)
        )

        # 4) Output gating: y = C * conv_out; align shapes (B, S, H)
        # C is (B, S, H); conv_out is (B, H, S). Transpose conv_out to (B, S, H) then gate.
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_2d(Cv, conv_out_T)        # (B, S, H), float32

        # 5) Final out-proj
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H), float32

        return output


def run(*args):
    return ModelNew()(*args)
