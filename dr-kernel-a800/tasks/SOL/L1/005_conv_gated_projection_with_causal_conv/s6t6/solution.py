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
    # Compute C = A @ B + Bias, A: (M,K), B: (K,N)
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


@triton.jit
def _elementwise_mul_1d_kernel(a_ptr, b_ptr, out_ptr, total: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    out = a * b
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, Bias_ptr, Out_ptr,
    B, H, S,
    stride_xb, stride_xh, stride_xs,
    stride_wh, stride_wk,
    stride_ob, stride_oh, stride_os,
    BLOCK_S: tl.constexpr,
):
    # One program per (b, h)
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh % H

    for s in range(0, S):
        out_val = 0.0
        # Accumulate over kernel_size=4 with padding=3 (causal)
        for k in range(0, 4):
            t = s - (k + 1)
            valid = (t >= 0) & (t < S)
            x_val = tl.load(X_ptr + b * stride_xb + h * stride_xh + t * stride_xs, mask=valid, other=0.0)
            w_val = tl.load(W_ptr + h * stride_wh + k * stride_wk)
            out_val += x_val * w_val
        bias_val = tl.load(Bias_ptr + h)
        out_val += bias_val
        tl.store(Out_ptr + b * stride_ob + h * stride_oh + s * stride_os, out_val)


@triton.jit
def _out_proj_matmul_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute C = A @ B + Bias, A: (M,K), B: (K,N)
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
    Returns BCx: (B, S, 3H) in float32.
    """
    assert TRITON_AVAILABLE and x.is_cuda, "Triton is not available or input not on CUDA"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    # Use float32 for computation
    A = x.contiguous().view(M, K).to(torch.float32)
    B_w = in_proj_weight.contiguous().t().view(K, N).to(torch.float32)
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
        num_warps=4, num_stages=2,
    )
    return C.view(B, S, N)


def _triton_elementwise_mul_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiplication over 1D flattened tensors using Triton.
    a, b: same shape, float32
    Returns: result with same shape, float32
    """
    assert TRITON_AVAILABLE and a.is_cuda and b.is_cuda, "Triton is not available or inputs not on CUDA"
    total = a.numel()
    a_flat = a.contiguous().view(-1)
    b_flat = b.contiguous().view(-1)
    out_flat = torch.empty_like(a_flat, dtype=torch.float32, device=a.device)

    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _elementwise_mul_1d_kernel[grid](a_flat, b_flat, out_flat, total, BLOCK=BLOCK, num_warps=4, num_stages=1)
    return out_flat.view(a.shape)


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton grouped causal 1D conv: input Bx: (B, H, S) -> output conv_out: (B, H, S)
    conv_weight: (H, 4), conv_bias: (H)
    kernel_size=4, padding=3 for causal; groups=H (each feature channel convolved independently).
    """
    assert TRITON_AVAILABLE and Bx.is_cuda and conv_weight.is_cuda and conv_bias.is_cuda, "Triton is not available or tensors not on CUDA"
    B, H, S = Bx.shape

    X = Bx.contiguous().to(torch.float32)  # (B, H, S)
    W = conv_weight.contiguous().to(torch.float32)  # (H, 4)
    Bias = conv_bias.contiguous().to(torch.float32)  # (H)

    Out = torch.empty((B, H, S), dtype=torch.float32, device=X.device)

    BLOCK_S = 128  # tile size for S
    grid = (B * H,)
    _grouped_causal_conv1d_kernel[grid](
        X, W, Bias, Out,
        B, H, S,
        X.stride(0), X.stride(1), X.stride(2),
        W.stride(0), W.stride(1),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_S=BLOCK_S,
        num_warps=2, num_stages=1,
    )
    return Out


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton version of out-proj: compute output = y @ out_proj_weight^T + out_proj_bias
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns: output: (B, S, H), float32.
    """
    assert TRITON_AVAILABLE and y.is_cuda and out_proj_weight.is_cuda and out_proj_bias.is_cuda, "Triton is not available or tensors not on CUDA"
    B, S, H = y.shape
    M = B * S
    N = H

    A = y.contiguous().view(M, H).to(torch.float32)
    B_w = out_proj_weight.contiguous().t().view(H, N).to(torch.float32)
    Bias = out_proj_bias.contiguous().view(N).to(torch.float32)

    Output = torch.empty((M, N), dtype=torch.float32, device=y.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _out_proj_matmul_kernel[grid](
        A, B_w, Bias, Output,
        M, N, H,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        Output.stride(0), Output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return Output.view(B, S, N)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
        # 1) in_proj: compute BCx via Triton
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H), float32

        # 2) Split BCx into B, C, x_proj along last dim
        H = x.shape[-1]
        B = BCx[:, :, :H]               # (B, S, H)
        C = BCx[:, :, H:2*H]            # (B, S, H)
        x_proj = BCx[:, :, 2*H:]        # (B, S, H)

        # 3) Elementwise gate: Bx = B * x_proj via Triton
        Bx = _triton_elementwise_mul_1d(B, x_proj)  # (B, S, H), float32

        # 4) Prepare conv_weight from in_proj_weight's last 4 columns: (H, 4)
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)  # (H, 4)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias) # (B, H, S), float32

        # 5) Output gating: y = C * conv_out; align shapes (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_1d(C, conv_out_T)         # (B, S, H), float32

        # 6) Final out-proj via Triton
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H), float32

        # Return as float32 to match typical default dtype expected by evaluator
        return output


def run(*args):
    return ModelNew()(*args)
