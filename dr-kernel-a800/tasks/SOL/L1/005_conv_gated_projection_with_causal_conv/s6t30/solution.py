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
    """
    Compute C = A @ B + Bias
    A: (M, K), row-major, float32
    B: (K, N), row-major, float32
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
    Returns BCx: (B, S, 3H), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    # Ensure float32 for computation
    A = x.contiguous().view(M, H).to(torch.float32)            # (M, K)
    B_w = in_proj_weight.t().contiguous().view(H, N).to(torch.float32)  # (K, N)
    Bias = in_proj_bias.contiguous().view(N).to(torch.float32)  # (N)

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


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    C = A * B, elementwise 2D
    A: (M, N), float32
    B: (M, N), float32
    C: (M, N), float32
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(a_ptrs, mask=mask, other=1.0)
    b = tl.load(b_ptrs, mask=mask, other=1.0)

    c = a * b
    tl.store(c_ptrs, c, mask=mask)


def _triton_elementwise_mul_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiply a and b. Assumes both (M, N), returns (M, N), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert a.shape == b.shape, "a and b must have the same shape"
    M, N = a.shape
    A = a.contiguous().to(torch.float32)
    B = b.contiguous().to(torch.float32)
    C = torch.empty((M, N), dtype=torch.float32, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _elementwise_mul_2d_kernel[grid](
        A, B, C,
        M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return C


@triton.jit
def _grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, Bias_ptr, Out_ptr,
    B, H, S, P,  # P = S + pad = S + 3
    stride_xb, stride_xh, stride_xs,
    stride_wh, stride_wk,
    stride_outb, stride_outh, stride_outs,
    BLOCK_S: tl.constexpr,
):
    """
    Grouped causal 1D convolution:
    - Input X: (B, H, P) where P = S + pad_left (pad_left = 3), float32
    - Weight W: (H, 4), float32
    - Bias: (H,), float32
    - Output Out: (B, H, S), float32
    Groups are handled by iterating h in [0, H) and writing to Out[b, h, s] for s in [0, S).
    """
    pid_bh = tl.program_id(0)  # over B * H
    b = pid_bh // H
    h = pid_bh % H

    # Prepare output base pointer
    out_base = Out_ptr + b * stride_outb + h * stride_outh

    # Loop over sequence positions s=0..S-1 in tiles
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S

        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # kernel_size=4, causal padding left of 3
        # conv_out[b, h, s] = sum_{k=0..3} X[b, h, s + 3 - k] * W[h, k] + Bias[h]
        for k in range(0, 4):
            pos = offs_s + 3 - k
            x_ptrs = X_ptr + b * stride_xb + h * stride_xh + pos * stride_xs
            x_vals = tl.load(x_ptrs, mask=mask_s, other=0.0)  # (BLOCK_S,)
            w_val = tl.load(W_ptr + h * stride_wh + k * stride_wk)  # scalar
            acc += x_vals * w_val

        # Add bias
        bias_val = tl.load(Bias_ptr + h)  # scalar
        acc += bias_val

        out_ptrs = out_base + offs_s * stride_outs
        tl.store(out_ptrs, acc, mask=mask_s)


def _triton_grouped_causal_conv1d(x_padded: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute grouped causal 1D conv:
    x_padded: (B, H, S+3), float32
    conv_weight: (H, 4), float32
    conv_bias: (H,), float32
    returns: (B, H, S), float32
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, P = x_padded.shape
    S = P - 3
    Out = torch.empty((B, H, S), dtype=torch.float32, device=x_padded.device)

    # Strides (note: we pass element strides, Triton uses element-based addressing)
    stride_xb, stride_xh, stride_xs = x_padded.stride()
    stride_wh, stride_wk = conv_weight.stride()
    stride_outb, stride_outh, stride_outs = Out.stride()

    # Launch grid: one program per (b, h)
    grid = (B * H,)
    BLOCK_S = 128
    _grouped_causal_conv1d_kernel[grid](
        x_padded, conv_weight, conv_bias, Out,
        B, H, S, P,
        stride_xb, stride_xh, stride_xs,
        stride_wh, stride_wk,
        stride_outb, stride_outh, stride_outs,
        BLOCK_S=BLOCK_S,
    )
    return Out


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


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute output = y @ out_proj_weight^T + out_proj_bias.
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns output: (B, S, H), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    M = B * S
    K = H
    N = H

    A = y.contiguous().view(M, K).to(torch.float32)            # (M, K)
    B_w = out_proj_weight.t().contiguous().view(K, N).to(torch.float32)  # (K, N)
    Bias = out_proj_bias.contiguous().view(N).to(torch.float32)  # (N)

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
    return C.view(B, S, H)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
        """
        Triton-only fused forward:
        1) BCx = in_proj(x)
        2) Split into B, C, x_proj and elementwise gate Bx = B * x_proj
        3) Grouped causal conv with kernel_size=4 using conv_weight derived from in_proj_weight's last 4 columns
        4) Output gating: y = C * conv_out
        5) Final out-proj: output = y @ out_proj_weight^T + out_proj_bias
        """
        # 1) in-proj: BCx = x @ in_proj_weight^T + in_proj_bias, return float32
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H), float32

        # Split into B, C, x_proj along last dim
        B = BCx[:, :, :x.shape[-1]]              # (B, S, H)
        C = BCx[:, :, x.shape[-1]:2 * x.shape[-1]]  # (B, S, H)
        x_proj = BCx[:, :, 2 * x.shape[-1]:]     # (B, S, H)

        # 2) Elementwise gate Bx = B * x_proj
        Bx = _triton_elementwise_mul_2d(B, x_proj)  # (B, S, H), float32

        # 3) Grouped causal 1D conv on Bx with kernel_size=4 and groups=H
        # conv_weight in original is conv_weight (given), not derived. But original code uses conv_weight as (H, 1, 4).
        # We construct W as (H, 4) by reshaping: conv_weight.view(H, 4)
        conv_w_flat = conv_weight.view(-1)  # length H*4 -> reshape to (H, 4)
        H_weight = int((conv_w_flat.numel()) // 4)
        conv_w = conv_w_flat.view(H_weight, 4).to(torch.float32)  # (H, 4), float32
        conv_b = conv_bias.to(torch.float32)  # (H,), float32

        # Pad Bx on the sequence dimension by 3 zeros on the left: (B, H, S+3)
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # (B, H, S+3)

        conv_out = _triton_grouped_causal_conv1d(Bx_padded, conv_w, conv_b)  # (B, H, S), float32

        # 4) Output gating: y = C * conv_out. Align shapes by transposing conv_out to (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H), float32
        y = _triton_elementwise_mul_2d(C, conv_out_T)         # (B, S, H), float32

        # 5) Final out-proj: y @ out_proj_weight^T + out_proj_bias
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H), float32

        return output


def run(*args):
    return ModelNew()(*args)
