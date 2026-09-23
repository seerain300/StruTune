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

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,  # A and B are 2D tensors of shape (M, N)
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Compute C = A * B elementwise
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


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, conv_weight_ptr, Bias_ptr, Out_ptr,
    B, H, S,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_w_h, stride_w_k,
    stride_out_b, stride_out_h, stride_out_s,
    BLOCK_S: tl.constexpr,
):
    # Compute grouped causal conv: Out[b, h, s] = sum_{k=0..3} Bx[b, h, s+3-k] * conv_weight[h, k] + Bias[h]
    # Input Bx assumed to be (B, H, S) without padding (we will write padded values inside this kernel for valid s).
    # Launch one program per (b, h)
    pid = tl.program_id(0)  # 0..(B*H-1)
    b = pid // H
    h = pid % H

    # Output vector of length S
    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # k loop: 0..3
    for k in range(4):
        s_in = offs_s + 3 - k  # corresponds to s+3-k
        # Mask valid positions
        valid = (s_in >= 0) & (s_in < S) & mask_s
        bx_ptrs = Bx_ptr + (b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s)
        bx = tl.load(bx_ptrs, mask=valid, other=0.0)
        w = tl.load(conv_weight_ptr + (h * stride_w_h + k * stride_w_k))
        acc += bx * w

    # Add bias[h]
    bias_h = tl.load(Bias_ptr + h)
    acc += bias_h

    out_ptrs = Out_ptr + (b * stride_out_b + h * stride_out_h + offs_s * stride_out_s)
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

    # Cast to float32 for stable accumulation
    A = x.contiguous().view(M, H).to(torch.float32)
    B_w = in_proj_weight.t().contiguous().view(H, N).to(torch.float32)  # (K, N)
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


def _triton_elementwise_mul_2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiply: C = A * B, A, B: (B, S, H)
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = A.shape
    M = B * S
    # Flatten to 2D (M, H)
    A_flat = A.contiguous().view(M, H).to(torch.float32)
    B_flat = B.contiguous().view(M, H).to(torch.float32)
    C_flat = torch.empty((M, H), dtype=torch.float32, device=A.device)
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
    _elementwise_mul_2d_kernel[grid](
        A_flat, B_flat, C_flat,
        M, H,
        A_flat.stride(0), A_flat.stride(1),
        B_flat.stride(0), B_flat.stride(1),
        C_flat.stride(0), C_flat.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return C_flat.view(B, S, H)


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute grouped causal conv:
    Bx: (B, H, S), conv_weight: (H, 4), conv_bias: (H,)
    Returns conv_out: (B, H, S)
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S = Bx.shape
    # conv_weight: (H, 4), conv_bias: (H,)
    conv_weight = conv_weight.contiguous().to(torch.float32)
    conv_bias = conv_bias.contiguous().to(torch.float32)

    Out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)

    grid = (B * H,)
    BLOCK_S = 128
    _grouped_causal_conv1d_kernel[grid](
        Bx, conv_weight, conv_bias, Out,
        B, H, S,
        Bx.stride(0), Bx.stride(1), Bx.stride(2),
        conv_weight.stride(0), conv_weight.stride(1),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_S=BLOCK_S,
    )
    return Out


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    output = y @ out_proj_weight^T + out_proj_bias
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns output: (B, S, H), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
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
        Triton-only forward. All computations via Triton kernels.
        1) BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3H) via _matmul_linear_kernel
        2) Split BCx into B, C, x_proj and elementwise gate Bx = B * x_proj via _elementwise_mul_2d
        3) Grouped causal conv with kernel_size=4, padding=3, groups=H:
             - conv_weight from in_proj_weight last 4 columns: (H, 1, 4) -> (H, 4)
             - conv_out = _grouped_causal_conv1d_kernel(Bx, conv_weight, conv_bias) -> (B, H, S)
             - y = C * conv_out (C is (B, S, H), conv_out transposed to (B, S, H))
        4) Final out-proj: output = y @ out_proj_weight^T + out_proj_bias via _matmul_linear_kernel_out_proj
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        # 1) in_proj
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H), float32

        # 2) Split BCx and elementwise gate
        B = BCx[:, :, :H]
        C = BCx[:, :, H:2 * H]
        x_proj = BCx[:, :, 2 * H:]
        Bx = _triton_elementwise_mul_2d(B, x_proj)  # (B, S, H)

        # 3) Grouped causal conv: kernel_size=4, groups=H
        # conv_weight: take last 4 columns from in_proj_weight: (3H, 4)
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)  # (H, 4)
        conv_bias = conv_bias.contiguous().to(torch.float32)           # (H,)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias)  # (B, H, S)

        # Output gating: y = C * conv_out. Align shapes: conv_out transpose to (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()           # (B, S, H)
        y = _triton_elementwise_mul_2d(C, conv_out_T)                  # (B, S, H)

        # 4) Final out-proj
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)   # (B, S, H), float32

        return output


def run(*args):
    return ModelNew()(*args)
