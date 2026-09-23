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

    # Cast to desired output dtype before storing
    if OUT_DTYPE == 0:
        out_val = acc
    elif OUT_DTYPE == 1:
        out_val = acc.to(tl.float16)
    elif OUT_DTYPE == 2:
        out_val = acc.to(tl.bfloat16)
    else:
        out_val = acc  # default to float32

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out_val, mask=c_mask)


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Compute C = A * B, all shape (M, N), A,B,C strides accordingly
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_m[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    prod = a * b
    if OUT_DTYPE == 0:
        out_val = prod
    elif OUT_DTYPE == 1:
        out_val = prod.to(tl.float16)
    elif OUT_DTYPE == 2:
        out_val = prod.to(tl.bfloat16)
    else:
        out_val = prod
    tl.store(c_ptrs, out_val, mask=mask)


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton matmul for in_proj: BCx = x @ in_proj_weight^T + in_proj_bias.
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns BCx: (B, S, 3H), same dtype as x.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    A = x.contiguous().view(M, K)  # (M,K)
    B_w = in_proj_weight.t().contiguous().view(K, N)  # (K,N)
    Bias = in_proj_bias.contiguous().view(N)  # (N,)
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Map dtype to OUT_DTYPE for kernel
    if C.dtype == torch.float32:
        OUT_DTYPE = 0
    elif C.dtype == torch.float16:
        OUT_DTYPE = 1
    elif C.dtype == torch.bfloat16:
        OUT_DTYPE = 2
    else:
        OUT_DTYPE = 0

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
        OUT_DTYPE=OUT_DTYPE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C.view(B, S, N)


def _triton_elementwise_mul_2d(B: torch.Tensor, Cx: torch.Tensor) -> torch.Tensor:
    """
    Triton elementwise 2D multiplication: output = B * Cx
    B: (B, S, H), Cx: (B, S, H) -> output: (B, S, H)
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B_out = torch.empty_like(B)
    M = B.numel()
    K = B.shape[-1]
    # Use default OUT_DTYPE=0 (float32) since we allocate B_out with same dtype as B
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
    _elementwise_mul_2d_kernel[grid](
        B, Cx, B_out,
        M, K, K,
        B.stride(0), B.stride(1),
        Cx.stride(0), Cx.stride(1),
        B_out.stride(0), B_out.stride(1),
        OUT_DTYPE=0,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return B_out


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton matmul for out_proj: output = y @ out_proj_weight^T + out_proj_bias
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns output: (B, S, H), same dtype as y.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    K = H
    M = B * S

    A = y.contiguous().view(M, K)  # (M,K)
    B_w = out_proj_weight.t().contiguous().view(K, K)  # (K,K)
    Bias = out_proj_bias.contiguous().view(K)  # (K,)
    C = torch.empty((M, K), dtype=A.dtype, device=A.device)

    if C.dtype == torch.float32:
        OUT_DTYPE = 0
    elif C.dtype == torch.float16:
        OUT_DTYPE = 1
    elif C.dtype == torch.bfloat16:
        OUT_DTYPE = 2
    else:
        OUT_DTYPE = 0

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
    _matmul_linear_kernel[grid](
        A, B_w, Bias, C,
        M, K, K,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        C.stride(0), C.stride(1),
        OUT_DTYPE=OUT_DTYPE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C.view(B, S, K)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that matches the original computation:
        1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3H)
        2) Split BCx into B, C, x_proj -> (B, S, H), then elementwise Bx = B * x_proj
        3) Grouped causal conv with kernel_size=4, using conv_weight derived from in_proj_weight[:, -4:], groups=H
        4) Output gating: y = C * conv_out (after transposing conv_out to (B, S, H))
        5) Final out-proj: output = F.linear(y, out_proj_weight, out_proj_bias) -> (B, S, H)
        Triton is used for in_proj, elementwise gates, and out_proj. Convolution uses PyTorch for correctness.
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        # 1) in_proj via Triton
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)

        # Split along last dim: H = x.shape[-1]
        B, S, H = x.shape
        N = 3 * H
        B_mat = BCx[:, :, :H].contiguous()      # (B, S, H)
        C_mat = BCx[:, :, H:2*H].contiguous()   # (B, S, H)
        x_proj = BCx[:, :, 2*H:].contiguous()   # (B, S, H)

        # 2) Elementwise gate: Bx = B_mat * x_proj (Triton)
        Bx = _triton_elementwise_mul_2d(B_mat, x_proj)  # (B, S, H)

        # 3) Grouped causal conv:
        # Derive conv_weight from in_proj_weight's last 4 columns
        conv_w = in_proj_weight[:, -4:].contiguous()  # (H, 4), float32
        # Pad along sequence dimension for causal conv: (B, H, S+3)
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # (B, H, S+3)
        # Use PyTorch conv1d for correctness and to avoid Triton kernel pitfalls
        conv_out = torch.nn.functional.conv1d(
            Bx_padded, conv_w, conv_bias, stride=1, padding=0, dilation=1, groups=H
        )  # (B, H, S)

        # 4) Output gating: y = C_mat * conv_out; align shapes (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_2d(C_mat, conv_out_T)     # (B, S, H)

        # 5) Final out-proj via Triton
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
