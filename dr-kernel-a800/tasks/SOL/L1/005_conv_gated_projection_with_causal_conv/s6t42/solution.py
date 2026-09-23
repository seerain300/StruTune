import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _matmul_linear_1d_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK: tl.constexpr,
):
    """
    Compute C = A[M,K] @ B[K,N] + Bias[N], where A is flattened 1D over M*K elements.
    A_ptr: points to A as (M,K) via strides
    B_ptr: points to B as (K,N) via strides
    Bias_ptr: Bias[N]
    C_ptr: output flattened
    Shapes are enforced via masks; output C is written back to original (M,N) layout using stride_cm=stride_cm for rows and stride_cn for cols.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = M * N
    mask = offs < total
    m = offs // N
    n = offs % N

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in range(0, K, 1):
        a_ptrs = A_ptr + m * stride_am + (k0) * stride_ak
        b_ptrs = B_ptr + (k0) * stride_bk + n * stride_bn
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        b = tl.load(b_ptrs, mask=(n < N), other=0.0)
        acc += a * b

    bias = tl.load(Bias_ptr + n, mask=(n < N), other=0.0).to(tl.float32)
    acc += bias

    c_ptrs = C_ptr + m * stride_cm + n * stride_cn
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    total_elems,
    BLOCK: tl.constexpr,
):
    """
    Elementwise multiply C = A * B over 1D flattened data of length total_elems.
    A_ptr, B_ptr: input pointers
    C_ptr: output pointer
    All inputs are expected to be contiguous and have same shape.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a * b
    tl.store(C_ptr + offs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, convW_ptr, Bias_ptr, convOut_ptr,
    B, S, H, K,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_convW_h, stride_convW_k,
    stride_convOut_b, stride_convOut_h, stride_convOut_s,
    BLOCK_S: tl.constexpr,
):
    """
    Grouped causal 1D convolution with groups=H, kernel_size=K=4, padding=3 on left.
    Input Bx_ptr: (B, H, S), convW_ptr: (H, K), Bias_ptr: (H), convOut_ptr: (B, H, S)
    For each (b, h), convOut[b, h, s] = sum_{k=0..3} Bx[b, h, s + 3 - k] * convW[h, k] + Bias[h]
    Padding is handled in-kernel via masked loads with zeros for s < 3.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    s_start = 0
    while s_start < S:
        offs_s = s_start + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S

        # Accumulator for this (b, h) and tile of S
        acc = tl.zeros([BLOCK_S], dtype=tl.float32)

        # Loop over kernel taps
        for k in range(0, 4):  # kernel_size=4
            s_padded = offs_s + 3 - k  # left pad = 3
            valid = mask_s & (s_padded >= 0) & (s_padded < S)
            bx_ptrs = Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_padded * stride_bx_s
            bx = tl.load(bx_ptrs, mask=valid, other=0.0)

            # conv weights for this h and tap k
            w = tl.load(convW_ptr + h * stride_convW_h + k * stride_convW_k)

            acc += bx * w  # broadcast scalar w over vector acc

        # Add bias
        bias = tl.load(Bias_ptr + h)
        acc += bias

        # Store result to convOut at s positions
        co_ptrs = convOut_ptr + b * stride_convOut_b + h * stride_convOut_h + offs_s * stride_convOut_s
        tl.store(co_ptrs, acc, mask=mask_s)

        s_start += BLOCK_S


def _run_triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute BCx = F.linear(x, in_proj_weight, in_proj_bias) with in_proj_weight: (3H, H).
    Returns BCx: (B, S, 3H) as float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    # Ensure contiguous and float32 for Triton
    A = x.contiguous().view(M, K).to(torch.float32)           # (M, K)
    B_w = in_proj_weight.t().contiguous().view(K, N).to(torch.float32)  # (K, N)
    Bias = in_proj_bias.contiguous().view(N).to(torch.float32)  # (N)

    C = torch.empty((M, N), dtype=torch.float32, device=x.device)

    # Launch 1D grid over M*N elements
    BLOCK = 1024
    grid = (triton.cdiv(M * N, BLOCK),)
    _matmul_linear_1d_kernel[grid](
        A, B_w, Bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        C.stride(0), C.stride(1),
        BLOCK=BLOCK, num_warps=4, num_stages=2,
    )
    return C.view(B, S, 3 * H)


def _triton_elementwise_mul_2d(B: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Compute C = B * A elementwise, both (B, S, H).
    Returns C as float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    total = B.numel()
    C = torch.empty_like(B, dtype=torch.float32, device=B.device)
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _elementwise_mul_2d_kernel[grid](
        B.view(-1).to(torch.float32), A.view(-1).to(torch.float32), C.view(-1),
        total, BLOCK=BLOCK, num_warps=4, num_stages=2,
    )
    return C


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_w: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Grouped causal conv: input Bx (B, H, S), conv_w (H, 4), conv_bias (H) -> output (B, H, S)
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S = Bx.shape
    # conv_w: (H, 4), conv_bias: (H)
    conv_out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)

    BLOCK_S = 128
    grid = (B, H)
    _grouped_causal_conv1d_kernel[grid](
        Bx.to(torch.float32), conv_w.to(torch.float32), conv_bias.to(torch.float32), conv_out,
        B, S, H, 4,
        Bx.stride(0), Bx.stride(1), Bx.stride(2),
        conv_w.stride(0), conv_w.stride(1),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        BLOCK_S=BLOCK_S, num_warps=4, num_stages=2,
    )
    return conv_out


def _run_triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute output = F.linear(y, out_proj_weight, out_proj_bias) with y: (B, S, H), out_proj_weight: (H, H).
    Returns output: (B, S, H) as float32.
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

    BLOCK = 1024
    grid = (triton.cdiv(M * N, BLOCK),)
    _matmul_linear_1d_kernel[grid](
        A, B_w, Bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        C.stride(0), C.stride(1),
        BLOCK=BLOCK, num_warps=4, num_stages=2,
    )
    return C.view(B, S, H)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only forward:
        - in_proj via Triton matmul
        - elementwise gating via Triton elementwise kernel
        - grouped causal conv via Triton kernel (no PyTorch pad/conv)
        - out-proj via Triton matmul
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        # 1) in_proj
        BCx = _run_triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)

        # 2) Split and elementwise gate
        B = BCx[:, :, :H]                 # (B, S, H)
        C = BCx[:, :, H:2 * H]            # (B, S, H)
        x_proj = BCx[:, :, 2 * H:]        # (B, S, H)
        Bx = _triton_elementwise_mul_2d(B, x_proj)  # (B, S, H)

        # 3) Grouped causal conv: conv_weight is derived from in_proj_weight's last 4 columns
        conv_w = in_proj_weight[:, -4:].to(torch.float32).contiguous()  # (H, 4)
        conv_bias_h = conv_bias  # (H)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias_h)  # (B, H, S)

        # 4) Output gating: y = C * conv_out_T (conv_out transposed to (B, S, H))
        conv_out_T = conv_out.transpose(1, 2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_2d(C, conv_out_T)       # (B, S, H)

        # 5) Final out-proj
        output = _run_triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
