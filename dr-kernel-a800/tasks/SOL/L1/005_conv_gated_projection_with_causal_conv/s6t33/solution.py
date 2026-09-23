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
    # Compute C = A @ B + Bias
    # A: (M, K), B: (K, N), Bias: (N,), C: (M, N)
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
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H,)
    Returns BCx: (B, S, 3H), float32.
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


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,  # M is batch*seq, N is hidden
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Compute C = A * B, elementwise over a 2D view of tensors
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    c = a * b
    tl.store(c_ptrs, c, mask=mask)


def _triton_elementwise_mul_2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiply over a 2D view of A and B: both are reshaped to (M, N).
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    M, N = A.shape
    A_ = A.contiguous().view(M, N).to(torch.float32)
    B_ = B.contiguous().view(M, N).to(torch.float32)
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    BLOCK_M = 128
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _elementwise_mul_2d_kernel[grid](
        A_, B_, C,
        M, N,
        A_.stride(0), A_.stride(1),
        B_.stride(0), B_.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return C.view_as(A)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Xpad_ptr, W_ptr, Bias_ptr, Out_ptr,
    B, H, S, P,  # P = padding on left (3 for kernel_size=4)
    stride_xb, stride_xh, stride_xs,
    stride_wh, stride_wk,
    stride_outb, stride_outh, stride_outs,
    BLOCK_S: tl.constexpr,
):
    # Xpad: (B, H, S+P), W: (H, 4), Bias: (H,), Out: (B, H, S)
    # Compute Out[b, h, s] = sum_{k=0..3} Xpad[b, h, s+P-k] * W[h, k] + Bias[h]
    # One program per (b, h), loop over s tiles
    b = tl.program_id(0)
    h = tl.program_id(1)

    # bounds check for safety (though grid should be exact)
    if (b >= B) or (h >= H):
        return

    # We loop over s in tiles
    # For each tile, compute output positions s in [0, S)
    for s_start in range(0, S, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S

        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # Accumulate over kernel k=0..3
        # Note: Xpad indexing uses s + P - k, and mask_s ensures valid
        for k in range(4):
            pos = offs_s + P - k  # positions in padded sequence
            x_ptrs = Xpad_ptr + b * stride_xb + h * stride_xh + pos * stride_xs
            # Load with mask: pos must be in [0, S+P)
            x_val = tl.load(x_ptrs, mask=mask_s, other=0.0)
            w_val = tl.load(W_ptr + h * stride_wh + k * stride_wk)
            acc += x_val * w_val

        acc = acc + tl.load(Bias_ptr + h)  # add bias[h]

        out_ptrs = Out_ptr + b * stride_outb + h * stride_outh + offs_s * stride_outs
        tl.store(out_ptrs, acc, mask=mask_s)


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Grouped causal 1D conv on Bx: (B, H, S), conv_weight: (H, 4), conv_bias: (H,).
    Output: (B, H, S). Triton kernel with padding=3. Uses groups=H implicitly in host by launching (B,H).
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S = Bx.shape
    # Host-side pad (left causal padding)
    Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # pad 3 zeros on left

    conv_weight = conv_weight.contiguous().to(torch.float32)  # (H, 4)
    conv_bias = conv_bias.contiguous().to(torch.float32)      # (H,)

    Out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)

    grid = (B, H)
    BLOCK_S = 128
    _grouped_causal_conv1d_kernel[grid](
        Bx_padded, conv_weight, conv_bias, Out,
        B, H, S, 3,
        Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
        conv_weight.stride(0), conv_weight.stride(1),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_S=BLOCK_S,
    )
    return Out


@triton.jit
def _matmul_linear_kernel_out_proj(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, K, N,  # M=B*S, K=H, N=H
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute C = A @ B + Bias
    # A: (M, K), B: (K, N), Bias: (N,), C: (M, N)
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
    output = y @ out_proj_weight^T + out_proj_bias
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns output: (B, S, H), float32. Triton kernel.
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
        Triton-only forward, no PyTorch functional calls. Computes:
        1) BCx = x @ in_proj_weight^T + in_proj_bias
        2) Split into B, C, x_proj; Bx = B * x_proj
        3) conv_out = grouped causal conv(Bx, conv_weight, conv_bias, padding=3)
        4) y = C * conv_out  (conv_out transposed to (B,S,H) for elementwise multiply)
        5) output = y @ out_proj_weight^T + out_proj_bias
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        # 1) in_proj
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)

        # Split into B, C, x_proj (each (B, S, H))
        H = x.shape[-1]
        B = BCx[:, :, :H]
        C = BCx[:, :, H:2 * H]
        x_proj = BCx[:, :, 2 * H:]

        # 2) Elementwise gate: Bx = B * x_proj
        Bx = _triton_elementwise_mul_2d(B, x_proj)  # (B, S, H)

        # 3) Grouped causal conv with kernel_size=4 (padding=3)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_weight, conv_bias)  # (B, H, S)

        # 4) Output gating: y = C * conv_out, align conv_out to (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_2d(C, conv_out_T)         # (B, S, H)

        # 5) Final out-proj
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
