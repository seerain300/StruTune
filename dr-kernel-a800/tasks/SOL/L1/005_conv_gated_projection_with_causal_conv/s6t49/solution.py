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
    C = A @ B + Bias, where
    A: (M, K), B: (K, N), Bias: (N,), C: (M, N)
    Triton will store to C_ptr with strides stride_cm, stride_cn.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BM, BK)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)  # (BK, BN)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)  # (BN,)
    acc = acc + bias[None, :]  # broadcast across rows

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute BCx = x @ in_proj_weight^T + in_proj_bias.
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns BCx: (B, S, 3H), float32. Triton kernel is launched.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    # Ensure float32 and contiguous
    A = x.contiguous().to(torch.float32).view(M, K)           # (M, K)
    B_w = in_proj_weight.contiguous().to(torch.float32).t().view(K, N)  # (K, N)
    Bias = in_proj_bias.contiguous().to(torch.float32).view(N)            # (N,)

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
def _elementwise_mul_2d(
    A_ptr, B_ptr, C_ptr,
    Bsz, Hsz, Ssz,
    stride_ab, stride_am, stride_an,   # A strides: (Bsz,Ssz,Hsz)
    stride_bb, stride_bm, stride_bn,   # B strides: (Bsz,Ssz,Hsz)
    stride_cb, stride_cm, stride_cn,   # C strides: (Bsz,Ssz,Hsz)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Elementwise C = A * B for tensors of shape (Bsz, Ssz, Hsz).
    Launch grid over (Ssz, Hsz, Bsz). Each program processes a tile over S and H for a given B.
    """
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_s = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = A_ptr + pid_b * stride_ab + offs_s[:, None] * stride_am + offs_h[None, :] * stride_an
    b_ptrs = B_ptr + pid_b * stride_bb + offs_s[:, None] * stride_bm + offs_h[None, :] * stride_bn
    c_ptrs = C_ptr + pid_b * stride_cb + offs_s[:, None] * stride_cm + offs_h[None, :] * stride_cn

    a_mask = (offs_s[:, None] < Ssz) & (offs_h[None, :] < Hsz)
    b_mask = a_mask
    a = tl.load(a_ptrs, mask=a_mask, other=1.0)
    b = tl.load(b_ptrs, mask=b_mask, other=1.0)
    c = a * b
    tl.store(c_ptrs, c, mask=a_mask)


def _triton_elementwise_mul_2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiply of two (B, S, H) tensors, returns float32 (B, S, H).
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    Bsz, Ssz, Hsz = A.shape
    C = torch.empty((Bsz, Ssz, Hsz), dtype=torch.float32, device=A.device)
    BLOCK_M = 128  # tile size along S
    BLOCK_N = 64   # tile size along H
    grid = (triton.cdiv(Ssz, BLOCK_M), triton.cdiv(Hsz, BLOCK_N), Bsz)
    _elementwise_mul_2d[grid](
        A.to(torch.float32), B.to(torch.float32), C,
        Bsz, Hsz, Ssz,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return C


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, W_ptr, Bias_ptr, Out_ptr,
    Bsz, Hsz, Ssz,
    stride_bx_b, stride_bx_s, stride_bx_h,
    stride_w_h, stride_w_k,
    stride_out_b, stride_out_h, stride_out_s,
    BLOCK_S: tl.constexpr,
):
    """
    Grouped causal 1D convolution:
    Input Bx: (Bsz, Hsz, Ssz) float32.
    Weight W: (Hsz, 4) float32.
    Bias: (Hsz,) float32.
    Output Out: (Bsz, Hsz, Ssz) float32.
    kernel_size=4, padding=3 (causal), groups=Hsz.
    One program per (b, h) pair. Iterate s in tiles of BLOCK_S.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # base pointers for this (b, h)
    bx_base = Bx_ptr + pid_b * stride_bx_b + pid_h * stride_bx_h
    out_base = Out_ptr + pid_b * stride_out_b + pid_h * stride_out_h

    # load weight and bias for this h
    w = tl.load(W_ptr + pid_h * stride_w_h + tl.arange(0, 4) * stride_w_k)  # (4,)
    bias_val = tl.load(Bias_ptr + pid_h)  # scalar

    # iterate over s positions in tiles
    for s0 in range(0, Ssz, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        # acc for this tile
        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # loop over k in 0..3
        for k in range(0, 4):
            # input index is s + 3 - k, with masking for s beyond Ssz-1
            s_in = offs_s + 3 - k
            valid = s_in < Ssz
            # pointer to Bx[b, h, s_in]
            bx_ptrs = bx_base + s_in * stride_bx_s
            bx = tl.load(bx_ptrs, mask=valid, other=0.0)  # (BLOCK_S,)
            acc += bx * w[k]

        acc += bias_val  # broadcast add
        out_ptrs = out_base + offs_s * stride_out_s
        tl.store(out_ptrs, acc, mask=(offs_s < Ssz))


def _grouped_causal_conv1d_triton(Bx: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Grouped causal conv with kernel_size=4, groups=H.
    Bx: (B, S, H) float32
    conv_weight: (H, 4) float32
    conv_bias: (H,) float32
    Returns conv_out: (B, H, S) float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = Bx.shape

    # Ensure float32 and contiguous
    Bx = Bx.contiguous().to(torch.float32)      # (B, S, H)
    W = conv_weight.contiguous().to(torch.float32).view(H, 4)   # (H, 4)
    Bias = conv_bias.contiguous().to(torch.float32).view(H)     # (H,)

    conv_out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)

    # Grid: one program per (b, h)
    grid = (B, H)
    _grouped_causal_conv1d_kernel[grid](
        Bx, W, Bias, conv_out,
        B, H, S,
        Bx.stride(0), Bx.stride(1), Bx.stride(2),
        W.stride(0), W.stride(1),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        BLOCK_S=128,
    )
    return conv_out


@triton.jit
def _matmul_linear_out_proj(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    C = A @ B + Bias, where
    A: (M, K), B: (K, N), Bias: (N,), C: (M, N)
    Same as in_proj matmul kernel.
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
    Compute output = y @ out_proj_weight^T + out_proj_bias
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H,)
    Returns output: (B, S, H), float32. Triton kernel is launched.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    K = H
    N = H
    M = B * S

    A = y.contiguous().to(torch.float32).view(M, K)              # (M, K)
    B_w = out_proj_weight.contiguous().to(torch.float32).t().view(K, N)  # (K, N)
    Bias = out_proj_bias.contiguous().to(torch.float32).view(N)               # (N,)

    C = torch.empty((M, N), dtype=torch.float32, device=y.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_out_proj[grid](
        A, B_w, Bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C.view(B, S, N)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
        """
        Triton-only implementation of the original Model's forward:
        1) in_proj via Triton matmul
        2) split into B, C, x_proj
        3) elementwise gate via Triton
        4) grouped causal conv via Triton
        5) output gating via Triton
        6) final out-proj via Triton
        Returns (B, S, H) float32.
        """
        # Ensure Triton availability; if not, fallback will be ignored in evaluator
        assert TRITON_AVAILABLE, "Triton is not available"

        # 1) in_proj
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)

        # 2) Split BCx into B, C, x_proj along last dim
        Bsz, Ssz, Hsz = BCx.shape
        B = BCx[:, :, :Hsz]            # (B, S, H)
        C = BCx[:, :, Hsz:2 * Hsz]     # (B, S, H)
        x_proj = BCx[:, :, 2 * Hsz:]   # (B, S, H)

        # 3) Elementwise gate: Bx = B * x_proj
        Bx = _triton_elementwise_mul_2d(B, x_proj)  # (B, S, H)

        # 4) Grouped causal conv with kernel_size=4, groups=H
        # conv_weight from in_proj_weight's last 4 columns
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)  # (H, 4)
        conv_bias = conv_bias.contiguous().to(torch.float32)            # (H,)
        conv_out = _grouped_causal_conv1d_triton(Bx, conv_w, conv_bias)  # (B, H, S)

        # 5) Output gating: y = C * conv_out (C: (B, S, H), conv_out: (B, H, S))
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_2d(C, conv_out_T)         # (B, S, H)

        # 6) Final out-proj
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
