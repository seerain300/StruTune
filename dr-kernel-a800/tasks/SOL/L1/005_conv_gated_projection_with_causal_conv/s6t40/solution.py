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
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BM, BK)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)  # (BK, BN)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_1d_kernel(X_ptr, Y_ptr, O_ptr, N, BLOCK: tl.constexpr):
    # Compute O = X * Y for flattened arrays of length N
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offs, mask=mask, other=0.0)
    o = x * y
    tl.store(O_ptr + offs, o, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Xpad_ptr,  # input padded: (B, H, S+pad) with pad=3, contiguous
    W_ptr,     # conv weights: (H, 4), contiguous
    Bias_ptr,  # conv bias: (H), contiguous
    Out_ptr,   # output: (B, H, S), contiguous
    S,         # output seq length
    BLOCK_S: tl.constexpr,
):
    # One program per (b, h)
    pid_bh = tl.program_id(0)  # 0..(B*H - 1)
    b = pid_bh // H
    h = pid_bh % H

    # Accumulator for output positions
    acc = tl.zeros((S,), dtype=tl.float32)

    # Preload conv weights for this h
    w0 = tl.load(W_ptr + h * 4 + 0)
    w1 = tl.load(W_ptr + h * 4 + 1)
    w2 = tl.load(W_ptr + h * 4 + 2)
    w3 = tl.load(W_ptr + h * 4 + 3)

    # Loop over s in tiles
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S

        # Base offset for this (b, h) in padded input: we index Xpad as [b, h, s+3]
        base = b * (H * (S + 3)) + h * (S + 3)
        x_ptrs = Xpad_ptr + base + offs_s  # vector of s indices into padded input
        x_vals = tl.load(x_ptrs, mask=mask_s, other=0.0)

        # Accumulate convolution: x[s+3], x[s+2], x[s+1], x[s]
        # Note: x_vals corresponds to s in [s0, s0+BLOCK_S-1]. We read shifted indices from padded input.
        acc += w0 * x_vals + w1 * tl.load(Xpad_ptr + base + offs_s - 1, mask=mask_s, other=0.0) + \
               w2 * tl.load(Xpad_ptr + base + offs_s - 2, mask=mask_s, other=0.0) + \
               w3 * tl.load(Xpad_ptr + base + offs_s - 3, mask=mask_s, other=0.0)

    # Add bias
    bias = tl.load(Bias_ptr + h)
    acc = acc + bias

    # Store output: Out[b, h, s] for s in [0..S-1]
    Out_base = (b * H + h) * S
    Out_ptrs = Out_ptr + Out_base + tl.arange(0, S)
    tl.store(Out_ptrs, acc, mask=True)


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute BCx = x @ in_proj_weight^T + in_proj_bias.
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns BCx: (B, S, 3H), float32
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    # Ensure float32 for numeric stability, contiguous inputs
    x_c = x.contiguous().to(torch.float32)
    in_w_c = in_proj_weight.contiguous().to(torch.float32)  # (3H, H)
    bias_c = in_proj_bias.contiguous().to(torch.float32)

    A = x_c.view(M, K)                                  # (M, K)
    B_w = in_w_c.t().view(K, N)                        # (K, N)

    C = torch.empty((M, N), dtype=torch.float32, device=x.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        A, B_w, bias_c, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )
    # Return in original shape (B, S, 3H)
    return C.view(B, S, N)


def _triton_elementwise_mul_1d(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Elementwise O = X * Y for 1D flattened tensors.
    Returns O: same shape as X/Y, dtype=float32
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    X_c = X.contiguous().to(torch.float32)
    Y_c = Y.contiguous().to(torch.float32)
    N = X_c.numel()
    O = torch.empty(N, dtype=torch.float32, device=X_c.device)
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _elementwise_mul_1d_kernel[grid](X_c.view(-1), Y_c.view(-1), O, N, BLOCK=BLOCK, num_warps=4)
    return O.view(X.shape)


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute grouped causal 1D convolution for each (b, h):
    Input Bx: (B, H, S), conv_weight: (H, 4), conv_bias: (H)
    Output conv_out: (B, H, S)
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S = Bx.shape
    pad = 3  # since kernel_size=4 and causal
    Xpad = torch.nn.functional.pad(Bx.to(torch.float32), (pad, 0))  # (B, H, S+pad)
    # Ensure contiguous layout for Triton
    Xpad = Xpad.contiguous()
    W = conv_weight.contiguous().to(torch.float32)  # (H, 4)
    Bias = conv_bias.contiguous().to(torch.float32) # (H)

    Out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)

    BLOCK_S = 128
    grid = (B * H,)
    _grouped_causal_conv1d_kernel[grid](
        Xpad, W, Bias, Out, S,
        BLOCK_S=BLOCK_S,
        num_warps=4, num_stages=2
    )
    return Out


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute output = y @ out_proj_weight^T + out_proj_bias.
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns output: (B, S, H), float32
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    M = B * S
    K = H
    N = H

    y_c = y.contiguous().to(torch.float32)             # (M, K)
    out_w_c = out_proj_weight.t().contiguous().to(torch.float32)  # (K, N)
    bias_c = out_proj_bias.contiguous().to(torch.float32)          # (N)

    C = torch.empty((M, N), dtype=torch.float32, device=y.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        y_c, out_w_c, bias_c, C,
        M, N, K,
        y_c.stride(0), y_c.stride(1),
        out_w_c.stride(0), out_w_c.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )
    return C.view(B, S, N)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only fused implementation:
        1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3H) via Triton matmul
        2) Split BCx into B, C, x_proj, then Bx = B * x_proj (Triton elementwise)
        3) Grouped causal conv with kernel_size=4, groups=H using conv_weight derived from in_proj_weight's last 4 columns
        4) y = C * conv_out (Triton elementwise)
        5) Final out-proj: output = F.linear(y, out_proj_weight, out_proj_bias) via Triton matmul
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        # 1) in_proj
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H), float32

        # 2) Split into B, C, x_proj
        H = x.shape[-1]
        B = BCx[:, :, :H]                 # (B, S, H)
        C = BCx[:, :, H:2*H]              # (B, S, H)
        x_proj = BCx[:, :, 2*H:]          # (B, S, H)

        # 3) Elementwise gate
        Bx = _triton_elementwise_mul_1d(B, x_proj)           # (B, S, H)

        # 4) Grouped causal conv: prepare conv_weight from in_proj_weight's last 4 columns
        conv_w = in_proj_weight[:, -4:].to(torch.float32).contiguous()  # (H, 4)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias) # (B, H, S)

        # 5) Output gating: y = C * conv_out, align shapes (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_1d(C, conv_out_T)         # (B, S, H)

        # 6) Final out-proj
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
