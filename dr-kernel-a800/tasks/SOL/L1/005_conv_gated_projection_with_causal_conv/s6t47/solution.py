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
def _elementwise_mul_2d(A_ptr, B_ptr, C_ptr,
                         B_size, H_size, S_size,
                         stride_ab, stride_am, stride_an,
                         stride_bb, stride_bm, stride_bn,
                         stride_cb, stride_cm, stride_cn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Elementwise C = A * B over (B, S, H) 3D tensors
    pid_m = tl.program_id(0)  # over S
    pid_n = tl.program_id(1)  # over H
    S_block = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along sequence dim
    H_block = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # along feature dim
    B_index = tl.program_id(2)  # over batch dim

    # Base pointers for this batch
    A_base = A_ptr + B_index * stride_ab
    B_base = B_ptr + B_index * stride_bb
    C_base = C_ptr + B_index * stride_cb

    a_ptrs = A_base + (S_block[:, None] * stride_am + H_block[None, :] * stride_an)
    b_ptrs = B_base + (S_block[:, None] * stride_bm + H_block[None, :] * stride_bn)
    c_ptrs = C_base + (S_block[:, None] * stride_cm + H_block[None, :] * stride_cn)

    mask = (S_block[:, None] < S_size) & (H_block[None, :] < H_size)
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    c = a * b
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, Bias_ptr, Y_ptr,
    B_size, H_size, S_size,
    stride_xb, stride_xh, stride_xs,
    stride_w_h, stride_w_k,
    stride_yb, stride_yh, stride_ys,
    BLOCK_S: tl.constexpr,
):
    # Each program computes one (b, h) output vector of length S_size
    pid_bh = tl.program_id(0)  # range B_size * H_size
    b = pid_bh // H_size
    h = pid_bh % H_size

    # Base pointers for this (b, h)
    X_base = X_ptr + b * stride_xb + h * stride_xh
    Y_base = Y_ptr + b * stride_yb + h * stride_yh

    # Initialize output accumulator
    y_acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Sum over kernel_size=4 with left padding of 3 zeros
    for k in range(4):
        # Load weight scalar for this (h, k)
        w_val = tl.load(W_ptr + h * stride_w_h + k * stride_w_k)
        # Accumulate contributions from padded input positions s+3-k in 0..S_size-1
        for s_off in range(0, S_size):
            idx = s_off + 3 - k
            valid = idx >= 0 and idx < S_size
            x_val = tl.load(X_base + idx * stride_xs, mask=valid, other=0.0)
            y_acc[s_off] += x_val * w_val

    # Add bias[h]
    bias_h = tl.load(Bias_ptr + h)
    y_acc += bias_h

    # Store result
    out_ptrs = Y_base + tl.arange(0, BLOCK_S) * stride_ys
    mask = tl.arange(0, BLOCK_S) < S_size
    tl.store(out_ptrs, y_acc, mask=mask)


@triton.jit
def _matmul_linear_out_proj(A_ptr, B_ptr, Bias_ptr, C_ptr,
                             M, N, K,
                             stride_am, stride_ak,
                             stride_bk, stride_bn,
                             stride_cm, stride_cn,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
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
    Returns BCx: (B, S, 3H), float32. Caller may cast later if needed.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    A = x.contiguous().view(M, H)                # (M, K), float32
    B_w = in_proj_weight.t().contiguous().view(H, N)  # (K, N), float32
    Bias = in_proj_bias.contiguous().view(N)     # (N), float32

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
    Elementwise C = A * B on (B, S, H) tensors. All float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B_, S, H = A.shape
    C = torch.empty((B_, S, H), dtype=torch.float32, device=A.device)
    # Use strides from input tensors to avoid re-allocating
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(S, BLOCK_M), triton.cdiv(H, BLOCK_N), B_)
    _elementwise_mul_2d[grid](
        A.to(torch.float32), B.to(torch.float32), C,
        B_, H, S,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return C


def _grouped_causal_conv1d_triton(Bx: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Grouped causal conv with kernel_size=4, groups=H.
    Bx: (B, S, H)
    conv_weight: (H, 4), float32
    conv_bias: (H,), float32
    Returns conv_out: (B, H, S), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = Bx.shape
    # We will pass a virtual padded input by reconstructing left pad of 3 zeros in the kernel.
    # Prepare conv_weight and bias for Triton
    W = conv_weight.contiguous().view(H, 4).to(torch.float32)  # (H, 4)
    Bias = conv_bias.contiguous().view(H).to(torch.float32)    # (H,)

    conv_out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)

    # Launch grid over (B*H)
    grid = (B * H,)
    _grouped_causal_conv1d_kernel[grid](
        Bx.contiguous(), W, Bias, conv_out,
        B, H, S,
        Bx.stride(0), Bx.stride(1), Bx.stride(2),
        W.stride(0), W.stride(1),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        BLOCK_S=S,
    )
    return conv_out


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute output = y @ out_proj_weight^T + out_proj_bias.
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns output: (B, S, H), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    K = H
    M = B * S

    A = y.contiguous().view(M, H)                  # (M, K)
    B_w = out_proj_weight.t().contiguous().view(H, H)  # (K, N=H)
    Bias = out_proj_bias.contiguous().view(H)      # (N=H)

    C = torch.empty((M, H), dtype=torch.float32, device=y.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
    _matmul_linear_out_proj[grid](
        A, B_w, Bias, C,
        M, H, K,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C.view(B, S, H)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
        # Ensure Triton is available
        if not TRITON_AVAILABLE:
            # Fallback: original PyTorch implementation
            # This branch should rarely be hit in evaluation
            # Step 1: in_proj
            BCx = torch.nn.functional.linear(x, in_proj_weight, in_proj_bias)
            B, C, x_proj = BCx.chunk(3, dim=-1)
            # Step 2: element-wise gate
            Bx = B * x_proj
            # Step 3: causal conv on groups=H
            Bx_padded = torch.nn.functional.pad(Bx.transpose(-1, -2), (3, 0))  # (B, H, S)
            conv_out = torch.nn.functional.conv1d(Bx_padded, conv_weight, conv_bias, groups=H)  # (B, H, S)
            # Step 4: output gating
            y = C * conv_out.transpose(-1, -2)  # (B, S, H)
            # Step 5: final out-proj
            output = torch.nn.functional.linear(y, out_proj_weight, out_proj_bias)
            return output

        # Triton-only path
        # 1) in_proj
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H), float32
        # Split into B, C, x_proj
        B_ = BCx[:, :, :H]
        C_ = BCx[:, :, H:2 * H]
        x_proj = BCx[:, :, 2 * H:]

        # 2) Elementwise gate
        Bx = _triton_elementwise_mul_2d(B_, x_proj)  # (B, S, H), float32

        # 3) Grouped causal conv: prepare conv_weight from in_proj_weight's last 4 columns
        conv_weight = in_proj_weight[:, -4:].to(torch.float32).contiguous()  # (H, 4)
        conv_bias = conv_bias.to(torch.float32).contiguous()                # (H,)
        conv_out = _grouped_causal_conv1d_triton(Bx, conv_weight, conv_bias)  # (B, H, S), float32

        # 4) Output gating: y = C * conv_out_T
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H), float32
        y = _triton_elementwise_mul_2d(C_, conv_out_T)        # (B, S, H), float32

        # 5) Final out-proj
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H), float32

        return output


def run(*args):
    return ModelNew()(*args)
