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

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _matmul_outproj_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,  # M = B*S, N = H, K = H
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute C = A @ B + Bias (A: (M,K), B: (K,N))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(0)  # we tile only along N, M is large
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

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # C = A * B
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    for n0 in range(0, N, BLOCK_N):
        idx_n = n0 + offs_n
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + idx_n[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_m[:, None] * stride_bk + idx_n[None, :] * stride_bn)
        mask = (offs_m[:, None] < M) & (idx_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        b = tl.load(b_ptrs, mask=mask, other=0.0)
        c = a * b
        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + idx_n[None, :] * stride_cn)
        tl.store(c_ptrs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr,  # input Bx: (B, H, S), viewed as (B*S*H)
    conv_w_ptr,  # conv_weight: (H, 1, 4) contiguous
    conv_b_ptr,  # conv_bias: (H)
    out_ptr,     # output conv_out: (B, H, S), viewed as (B*S*H)
    B, H, S, K,  # K = kernel_size = 4
    BLOCK_S: tl.constexpr,
):
    # One program per (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # s tile
    offs_s = tl.arange(0, BLOCK_S)
    for s0 in range(0, S, BLOCK_S):
        s_idx = s0 + offs_s  # vector of positions
        # Accumulator for this (b, h) over BLOCK_S
        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # Iterate over kernel taps
        for k in range(0, K):
            pos_in = s_idx + K - 1 - k  # causal: output at s depends on input at s + K - 1 - k
            valid = (pos_in >= 0) & (pos_in < S)
            # Gather input vector Bx[b, h, pos_in]
            # Flatten indexing: idx = b*S*H + h*S + pos_in
            idx_in = b * S * H + h * S + pos_in
            # Load with mask
            bx = tl.load(Bx_ptr + idx_in, mask=valid, other=0.0)
            # Load conv weight for this h and tap k
            # conv_w layout is (H, 1, 4) contiguous => index h * (1*4) + k
            w = tl.load(conv_w_ptr + h * (1 * K) + k)
            acc += bx * w

        # Add bias
        bias = tl.load(conv_b_ptr + h)
        acc += bias

        # Store to output conv_out[b, h, s_idx]
        idx_out = b * S * H + h * S + s_idx
        tl.store(out_ptr + idx_out, acc, mask=(s_idx < S))


def _triton_inproj(x, in_proj_weight, in_proj_bias):
    """
    Triton version of in_proj: compute BCx = x @ in_proj_weight^T + in_proj_bias
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns: BCx: (B, S, 3H), float32, contiguous
    """
    B, S, H = x.shape
    N = in_proj_weight.shape[0]  # 3H
    K = in_proj_weight.shape[1]  # H

    # Flatten x to (M, K)
    x_flat = x.reshape(-1, H).contiguous()  # M = B*S
    M = x_flat.shape[0]

    # Allocate output (M, N)
    BCx = torch.empty((M, N), dtype=torch.float32, device=x.device)

    # Launch Triton matmul + bias
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        x_flat, in_proj_weight, in_proj_bias, BCx,
        M, N, K,
        x_flat.stride(0), x_flat.stride(1),
        in_proj_weight.stride(0), in_proj_weight.stride(1),
        BCx.stride(0), BCx.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    # Reshape back to (B, S, 3H)
    return BCx.reshape(B, S, N)


def _triton_outproj(y, out_proj_weight, out_proj_bias):
    """
    Triton version of out_proj: compute output = y @ out_proj_weight^T + out_proj_bias
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns: output: (B, S, H), float32, contiguous
    """
    B, S, H = y.shape
    M = B * S
    N = H  # output channels = H

    y_flat = y.reshape(M, H).contiguous()
    output = torch.empty((M, N), dtype=torch.float32, device=y.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_outproj_kernel[grid](
        y_flat, out_proj_weight, out_proj_bias, output,
        M, N, H,
        y_flat.stride(0), y_flat.stride(1),
        out_proj_weight.stride(0), out_proj_weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return output.reshape(B, S, H)


def _triton_elementwise_mul(a, b):
    """
    Triton elementwise multiplication: C = A * B
    a, b: tensors of shape (B, S, H), float32, contiguous
    returns C: same shape, float32
    """
    B, S, H = a.shape
    M = B * S
    N = H
    a_flat = a.reshape(M, N).contiguous()
    b_flat = b.reshape(M, N).contiguous()
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _elementwise_mul_kernel[grid](
        a_flat, b_flat, c,
        M, N,
        a_flat.stride(0), a_flat.stride(1),
        b_flat.stride(0), b_flat.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=1,
    )
    return c.reshape(B, S, H)


def _triton_grouped_causal_conv1d(Bx, conv_weight, conv_bias):
    """
    Triton grouped causal 1D conv: input Bx: (B, H, S) -> output conv_out: (B, H, S)
    conv_weight: (H, 1, 4), conv_bias: (H)
    Kernel size fixed at 4; groups=H.
    """
    B, H, S = Bx.shape
    K = 4

    # View Bx as (B*S*H) for linear indexing
    Bx_flat = Bx.reshape(B * S * H).contiguous()

    # conv_weight is (H, 1, 4); make sure contiguous and flattened per (h, k)
    # conv_weight_flat: (H, K)
    conv_weight_flat = conv_weight.reshape(H, K).contiguous()

    conv_out_flat = torch.empty((B * S * H), dtype=torch.float32, device=Bx.device)

    # Launch grid over (B, H)
    grid = (B, H)
    _grouped_causal_conv1d_kernel[grid](
        Bx_flat, conv_weight_flat, conv_bias, conv_out_flat,
        B, H, S, K,
        BLOCK_S=128,  # tile over sequence length
        num_warps=4, num_stages=2,
    )
    return conv_out_flat.reshape(B, H, S)


class ModelNew(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        """
        Triton-only forward: implement the full computation without torch.nn.functional.
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        # Ensure CUDA and dtype
        device = x.device
        dtype = torch.float32  # use float32 for Triton kernels

        # 1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3H)
        # Cast to float32 and make contiguous
        x_f = x.contiguous().to(torch.float32)
        in_proj_weight_f = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias_f = in_proj_bias.contiguous().to(torch.float32)
        BCx = _triton_inproj(x_f, in_proj_weight_f, in_proj_bias_f)  # (B, S, 3H)

        # 2) Split BCx into B, C, x_proj of shape (B, S, H)
        # BCx has last dim = 3H; we can take chunked views to match original semantics
        H = in_proj_weight_f.shape[1]
        # B: last dim index 0..H-1, C: H..2H-1, x_proj: 2H..3H-1
        B = BCx[:, :, :H]                  # (B, S, H)
        C = BCx[:, :, H:2 * H]             # (B, S, H)
        x_proj = BCx[:, :, 2 * H:3 * H]    # (B, S, H)

        # 3) Elementwise gate: Bx = B * x_proj
        Bx = _triton_elementwise_mul(B, x_proj)  # (B, S, H)

        # 4) Grouped causal conv: conv_out = conv(Bx, conv_weight, conv_bias)
        # conv_weight: (H, 1, 4), conv_bias: (H)
        conv_weight_f = conv_weight.contiguous().to(torch.float32)  # (H, 1, 4)
        conv_bias_f = conv_bias.contiguous().to(torch.float32)      # (H,)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_weight_f, conv_bias_f)  # (B, H, S)

        # 5) Output gating: y = C * conv_out
        y = _triton_elementwise_mul(C, conv_out)  # (B, H, S)

        # 6) Final out-proj: output = F.linear(y, out_proj_weight, out_proj_bias)
        # y: (B, H, S) but for linear we need (B, S, H)
        y_T = y.transpose(1, 2).contiguous()  # (B, S, H)
        out_proj_weight_f = out_proj_weight.contiguous().to(torch.float32)  # (H, H)
        out_proj_bias_f = out_proj_bias.contiguous().to(torch.float32)      # (H,)
        output = _triton_outproj(y_T, out_proj_weight_f, out_proj_bias_f)   # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
