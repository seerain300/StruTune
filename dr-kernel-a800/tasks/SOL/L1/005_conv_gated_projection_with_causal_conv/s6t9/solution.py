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


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute BCx = x @ in_proj_weight^T + in_proj_bias, using Triton.
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns BCx: (B, S, 3H), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    # Ensure float32 for kernels
    A = x.contiguous().view(M, K).to(torch.float32)            # (M, K)
    B_w = in_proj_weight.t().contiguous().view(K, N).to(torch.float32)  # (K, N)
    Bias = in_proj_bias.contiguous().view(N).to(torch.float32) # (N)

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


@triton.jit
def _elementwise_mul_1d_kernel(A_ptr, B_ptr, C_ptr, total_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = a * b
    tl.store(C_ptr + offs, c, mask=mask)


def _triton_elementwise_mul_1d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Elementwise A * B via Triton 1D kernel.
    Returns result of same shape as A (float32).
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    total = A.numel()
    A_flat = A.contiguous().to(torch.float32).view(-1)
    B_flat = B.contiguous().to(torch.float32).view(-1)
    C = torch.empty_like(A_flat, dtype=torch.float32, device=A.device)

    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _elementwise_mul_1d_kernel[grid](A_flat, B_flat, C, total, BLOCK=BLOCK, num_warps=4, num_stages=1)
    return C.view(A.shape)


@triton.jit
def _grouped_causal_conv1d_kernel(Bx_ptr, conv_w_ptr, conv_b_ptr, conv_out_ptr,
                                  B, H, S, K,
                                  stride_bx_b, stride_bx_h, stride_bx_s,
                                  stride_w_h, stride_w_k,
                                  stride_co_b, stride_co_h, stride_co_s,
                                  BLOCK_S: tl.constexpr):
    # Bx_ptr: (B, H, S+K-1), conv_w_ptr: (H, K), conv_b_ptr: (H)
    pid_b = tl.program_id(0)  # batch
    pid_h = tl.program_id(1)  # channel
    # Loop over output positions s in 0..S-1
    for s_out in range(0, S):
        # Accumulator for this (b, h, s_out)
        acc = tl.zeros((), dtype=tl.float32)
        # Sum over kernel elements
        for k in range(0, K):
            pos = s_out + K - 1 - k  # corresponds to padded input index
            # Guard against negative pos by masking (shouldn't happen for s_out in [0..S-1] with K padding)
            valid = pos >= 0
            val = tl.load(Bx_ptr + pid_b * stride_bx_b + pid_h * stride_bx_h + pos * stride_bx_s, mask=valid, other=0.0)
            w = tl.load(conv_w_ptr + pid_h * stride_w_h + k * stride_w_k)
            acc += val * w
        # Add bias
        b = tl.load(conv_b_ptr + pid_h)
        acc += b
        # Store result
        tl.store(conv_out_ptr + pid_b * stride_co_b + pid_h * stride_co_h + s_out * stride_co_s, acc)


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton grouped causal 1D conv on Bx: input (B, H, S) -> output (B, H, S)
    conv_weight: (H, 4), conv_bias: (H), float32
    Kernel size fixed at 4, groups=H (depthwise grouped).
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S = Bx.shape
    # Pre-pad Bx with 3 zeros on the left along sequence dimension to implement causal conv
    Bx_padded = torch.nn.functional.pad(Bx, (3, 0), mode='constant', value=0.0).to(torch.float32).contiguous()  # (B, H, S+3)
    conv_out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)  # output (B, H, S), float32

    conv_w = conv_weight.contiguous().to(torch.float32)  # (H, 4)
    conv_b = conv_bias.contiguous().to(torch.float32)    # (H)

    grid = (B, H)
    _grouped_causal_conv1d_kernel[grid](
        Bx_padded, conv_w, conv_b, conv_out,
        B, H, S, 4,
        Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
        conv_w.stride(0), conv_w.stride(1),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        BLOCK_S=128, num_warps=2, num_stages=1,
    )
    return conv_out  # (B, H, S), float32


@triton.jit
def _out_proj_linear_kernel(A_ptr, B_ptr, Bias_ptr, C_ptr,
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


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton out-proj: output = y @ out_proj_weight^T + out_proj_bias
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H), all float32
    Returns output: (B, S, H), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    M = B * S
    N = H  # output channels = H

    y_f32 = y.contiguous().to(torch.float32)  # (B, S, H)
    A = y_f32.view(M, H).contiguous()          # (M, K), K=H
    B_w = out_proj_weight.t().contiguous().view(H, H).to(torch.float32)  # (K, N), K=H, N=H
    Bias = out_proj_bias.contiguous().view(H).to(torch.float32)          # (N)

    output = torch.empty((M, H), dtype=torch.float32, device=y.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
    _out_proj_linear_kernel[grid](
        A, B_w, Bias, output,
        M, H, H,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return output.view(B, S, H)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # 1) in_proj via Triton
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H), float32

        # 2) Split BCx into B, C, x_proj
        B = BCx[:, :, :H].contiguous()            # (B, S, H)
        C = BCx[:, :, H:2*H].contiguous()         # (B, S, H)
        x_proj = BCx[:, :, 2*H:].contiguous()     # (B, S, H)

        # 3) Elementwise gate: Bx = B * x_proj
        Bx = _triton_elementwise_mul_1d(B, x_proj)  # (B, S, H), float32

        # 4) Grouped causal conv: prepare conv_weight from in_proj_weight's last 4 columns
        # Note: conv_weight in original is derived from in_proj_weight[:, -4:], reshaped to (H, 1, 4)
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)  # (H, 4)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias) # (B, H, S), float32

        # 5) Output gating: y = C * conv_out; align shapes (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_1d(C, conv_out_T)         # (B, S, H), float32

        # 6) Final out-projection via Triton
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H), float32

        return output


def run(*args):
    return ModelNew()(*args)
