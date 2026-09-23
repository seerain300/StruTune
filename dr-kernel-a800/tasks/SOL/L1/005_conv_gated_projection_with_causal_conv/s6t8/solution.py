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
    # Compute C = A @ B + Bias, where:
    # A: (M, K), B: (K, N)
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
def _elementwise_mul_1d_kernel(a_ptr, b_ptr, c_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    # 1D elementwise multiply: c[i] = a[i] * b[i]
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    c = a * b
    tl.store(c_ptr + offs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr,  # input pointer to (B, H, S), float32
    conv_w_ptr,  # conv_weight pointer to (H, 4), float32
    conv_b_ptr,  # conv_bias pointer to (H), float32
    conv_out_ptr,  # output pointer to (B, H, S), float32
    B, H, S, K: tl.constexpr,  # K=4
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_w_h, stride_w_k,
    stride_out_b, stride_out_h, stride_out_s,
):
    # Launch grid: (B, H)
    b = tl.program_id(0)
    h = tl.program_id(1)
    # We will iterate over s in tiles and compute conv_out[b, h, s] for s in [0..S-1]
    # Each output position s uses the sliding window: Bx[b, h, s+3 - k] for k in [0..3]
    # groups=H implies conv per channel independently; conv_w[h, k] is used for channel h.

    for s_start in range(0, S, 1):
        s = s_start
        # Accumulate over kernel size K=4
        acc = tl.zeros((), dtype=tl.float32)
        # conv_w[h, k] for k=0..3
        # conv_w_ptr indexing: offset = h*stride_w_h + k*stride_w_k
        w0 = tl.load(conv_w_ptr + h * stride_w_h + 0 * stride_w_k).to(tl.float32)
        w1 = tl.load(conv_w_ptr + h * stride_w_h + 1 * stride_w_k).to(tl.float32)
        w2 = tl.load(conv_w_ptr + h * stride_w_h + 2 * stride_w_k).to(tl.float32)
        w3 = tl.load(conv_w_ptr + h * stride_w_h + 3 * stride_w_k).to(tl.float32)

        # For causal, use indices s+3 - k, with zero padding when index < 0
        idx0 = s + 3 - 0
        idx1 = s + 3 - 1
        idx2 = s + 3 - 2
        idx3 = s + 3 - 3

        val0 = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + idx0 * stride_bx_s, mask=(idx0 >= 0), other=0.0)
        val1 = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + idx1 * stride_bx_s, mask=(idx1 >= 0), other=0.0)
        val2 = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + idx2 * stride_bx_s, mask=(idx2 >= 0), other=0.0)
        val3 = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + idx3 * stride_bx_s, mask=(idx3 >= 0), other=0.0)

        acc = w0 * val0 + w1 * val1 + w2 * val2 + w3 * val3
        # Add bias
        acc = acc + tl.load(conv_b_ptr + h).to(tl.float32)

        # Store conv_out[b, h, s]
        tl.store(conv_out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s, acc)


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute BCx = x @ in_proj_weight^T + in_proj_bias via Triton.
    x: (B, S, H), float32, contiguous
    in_proj_weight: (3H, H), float32, contiguous
    in_proj_bias: (3H), float32, contiguous
    Returns BCx: (B, S, 3H), float32, contiguous
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    A = x.contiguous().to(torch.float32).view(M, K)                  # (M, K)
    B_w = in_proj_weight.contiguous().to(torch.float32).view(K, N)   # (K, N)
    Bias = in_proj_bias.contiguous().to(torch.float32).view(N)       # (N)

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


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton out-proj: output = y @ out_proj_weight^T + out_proj_bias
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H), float32
    Returns output: (B, S, H), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    M = B * S
    N = H

    y_f32 = y.contiguous().to(torch.float32)                       # (B, S, H)
    W_t = out_proj_weight.contiguous().to(torch.float32).view(H, N)  # (H, N), out_proj_weight (H, H) -> (H, H)
    Bias = out_proj_bias.contiguous().to(torch.float32).view(N)     # (N)

    C = torch.empty((M, N), dtype=torch.float32, device=y.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        y_f32.view(M, H), W_t, Bias, C,
        M, N, H,
        y_f32.stride(0), y_f32.stride(1),
        W_t.stride(0), W_t.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C.view(B, S, N)


def _triton_elementwise_mul_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton 1D elementwise multiply: C = A * B
    a, b: any shape, float32
    returns C with same shape as a, float32
    """
    total = a.numel()
    a_flat = a.contiguous().to(torch.float32).view(-1)
    b_flat = b.contiguous().to(torch.float32).view(-1)
    c = torch.empty_like(a_flat, dtype=torch.float32, device=a.device)

    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _elementwise_mul_1d_kernel[grid](
        a_flat, b_flat, c, total, BLOCK=BLOCK, num_warps=4, num_stages=1,
    )
    return c.view(a.shape)


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton grouped causal 1D conv: input Bx: (B, H, S) -> output conv_out: (B, H, S)
    conv_weight: (H, 4), conv_bias: (H), float32
    Kernel size fixed at 4; groups=H (depthwise).
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S = Bx.shape

    # Ensure dtype and contiguity
    Bx_f32 = Bx.contiguous().to(torch.float32)                # (B, H, S)
    conv_w_f32 = conv_weight.contiguous().to(torch.float32)   # (H, 4)
    conv_b_f32 = conv_bias.contiguous().to(torch.float32)     # (H)

    conv_out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)  # (B, H, S)

    grid = (B, H)
    _grouped_causal_conv1d_kernel[grid](
        Bx_f32, conv_w_f32, conv_b_f32, conv_out,
        B, H, S, 4,
        Bx_f32.stride(0), Bx_f32.stride(1), Bx_f32.stride(2),
        conv_w_f32.stride(0), conv_w_f32.stride(1),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        num_warps=2, num_stages=1,
    )
    return conv_out


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        """
        Triton-optimized forward:
        - in_proj via matmul
        - elementwise gating via Triton
        - grouped causal conv via Triton (no F.pad/F.conv1d)
        - final out-proj via matmul
        All tensors are float32. Triton kernels are invoked for all heavy ops.
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)   # (B, S, 3H)

        # 2) Split into B, C, x_proj
        B = BCx[:, :, :H]        # (B, S, H)
        C = BCx[:, :, H:2*H]     # (B, S, H)
        x_proj = BCx[:, :, 2*H:] # (B, S, H)

        # 3) Elementwise gate: Bx = B * x_proj
        Bx = _triton_elementwise_mul_1d(B, x_proj)               # (B, S, H)

        # 4) Grouped causal 1D conv on Bx (kernel_size=4), groups=H
        #    Original code sets conv_weight = in_proj_weight[:, -4:], reshaped (H, 1, 4), and padding=3
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)  # (H, 4)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias) # (B, H, S)

        # 5) Output gating: y = C * conv_out
        #    conv_out_T: (B, S, H) via transpose and contiguous
        conv_out_T = conv_out.transpose(-1, -2).contiguous()    # (B, S, H)
        y = _triton_elementwise_mul_1d(C, conv_out_T)           # (B, S, H)

        # 6) Final out-proj: output = y @ out_proj_weight^T + out_proj_bias
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
