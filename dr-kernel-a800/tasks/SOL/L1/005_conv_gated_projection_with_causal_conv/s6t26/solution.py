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
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,  # here M = B, N = S*H (flattened)
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols (flattened)
    # Note: inputs are flattened to 2D. We infer mapping as A[offs_m, offs_n], B[offs_m, offs_n].
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    c = a * b
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    X_ptr,  # Bx_padded: (B*H, S+3) laid out as row-major with rows=b*H + h
    W_ptr,  # conv_weight: (H, 4)
    Bias_ptr,  # conv_bias: (H)
    Out_ptr,  # conv_out: (B*H, S)
    B, S, H,
    stride_xm, stride_xs,  # X strides: m=rows=B*H, s=cols=S+3
    stride_om, stride_os,  # Out strides: m=rows=B*H, s=cols=S
    BLOCK_S: tl.constexpr,
):
    pid_m = tl.program_id(0)  # over B*H rows
    pid_s = tl.program_id(1)  # over tiles of S
    h = pid_m % H
    b = pid_m // H

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    # Initialize accumulator for this (b, h)
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # We will loop over kernel k=0..3 and accumulate:
    # x_idx = (b*H + h) * (S+3) + (offs_s + 3 - k)
    # conv_out[b, h, s] = sum_k X[b,h,x_idx] * W[h,k] + Bias[h]
    # Note: X pointer is flat; we compute offset with strides.
    for k in range(4):
        x_idx = (b * H + h) * (S + 3) + (offs_s + 3 - k)
        x_val = tl.load(X_ptr + x_idx, mask=mask_s, other=0.0)
        w_val = tl.load(W_ptr + h * 4 + k)
        acc += x_val * w_val

    bias_val = tl.load(Bias_ptr + h)
    acc += bias_val

    out_idx = pid_m * S + offs_s
    tl.store(Out_ptr + out_idx, acc, mask=mask_s)


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute BCx = x @ in_proj_weight^T + in_proj_bias.
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns BCx: (B, S, 3H) as float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    A = x.contiguous().view(M, K).to(torch.float32)              # (M, K), float32
    B_w = in_proj_weight.t().contiguous().view(K, N).to(torch.float32)  # (K, N), float32
    Bias = in_proj_bias.contiguous().view(N).to(torch.float32)   # (N), float32

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


def _triton_elementwise_mul_2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiply of A and B, both (B, S, H) float32, returns (B, S, H) float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    Bsz, S, H = A.shape
    M = Bsz
    N = S * H
    A_flat = A.contiguous().view(M, N).to(torch.float32)
    B_flat = B.contiguous().view(M, N).to(torch.float32)
    C_flat = torch.empty((M, N), dtype=torch.float32, device=A.device)

    BLOCK_M = 64
    BLOCK_N = 256
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _elementwise_mul_2d_kernel[grid](
        A_flat, B_flat, C_flat,
        M, N,
        A_flat.stride(0), A_flat.stride(1),
        B_flat.stride(0), B_flat.stride(1),
        C_flat.stride(0), C_flat.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return C_flat.view(Bsz, S, H)


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_w: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Grouped causal 1D conv with kernel_size=4, padding=3, groups=H.
    Bx: (B, H, S) float32 (original conv input, no padding on host).
    conv_w: (H, 4) float32.
    conv_bias: (H) float32.
    Returns conv_out: (B, H, S) float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S = Bx.shape
    # Build Bx_padded: (B, H, S+3) with 3 zeros on the left (causal)
    S_padded = S + 3
    Bx_padded = torch.nn.functional.pad(Bx, (3, 0), mode='constant', value=0.0).to(torch.float32)  # (B, H, S+3)
    # Flatten rows: (B*H, S+3)
    X = Bx_padded.view(B * H, S_padded).contiguous()
    W = conv_w.contiguous().to(torch.float32)                   # (H, 4)
    Bias = conv_bias.contiguous().to(torch.float32)             # (H)
    Out = torch.empty((B * H, S), dtype=torch.float32, device=Bx.device)

    BLOCK_S = 128
    grid = (B * H, triton.cdiv(S, BLOCK_S))
    _grouped_causal_conv1d_kernel[grid](
        X, W, Bias, Out,
        B, S, H,
        X.stride(0), X.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_S=BLOCK_S,
        num_warps=2, num_stages=2,
    )
    return Out.view(B, H, S)


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Compute output = y @ out_proj_weight^T + out_proj_bias.
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns output: (B, S, H) as float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    K = H
    N = H
    M = B * S

    A = y.contiguous().view(M, K).to(torch.float32)             # (M, K)
    B_w = out_proj_weight.t().contiguous().view(K, N).to(torch.float32)  # (K, N)
    Bias = out_proj_bias.contiguous().view(N).to(torch.float32) # (N)

    C = torch.empty((M, N), dtype=torch.float32, device=y.device)

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


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor) -> torch.Tensor:
        """
        Triton-only fused forward:
        1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        2) Split BCx into B, C, x_proj
        3) Elementwise gate: Bx = B * x_proj
        4) Grouped causal conv with kernel_size=4, padding=3, groups=H on Bx:
           conv_weight derived from in_proj_weight's last 4 columns: in_proj_weight[:, -4:], (H, 4)
        5) Output gating: y = C * conv_out (conv_out transposed to (B, S, H))
        6) Final out-proj: y @ out_proj_weight^T + out_proj_bias
        Returns output: (B, S, H), float32.
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H), float32

        # 2) Split BCx into B, C, x_proj
        B_part = BCx[:, :, :H]                                  # (B, S, H), float32
        C_part = BCx[:, :, H:2 * H]                            # (B, S, H), float32
        x_proj = BCx[:, :, 2 * H:]                             # (B, S, H), float32

        # 3) Elementwise gate: Bx = B_part * x_proj
        Bx = _triton_elementwise_mul_2d(B_part, x_proj)        # (B, S, H), float32

        # 4) Grouped causal conv: derive conv_weight from in_proj_weight's last 4 columns
        conv_w = in_proj_weight[:, -4:].contiguous()           # (H, 4)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias)  # (B, H, S), float32

        # 5) Output gating: y = C_part * conv_out (align shapes: conv_out.T -> (B, S, H))
        conv_out_T = conv_out.transpose(-1, -2).contiguous()   # (B, S, H), float32
        y = _triton_elementwise_mul_2d(C_part, conv_out_T)     # (B, S, H), float32

        # 6) Final out-proj
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H), float32

        return output


def run(*args):
    return ModelNew()(*args)
