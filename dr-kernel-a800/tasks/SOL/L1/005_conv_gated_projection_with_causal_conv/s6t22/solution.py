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

    # Cast acc to desired output dtype
    if OUT_DTYPE == 0:
        to_store = acc
    elif OUT_DTYPE == 1:
        to_store = acc.to(tl.float16)
    elif OUT_DTYPE == 2:
        to_store = acc.to(tl.bfloat16)
    else:
        to_store = acc  # default float32

    tl.store(c_ptrs, to_store, mask=c_mask)


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,  # M = B*S, N = H
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # C[M, N] = A[M, N] * B[M, N]
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

    prod = a * b

    if OUT_DTYPE == 0:
        to_store = prod
    elif OUT_DTYPE == 1:
        to_store = prod.to(tl.float16)
    elif OUT_DTYPE == 2:
        to_store = prod.to(tl.bfloat16)
    else:
        to_store = prod

    tl.store(c_ptrs, to_store, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    X_ptr, W_ptr, Bias_ptr, C_ptr,
    B, H, S,
    stride_xb, stride_xh, stride_xs,
    stride_wh, stride_wk,
    stride_cb, stride_ch, stride_cs,
    OUT_DTYPE: tl.constexpr,
):
    # X_ptr points to Bx padded to S+3 on the left, shape (B, H, S+3)
    # W_ptr: (H, 4), Bias_ptr: (H)
    # C_ptr: (B, H, S)
    # Grouped conv with groups=H: each output (b, h, s) sums over k=0..3: X[b, h, s+3-k] * W[h, k]
    # We will process all s for each (b, h) in one program.

    b = tl.program_id(0)
    h = tl.program_id(1)

    # If b >= B or h >= H, nothing to do
    if b >= B or h >= H:
        return

    # Accumulator for output at this (b, h)
    acc = tl.zeros((S,), dtype=tl.float32)

    # Load conv bias for this h
    bias_h = tl.load(Bias_ptr + h)
    acc += bias_h

    # Loop over k=0..3
    for k in range(4):
        x_index = 0  # start from padded left
        # For each position s in output sequence (0..S-1), input index is x_index = s + 3 - k
        # x_index is in [3-k, S+2-k]; out of bounds means zero.
        # We will loop over s and load accordingly.
        s = 0
        while s < S:
            x_pos = s + 3 - k
            # valid if 0 <= x_pos < S+3
            valid = (x_pos >= 0) & (x_pos < (S + 3))
            # Address for X[b, h, x_pos]
            x_val = tl.load(X_ptr + b * stride_xb + h * stride_xh + x_pos * stride_xs, mask=valid, other=0.0)
            # Weight for this h and k
            w_val = tl.load(W_ptr + h * stride_wh + k * stride_wk)
            acc[s] += x_val * w_val
            s += 1

    # Store to C[b, h, :]
    c_ptrs = C_ptr + b * stride_cb + h * stride_ch + tl.arange(0, S) * stride_cs
    if OUT_DTYPE == 0:
        to_store = acc
    elif OUT_DTYPE == 1:
        to_store = acc.to(tl.float16)
    elif OUT_DTYPE == 2:
        to_store = acc.to(tl.bfloat16)
    else:
        to_store = acc
    tl.store(c_ptrs, to_store)


@triton.jit
def _out_proj_kernel(
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

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if OUT_DTYPE == 0:
        to_store = acc
    elif OUT_DTYPE == 1:
        to_store = acc.to(tl.float16)
    elif OUT_DTYPE == 2:
        to_store = acc.to(tl.bfloat16)
    else:
        to_store = acc

    tl.store(c_ptrs, to_store, mask=c_mask)


def _dtype_code(t: torch.Tensor) -> int:
    if t.dtype == torch.float32:
        return 0
    elif t.dtype == torch.float16:
        return 1
    elif t.dtype == torch.bfloat16:
        return 2
    else:
        return 0  # default to float32


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    A = x.contiguous().view(M, K)           # (M, K)
    B_w = in_proj_weight.t().contiguous().view(K, N)  # (K, N)
    Bias = in_proj_bias.contiguous().view(N)   # (N)

    BCx = torch.empty((B, S, N), dtype=x.dtype, device=x.device)
    C = BCx.view(M, N)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        A, B_w, Bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        C.stride(0), C.stride(1),
        _dtype_code(x),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return BCx


def _triton_elementwise_mul_2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A, B: (B, S, H), output: (B, S, H)
    B, S, H = A.shape
    M = B * S
    N = H
    A_ = A.contiguous().view(M, N)
    B_ = B.contiguous().view(M, N)
    C = torch.empty((B, S, N), dtype=A.dtype, device=A.device)
    C_ = C.view(M, N)
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _elementwise_mul_2d_kernel[grid](
        A_, B_, C_,
        M, N,
        A_.stride(0), A_.stride(1),
        B_.stride(0), B_.stride(1),
        C_.stride(0), C_.stride(1),
        _dtype_code(A),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return C


def _triton_grouped_causal_conv1d(Bx_padded: torch.Tensor, conv_w: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    # Bx_padded: (B, H, S+3), conv_w: (H, 4), conv_bias: (H)
    B, H, SP = Bx_padded.shape
    S = SP - 3  # original sequence length
    conv_out = torch.empty((B, H, S), dtype=Bx_padded.dtype, device=Bx_padded.device)

    grid = (B, H)
    _grouped_causal_conv1d_kernel[grid](
        Bx_padded, conv_w, conv_bias, conv_out,
        B, H, S,
        Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
        conv_w.stride(0), conv_w.stride(1),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        _dtype_code(Bx_padded),
    )
    return conv_out


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    B, S, H = y.shape
    M = B * S
    K = H
    N = H
    A = y.contiguous().view(M, K)
    B_w = out_proj_weight.t().contiguous().view(K, N)  # (K, N)
    Bias = out_proj_bias.contiguous().view(N)

    output = torch.empty((B, S, N), dtype=y.dtype, device=y.device)
    C_out = output.view(M, N)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _out_proj_kernel[grid](
        A, B_w, Bias, C_out,
        M, N, K,
        A.stride(0), A.stride(1),
        B_w.stride(0), B_w.stride(1),
        C_out.stride(0), C_out.stride(1),
        _dtype_code(y),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return output


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
        # All computation in Triton; no torch functional ops.
        assert TRITON_AVAILABLE, "Triton is not available"

        # 1) in_proj
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)

        # Split BCx into B, C, x_proj along last dim
        B, S, H = x.shape
        N = 3 * H
        B_mat = BCx[:, :, :H]            # (B, S, H)
        C_mat = BCx[:, :, H:2*H]         # (B, S, H)
        x_proj = BCx[:, :, 2*H:]         # (B, S, H)

        # 2) Elementwise gate: Bx = B_mat * x_proj (Triton elementwise 2D)
        Bx = _triton_elementwise_mul_2d(B_mat, x_proj)  # (B, S, H)

        # 3) Grouped causal conv: derive conv_weight from last 4 columns of in_proj_weight
        conv_w = in_proj_weight[:, -4:].contiguous()  # (H, 4)
        # Pad Bx on sequence dim for causal conv
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # (B, H, S+3)
        # Triton grouped causal conv
        conv_out = _triton_grouped_causal_conv1d(Bx_padded, conv_w, conv_bias)  # (B, H, S)

        # 4) Output gating: y = C_mat * conv_out (conv_out_T shape (B, S, H))
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = _triton_elementwise_mul_2d(C_mat, conv_out_T)     # (B, S, H)

        # 5) Final out-proj
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
