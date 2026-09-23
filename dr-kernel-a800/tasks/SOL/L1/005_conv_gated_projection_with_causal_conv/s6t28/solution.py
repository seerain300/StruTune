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
    # Computes C = A @ B + Bias, A: (M, K), B: (K, N)
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
    Bsz, S, H,
    stride_ab, stride_bb, stride_cb,
    BLOCK: tl.constexpr,
):
    # Multiplies A[Bsz, S, H] and B[Bsz, S, H] elementwise, writes C[Bsz, S, H]
    pid = tl.program_id(0)
    total = Bsz * S * H
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    a = tl.load(A_ptr + offs * stride_ab, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs * stride_bb, mask=mask, other=0.0)
    c = a * b
    tl.store(C_ptr + offs * stride_cb, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    Bx_ptr, ConvW_ptr, Bias_ptr, Out_ptr,
    Bsz, S, H,
    # Bx is (Bsz, H, S), ConvW is (H, 4), Bias is (H)
    stride_bx_row, stride_bx_col, stride_bx_seq,
    stride_ow_row, stride_ow_col, stride_ow_seq,
    BLOCK_S: tl.constexpr,
):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # Accumulator for this (b, h)
    acc = tl.zeros((1,), dtype=tl.float32)

    # Loop over output sequence positions
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S

        # For each k in kernel, compute padded index s_pos = s + 3 - k
        # Accumulate sum over k=0..3
        # conv_weight index: ConvW[h, k] = ConvW_ptr + h * stride_w_row + k * stride_w_col
        # Load conv weights for this h
        conv_w0 = tl.load(ConvW_ptr + h * 1 + 0 * 1)  # conv_weight[h, 0]
        conv_w1 = tl.load(ConvW_ptr + h * 1 + 1 * 1)  # conv_weight[h, 1]
        conv_w2 = tl.load(ConvW_ptr + h * 1 + 2 * 1)  # conv_weight[h, 2]
        conv_w3 = tl.load(ConvW_ptr + h * 1 + 3 * 1)  # conv_weight[h, 3]

        sum_k = tl.zeros((BLOCK_S,), dtype=tl.float32)
        # k=0
        s_pos0 = offs_s + 3 - 0
        mask0 = (s_pos0 < S) & mask_s
        ptr0 = Bx_ptr + b * stride_bx_row + h * stride_bx_col + s_pos0 * stride_bx_seq
        val0 = tl.load(ptr0, mask=mask0, other=0.0)
        sum_k += val0 * conv_w0

        # k=1
        s_pos1 = offs_s + 3 - 1
        mask1 = (s_pos1 < S) & mask_s
        ptr1 = Bx_ptr + b * stride_bx_row + h * stride_bx_col + s_pos1 * stride_bx_seq
        val1 = tl.load(ptr1, mask=mask1, other=0.0)
        sum_k += val1 * conv_w1

        # k=2
        s_pos2 = offs_s + 3 - 2
        mask2 = (s_pos2 < S) & mask_s
        ptr2 = Bx_ptr + b * stride_bx_row + h * stride_bx_col + s_pos2 * stride_bx_seq
        val2 = tl.load(ptr2, mask=mask2, other=0.0)
        sum_k += val2 * conv_w2

        # k=3
        s_pos3 = offs_s + 3 - 3
        mask3 = (s_pos3 < S) & mask_s
        ptr3 = Bx_ptr + b * stride_bx_row + h * stride_bx_col + s_pos3 * stride_bx_seq
        val3 = tl.load(ptr3, mask=mask3, other=0.0)
        sum_k += val3 * conv_w3

        # Add bias[h]
        bias_h = tl.load(Bias_ptr + h, mask=True, other=0.0).to(tl.float32)
        sum_k += bias_h

        # Store results to Out[b, h, s] at positions s in this tile
        out_ptrs = Out_ptr + b * stride_ow_row + h * stride_ow_col + offs_s * stride_ow_seq
        tl.store(out_ptrs, sum_k, mask=mask_s)


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of F.linear: BCx = x @ in_proj_weight^T + in_proj_bias
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns BCx: (B, S, 3H) as float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = x.shape
    K = H
    N = 3 * H
    M = B * S

    # Ensure contiguous and float32
    x_contig = x.contiguous()
    A = x_contig.view(M, H).to(torch.float32)  # (M, K)
    Wt = in_proj_weight.t().contiguous().to(torch.float32)  # (K, N)
    Bias = in_proj_bias.contiguous().to(torch.float32)      # (N)

    C = torch.empty((M, N), dtype=torch.float32, device=x.device)

    # Launch config
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        A, Wt, Bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        Wt.stride(0), Wt.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    # Reshape to (B, S, 3H)
    return C.view(B, S, 3 * H)


def _triton_elementwise_mul_2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiplication of two tensors A, B of shape (Bsz, S, H).
    Returns C (Bsz, S, H) as float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    Bsz, S, H = A.shape
    out = torch.empty_like(A, dtype=torch.float32, device=A.device)
    total = Bsz * S * H
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    _elementwise_mul_2d_kernel[grid](
        A.contiguous(), B.contiguous(), out,
        Bsz, S, H,
        A.stride(0), B.stride(0), out.stride(0),
        BLOCK=BLOCK,
        num_warps=2, num_stages=1,
    )
    return out


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of grouped causal 1D conv:
    Input Bx: (B, H, S), conv_weight: (H, 4), conv_bias: (H)
    Output conv_out: (B, H, S)
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S = Bx.shape

    Bx_f = Bx.contiguous().to(torch.float32)
    ConvW_f = conv_weight.contiguous().to(torch.float32)  # (H, 4)
    Bias_f = conv_bias.contiguous().to(torch.float32)     # (H)

    conv_out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)

    # One program per (b, h)
    grid = (B * H,)
    _grouped_causal_conv1d_kernel[grid](
        Bx_f, ConvW_f, Bias_f, conv_out,
        B, S, H,
        Bx_f.stride(0), Bx_f.stride(1), Bx_f.stride(2),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        BLOCK_S=128,
        num_warps=2, num_stages=2,
    )
    return conv_out


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of F.linear: output = y @ out_proj_weight^T + out_proj_bias
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns output: (B, S, H) as float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    K = H
    N = H  # out_proj weight is (H, H)
    M = B * S

    A = y.contiguous().view(M, K).to(torch.float32)            # (M, K)
    Wt = out_proj_weight.t().contiguous().to(torch.float32)    # (K, N)
    Bias = out_proj_bias.contiguous().to(torch.float32)        # (N)

    C = torch.empty((M, N), dtype=torch.float32, device=y.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        A, Wt, Bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        Wt.stride(0), Wt.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C.view(B, S, H)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor) -> torch.Tensor:
        """
        Triton-only forward implementing the original logic:
        1) BCx = x @ in_proj_weight^T + in_proj_bias
        2) Split BCx into B, C, x_proj
        3) Bx = B * x_proj
        4) conv_out = grouped causal conv of Bx with kernel_size=4 and groups=H, conv_weight from in_proj_weight[:, -4:]
        5) y = C * conv_out (after conv_out.T)
        6) output = y @ out_proj_weight^T + out_proj_bias
        All tensors are float32 and returned as float32.
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        # 1) in_proj
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)
        # Split into parts
        Bsz, S, H = x.shape
        # We can recover H from in_proj_weight (last dim of x is H)
        # Split: use last dim of BCx to split. But we need H. Original code uses H = x.shape[-1].
        H = x.shape[-1]
        # Slice along last dim
        B = BCx[:, :, :H]
        C = BCx[:, :, H:2 * H]
        x_proj = BCx[:, :, 2 * H:3 * H]

        # 2) Elementwise gate
        Bx = _triton_elementwise_mul_2d(B, x_proj)  # (B, S, H)

        # 3) Grouped causal conv
        # Prepare conv_weight from last 4 columns of in_proj_weight: (H, 4)
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)   # (H, 4)
        conv_bias_h = conv_bias.contiguous().to(torch.float32)           # (H)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias_h)  # (B, H, S)

        # 4) Output gating
        # y = C * conv_out after transposing conv_out to (B, S, H)
        conv_out_T = conv_out.transpose(1, 2).contiguous()   # (B, S, H)
        y = _triton_elementwise_mul_2d(C, conv_out_T)        # (B, S, H)

        # 5) Final out-proj
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
