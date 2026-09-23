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
    # A: (M, K), row-major: A[i, j] at ptr + i*stride_am + j*stride_ak
    # B: (K, N), row-major: B[k, n] at ptr + k*stride_bk + n*stride_bn
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
        # a: (BM, BK), b: (BK, BN) -> acc += a @ b
        acc += tl.dot(a, b)

    # Add bias: Bias is (N,) -> broadcast along rows
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    Bsz, Sz, H,
    stride_ab, stride_am, stride_an,
    stride_bb, stride_bm, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK: tl.constexpr,
):
    # Elementwise C = A * B, all tensors have shape (Bsz, Sz, H)
    pid = tl.program_id(0)
    total = Bsz * Sz * H
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    # Map linear index -> (b, s, h)
    Ht = H
    SzH = Sz * Ht
    b = offs // (SzH)
    rem = offs % (SzH)
    s = rem // Ht
    h = rem % Ht

    a_ptrs = A_ptr + b * stride_ab + s * stride_am + h * stride_an
    b_ptrs = B_ptr + b * stride_bb + s * stride_bm + h * stride_bn
    c_ptrs = C_ptr + b * stride_cb + s * stride_cm + h * stride_cn

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b_val = tl.load(b_ptrs, mask=mask, other=0.0)
    c = a * b_val
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    In_ptr, Weight_ptr, Bias_ptr, Out_ptr,
    Bsz, Hsz, S, KS,  # KS = 4
    stride_in_b, stride_in_h, stride_in_s,
    stride_w_h, stride_w_k,
    stride_out_b, stride_out_h, stride_out_s,
):
    # Each program handles one (b, h) pair, computes conv_out[b, h, 0..S-1]
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Iterate over positions s = 0..S-1
    for s in range(0, S):
        acc = 0.0
        # Fixed kernel_size=4, causal: input index = s + 3 - k, guarded by mask
        for k in range(0, 4):
            in_idx = s + 3 - k
            mask_pos = in_idx >= 0
            in_ptr = In_ptr + pid_b * stride_in_b + pid_h * stride_in_h + in_idx * stride_in_s
            val = tl.load(in_ptr, mask=mask_pos, other=0.0)
            w_ptr = Weight_ptr + pid_h * stride_w_h + k * stride_w_k
            w = tl.load(w_ptr)
            acc += val * w
        # Add bias
        bias_ptr = Bias_ptr + pid_h
        bias = tl.load(bias_ptr)
        acc += bias
        out_ptr = Out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + s * stride_out_s
        tl.store(out_ptr, acc)


def _triton_in_proj(x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
    x: (B, S, H), in_proj_weight: (3H, H), in_proj_bias: (3H)
    Returns BCx: (B, S, 3H), float32. We keep float32 throughout for stability.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    # Ensure dtype float32 and contiguous
    x_f32 = x.contiguous().to(torch.float32)
    B, S, H = x_f32.shape
    K = H
    N = 3 * H
    M = B * S

    # A: (M, K) = x.view(M, H), B: (K, N) = in_proj_weight.T.view(H, 3H)
    A = x_f32.view(M, K)  # float32
    B_w = in_proj_weight.contiguous().to(torch.float32).t().view(K, N)  # (K, N), float32
    Bias = in_proj_bias.contiguous().to(torch.float32).view(N)          # (N), float32

    C = torch.empty((M, N), dtype=torch.float32, device=x_f32.device)

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
    # Reshape back to (B, S, 3H)
    BCx = C.view(B, S, 3 * H)
    return BCx  # float32


def _triton_elementwise_mul_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton elementwise multiplication over flattened tensors: C = A * B
    a, b: tensors of any shape, float32 recommended
    returns C with same shape as a, float32
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    a_f32 = a.contiguous().to(torch.float32)
    b_f32 = b.contiguous().to(torch.float32)
    total = a_f32.numel()
    c = torch.empty_like(a_f32, dtype=torch.float32, device=a_f32.device)

    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    _elementwise_mul_1d_kernel[grid](
        a_f32.view(-1), b_f32.view(-1), c.view(-1),
        total,
        BLOCK=BLOCK,
        num_warps=4, num_stages=1,
    )
    return c.reshape(a.shape)  # float32


def _triton_grouped_causal_conv1d(Bx: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton grouped causal 1D conv on Bx: input (B, H, S) -> output (B, H, S)
    conv_weight: (H, 4), conv_bias: (H)
    Kernel size fixed at 4, groups=H (depthwise grouped).
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, H, S = Bx.shape
    # Pre-pad Bx with 3 zeros on the left along sequence dimension to implement causal conv
    Bx_padded = torch.nn.functional.pad(Bx, (3, 0), mode='constant', value=0.0).to(torch.float32).contiguous()  # (B, H, S+3)
    # Allocate output (B, H, S), float32
    conv_out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)

    # Ensure conv_weight and conv_bias are float32 and contiguous
    conv_w = conv_weight.contiguous().to(torch.float32)  # (H, 4)
    conv_b = conv_bias.contiguous().to(torch.float32)    # (H)

    # Launch Triton kernel: one program per (b, h)
    grid = (B, H)
    _grouped_causal_conv1d_kernel[grid](
        Bx_padded, conv_w, conv_b, conv_out,
        B, H, S, 4,
        Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
        conv_w.stride(0), conv_w.stride(1),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        num_warps=1, num_stages=1,
    )
    return conv_out  # float32


def _triton_out_proj(y: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
    """
    Triton out-proj: output = y @ out_proj_weight^T + out_proj_bias
    y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H)
    Returns output: (B, S, H), float32.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    B, S, H = y.shape
    M = B * S
    N = H  # output channels = H

    y_f32 = y.contiguous().to(torch.float32)  # (B, S, H)
    A = y_f32.reshape(M, H)                    # (M, K)

    # out_proj_weight: (H, H) -> transpose to (K, N)
    W = out_proj_weight.contiguous().to(torch.float32).t().view(H, H)  # (K, N)

    Bias = out_proj_bias.contiguous().to(torch.float32).view(N)         # (N)

    Output = torch.empty((M, N), dtype=torch.float32, device=y_f32.device)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_linear_kernel[grid](
        A, W, Bias, Output,
        M, N, H,
        A.stride(0), A.stride(1),
        W.stride(0), W.stride(1),
        Output.stride(0), Output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return Output.reshape(B, S, H)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized fused forward:
        1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias, (B, S, 3H)
        2) Split into B, C, x_proj of shape (B, S, H)
        3) Elementwise gate: Bx = B * x_proj
        4) Grouped causal conv: conv_out = conv(Bx) with kernel_size=4, groups=H
        5) Output gating: y = C * conv_out  (conv_out must be (B, S, H))
        6) out_proj: output = y @ out_proj_weight^T + out_proj_bias
        Returns: output (B, S, H), float32.
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        # 1) in_proj via Triton matmul
        BCx = _triton_in_proj(x, in_proj_weight, in_proj_bias)  # (B, S, 3H), float32
        # Split into B, C, x_proj
        H = x.shape[-1]
        B = BCx[:, :, :H]           # (B, S, H)
        C = BCx[:, :, H:2*H]        # (B, S, H)
        x_proj = BCx[:, :, 2*H:]    # (B, S, H)

        # 2) Elementwise gate Bx = B * x_proj
        Bx = _triton_elementwise_mul_1d(B, x_proj)  # (B, S, H), float32

        # 3) Grouped causal conv: build conv_weight from in_proj_weight's last 4 columns
        # Original code constructs conv_weight = in_proj_weight[:, -4:], reshaped to (H, 1, 4).
        # Here we use (H, 4) directly in Triton kernel with groups=H via per-(b,h) indexing.
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)  # (H, 4)
        conv_out = _triton_grouped_causal_conv1d(Bx, conv_w, conv_bias)  # (B, H, S), float32

        # 4) Output gating: y = C * conv_out, align shapes: conv_out.transpose(-1, -2) -> (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H), float32
        y = _triton_elementwise_mul_1d(C, conv_out_T)         # (B, S, H), float32

        # 5) Final out-proj via Triton matmul
        output = _triton_out_proj(y, out_proj_weight, out_proj_bias)  # (B, S, H), float32

        return output


def run(*args):
    return ModelNew()(*args)
