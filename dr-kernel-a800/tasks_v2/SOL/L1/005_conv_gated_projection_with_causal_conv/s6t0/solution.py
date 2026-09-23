import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton matmul kernel: C[M, N] = A[M, K] @ B[K, N] + bias[N]
# A is x, B is in_proj_weight. We'll launch over M and N tiles.
@triton.jit
def _matmul_linear_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Pointers for A and B tiles
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for boundary
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc = acc + bias[None, :]

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Triton matmul kernel for final output projection:
# C[M, N] = A[M, K] @ B[K, N] + bias[N]
# Here A is y (B*seq_len, hidden_size), B is out_proj_weight (hidden_size, hidden_size)
@triton.jit
def _matmul_outproj_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,  # M = B*seq_len, N = hidden_size, K = hidden_size
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
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


def _triton_inproj(x, in_proj_weight, in_proj_bias):
    """
    Compute BCx = in_proj(x) where x: (B, S, H), in_proj_weight: (3*H, H),
    returns: (B, S, 3*H)
    """
    assert x.dim() == 3, "x must be (B, S, H)"
    assert in_proj_weight.dim() == 2, "in_proj_weight must be (N, K) with N=3*H, K=H"
    B, S, H = x.shape
    N = in_proj_weight.shape[0]
    K = in_proj_weight.shape[1]
    assert N == 3 * H and K == H, "in_proj_weight shape must be (3*H, H)"

    # Flatten to (M, K) for Triton, M = B*S
    x_2d = x.reshape(B * S, H).contiguous()
    in_proj_weight_2d = in_proj_weight.contiguous()
    in_proj_bias_2d = in_proj_bias.contiguous()

    # Allocate output (M, N)
    out_2d = torch.empty((B * S, N), device=x.device, dtype=torch.float32)

    # Launch grid
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(B * S, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _matmul_linear_kernel[grid](
        x_2d, in_proj_weight_2d, in_proj_bias_2d, out_2d,
        B * S, N, H,
        x_2d.stride(0), x_2d.stride(1),
        in_proj_weight_2d.stride(0), in_proj_weight_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # Reshape back to (B, S, 3*H)
    return out_2d.reshape(B, S, 3 * H)


def _triton_outproj(y, out_proj_weight, out_proj_bias):
    """
    Compute output = out_proj(y) where y: (B, S, H), out_proj_weight: (H, H),
    returns: (B, S, H)
    """
    assert y.dim() == 3, "y must be (B, S, H)"
    assert out_proj_weight.dim() == 2, "out_proj_weight must be (N, K) with N=H, K=H"
    B, S, H = y.shape
    N = out_proj_weight.shape[0]
    K = out_proj_weight.shape[1]
    assert N == H and K == H, "out_proj_weight must be (H, H)"

    # Flatten y to (M, K), M = B*S
    y_2d = y.reshape(B * S, H).contiguous()
    out_proj_weight_2d = out_proj_weight.contiguous()
    out_proj_bias_2d = out_proj_bias.contiguous()

    output_2d = torch.empty((B * S, H), device=y.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(B * S, BLOCK_M), triton.cdiv(H, BLOCK_N))

    _matmul_outproj_kernel[grid](
        y_2d, out_proj_weight_2d, out_proj_bias_2d, output_2d,
        B * S, H, H,
        y_2d.stride(0), y_2d.stride(1),
        out_proj_weight_2d.stride(0), out_proj_weight_2d.stride(1),
        output_2d.stride(0), output_2d.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    return output_2d.reshape(B, S, H)


@triton.jit
def _gate_mul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 64 + tl.arange(0, 64)
    offs_n = pid_n * 64 + tl.arange(0, 64)
    acc = tl.zeros((64, 64), dtype=tl.float32)
    for k0 in range(0, N, 64):
        offs_k = k0 + tl.arange(0, 64)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < N)
        b_mask = (offs_k[:, None] < N) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += a * b

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
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 64 + tl.arange(0, 64)
    offs_n = pid_n * 64 + tl.arange(0, 64)
    for k0 in range(0, N, 64):
        offs_k = k0 + tl.arange(0, 64)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < N)
        b_mask = (offs_k[:, None] < N) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        c = a * b
        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    """
    Triton-optimized version:
      - In-proj linear computed by Triton matmul kernel.
      - Grouped causal conv1d handled by PyTorch F.conv1d (groups=hidden_size).
      - Elementwise gating (Bx = B * x_proj) implemented as Triton elementwise kernels.
      - Output gating (y = C * conv_out) implemented as Triton elementwise kernel.
      - Final out-proj linear computed by Triton matmul kernel.
    """
    assert TRITON_AVAILABLE, "Triton is required but not available"

    # 1) In-proj linear via Triton
    # x: (B, S, H), in_proj_weight: (3*H, H)
    BCx = _triton_inproj(x, in_proj_weight, in_proj_bias)  # (B, S, 3*H)

    # 2) Split into B, C, x_proj: each (B, H, S)
    # chunk along last dim=3*H
    H = in_proj_weight.shape[1]  # original hidden_size
    B = BCx[:, :, :H]            # (B, S, H)
    x_proj = BCx[:, :, H:2*H]    # (B, S, H)
    C = BCx[:, :, 2*H:]          # (B, S, H)

    # 3) Elementwise gating: Bx = B * x_proj
    # Make them 2D (M, H) for Triton
    B_2d = B.reshape(-1, H).contiguous()
    x_proj_2d = x_proj.reshape(-1, H).contiguous()
    Bx = torch.empty_like(B_2d, device=x.device, dtype=torch.float32)
    M = B.shape[0] * S
    grid_mul = (triton.cdiv(M, 64), triton.cdiv(H, 64))
    _gate_mul_kernel[grid_mul](
        B_2d, x_proj_2d, Bx,
        M, H,
        B_2d.stride(0), B_2d.stride(1),
        x_proj_2d.stride(0), x_proj_2d.stride(1),
        Bx.stride(0), Bx.stride(1),
        num_warps=4, num_stages=2,
    )
    Bx = Bx.reshape(B.shape[0], S, H)  # (B, S, H), same dtype as inputs (float32)

    # 4) Grouped causal 1D convolution using PyTorch (as in the original)
    # conv_weight: (H, 1, 4), conv_bias: (H,)
    # Input for conv: (B, H, S) -> pad 3 on left (kernel_size=4 => causal pad=3)
    Bx_padded = F.pad(Bx, (3, 0))
    conv_out = F.conv1d(Bx_padded, conv_weight, conv_bias, groups=H)

    # 5) Output gating: y = C * conv_out
    C_2d = C.reshape(-1, H).contiguous()
    conv_out_2d = conv_out.reshape(-1, H).contiguous()  # (B*S, H)
    y = torch.empty_like(C_2d, device=x.device, dtype=torch.float32)
    grid_mul2 = (triton.cdiv(B.shape[0] * S, 64), triton.cdiv(H, 64))
    _elementwise_mul_kernel[grid_mul2](
        C_2d, conv_out_2d, y,
        B.shape[0] * S, H,
        C_2d.stride(0), C_2d.stride(1),
        conv_out_2d.stride(0), conv_out_2d.stride(1),
        y.stride(0), y.stride(1),
        num_warps=4, num_stages=2,
    )
    y = y.reshape(B.shape[0], S, H)  # (B, S, H)

    # 6) Final out-proj linear via Triton
    output = _triton_outproj(y, out_proj_weight, out_proj_bias)  # (B, S, H)

    return output


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Ensure inputs are on the same device and dtype (float32 for Triton kernels)
        # We cast to float32 for numerical stability and Triton expectations.
        # If you want exact dtype preservation, adjust accordingly.
        x = x.to(device='cuda', dtype=torch.float32)
        in_proj_weight = in_proj_weight.to(device='cuda', dtype=torch.float32)
        in_proj_bias = in_proj_bias.to(device='cuda', dtype=torch.float32)
        conv_weight = conv_weight.to(device='cuda', dtype=torch.float32)
        conv_bias = conv_bias.to(device='cuda', dtype=torch.float32)
        out_proj_weight = out_proj_weight.to(device='cuda', dtype=torch.float32)
        out_proj_bias = out_proj_bias.to(device='cuda', dtype=torch.float32)
        return run(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias)


def run(*args):
    return ModelNew()(*args)
