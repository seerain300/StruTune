import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


# Triton kernels for GEMM:
# A: (M, K), B: (K, N) where B is conv_out_weight (N_out, K_in) = (1024, 3840).
# We will index B as b_k = conv_out_weight[n, k] to represent (K, N). Inside kernel we load B tiles using conv_out_weight[n, k].
# Output: C_fp32 (M, N) in fp32, which we'll cast to bf16 after kernel in PyTorch.

@triton.jit
def matmul_bf16_f32(
    A_ptr,  # *bf16, shape (M, K)
    B_ptr,  # *bf16, shape (N_out, K_in) = (1024, 3840); we index B as (K, N) via (k, n)
    C_ptr,  # *fp32, shape (M, N) output
    M, N, K,
    stride_am, stride_ak,  # strides for A
    stride_bn, stride_bk,  # strides for B; here, we access conv_out_weight[n, k] => stride over n and k
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K tile
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=tl.zeros((), dtype=tl.bfloat16))
        # Load B as (BLOCK_K, BLOCK_N): k index along rows, n along cols
        b = tl.load(B_ptrs, mask=(offs_n[None, :] < N) & (k + offs_k[:, None] < K), other=tl.zeros((), dtype=tl.bfloat16))
        # Accumulate
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        # Advance pointers
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Store results
    C_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=mask_out)


@triton.jit
def matmul_bf16_bf16(
    A_ptr,  # *bf16, shape (M, K)
    B_ptr,  # *bf16, shape (N_out, K_in) = (1024, 3840)
    C_ptr,  # *bf16, shape (M, N) output
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=tl.zeros((), dtype=tl.bfloat16))
        b = tl.load(B_ptrs, mask=(offs_n[None, :] < N) & (k + offs_k[:, None] < K), other=tl.zeros((), dtype=tl.bfloat16))
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    C_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Cast acc (fp32) to bf16 for storage
    tl.store(C_ptrs, acc.to(tl.bfloat16), mask=mask_out)


def triton_linear_bf16_fp32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute A @ B^T with A: (M, K) bfloat16, B: (N_out, K_in) bfloat16.
    Returns C_fp32: (M, N_out) float32.
    """
    assert A.is_cuda and B.is_cuda, "Triton kernels require CUDA tensors."
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16, "Inputs must be bfloat16."
    M, K = A.shape
    N_out, K_in = B.shape
    assert K == K_in, "Incompatible shapes for GEMM."

    # Output buffer in fp32
    C = torch.empty((M, N_out), device=A.device, dtype=torch.float32)

    # Strides
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bn = B.stride(0)  # stride along N_out (rows)
    stride_bk = B.stride(1)  # stride along K_in (cols)

    # Grid
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_out, meta['BLOCK_N']))

    # Launch kernel
    matmul_bf16_f32[grid](
        A, B, C,
        M, N_out, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=128,
        num_warps=4, num_stages=2
    )

    return C


def triton_linear_bf16_bf16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute A @ B^T with A: (M, K) bfloat16, B: (N_out, K_in) bfloat16.
    Returns C_bf16: (M, N_out) bfloat16.
    """
    assert A.is_cuda and B.is_cuda, "Triton kernels require CUDA tensors."
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16, "Inputs must be bfloat16."
    M, K = A.shape
    N_out, K_in = B.shape
    assert K == K_in, "Incompatible shapes for GEMM."

    C = torch.empty((M, N_out), device=A.device, dtype=torch.bfloat16)

    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bn = B.stride(0)
    stride_bk = B.stride(1)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_out, meta['BLOCK_N']))

    matmul_bf16_bf16[grid](
        A, B, C,
        M, N_out, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=128,
        num_warps=4, num_stages=2
    )

    return C


class ModelNew(torch.nn.Module):
    def forward(self, input_features: torch.Tensor,
                conv2d1_weight: torch.Tensor, conv2d1_bias: torch.Tensor,
                conv2d2_weight: torch.Tensor, conv2d2_bias: torch.Tensor,
                conv2d3_weight: torch.Tensor, conv2d3_bias: torch.Tensor,
                conv_out_weight: torch.Tensor,
                positional_embedding: torch.Tensor,
                embed_scale: float):
        # Convolution Stage 1
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = F.gelu(x)
        # Convolution Stage 2
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = F.gelu(x)
        # Convolution Stage 3
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Reshape to (B, t, channels*freq)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # (B, t, 15360)

        # Ensure dtype/device consistency
        # We keep everything in bfloat16 for compute; conv_out_weight is (1024, 3840) bfloat16
        assert conv_out_weight.is_cuda and conv_out_weight.dtype == torch.bfloat16, "conv_out_weight must be CUDA bfloat16"
        assert x.is_cuda, "Input tensor x must be CUDA for Triton kernel"

        # Triton linear: x @ conv_out_weight.T -> output (B, t, 1024)
        # x is (B, t, K_in=15360), conv_out_weight is (N_out=1024, K_in=3840), but we need K_in=15360 -> cast/reshape conv_out_weight or use bf16_f32 path and cast back.
        # Note: conv_out_weight has 3840 columns; we need K_in=15360. The original code uses conv_out_weight of shape (d_model=1024, conv_out_dim=3840), but we must ensure it matches K_in.
        # Here, conv_out_weight must be shape (1024, 15360) to match the view (B, t, 15360) @ (1024, 15360)^T -> (B, t, 1024). Given the provided setup, it is (1024, 3840). This implies the original code implicitly relies on conv_out_dim matching C*F.
        # To ensure correctness across workloads, we adjust conv_out_weight to have second dim equal to x.shape[-1] (i.e., 15360). If it doesn't, we fall back to PyTorch linear for correctness.
        K_in = x.shape[-1]  # 15360
        N_out = conv_out_weight.shape[0]  # 1024

        if conv_out_weight.shape[1] != K_in:
            # Fallback to PyTorch if shapes are incompatible (defensive)
            y = F.linear(x, conv_out_weight, bias=None)
        else:
            # Cast to bfloat16 if not already
            if conv_out_weight.dtype != torch.bfloat16:
                conv_out_weight = conv_out_weight.to(torch.bfloat16)
            # Launch Triton kernel in fp32 store mode, then cast to bf16 at the end to match original pipeline
            x_bf16 = x.to(torch.bfloat16)
            C_fp32 = triton_linear_bf16_fp32(x_bf16, conv_out_weight)
            y = C_fp32.to(torch.bfloat16)

        # Scale embeddings
        y = y * embed_scale

        # Add positional embedding
        # positional_embedding: (max_source_positions=1500, d_model=1024), dtype bfloat16, device CUDA
        assert positional_embedding.is_cuda and positional_embedding.dtype == torch.bfloat16
        seq_len = y.shape[1]  # equals t
        pos_embed = positional_embedding[:seq_len, :].unsqueeze(0)  # (1, t, 1024)
        y = y + pos_embed

        return y


def run(*args):
    return ModelNew()(*args)
