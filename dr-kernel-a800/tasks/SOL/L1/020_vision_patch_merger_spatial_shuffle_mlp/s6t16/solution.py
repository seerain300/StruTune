import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_affine_kernel(
    x_ptr,        # *bf16, (M, K)
    w_ptr,        # *bf16, (K,)
    b_ptr,        # *bf16, (K,)
    y_ptr,        # *bf16, (M, K)
    M: tl.constexpr,  # number of rows (num_patches)
    K: tl.constexpr,  # hidden size (1536)
    eps: tl.constexpr,
    BLOCK: tl.constexpr,  # block size over K, set to 1024
):
    # One program per row
    row = tl.program_id(0)
    if row >= M:
        return

    # First pass: compute mean and variance in FP32
    sum_ = 0.0
    sumsq_ = 0.0
    for k0 in range(0, K, BLOCK):
        k = k0 + tl.arange(0, BLOCK)
        mask = k < K
        x = tl.load(x_ptr + row * K + k, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_ += tl.sum(x, axis=0)
        sumsq_ += tl.sum(x * x, axis=0)
    mean = sum_ / K
    var = sumsq_ / K - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for k0 in range(0, K, BLOCK):
        k = k0 + tl.arange(0, BLOCK)
        mask = k < K
        x = tl.load(x_ptr + row * K + k, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = (x - mean) * inv_std
        w = tl.load(w_ptr + k, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + k, mask=mask, other=0.0).to(tl.float32)
        y = y * w + b
        # Store as BF16
        y_out = y.to(tl.bfloat16)
        tl.store(y_ptr + row * K + k, y_out, mask=mask)


@triton.jit
def _gemm_bias_kernel(
    A_ptr,        # *bf16, (M, K) input
    W_ptr,        # *bf16, (N, K) weight
    B_ptr,        # *bf16, (M, N) output
    bias_ptr,     # *bf16, (N,) bias
    M: tl.constexpr,    # rows of A (num_merged_patches)
    N: tl.constexpr,    # cols of output
    K: tl.constexpr,    # inner dim (6144)
    BLOCK_M: tl.constexpr,  # tile size for rows
    BLOCK_N: tl.constexpr,  # tile size for cols
    BLOCK_K: tl.constexpr,  # tile size for inner
):
    # 2D grid over output rows and cols
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= triton.cdiv(M, BLOCK_M) or pid_n >= triton.cdiv(N, BLOCK_N):
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # A_tile: (BLOCK_M, BLOCK_K), row-major
        a_ptrs = A_ptr + (offs_m[:, None] * K + offs_k[None, :])
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # W_tile as (BLOCK_K, BLOCK_N): W_ptr[n, k]
        # We want B = A @ W^T => sum over k of A[i,k] * W[j,k]
        w_ptrs = W_ptr + (offs_n[None, :] * K + offs_k[:, None])
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # acc += A_tile @ W_tile
        acc += tl.dot(a, w)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store as BF16
    out_ptrs = B_ptr + (offs_m[:, None] * N + offs_n[None, :])
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@triton.jit
def _gelu_kernel(
    X_ptr,        # *bf16, (M, N) input
    Y_ptr,        # *bf16, (M, N) output
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= triton.cdiv(N, BLOCK_N):
        return
    offs_m = pid_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N
    x = tl.load(X_ptr + offs_m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    # GELU tanh approximation:
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = x * x * x
    inner = c0 * (x + c1 * x3)
    gelu = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(Y_ptr + offs_m * N + offs_n, gelu.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor,
                eps: float):
        # Ensure CUDA and dtype
        assert hidden.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda and \
               fc1_weight.is_cuda and fc1_bias.is_cuda and \
               fc2_weight.is_cuda and fc2_bias.is_cuda, "All inputs must be CUDA tensors"

        M = hidden.shape[0]
        K = hidden.shape[1]  # 1536

        # 1) LayerNorm + affine (Triton)
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)

        BLOCK_ln = 1024
        grid_ln = (M,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M=M, K=K, eps=1e-6,
            BLOCK=BLOCK_ln,
            num_warps=4, num_stages=2
        )

        # 2) Packing: reshape normalized hidden to (M_out, 4*K)
        M_out = M // 4  # invariant from provided inputs
        K_expanded = 4 * K  # 6144
        packed = ln_out.view(M_out, K_expanded)

        # 3) fc1: GEMM + bias (Triton), output shape (M_out, 6144)
        N1 = fc1_weight.shape[0]  # 6144
        K1 = K_expanded  # 6144
        fc1_out = torch.empty((M_out, N1), dtype=torch.bfloat16, device=hidden.device)

        BLOCK_M1 = 128
        BLOCK_N1 = 128
        BLOCK_K1 = 64
        grid_fc1 = (triton.cdiv(M_out, BLOCK_M1), triton.cdiv(N1, BLOCK_N1))
        _gemm_bias_kernel[grid_fc1](
            packed, fc1_weight, fc1_out, fc1_bias,
            M=M_out, N=N1, K=K1,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
            num_warps=4, num_stages=3
        )

        # 4) GELU activation (Triton)
        K_after_gelu = fc1_out.shape[1]  # 6144
        fc1_after_gelu = torch.empty_like(fc1_out, dtype=torch.bfloat16, device=hidden.device)

        BLOCK_N_gelu = 256
        grid_gelu = (M_out, triton.cdiv(K_after_gelu, BLOCK_N_gelu))
        _gelu_kernel[grid_gelu](
            fc1_out, fc1_after_gelu,
            M=M_out, N=K_after_gelu,
            BLOCK_N=BLOCK_N_gelu,
            num_warps=4, num_stages=2
        )

        # 5) fc2: GEMM + bias (Triton), output shape (M_out, 3584)
        N2 = fc2_weight.shape[0]  # 3584
        K2 = fc1_after_gelu.shape[1]  # 6144
        out = torch.empty((M_out, N2), dtype=torch.bfloat16, device=hidden.device)

        BLOCK_M2 = 128
        BLOCK_N2 = 128
        BLOCK_K2 = 64
        grid_fc2 = (triton.cdiv(M_out, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        _gemm_bias_kernel[grid_fc2](
            fc1_after_gelu, fc2_weight, out, fc2_bias,
            M=M_out, N=N2, K=K2,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=3
        )

        return out


def run(*args):
    return ModelNew()(*args)
