import torch
import triton
import triton.language as tl


# LayerNorm + affine: one program per row (patch)
@triton.jit
def layernorm_affine_kernel(
    x_ptr,              # *float32, input [num_patches, hidden_size]
    out_ptr,            # *float32, output [num_patches, hidden_size]
    ln_weight_ptr,      # *float32, [hidden_size]
    ln_bias_ptr,        # *float32, [hidden_size]
    hidden_size: tl.constexpr,  # 1536
    eps,                # float32 scalar
    BLOCK_SIZE: tl.constexpr,   # tile size for reduction (e.g., 1024)
):
    pid = tl.program_id(0)  # one program per row
    total = 0.0
    total2 = 0.0

    # Compute mean and variance
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(x_ptr + pid * hidden_size + offs, mask=mask, other=0.0)
        x = tl.where(mask, x, 0.0)
        total += tl.sum(x, axis=0)
        total2 += tl.sum(x * x, axis=0)

    mean = total / hidden_size
    var = total2 / hidden_size - mean * mean
    inv_std = tl.math.rsqrt(var + eps)

    # Normalize and apply affine
    for col in range(0, hidden_size, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(x_ptr + pid * hidden_size + offs, mask=mask, other=0.0)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        out = y * w + b
        tl.store(out_ptr + pid * hidden_size + offs, out, mask=mask)


# GEMM + bias: C[M, N] = A[M, K] @ B[K, N] (+ Bias)
@triton.jit
def gemm_bias_kernel(
    A_ptr,              # *float32, [M, K]
    B_ptr,              # *float32, [K, N]
    Bias_ptr,           # *float32, [N]
    C_ptr,              # *float32, [M, N]
    M, K, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile along M
    pid_n = tl.program_id(1)  # tile along N
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load A tile: [BLOCK_M, BLOCK_K]
        a = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for i in range(0, BLOCK_M):
            a_i = tl.load(A_ptr + (m0 + i) * K + k_range, mask=(m0 + i) < M, other=0.0)
            a[i, :] = a_i
        # Load B tile: [BLOCK_K, BLOCK_N]
        b = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        for j in range(0, BLOCK_N):
            b_cols = tl.load(B_ptr + k_range * N + (n0 + j), mask=(n0 + j) < N, other=0.0)
            b[:, j] = b_cols
        # Accumulate
        acc += tl.dot(a, b)

    # Add bias: broadcast bias across rows
    bias = tl.load(Bias_ptr + n0 + tl.arange(0, BLOCK_N), mask=(n0 + tl.arange(0, BLOCK_N)) < N, other=0.0)
    acc += bias[None, :]

    # Store result
    for i in range(0, BLOCK_M):
        row_ptr = C_ptr + (m0 + i) * N + n0
        tl.store(row_ptr + tl.arange(0, BLOCK_N), acc[i, :], mask=(m0 + i) < M)


# Elementwise GELU on fp32
@triton.jit
def gelu_kernel(
    x_ptr,              # *float32, input [M, K]
    out_ptr,            # *float32, output [M, K]
    M, K,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < K
    x = tl.load(x_ptr + pid_m * K + offs, mask=mask, other=0.0)
    # GELU approximation
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    t = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.math.tanh(t))
    tl.store(out_ptr + pid_m * K + offs, y, mask=mask)


# Launch helper for LayerNorm + affine
def _launch_layernorm_affine(x_fp32, ln_weight_fp32, ln_bias_fp32, hidden_size, eps):
    num_patches = x_fp32.shape[0]
    out = torch.empty_like(x_fp32)
    grid = (num_patches,)
    layernorm_affine_kernel[grid](
        x_fp32, out, ln_weight_fp32, ln_bias_fp32,
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_SIZE=1024,
        num_warps=4,
    )
    return out


# Launch helper for GEMM + bias
def _launch_gemm_bias(A_fp32, B_fp32, Bias_fp32, M, K, N):
    out = torch.empty((M, N), dtype=torch.float32, device=A_fp32.device)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    gemm_bias_kernel[grid](
        A_fp32, B_fp32, Bias_fp32,
        out, M, K, N,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        num_warps=4,
    )
    return out


# Launch helper for GELU
def _launch_gelu(x_fp32, M, K):
    out = torch.empty_like(x_fp32)
    grid = (M, triton.cdiv(K, 256))
    gelu_kernel[grid](
        x_fp32, out, M, K,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect inputs: hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps
        hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps = args

        # Ensure tensors are on CUDA and dtype fp32 for Triton
        device = hidden.device
        assert hidden.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda and fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, "All tensors must be CUDA."
        assert hidden.dtype == torch.float32 and ln_weight.dtype == torch.float32 and ln_bias.dtype == torch.float32 and fc1_weight.dtype == torch.float32 and fc1_bias.dtype == torch.float32 and fc2_weight.dtype == torch.float32 and fc2_bias.dtype == torch.float32, "All tensors must be float32 for Triton kernels."

        # LayerNorm + affine
        hidden_norm = _launch_layernorm_affine(hidden, ln_weight, ln_bias, hidden.shape[1], eps)

        # fc1: A [num_merged_patches, 6144] @ B [6144, 6144] (+ bias)
        M = hidden_norm.shape[0]
        K = hidden_norm.shape[1]  # 6144
        fc1_out = _launch_gemm_bias(hidden_norm, fc1_weight, fc1_bias, M, K, K)

        # GELU activation
        fc1_out = _launch_gelu(fc1_out, M, K)

        # fc2: [M, 6144] @ fc2_weight [3584, 6144] (+ bias)
        N = fc2_weight.shape[0]  # 3584
        fc2_out = _launch_gemm_bias(fc1_out, fc2_weight, fc2_bias, M, K, N)

        return fc2_out


def run(*args):
    return ModelNew()(*args)
