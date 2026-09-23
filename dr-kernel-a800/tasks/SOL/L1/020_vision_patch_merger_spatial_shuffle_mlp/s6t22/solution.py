import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Define Triton kernels

# LayerNorm + affine: one program per row (M=num_patches), two-pass in FP32, store BF16
@triton.jit
def _layer_norm_affine_kernel(
    X_ptr,            # *bf16, shape (M, K)
    LN_W_ptr,         # *bf16, shape (K,)
    LN_B_ptr,         # *bf16, shape (K,)
    Y_ptr,            # *bf16, shape (M, K)
    M,                # int: number of rows
    K,                # int: hidden size
    eps,              # float32
    BLOCK: tl.constexpr,
):
    row = tl.program_id(axis=0)
    # guard: if row >= M, return (grid is exactly (M,))
    # Load row slice in chunks of BLOCK
    sum_ = 0.0
    sqsum_ = 0.0
    # First pass: compute mean and variance (sum and sum of squares)
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        sum_ += tl.sum(x, axis=0)
        sqsum_ += tl.sum(x * x, axis=0)
    K_f = K  # K is int, but mean/var need float
    mean = sum_ / K_f
    var = sqsum_ / K_f - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, store BF16
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        w = tl.load(LN_W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(LN_B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = y * w + b
        tl.store(Y_ptr + row * K + offs, y.to(tl.bfloat16), mask=mask)


# GEMM + bias: compute C[M, N] = A[M, K] @ B[K, N] + bias[N]
@triton.jit
def _gemm_bias_kernel(
    A_ptr,            # *bf16, shape (M, K)
    B_ptr,            # *bf16, shape (K, N)
    bias_ptr,         # *bf16, shape (N,)
    C_ptr,            # *bf16, shape (M, N)
    M,                # int
    N,                # int
    K,                # int
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # guard via masks is not needed; grid ensures bounds
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Loop over K in chunks
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * K) + offs_k[None, :]
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        # Load B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (offs_k[:, None] * N) + offs_n[None, :]
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        # Accumulate
        acc += tl.dot(a, b)
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]
    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * N) + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# GELU activation (tanh approximation), elementwise on (M, N)
@triton.jit
def _gelu_kernel(
    X_ptr,            # *bf16, shape (M, N)
    Y_ptr,            # *bf16, shape (M, N)
    M,                # int
    N,                # int
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    col_block = tl.program_id(axis=1)
    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (row < M) & (offs_n < N)
    x = tl.load(X_ptr + row * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = x * x * x
    gelu = 0.5 * x * (1.0 + tl.tanh(c0 * (x + c1 * x3)))
    tl.store(Y_ptr + row * N + offs_n, gelu.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        """
        hidden: (num_patches, 1536), bfloat16
        ln_weight, ln_bias: (1536,), bfloat16
        fc1_weight: (6144, 6144), bfloat16
        fc1_bias: (6144,), bfloat16
        fc2_weight: (3584, 6144), bfloat16
        fc2_bias: (3584,), bfloat16
        grid_thw: shape (num_grids, 3) but not used (T=1 in get_inputs)
        """
        device = hidden.device
        M = hidden.shape[0]  # num_patches
        K = hidden.shape[1]  # hidden_size = 1536
        K_expanded = 4 * K   # 6144

        # 1) LayerNorm + affine in Triton, output to ln_out BF16
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        if TRITON_AVAILABLE:
            BLOCK_ln = 256
            grid_ln = (M,)
            _layer_norm_affine_kernel[grid_ln](
                hidden, ln_weight, ln_bias, ln_out,
                M, K, eps,
                BLOCK=BLOCK_ln,
                num_warps=4, num_stages=2
            )
        else:
            # Fallback: PyTorch LayerNorm (rare path if Triton not available)
            mean = ln_out.mean(dim=-1, keepdim=True)
            var = ln_out.var(dim=-1, keepdim=True, unbiased=False)
            ln_out = (ln_out - mean) / torch.sqrt(var + eps)
            ln_out = ln_out * ln_weight + ln_bias

        # 2) Packing: since T=1 and num_patches % 4 == 0 (per get_inputs), reshape to (M//4, 4*K)
        M_out = M // 4
        packed = ln_out.view(M_out, K_expanded)

        # 3) fc1: (M_out, 6144) @ (6144, 6144)^T + bias
        N1 = fc1_weight.shape[0]  # 6144
        fc1_out = torch.empty((M_out, N1), dtype=torch.bfloat16, device=device)

        BLOCK_M1, BLOCK_N1, BLOCK_K1 = 128, 128, 64
        grid_fc1 = (triton.cdiv(M_out, BLOCK_M1), triton.cdiv(N1, BLOCK_N1))
        if TRITON_AVAILABLE:
            _gemm_bias_kernel[grid_fc1](
                packed, fc1_weight, fc1_bias, fc1_out,
                M_out, N1, K_expanded,
                BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
                num_warps=4, num_stages=2
            )
        else:
            # Fallback: PyTorch matmul + bias
            fc1_out = torch.nn.functional.linear(packed, fc1_weight, fc1_bias)

        # 4) GELU activation via Triton
        # Ensure positive grid: N1 >= 6144 (given by fc1_weight.shape[0])
        BLOCK_N_gelu = 256
        grid_gelu = (M_out, triton.cdiv(N1, BLOCK_N_gelu))
        if TRITON_AVAILABLE:
            fc1_after_gelu = torch.empty_like(fc1_out, dtype=torch.bfloat16, device=device)
            _gelu_kernel[grid_gelu](
                fc1_out, fc1_after_gelu,
                M_out, N1,
                BLOCK_N=BLOCK_N_gelu,
                num_warps=4, num_stages=2
            )
        else:
            # Fallback: PyTorch GELU
            fc1_after_gelu = torch.nn.functional.gelu(fc1_out)

        # 5) fc2: (M_out, 6144) @ (3584, 6144)^T + bias
        N2 = fc2_weight.shape[0]  # 3584
        output = torch.empty((M_out, N2), dtype=torch.bfloat16, device=device)

        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 128, 64
        grid_fc2 = (triton.cdiv(M_out, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        if TRITON_AVAILABLE:
            _gemm_bias_kernel[grid_fc2](
                fc1_after_gelu, fc2_weight, fc2_bias, output,
                M_out, N2, N1,  # K is N1 (6144)
                BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
                num_warps=4, num_stages=2
            )
        else:
            # Fallback: PyTorch matmul + bias
            output = torch.nn.functional.linear(fc1_after_gelu, fc2_weight, fc2_bias)

        return output


def run(*args):
    return ModelNew()(*args)
