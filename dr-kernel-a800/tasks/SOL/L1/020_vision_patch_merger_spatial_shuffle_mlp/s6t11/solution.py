import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_affine_kernel(
    X_ptr,      # *bf16, input of shape [M, K], row-major
    W_ptr,      # *bf16, ln_weight of shape [K]
    B_ptr,      # *bf16, ln_bias of shape [K]
    Out_ptr,    # *bf16, output of shape [M, K]
    M: tl.constexpr,   # number of rows (patches)
    K: tl.constexpr,   # feature size (hidden_size)
    eps: tl.constexpr, # epsilon for var + eps
    BLOCK: tl.constexpr,  # tile along K
):
    # one program per row
    row = tl.program_id(0)
    row_base = row * K
    # First pass: compute mean/var in FP32
    sum_val = 0.0
    sum_sq = 0.0
    for col in range(0, K, BLOCK):
        offs = col + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    n = K
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for col in range(0, K, BLOCK):
        offs = col + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(Out_ptr + row_base + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _gemm_bias_kernel(
    A_ptr,           # *bf16, input matrix, shape [M, K]
    B_ptr,           # *bf16, weight matrix, shape [N, K] (compute A @ B^T)
    Bias_ptr,        # *bf16, bias, shape [N]
    C_ptr,           # *bf16, output matrix, shape [M, N]
    M: tl.constexpr, # number of rows in A (and C)
    N: tl.constexpr, # number of columns in C (and number of rows in Bias)
    K: tl.constexpr, # feature dimension for A
    BLOCK_M: tl.constexpr,  # e.g., 128
    BLOCK_N: tl.constexpr,  # e.g., 128
    BLOCK_K: tl.constexpr,  # e.g., 64
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m0 * K + (tl.arange(0, BLOCK_M)[:, None]) * K + k_ids[None, :]
        a = tl.load(a_ptrs, mask=(tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & (k_ids[None, :] < K), other=0.0).to(tl.float32)
        # Load B tile as W[n, k] with B_ptr[n*K + k]
        b_ptrs = B_ptr + n0 * K + k_ids[None, :] * N + tl.arange(0, BLOCK_N)[:, None]
        b = tl.load(b_ptrs, mask=(tl.arange(0, BLOCK_N)[:, None] < BLOCK_N) & (k_ids[None, :] < K), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)  # (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)

    # Add bias
    bias = tl.load(Bias_ptr + n0 + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < BLOCK_N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store
    c_ptrs = C_ptr + m0 * N + (tl.arange(0, BLOCK_M)[:, None]) * N + (n0 + tl.arange(0, BLOCK_N)[None, :])
    mask_c = (tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & (tl.arange(0, BLOCK_N)[None, :] < BLOCK_N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_c)


@triton.jit
def _gelu_kernel(
    In_ptr,     # *bf16, input of shape [M, N]
    Out_ptr,    # *bf16, output of shape [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m
    n0 = pid_n * BLOCK_N
    x = tl.load(In_ptr + m * N + n0 + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < N, other=0.0).to(tl.float32)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    e = tl.exp(-x * x)
    t = tl.tanh(c * (x + 0.044715 * x3))
    y = 0.5 * x * (1.0 + t)
    tl.store(Out_ptr + m * N + n0 + tl.arange(0, BLOCK_N), y.to(tl.bfloat16), mask=tl.arange(0, BLOCK_N) < N)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        # Device and shapes
        device = hidden.device
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]  # 1536

        # 1) LayerNorm + affine (pre-shuffle)
        ln_out = torch.empty((num_patches, hidden_size), dtype=torch.bfloat16, device=device)
        BLOCK_ln = 256
        grid_ln = (num_patches,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M=num_patches, K=hidden_size, eps=eps,
            BLOCK=BLOCK_ln, num_warps=4, num_stages=2
        )

        # 2) Pack 2x2 features to expanded feature dimension via view (num_patches % 4 == 0)
        num_merged_patches = num_patches // 4
        K = hidden_size
        K_expanded = 4 * K
        # Reshape ln_out to (num_patches//4, 4*K) — this is equivalent to packing 2x2 since T=1 and num_patches % 4 == 0
        packed = ln_out.view(num_merged_patches, K_expanded)

        # 3) FC1: (M_out, 6144) @ (6144, 6144)^T + bias
        K1 = packed.shape[1]  # 4*K = 6144
        N1 = fc1_weight.shape[0]  # 6144
        fc1_out = torch.empty((num_merged_patches, N1), dtype=torch.bfloat16, device=device)
        BLOCK_M1, BLOCK_N1, BLOCK_K1 = 128, 128, 64
        grid_fc1 = (triton.cdiv(num_merged_patches, BLOCK_M1), triton.cdiv(N1, BLOCK_N1))
        _gemm_bias_kernel[grid_fc1](
            packed, fc1_weight, fc1_bias, fc1_out,
            M=num_merged_patches, N=N1, K=K1,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
            num_warps=4, num_stages=2
        )

        # 4) GELU activation
        K_after_gelu = fc1_out.shape[1]  # 6144
        fc1_after_gelu = torch.empty((num_merged_patches, K_after_gelu), dtype=torch.bfloat16, device=device)
        BLOCK_N_gelu = 256
        grid_gelu = (num_merged_patches, triton.cdiv(K_after_gelu, BLOCK_N_gelu))
        _gelu_kernel[grid_gelu](
            fc1_out, fc1_after_gelu,
            M=num_merged_patches, N=K_after_gelu, BLOCK_N=BLOCK_N_gelu,
            num_warps=4, num_stages=2
        )

        # 5) FC2: fc1_after_gelu @ fc2_weight^T + fc2_bias
        K2_in = K_after_gelu  # 6144
        N2 = fc2_weight.shape[0]  # 3584
        out = torch.empty((num_merged_patches, N2), dtype=torch.bfloat16, device=device)
        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 64, 64
        grid_fc2 = (triton.cdiv(num_merged_patches, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        _gemm_bias_kernel[grid_fc2](
            fc1_after_gelu, fc2_weight, fc2_bias, out,
            M=num_merged_patches, N=N2, K=K2_in,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
