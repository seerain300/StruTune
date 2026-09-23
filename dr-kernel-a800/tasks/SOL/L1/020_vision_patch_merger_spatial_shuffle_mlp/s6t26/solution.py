import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_affine_kernel(
    x_ptr,         # *bf16, input [M, K]
    weight_ptr,    # *bf16, [K]
    bias_ptr,      # *bf16, [K]
    y_ptr,         # *bf16, output [M, K]
    M, K, eps,     # int32, float32
    BLOCK: tl.constexpr,
):
    # One program per row
    row = tl.program_id(axis=0)
    if row >= M:
        return
    # First pass: compute mean and variance in fp32
    acc1 = tl.zeros((), dtype=tl.float32)
    acc2 = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        acc1 += tl.sum(x, axis=0)
        acc2 += tl.sum(x * x, axis=0)
    mean = acc1 / K
    var = acc2 / K - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(y_ptr + row * K + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _gemm_bias_kernel(
    A_ptr,         # *bf16, input [M, K]
    W_ptr,         # *bf16, weight [N, K] (row-major: N is fast dim)
    B_ptr,         # *bf16, bias [N]
    C_ptr,         # *bf16, output [M, N]
    M, N, K,       # int32 sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m0 * K + (tl.arange(0, BLOCK_M)[:, None] * K + k[None, :])
        a_mask = (m0 + tl.arange(0, BLOCK_M))[:, None] < M and (k0 + tl.arange(0, BLOCK_K))[None, :] < K
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        # Load W tile as [BLOCK_K, BLOCK_N], W is [N, K] row-major
        w_ptrs = W_ptr + n0 * K + (k[:, None] * K + (tl.arange(0, BLOCK_N)[None, :] * K))
        w_mask = (k0 + tl.arange(0, BLOCK_K))[:, None] < K and (n0 + tl.arange(0, BLOCK_N))[None, :] < N
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # acc += A_tile @ W_tile
        acc += tl.dot(a, w)

    # Add bias
    bias = tl.load(B_ptr + n0 + tl.arange(0, BLOCK_N), mask=(n0 + tl.arange(0, BLOCK_N)) < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store result as bf16
    c_ptrs = C_ptr + m0 * N + (tl.arange(0, BLOCK_M)[:, None] * N + (n0 + tl.arange(0, BLOCK_N))[None, :])
    c_mask = (m0 + tl.arange(0, BLOCK_M))[:, None] < M and (n0 + tl.arange(0, BLOCK_N))[None, :] < N
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def _gelu_kernel(
    x_ptr,         # *bf16, input [M, N]
    y_ptr,         # *bf16, output [M, N]
    M, N,          # int32
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    m = pid_m
    n0 = pid_n * BLOCK_N
    for i in range(0, N, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        mask = (m < M) and (cols < N)
        x = tl.load(x_ptr + m * N + cols, mask=mask, other=0.0).to(tl.float32)
        # GELU tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        c = 0.7978845608028654  # sqrt(2/pi)
        x3 = x * x * x
        gelu = 0.5 * x * (1.0 + tl.tanh(c * (x + 0.044715 * x3)))
        tl.store(y_ptr + m * N + cols, gelu.to(tl.bfloat16), mask=mask)


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
        """
        Triton-only implementation of the original forward:
        1) LayerNorm + affine over hidden dimension
        2) Spatial packing (view): (num_patches//4, 4*hidden_size)
        3) fc1: GEMM + bias
        4) GELU
        5) fc2: GEMM + bias
        """
        device = hidden.device
        # Ensure tensors are on CUDA for Triton
        assert hidden.is_cuda, "Input tensor must be on CUDA device for Triton kernels."
        # LayerNorm + affine
        M = hidden.shape[0]  # num_patches
        K = hidden.shape[1]  # hidden_size = 1536
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        BLOCK_ln = 256
        grid_ln = (M,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M, K, eps,
            BLOCK=BLOCK_ln,
            num_warps=4, num_stages=2
        )

        # Step 2: Spatial packing by view (T=1 in provided inputs)
        assert M % 4 == 0, "num_patches must be divisible by 4 for 2x2 packing"
        M_out = M // 4
        K_expanded = 4 * K  # 6144
        # View packing: reshape to [M_out, 4, K] -> transpose -> [M_out, 4*K]
        packed = ln_out.view(M_out, 4, K).transpose(1, 2).reshape(M_out, K_expanded)

        # Step 3: fc1 (6144 -> 6144) GEMM + bias
        M_merged = M_out  # num_merged_patches (from axes)
        N1 = fc1_weight.shape[0]  # 6144
        K1 = packed.shape[1]      # 4*K = 6144
        fc1_out = torch.empty((M_merged, N1), dtype=torch.bfloat16, device=device)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        grid_fc1 = (triton.cdiv(M_merged, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        _gemm_bias_kernel[grid_fc1](
            packed, fc1_weight, fc1_bias, fc1_out,
            M_merged, N1, K1,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Step 4: GELU activation
        K_gelu = fc1_out.shape[1]  # 6144
        fc1_after_gelu = torch.empty((M_merged, K_gelu), dtype=torch.bfloat16, device=device)
        BLOCK_N_gelu = 256
        grid_gelu = (M_merged, triton.cdiv(K_gelu, BLOCK_N_gelu))
        _gelu_kernel[grid_gelu](
            fc1_out, fc1_after_gelu,
            M_merged, K_gelu,
            BLOCK_N=BLOCK_N_gelu,
            num_warps=4, num_stages=2
        )

        # Step 5: fc2 (6144 -> 3584) GEMM + bias
        M_final = M_merged  # num_merged_patches
        N2 = fc2_weight.shape[0]  # 3584
        K2 = fc1_after_gelu.shape[1]  # 6144
        output = torch.empty((M_final, N2), dtype=torch.bfloat16, device=device)
        BLOCK_M2 = 128
        BLOCK_N2 = 64
        BLOCK_K2 = 64
        grid_fc2 = (triton.cdiv(M_final, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        _gemm_bias_kernel[grid_fc2](
            fc1_after_gelu, fc2_weight, fc2_bias, output,
            M_final, N2, K2,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
