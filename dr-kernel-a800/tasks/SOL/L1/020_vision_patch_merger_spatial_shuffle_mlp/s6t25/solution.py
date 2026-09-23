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
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # A_tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * K) + offs_k[None, :]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # W_tile: [BLOCK_K, BLOCK_N], W is [N, K] row-major
        w_ptrs = W_ptr + (offs_n[None, :] * K) + offs_k[:, None]
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, w)

    # Add bias: broadcast over rows
    b = tl.load(B_ptr + n0 + tl.arange(0, BLOCK_N), mask=(n0 + tl.arange(0, BLOCK_N)) < N, other=0.0).to(tl.float32)
    acc += b[None, :]

    # Store to C as bf16
    c_ptrs = C_ptr + (offs_m[:, None] * N) + (n0 + tl.arange(0, BLOCK_N))[None, :]
    c_mask = (offs_m[:, None] < M) & ((n0 + tl.arange(0, BLOCK_N))[None, :] < N)
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
    offs_n = n0 + tl.arange(0, BLOCK_N)

    mask_m = m < M
    mask_n = offs_n < N
    mask = mask_m & mask_n

    x = tl.load(x_ptr + m * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    # GELU tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + c * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(y_ptr + m * N + offs_n, y.to(tl.bfloat16), mask=mask)


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
        # Ensure device and dtypes
        device = hidden.device
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()
        fc1_weight = fc1_weight.contiguous()
        fc1_bias = fc1_bias.contiguous()
        fc2_weight = fc2_weight.contiguous()
        fc2_bias = fc2_bias.contiguous()

        M = hidden.shape[0]  # num_patches
        K = hidden.shape[1]  # 1536

        # Step 1: LayerNorm + affine
        ln_out = torch.empty((M, K), dtype=torch.bfloat16, device=device)
        BLOCK_ln = 128
        grid_ln = (M,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M, K, eps,
            BLOCK=BLOCK_ln,
            num_warps=4, num_stages=2
        )

        # Step 2: Packing: view (T=1), each output row is 2x2 fused -> 4*K features
        assert (M % 4) == 0, "num_patches must be divisible by 4 for 2x2 merge"
        ln_view = ln_out.view(M // 4, 4, K).transpose(1, 2).reshape(M // 4, 4 * K)
        A = ln_view  # metadata-only view

        # Step 3: FC1 GEMM + bias: (M//4, 4*K) @ (4*K, 4*K)^T + bias -> (M//4, 4*K)
        M_out = M // 4  # num_merged_patches from get_inputs
        K1 = 4 * K      # 6144
        N1 = fc1_weight.shape[0]  # 6144
        fc1_out = torch.empty((M_out, N1), dtype=torch.bfloat16, device=device)

        BLOCK_M1, BLOCK_N1, BLOCK_K1 = 128, 128, 64
        grid_fc1 = (triton.cdiv(M_out, BLOCK_M1), triton.cdiv(N1, BLOCK_N1))
        _gemm_bias_kernel[grid_fc1](
            A, fc1_weight, fc1_bias, fc1_out,
            M_out, N1, K1,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
            num_warps=4, num_stages=2
        )

        # Step 4: GELU activation
        K_after = fc1_out.shape[1]  # 6144
        fc1_gelu = torch.empty((M_out, K_after), dtype=torch.bfloat16, device=device)
        BLOCK_N_gelu = 256
        grid_gelu = (M_out, triton.cdiv(K_after, BLOCK_N_gelu))
        _gelu_kernel[grid_gelu](
            fc1_out, fc1_gelu,
            M_out, K_after,
            BLOCK_N=BLOCK_N_gelu,
            num_warps=4, num_stages=2
        )

        # Step 5: FC2 GEMM + bias: (M_out, N1) @ (N2, N1)^T + bias -> (M_out, N2)
        N2 = fc2_weight.shape[0]  # 3584
        output = torch.empty((M_out, N2), dtype=torch.bfloat16, device=device)

        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 128, 64
        grid_fc2 = (triton.cdiv(M_out, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        _gemm_bias_kernel[grid_fc2](
            fc1_gelu, fc2_weight, fc2_bias, output,
            M_out, N2, N1,  # K for fc2 is N1 (6144)
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
