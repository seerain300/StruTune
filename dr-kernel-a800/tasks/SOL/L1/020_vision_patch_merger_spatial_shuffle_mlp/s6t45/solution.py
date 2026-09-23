import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_affine_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, K, eps,
    BLOCK: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)
    if row >= M:
        return

    # First pass: compute mean and variance over K
    sum_val = 0.0
    sum_sq = 0.0
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / K
    var = sum_sq / K - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        norm = (x - mean) * inv_std
        w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = norm * w + b
        tl.store(Y_ptr + row * K + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _pack_2x2_to_expanded_kernel(
    X_ptr, Y_ptr,
    M_in, K,  # X is (M_in, K), K=1536
    # Each output row corresponds to a fused 2x2; M_out=M_in//4
):
    # Grid = (M_out,)
    row_out = tl.program_id(0)
    if row_out < 0:
        return

    # Map output row to the corresponding input row in X
    # Since T=1 and num_patches % 4 == 0 (assumed by get_inputs),
    # we can directly compute the base input row:
    # For output index r, input base row is r * 4.
    base_in = row_out * 4

    # Copy features into four contiguous segments: [0:K), [K:2K), [2K:3K), [3K:4K)
    for s in range(4):
        start = s * K
        end = (s + 1) * K
        src = base_in * K + tl.arange(0, K)
        dst = row_out * (4 * K) + tl.arange(0, K) + start
        mask = (tl.arange(0, K) < K)  # always true
        vals = tl.load(X_ptr + src, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y_ptr + dst, vals.to(tl.bfloat16), mask=mask)


@triton.jit
def _gemm_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * K) + offs_k[None, :]
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Load B tile: B is [K, N], so element b[k, n] -> [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (offs_n[None, :] * K) + offs_k[:, None]
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store as BF16
    c_ptrs = C_ptr + (offs_m[:, None] * N) + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _gelu_tanh_kernel(X_ptr, Y_ptr, M, N, BLOCK_N: tl.constexpr):
    # 2D grid over rows and columns
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    offs = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = x * x * x
    inner = c0 * (x + c1 * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        """
        hidden: [num_patches, 1536], bfloat16
        ln_weight, ln_bias: [1536], bfloat16
        fc1_weight: [6144, 6144], bfloat16
        fc1_bias: [6144], bfloat16
        fc2_weight: [3584, 6144], bfloat16
        fc2_bias: [3584], bfloat16
        grid_thw is unused (original run didn't depend on it), but we keep signature.
        """
        device = hidden.device
        M = hidden.shape[0]
        K = hidden.shape[1]  # 1536

        # 1) LayerNorm + affine (Triton)
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        BLOCK_ln = 256
        grid_ln = (M,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M, K, eps,
            BLOCK=BLOCK_ln, num_warps=4, num_stages=2
        )

        # 2) Pack 2x2 features to expanded features: (M//4, 4*K)
        # get_inputs guarantees num_patches % 4 == 0
        M_out = M // 4
        K_expanded = 4 * K  # 6144
        packed = torch.empty((M_out, K_expanded), dtype=torch.bfloat16, device=device)
        grid_pack = (M_out,)
        _pack_2x2_to_expanded_kernel[grid_pack](
            ln_out, packed,
            M_in=M, K=K,
            num_warps=4, num_stages=2
        )

        # 3) FC1: (M_out, 6144) @ (6144, 6144) + bias  -> (M_out, 6144)
        K1 = packed.shape[1]  # 6144
        N1 = fc1_weight.shape[0]  # 6144
        fc1_out = torch.empty((M_out, N1), dtype=torch.bfloat16, device=device)
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        grid_fc1 = (triton.cdiv(M_out, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        _gemm_bias_kernel[grid_fc1](
            packed, fc1_weight, fc1_bias, fc1_out,
            M_out, N1, K1,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 4) GELU activation (Triton)
        K_after_gelu = fc1_out.shape[1]  # 6144
        fc1_after_gelu = torch.empty_like(fc1_out, dtype=torch.bfloat16, device=device)
        BLOCK_N_gelu = 256
        grid_gelu = (M_out, triton.cdiv(K_after_gelu, BLOCK_N_gelu))
        _gelu_tanh_kernel[grid_gelu](
            fc1_out, fc1_after_gelu, M_out, K_after_gelu,
            BLOCK_N=BLOCK_N_gelu, num_warps=4, num_stages=2
        )

        # 5) FC2: (M_out, 6144) @ (3584, 6144) + bias  -> (M_out, 3584)
        N2 = fc2_weight.shape[0]  # 3584
        output = torch.empty((M_out, N2), dtype=torch.bfloat16, device=device)
        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 128, 64
        grid_fc2 = (triton.cdiv(M_out, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        _gemm_bias_kernel[grid_fc2](
            fc1_after_gelu, fc2_weight, fc2_bias, output,
            M_out, N2, K_after_gelu,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
