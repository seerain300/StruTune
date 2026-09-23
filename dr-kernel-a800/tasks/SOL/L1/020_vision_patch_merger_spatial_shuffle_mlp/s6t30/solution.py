import math
import torch
import triton
import triton.language as tl

# Kernel 1: LayerNorm over last dim + affine
# Input: X[M, K], ln_weight[K], ln_bias[K], Output: Y[M, K]
@triton.jit
def _layer_norm_affine_kernel(
    X_ptr, ln_weight_ptr, ln_bias_ptr, Y_ptr,
    M: tl.int32, K: tl.int32, eps: tl.float32,
    BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    # bounds check
    if row >= M:
        return
    # First pass: compute mean and variance in FP32
    sum_ = 0.0
    sumsq_ = 0.0
    for off in range(0, K, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < K
        x = tl.load(X_ptr + row * K + cols, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sum_ += tl.sum(x32, axis=0)
        sumsq_ += tl.sum(x32 * x32, axis=0)
    mean = sum_ / K
    var = sumsq_ / K - mean * mean
    inv_std = tl.math.rsqrt(var + eps)

    # Second pass: normalize and apply affine
    for off in range(0, K, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < K
        x = tl.load(X_ptr + row * K + cols, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        y32 = (x32 - mean) * inv_std
        w = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y32 = y32 * w + b
        y = y32.to(tl.bfloat16)
        tl.store(Y_ptr + row * K + cols, y, mask=mask)


# Kernel 2: Pack normalized hidden into expanded feature dimension
# X is normalized hidden of shape (M, K=1536), write Y of shape (M_out, 4*K=6144).
# Each output row r corresponds to fused 2x2: write 4 contiguous segments of K.
@triton.jit
def _pack_2x2_to_expanded_kernel(
    X_ptr, Y_ptr,
    M: tl.int32,  # num_patches (from original hidden)
    K: tl.int32,  # hidden_size (1536)
    M_out: tl.int32,  # num_patches // 4
    BLOCK: tl.constexpr
):
    r = tl.program_id(0)  # output row index
    if r >= M_out:
        return
    # Which 2x2 block does r come from? Because M_out = M // 4, r -> block index (b, c) in [0,1]x[0,1]
    b = r // 4
    c = r % 4
    # The original indices in X: row_idx = b*2 + c // 2, col_idx = c % 2
    row_idx = b * 2 + (c // 2)
    col_idx = c % 2

    base = row_idx * K + col_idx * K
    for off in range(0, K, BLOCK):
        seg = off + tl.arange(0, BLOCK)
        mask = seg < K
        x = tl.load(X_ptr + base + seg, mask=mask, other=0.0).to(tl.bfloat16)
        # Y has 4*K columns; segment placement
        start = c * K
        tl.store(Y_ptr + r * (4 * K) + start + seg, x, mask=mask)


# Kernel 3: Triton GEMM + bias
# A[M, K], B[K, N], bias[N], Out[M, N]
@triton.jit
def _gemm_bias_kernel(
    A_ptr, B_ptr, bias_ptr, Out_ptr,
    M: tl.int32, K: tl.int32, N: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * K) + offs_k[None, :]
        b_ptrs = B_ptr + (offs_k[:, None] * N) + offs_n[None, :]
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    out = acc.to(tl.bfloat16)
    tl.store(Out_ptr + (offs_m[:, None] * N) + offs_n[None, :],
             out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Kernel 4: GELU elementwise (approximate)
@triton.jit
def _gelu_kernel(
    X_ptr, Y_ptr,
    M: tl.int32, N: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(X_ptr + (offs_m[:, None] * N) + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = x * x * x
    inner = c0 * (x + c1 * x3)
    gelu = 0.5 * x * (1.0 + tl.tanh(inner))
    y = gelu.to(tl.bfloat16)
    tl.store(Y_ptr + (offs_m[:, None] * N) + offs_n[None, :], y, mask=mask)


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
        Triton-only implementation:
        1) LayerNorm + affine (Triton)
        2) Pack normalized hidden into 4*hidden_size features (Triton)
        3) fc1: GEMM + bias (Triton)
        4) GELU (Triton)
        5) fc2: GEMM + bias (Triton)
        """
        device = hidden.device
        M = hidden.shape[0]  # num_patches
        K = hidden.shape[1]  # hidden_size (1536)
        # 1) LayerNorm + affine
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        BLOCK_ln = 256
        grid_ln = (M,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M, K, float(eps),
            BLOCK=BLOCK_ln,
            num_warps=4, num_stages=2
        )

        # 2) Pack 2x2 -> expanded
        M_out = M // 4  # invariant from get_inputs
        K_expanded = 4 * K  # 6144
        packed = torch.empty((M_out, K_expanded), dtype=torch.bfloat16, device=device)
        BLOCK_pack = 256
        grid_pack = (M_out,)
        _pack_2x2_to_expanded_kernel[grid_pack](
            ln_out, packed,
            M, K, M_out,
            BLOCK=BLOCK_pack,
            num_warps=4, num_stages=2
        )

        # 3) fc1: (M_out, 6144) @ (6144, 6144)^T + bias, output shape (M_out, 6144)
        M_merged = M_out  # num_merged_patches (from axes)
        N1 = fc1_weight.shape[0]  # 6144
        K1 = packed.shape[1]  # 4*K = 6144
        fc1_out = torch.empty((M_merged, N1), dtype=torch.bfloat16, device=device)
        BLOCK_M1, BLOCK_N1, BLOCK_K1 = 128, 128, 64
        grid_fc1 = (triton.cdiv(M_merged, BLOCK_M1), triton.cdiv(N1, BLOCK_N1))
        _gemm_bias_kernel[grid_fc1](
            packed, fc1_weight, fc1_bias, fc1_out,
            M_merged, K1, N1,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
            num_warps=4, num_stages=2
        )

        # 4) GELU activation
        K_after_gelu = fc1_out.shape[1]  # 6144
        fc1_after_gelu = torch.empty((M_merged, K_after_gelu), dtype=torch.bfloat16, device=device)
        BLOCK_M_gelu, BLOCK_N_gelu = 128, 256
        grid_gelu = (triton.cdiv(M_merged, BLOCK_M_gelu), triton.cdiv(K_after_gelu, BLOCK_N_gelu))
        _gelu_kernel[grid_gelu](
            fc1_out, fc1_after_gelu,
            M_merged, K_after_gelu,
            BLOCK_M=BLOCK_M_gelu, BLOCK_N=BLOCK_N_gelu,
            num_warps=4, num_stages=2
        )

        # 5) fc2: (M_merged, 6144) @ (3584, 6144)^T + bias, output shape (M_merged, 3584)
        M_out_fc2 = M_merged
        N2 = fc2_weight.shape[0]  # 3584
        K2 = fc1_after_gelu.shape[1]  # 6144
        out = torch.empty((M_out_fc2, N2), dtype=torch.bfloat16, device=device)
        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 128, 64
        grid_fc2 = (triton.cdiv(M_out_fc2, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        _gemm_bias_kernel[grid_fc2](
            fc1_after_gelu, fc2_weight, fc2_bias, out,
            M_out_fc2, K2, N2,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
