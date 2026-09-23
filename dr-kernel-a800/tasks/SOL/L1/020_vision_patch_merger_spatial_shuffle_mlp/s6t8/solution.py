import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_affine_kernel(
    X_ptr,      # *bf16, input of shape [M, K], where M=num_patches, K=hidden_size
    W_ptr,      # *bf16, ln_weight of shape [K]
    B_ptr,      # *bf16, ln_bias of shape [K]
    Out_ptr,    # *bf16, output of shape [M, K]
    M: tl.constexpr,   # number of rows (patches)
    K: tl.constexpr,   # hidden size (1536)
    BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    if row >= M:
        return
    sum_val = 0.0
    sum_sq = 0.0
    # Compute mean and variance in FP32
    for col in range(0, K, BLOCK):
        offs = col + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    n = K
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    inv_std = 1.0 / tl.sqrt(var + 1e-6)
    # Normalize and apply affine
    for col in range(0, K, BLOCK):
        offs = col + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(Out_ptr + row * K + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _pack_2x2_to_expanded_kernel(
    In_ptr,     # *bf16, input of shape [M, K] (normalized hidden)
    Out_ptr,    # *bf16, output of shape [M_out, 4*K] where M_out = M//4
    M_in: tl.constexpr,    # num_patches
    hidden_size: tl.constexpr,  # K
    BLOCK: tl.constexpr
):
    # 2D grid: (M_out, 4) for segments
    r = tl.program_id(0)
    seg = tl.program_id(1)
    if r >= (M_in // 4):
        return
    # Each output row corresponds to one 2x2 patch; pack 4 segments
    # Base row index in original input is r * 4
    base = r * 4
    K = hidden_size
    # Copy from input row 'base + (seg // 2) * 1 + (seg % 2)'
    kh = seg // 2
    kw = seg % 2
    src_row = base + kh * (M_in // 4) + kw
    offs = tl.arange(0, BLOCK)
    dst_col = seg * K + offs
    src_col = offs
    mask = dst_col < 4 * K
    x = tl.load(In_ptr + src_row * K + src_col, mask=mask, other=0.0).to(tl.bfloat16)
    tl.store(Out_ptr + r * (4 * K) + dst_col, x, mask=mask)


@triton.jit
def _gemm_bias_kernel(
    A_ptr,           # *bf16, input matrix A of shape [M, K] (e.g., packed)
    B_ptr,           # *bf16, weight matrix of shape [N, K] (we compute A @ B^T)
    Bias_ptr,        # *bf16, bias vector of shape [N]
    C_ptr,           # *bf16, output matrix of shape [M, N]
    M: tl.constexpr, # number of rows in A (and C)
    N: tl.constexpr, # number of columns in C (and length of Bias)
    K: tl.constexpr, # feature dimension
    BLOCK_M: tl.constexpr,  # e.g., 128
    BLOCK_N: tl.constexpr,  # e.g., 128
    BLOCK_K: tl.constexpr   # e.g., 64
):
    # 2D launch grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m0 * K + (tl.arange(0, BLOCK_M)[:, None]) * K + k_ids[None, :]
        a = tl.load(a_ptrs, mask=(tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & (k_ids[None, :] < K), other=0.0).to(tl.float32)
        # B tile as B[n, k]: pointer is B_ptr + n * K + k
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
    # tanh-based GELU approximation
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    t = c * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.tanh(t))
    tl.store(Out_ptr + m * N + n0 + tl.arange(0, BLOCK_N), gelu.to(tl.bfloat16), mask=tl.arange(0, BLOCK_N) < N)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args: hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps
        hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps = args
        device = hidden.device

        # 1) LayerNorm + affine
        M = hidden.shape[0]  # num_patches
        K = hidden.shape[1]  # hidden size (1536)
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        BLOCK_LN = 256
        grid_ln = (M,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M=M, K=K, BLOCK=BLOCK_LN,
            num_warps=4, num_stages=2
        )

        # 2) Pack 2x2 to expanded feature dimension: M_out = M // 4 (T=1 assumption)
        M_out = M // 4
        packed = torch.empty((M_out, 4 * K), dtype=torch.bfloat16, device=device)
        if M_out > 0:
            BLOCK_PACK = 256
            grid_pack = (M_out, 4)
            _pack_2x2_to_expanded_kernel[grid_pack](
                ln_out, packed,
                M_in=M, hidden_size=K, BLOCK=BLOCK_PACK,
                num_warps=4, num_stages=2
            )
        else:
            # If num_patches not divisible by 4 (shouldn't happen for T=1), guard.
            packed = torch.empty((1, 1), dtype=torch.bfloat16, device=device)

        # 3) FC1: packed @ fc1_weight^T + fc1_bias (Triton GEMM)
        # fc1_weight shape: [6144, 6144], fc1_bias: [6144]
        fc1_out = torch.empty((packed.shape[0], fc1_weight.shape[0]), dtype=torch.bfloat16, device=device)
        BLOCK_M1, BLOCK_N1, BLOCK_K1 = 128, 128, 64
        grid_fc1 = (triton.cdiv(packed.shape[0], BLOCK_M1), triton.cdiv(fc1_weight.shape[0], BLOCK_N1))
        _gemm_bias_kernel[grid_fc1](
            packed, fc1_weight, fc1_bias, fc1_out,
            M=packed.shape[0], N=fc1_weight.shape[0], K=fc1_weight.shape[1],
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
            num_warps=4, num_stages=2
        )

        # 4) GELU activation (Triton)
        K_after_gelu = fc1_out.shape[1]  # 6144
        fc1_after_gelu = torch.empty_like(fc1_out, dtype=torch.bfloat16, device=device)
        BLOCK_N_gelu = 256
        grid_gelu = (fc1_out.shape[0], triton.cdiv(K_after_gelu, BLOCK_N_gelu))
        _gelu_kernel[grid_gelu](
            fc1_out, fc1_after_gelu,
            M=fc1_out.shape[0], N=K_after_gelu, BLOCK_N=BLOCK_N_gelu,
            num_warps=4, num_stages=2
        )

        # 5) FC2: fc1_after_gelu @ fc2_weight^T + fc2_bias (Triton GEMM)
        # fc2_weight shape: [3584, 6144], fc2_bias: [3584]
        output = torch.empty((fc1_after_gelu.shape[0], fc2_weight.shape[0]), dtype=torch.bfloat16, device=device)
        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 64, 64
        grid_fc2 = (triton.cdiv(fc1_after_gelu.shape[0], BLOCK_M2), triton.cdiv(fc2_weight.shape[0], BLOCK_N2))
        _gemm_bias_kernel[grid_fc2](
            fc1_after_gelu, fc2_weight, fc2_bias, output,
            M=fc1_after_gelu.shape[0], N=fc2_weight.shape[0], K=fc2_weight.shape[1],
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
