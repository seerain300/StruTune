import math
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_affine_kernel(
    hidden_ptr,       # *bf16, [num_patches, hidden_size]
    ln_weight_ptr,    # *bf16, [hidden_size]
    ln_bias_ptr,      # *bf16, [hidden_size]
    out_ptr,          # *bf16, [num_patches, hidden_size]
    num_patches: tl.int32,
    hidden_size: tl.int32,
    eps: tl.float32,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= num_patches:
        return
    # Compute mean and variance in fp32 over all features
    sum_fp32 = 0.0
    sum_sq_fp32 = 0.0
    for c in range(0, hidden_size, BLOCK_C):
        offs = c + tl.arange(0, BLOCK_C)
        mask = offs < hidden_size
        h = tl.load(hidden_ptr + pid * hidden_size + offs, mask=mask, other=0.0)
        h = h.to(tl.float32)
        sum_fp32 += tl.sum(h, axis=0)
        sum_sq_fp32 += tl.sum(h * h, axis=0)
    mean = sum_fp32 / hidden_size
    var = sum_sq_fp32 / hidden_size - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine in fp32, store as bf16
    for c in range(0, hidden_size, BLOCK_C):
        offs = c + tl.arange(0, BLOCK_C)
        mask = offs < hidden_size
        h = tl.load(hidden_ptr + pid * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (h - mean) * inv_std
        y = y * w + b
        # Store bf16
        tl.store(out_ptr + pid * hidden_size + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def spatial_shuffle_to_fc1(
    ln_ptr,           # *bf16, [num_patches, hidden_size]
    grid_thw_ptr,     # *int64, [num_grids, 3], rows = [t, h, w]
    fc1_input_ptr,    # *bf16, [num_merged_patches, hidden_size_expanded]
    num_patches: tl.int32,
    num_grids: tl.int32,
    hidden_size: tl.int32,
    hidden_size_expanded: tl.int32,
    BLOCK_PATCH: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # We map patches across grids into fc1_input in a single kernel for simplicity.
    # The original code derives T,H,W per grid from grid_thw and reorders patches.
    # We emulate that reorder here by computing offsets from grid_thw for each grid.
    # For correctness, we recompute T,H,W per grid from grid_thw and write into fc1_input.
    pid_grid = tl.program_id(axis=0)
    if pid_grid >= num_grids:
        return

    t = tl.load(grid_thw_ptr + pid_grid * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + pid_grid * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + pid_grid * 3 + 2).to(tl.int32)
    h_merged = h // 2
    w_merged = w // 2
    num_patches_grid = t * h * w

    # We compute which original patch belongs to which merged position.
    # For each merged position (tm, hm, j2), there are four original (i0,j0):
    # i0 = 2*hm + di, j0 = 2*j2 + dj, di,dj in {0,1}
    # Input row in fc1_input is m = tm * (h_merged * w_merged) + (hm * w_merged + j2)
    # For each patch p, we compute i0,j0 and write into fc1_input[m, c].
    # We iterate over all patches and all features c.
    # Note: hidden_size_expanded equals hidden_size here, so we can load ln values directly.

    # Iterate over tm, hm, j2
    # Merged patches per grid: tm in [0, t), hm in [0, h_merged), j2 in [0, w_merged)
    for tm in range(0, t):
        for hm in range(0, h_merged):
            for j2 in range(0, w_merged):
                # Base patch index for this merged position
                base_patch = tm * (h * w) + (hm * 2) * w + j2 * 2
                for di in range(2):
                    for dj in range(2):
                        p = base_patch + di * w + dj  # original patch index in this grid
                        if p >= num_patches_grid:
                            continue
                        # Map original patch index to 2D indices
                        i0 = p // w
                        j0 = p % w
                        # Output row in fc1_input
                        m = tm * (h_merged * w_merged) + hm * w_merged + j2
                        # For each feature c
                        for c in range(0, hidden_size_expanded, BLOCK_C):
                            offs = c + tl.arange(0, BLOCK_C)
                            mask = offs < hidden_size_expanded
                            # Read ln value at (row = p, col = offs)
                            row_ptr = ln_ptr + p * hidden_size_expanded + offs
                            val = tl.load(row_ptr, mask=mask, other=0.0).to(tl.float32)
                            # Write to fc1_input at (m, offs)
                            out_ptr = fc1_input_ptr + m * hidden_size_expanded + offs
                            tl.store(out_ptr, val.to(tl.bfloat16), mask=mask)


@triton.jit
def matmul_bias_kernel(
    A_ptr,            # *bf16, [M, K]
    B_ptr,            # *bf16, [K, N]
    bias_ptr,         # *bf16, [N] or None (we pass zeros if no bias)
    C_ptr,            # *bf16, [M, N]
    M: tl.int32, N: tl.int32, K: tl.int32,
    A_stride0: tl.int32, A_stride1: tl.int32,
    B_stride0: tl.int32, B_stride1: tl.int32,
    C_stride0: tl.int32, C_stride1: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(
            A_ptr + m0 * A_stride0 + (k0 + tl.arange(0, BLOCK_K)) * A_stride1,
            mask=(m0 + tl.arange(0, BLOCK_M))[:, None] < M,
            other=0.0,
        ).to(tl.float32)  # [BM, BK]
        b = tl.load(
            B_ptr + (k0 + tl.arange(0, BLOCK_K)) * B_stride0 + n0 * B_stride1,
            mask=(n0 + tl.arange(0, BLOCK_N))[None, :] < N,
            other=0.0,
        ).to(tl.float32)  # [BK, BN]
        acc += tl.dot(a, b)
    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + n0 + tl.arange(0, BLOCK_N), mask=(n0 + tl.arange(0, BLOCK_N)) < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    # Store as bf16
    tl.store(
        C_ptr + m0 * C_stride0 + (n0 + tl.arange(0, BLOCK_N)) * C_stride1,
        acc.to(tl.bfloat16),
        mask=(m0 + tl.arange(0, BLOCK_M))[:, None] < M,
    )


@triton.jit
def gelu_tanh_kernel(
    inp_ptr,          # *bf16, [M, N]
    out_ptr,          # *bf16, [M, N]
    M: tl.int32, N: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    for i in range(0, M, BLOCK_M):
        for j in range(0, N, BLOCK_N):
            # Elementwise GELU (tanh approximation)
            # x = inp[i, j]; gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
            # We compute over tiles, but Triton prefers vectorized operations; we use broadcasting
            # Load tile
            offs_m = i + tl.arange(0, BLOCK_M)
            offs_n = j + tl.arange(0, BLOCK_N)
            mask_m = offs_m < M
            mask_n = offs_n < N
            x = tl.load(inp_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
            c0 = 0.7978845608028654  # sqrt(2/pi)
            c1 = 0.044715
            x3 = x * x * x
            inner = c0 * (x + c1 * x3)
            gelu = 0.5 * x * (1.0 + tl.tanh(inner))
            tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], gelu.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def _cdiv(a, b):
    return (a + b - 1) // b


def layernorm_affine(hidden: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float) -> torch.Tensor:
    num_patches, hidden_size = hidden.shape
    ln_out = torch.empty_like(hidden)
    BLOCK_C = 128
    grid = (num_patches,)
    layernorm_affine_kernel[grid](
        hidden, ln_weight, ln_bias, ln_out,
        num_patches, hidden_size, float(eps),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return ln_out


def spatial_shuffle_to_fc1(ln_out: torch.Tensor, grid_thw: torch.Tensor, hidden_size_expanded: int) -> torch.Tensor:
    # Compute num_merged_patches by summing T*H*W across grids from grid_thw
    num_merged_patches = 0
    for g in range(grid_thw.shape[0]):
        t = int(grid_thw[g, 0].item())
        h = int(grid_thw[g, 1].item())
        w = int(grid_thw[g, 2].item())
        num_merged_patches += t * h * (w // 2)  # 2x2 merge, W must be divisible by 2
    fc1_input = torch.empty((num_merged_patches, hidden_size_expanded), dtype=torch.bfloat16, device=ln_out.device)
    num_grids = grid_thw.shape[0]
    BLOCK_PATCH = 32
    BLOCK_C = 128
    grid = (num_grids,)
    spatial_shuffle_to_fc1[grid](
        ln_out, grid_thw, fc1_input,
        ln_out.shape[0], num_grids, ln_out.shape[1], hidden_size_expanded,
        BLOCK_PATCH=BLOCK_PATCH, BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return fc1_input


def first_linear(fc1_input: torch.Tensor, fc1_weight: torch.Tensor, fc1_bias: torch.Tensor) -> torch.Tensor:
    # fc1_input: [num_merged_patches, hidden_size_expanded]
    # fc1_weight: [hidden_size_expanded, hidden_size_expanded]
    M = fc1_input.shape[0]
    K = fc1_input.shape[1]
    N = fc1_weight.shape[1]
    assert fc1_weight.shape[0] == K, "fc1_weight dim0 must equal hidden_size_expanded"
    out = torch.empty((M, N), dtype=torch.bfloat16, device=fc1_input.device)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (_cdiv(M, BLOCK_M), _cdiv(N, BLOCK_N))
    matmul_bias_kernel[grid](
        fc1_input, fc1_weight, fc1_bias, out,
        M, N, K,
        fc1_input.stride(0), fc1_input.stride(1),
        fc1_weight.stride(0), fc1_weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out


def gelu(fc1_out: torch.Tensor) -> torch.Tensor:
    M, N = fc1_out.shape
    out = torch.empty_like(fc1_out)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (_cdiv(M, BLOCK_M), _cdiv(N, BLOCK_N))
    gelu_tanh_kernel[grid](
        fc1_out, out,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


def second_linear(fc1_out_gelu: torch.Tensor, fc2_weight: torch.Tensor, fc2_bias: torch.Tensor) -> torch.Tensor:
    M, K2 = fc1_out_gelu.shape
    N2, K2_w = fc2_weight.shape
    assert K2_w == K2, "fc2_weight dim1 must equal hidden_size_expanded"
    out = torch.empty((M, N2), dtype=torch.bfloat16, device=fc1_out_gelu.device)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (_cdiv(M, BLOCK_M), _cdiv(N2, BLOCK_N))
    matmul_bias_kernel[grid](
        fc1_out_gelu, fc2_weight, fc2_bias, out,
        M, N2, K2,
        fc1_out_gelu.stride(0), fc1_out_gelu.stride(1),
        fc2_weight.stride(0), fc2_weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor, fc2_weight: torch.Tensor, fc2_bias: torch.Tensor, eps: float):
        # 1) LayerNorm + affine in Triton
        ln_out = layernorm_affine(hidden, ln_weight, ln_bias, eps)  # [num_patches, hidden_size], bf16

        # 2) Spatial shuffle to first linear input in Triton
        fc1_input = spatial_shuffle_to_fc1(ln_out, grid_thw, fc1_weight.shape[0])  # fc1_weight.shape[0] == hidden_size_expanded

        # 3) First Linear: GEMM + bias in Triton
        fc1_out = first_linear(fc1_input, fc1_weight, fc1_bias)  # [num_merged_patches, hidden_size_expanded], bf16

        # 4) GELU in Triton (elementwise)
        fc1_out_gelu = gelu(fc1_out)

        # 5) Second Linear: GEMM + bias in Triton
        output = second_linear(fc1_out_gelu, fc2_weight, fc2_bias)  # [num_merged_patches, out_hidden_size], bf16

        return output


def run(*args):
    return ModelNew()(*args)
