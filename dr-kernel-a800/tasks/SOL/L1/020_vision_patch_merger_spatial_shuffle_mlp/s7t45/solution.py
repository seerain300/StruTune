import math
import torch
import triton
import triton.language as tl


# Kernel: LayerNorm + affine (pre-shuffle), one program per row (patch)
@triton.jit
def layernorm_affine_kernel(
    hidden_ptr,         # *bf16, [num_patches, hidden_size]
    ln_weight_ptr,      # *bf16, [hidden_size]
    ln_bias_ptr,        # *bf16, [hidden_size]
    out_ptr,            # *bf16, [num_patches, hidden_size]
    num_patches,        # int32
    hidden_size,        # int32
    eps,                # float32
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # row index
    # Compute mean and variance in fp32
    sum_val = 0.0
    sum_sq = 0.0
    for c in range(0, hidden_size, BLOCK_C):
        offs = c + tl.arange(0, BLOCK_C)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + pid * hidden_size + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / hidden_size
    var = sum_sq / hidden_size - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Normalize and apply affine, store as bf16
    for c in range(0, hidden_size, BLOCK_C):
        offs = c + tl.arange(0, BLOCK_C)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + pid * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        y = y * w + b
        y = y.to(tl.bfloat16)
        tl.store(out_ptr + pid * hidden_size + offs, y, mask=mask)


# Kernel: Spatial shuffle 2x2 to produce first linear input layout.
# For each grid, write LayerNorm outputs into out_fc1 according to 2x2 merge.
@triton.jit
def fill_X_fc1_from_ln(
    ln_out_ptr,     # *bf16, [num_patches, hidden_size]
    grid_thw_ptr,   # *int64, [num_grids, 3] => (t, h, w)
    out_fc1_ptr,    # *bf16, [num_merged_patches, hidden_size_expanded]
    num_grids,      # int32
    hidden_size,    # int32 (features per patch)
    hidden_size_expanded,  # int32
    merge_size,     # int32 (2 here)
):
    # One program per grid
    pid = tl.program_id(axis=0)
    t = tl.load(grid_thw_ptr + pid * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + pid * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + pid * 3 + 2).to(tl.int32)

    # Number of original patches handled by this grid
    num_patches_grid = t * h * w
    base_ln = (out_fc1_ptr.shape[0] - num_patches_grid) * hidden_size  # position in ln_out for this grid's first patch

    # For each original patch p in [0, num_patches_grid):
    for p in range(0, num_patches_grid):
        i0 = p // w
        j0 = p % w
        i2 = i0 // merge_size
        j2 = j0 // merge_size

        # Destination out_fc1 row index: (grid offset) + (t * i2 * (w//2) + j2) * hidden_size_expanded + feature
        grid_offset = (out_fc1_ptr.shape[0] - num_patches_grid) * hidden_size_expanded
        dest_row = grid_offset + (t * i2 * (w // merge_size) + j2) * hidden_size_expanded

        # Copy each feature c
        for c in range(0, hidden_size_expanded, 1):
            ln_val = tl.load(ln_out_ptr + (base_ln + p) * hidden_size + c).to(tl.bfloat16)
            tl.store(out_fc1_ptr + dest_row + c, ln_val)


# Triton GEMM with bias epilogue: A[M, K], B[K, N], bias[N], out[M, N]
@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store result in bf16
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Kernel: GELU (tanh approximation) elementwise on fp32 input, store bf16
@triton.jit
def gelu_tanh_kernel(
    inp_ptr,  # *fp32, flattened
    out_ptr,  # *bf16, flattened
    total_elems,  # int32
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    c = 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = x + c * x3
    gelu = 0.5 * x * (1.0 + tl.tanh(sqrt_2_over_pi * inner))
    tl.store(out_ptr + offs, gelu.to(tl.bfloat16), mask=mask)


def _cdiv(x, y):
    return (x + y - 1) // y


# Launch LayerNorm + affine in Triton
def layernorm_affine(hidden: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float) -> torch.Tensor:
    num_patches, hidden_size = hidden.shape
    out = torch.empty_like(hidden)
    BLOCK_C = 128
    grid = (num_patches,)
    layernorm_affine_kernel[grid](
        hidden, ln_weight, ln_bias, out,
        num_patches, hidden_size, eps,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


# Launch spatial shuffle to produce first linear input
def spatial_shuffle_to_fc1(ln_out: torch.Tensor, grid_thw: torch.Tensor, hidden_size_expanded: int) -> torch.Tensor:
    num_patches, hidden_size = ln_out.shape
    num_merged_patches = ln_out.shape[0] * hidden_size // hidden_size_expanded  # should match grid_thw-based count
    out_fc1 = torch.empty((num_merged_patches, hidden_size_expanded), dtype=torch.bfloat16, device=ln_out.device)
    grid = (grid_thw.shape[0],)
    fill_X_fc1_from_ln[grid](
        ln_out, grid_thw, out_fc1,
        grid_thw.shape[0], hidden_size, hidden_size_expanded, 2,
        num_warps=4,
    )
    return out_fc1


# Launch first linear (GEMM + bias) in Triton
def first_linear(inp: torch.Tensor, fc1_weight: torch.Tensor, fc1_bias: torch.Tensor) -> torch.Tensor:
    M, K = inp.shape
    K2, N = fc1_weight.shape
    assert K == K2, f"Input features {K} must match fc1_weight dim {K2}"
    out = torch.empty((M, N), dtype=torch.bfloat16, device=inp.device)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (_cdiv(M, BLOCK_M), _cdiv(N, BLOCK_N))
    matmul_bias_kernel[grid](
        inp, fc1_weight, fc1_bias, out,
        M, N, K,
        inp.stride(0), inp.stride(1),
        fc1_weight.stride(0), fc1_weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out


# Launch GELU (tanh approximation) elementwise in Triton
def gelu(inp: torch.Tensor) -> torch.Tensor:
    M, N = inp.shape
    inp_fp32 = inp.to(torch.float32)
    out = torch.empty_like(inp_fp32)
    BLOCK = 256
    grid = (_cdiv(M * N, BLOCK),)
    gelu_tanh_kernel[grid](inp_fp32, out, M * N, BLOCK=BLOCK, num_warps=4)
    return out.to(torch.bfloat16)


# Launch second linear (GEMM + bias) in Triton
def second_linear(inp: torch.Tensor, fc2_weight: torch.Tensor, fc2_bias: torch.Tensor) -> torch.Tensor:
    M, K = inp.shape
    N2, K2 = fc2_weight.shape
    assert K == K2, f"Input features {K} must match fc2_weight dim {K2}"
    out = torch.empty((M, N2), dtype=torch.bfloat16, device=inp.device)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (_cdiv(M, BLOCK_M), _cdiv(N2, BLOCK_N))
    matmul_bias_kernel[grid](
        inp, fc2_weight, fc2_bias, out,
        M, N2, K,
        inp.stride(0), inp.stride(1),
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
        fc1_out_fp32 = fc1_out.to(torch.float32)
        fc1_out_gelu = gelu(fc1_out_fp32)

        # 5) Second Linear: GEMM + bias in Triton
        output = second_linear(fc1_out_gelu, fc2_weight, fc2_bias)  # [num_merged_patches, out_hidden_size], bf16

        return output


def run(*args):
    return ModelNew()(*args)
