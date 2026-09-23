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
    # One program per row (patch)
    p = tl.program_id(0)
    if p >= num_patches:
        return

    # Compute sum and sum of squares over features in fp32
    sum_ = 0.0
    sumsq = 0.0
    # Loop over features in blocks
    for c0 in range(0, hidden_size, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        mask = c < hidden_size
        # Load hidden row p, features c
        h = tl.load(hidden_ptr + p * hidden_size + c, mask=mask, other=0.0)
        h = h.to(tl.float32)
        # Reduce
        sum_ += tl.sum(h, axis=0)
        sumsq += tl.sum(h * h, axis=0)

    mean = sum_ / hidden_size
    var = sumsq / hidden_size - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and affine
    for c0 in range(0, hidden_size, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        mask = c < hidden_size
        h = tl.load(hidden_ptr + p * hidden_size + c, mask=mask, other=0.0).to(tl.float32)
        ln_w = tl.load(ln_weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
        ln_b = tl.load(ln_bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
        y = (h - mean) * inv_std
        y = y * ln_w + ln_b
        # Cast to bf16 for storage
        y = y.to(tl.bfloat16)
        tl.store(out_ptr + p * hidden_size + c, y, mask=mask)


@triton.jit
def spatial_shuffle_to_fc1_kernel(
    ln_out_ptr,       # *bf16, [num_patches, hidden_size]
    grid_thw_ptr,     # *int64, [num_grids, 3] where 3 = [T, H, W]
    fc1_input_ptr,    # *bf16, [num_merged_patches, hidden_size_expanded]
    num_patches: tl.int32,
    num_grids: tl.int32,
    hidden_size_expanded: tl.int32,
    BLOCK_F: tl.constexpr,
):
    # One program per grid
    g = tl.program_id(0)
    if g >= num_grids:
        return

    # Read T, H, W from grid_thw[g]
    T = tl.load(grid_thw_ptr + g * 3 + 0)
    H = tl.load(grid_thw_ptr + g * 3 + 1)
    W = tl.load(grid_thw_ptr + g * 3 + 2)

    # Constants for merge size
    MERGE = 2
    h_merged = H // MERGE
    w_merged = W // MERGE
    num_patches_grid = T * H * W

    # Loop over original patches and features, write to fc1_input
    # fc1_input has rows indexed by merged patches: offset = g * num_patches_grid + p
    # For each original patch p in [0, num_patches_grid), map to (t, i, j):
    #   t = p // (H*W), rem = p % (H*W), i = rem // W, j = rem % W
    #   i2 = i // MERGE, j2 = j // MERGE
    # The merged patch index in the grid is t * (h_merged*w_merged) + i2 * w_merged + j2
    for p0 in range(0, num_patches_grid, BLOCK_F):
        p = p0 + tl.arange(0, BLOCK_F)
        mask_p = p < num_patches_grid

        # Compute (t, i, j) for each p
        HW = H * W
        t = p // HW
        rem = p % HW
        i = rem // W
        j = rem % W

        # Compute merged coordinates
        i2 = i // MERGE
        j2 = j // MERGE

        # Compute merged patch index in the grid
        merged_index = t * (h_merged * w_merged) + i2 * w_merged + j2

        # Compute the row offset in fc1_input for this grid
        row_offset = g * num_patches_grid

        # Load feature index c = 0..hidden_size_expanded-1
        # We will write ln_out[p, c] to fc1_input[row_offset + merged_index, c]
        # Note: hidden_size_expanded is the feature dimension; c spans features.
        # We iterate c in blocks
        for c0 in range(0, hidden_size_expanded, BLOCK_F):
            c = c0 + tl.arange(0, BLOCK_F)
            mask_c = c < hidden_size_expanded

            # Build 2D pointers for ln_out: [BLOCK_F, BLOCK_F]
            ln_ptr = ln_out_ptr + (p[:, None] * hidden_size + c[None, :])
            ln_mask = mask_p[:, None] & mask_c[None, :]
            vals = tl.load(ln_ptr, mask=ln_mask, other=0.0).to(tl.bfloat16)

            # Compute destination row for each p
            dst_rows = row_offset + merged_index
            dst_ptr = fc1_input_ptr + dst_rows[:, None] * hidden_size_expanded + c[None, :]

            tl.store(dst_ptr, vals, mask=ln_mask)


@triton.jit
def matmul_bias_kernel(
    A_ptr,            # *bf16, [M, K]
    B_ptr,            # *bf16, [K, N] (note: we pass W.T)
    bias_ptr,         # *bf16, [N]
    C_ptr,            # *bf16, [M, N]
    M: tl.int32, N: tl.int32, K: tl.int32,
    stride_am: tl.int64, stride_ak: tl.int64,
    stride_bk: tl.int64, stride_bn: tl.int64,
    stride_cm: tl.int64, stride_cn: tl.int64,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float16)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float16)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def gelu_tanh_kernel(
    x_ptr,            # *bf16, [M, N]
    y_ptr,            # *bf16, [M, N]
    M: tl.int32, N: tl.int32,
    stride_xm: tl.int64, stride_xn: tl.int64,
    stride_ym: tl.int64, stride_yn: tl.int64,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)

    # GELU tanh approximation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    c = 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + c * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))

    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, y.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _cdiv(a, b):
    return (a + b - 1) // b


def first_linear(inp: torch.Tensor, fc1_weight_t: torch.Tensor, fc1_bias: torch.Tensor) -> torch.Tensor:
    # inp: [num_merged_patches, hidden_size_expanded], fc1_weight_t: [hidden_size_expanded, hidden_size_expanded] (already transposed)
    M, K = inp.shape
    N = fc1_weight_t.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=inp.device)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (_cdiv(M, BLOCK_M), _cdiv(N, BLOCK_N))
    matmul_bias_kernel[grid](
        inp, fc1_weight_t, fc1_bias, out,
        M, N, K,
        inp.stride(0), inp.stride(1),
        fc1_weight_t.stride(0), fc1_weight_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out


def second_linear(inp: torch.Tensor, fc2_weight_t: torch.Tensor, fc2_bias: torch.Tensor) -> torch.Tensor:
    # inp: [num_merged_patches, hidden_size_expanded], fc2_weight_t: [hidden_size_expanded, out_hidden_size] (already transposed)
    M, K = inp.shape
    N = fc2_weight_t.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=inp.device)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (_cdiv(M, BLOCK_M), _cdiv(N, BLOCK_N))
    matmul_bias_kernel[grid](
        inp, fc2_weight_t, fc2_bias, out,
        M, N, K,
        inp.stride(0), inp.stride(1),
        fc2_weight_t.stride(0), fc2_weight_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor, fc2_weight: torch.Tensor, fc2_bias: torch.Tensor, eps: float):
        # 1) LayerNorm + affine in Triton
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)
        # Choose feature block size
        BLOCK_C = 128 if hidden.shape[1] >= 128 else 64
        grid_ln = (hidden.shape[0],)
        layernorm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            hidden.shape[0], hidden.shape[1], eps,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        # 2) Spatial shuffle to first linear input in Triton: launch for each grid
        # fc1_input: [num_merged_patches, hidden_size_expanded]
        num_patches = hidden.shape[0]
        num_grids = grid_thw.shape[0]
        hidden_size_expanded = fc1_weight.shape[1]  # second dim is features for this layer
        fc1_input = torch.empty((num_patches, hidden_size_expanded), dtype=torch.bfloat16, device=hidden.device)
        grid_shuffle = (num_grids,)
        spatial_shuffle_to_fc1_kernel[grid_shuffle](
            ln_out, grid_thw, fc1_input,
            num_patches, num_grids, hidden_size_expanded,
            BLOCK_F=1024,  # feature tile; large to minimize loop iterations
            num_warps=4,
        )

        # 3) First Linear: GEMM + bias in Triton
        fc1_input_T = fc1_input.transpose(0, 1)  # [hidden_size_expanded, hidden_size_expanded]
        fc1_out = first_linear(fc1_input_T, fc1_weight, fc1_bias)  # [num_merged_patches, hidden_size_expanded], bf16

        # 4) GELU in Triton (elementwise)
        gelu_in = fc1_out.contiguous()
        gelu_out = torch.empty_like(gelu_in, dtype=torch.bfloat16, device=gelu_in.device)
        M, N = gelu_in.shape
        BLOCK_M = 64
        BLOCK_N = 64
        grid_gelu = (_cdiv(M, BLOCK_M), _cdiv(N, BLOCK_N))
        gelu_tanh_kernel[grid_gelu](
            gelu_in, gelu_out,
            M, N,
            gelu_in.stride(0), gelu_in.stride(1),
            gelu_out.stride(0), gelu_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4,
        )

        # 5) Second Linear: GEMM + bias in Triton
        fc2_in_T = gelu_out.transpose(0, 1)  # [hidden_size_expanded, out_hidden_size]
        out_hidden_size = fc2_weight.shape[0]
        output = second_linear(fc2_in_T, fc2_weight, fc2_bias)  # [num_merged_patches, out_hidden_size], bf16

        return output


def run(*args):
    return ModelNew()(*args)
