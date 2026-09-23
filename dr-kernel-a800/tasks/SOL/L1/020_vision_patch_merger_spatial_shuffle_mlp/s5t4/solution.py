import math
import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel_three_pass(
    x_ptr,           # *ptr to input hidden (N, H), bfloat16
    y_ptr,           # *ptr to output normalized hidden (N, H), bfloat16
    ln_weight_ptr,   # *ptr to ln_weight (H), bfloat16
    ln_bias_ptr,     # *ptr to ln_bias (H), bfloat16
    N,               # number of rows
    H: tl.constexpr, # hidden_size (1536)
    eps,             # epsilon (float32)
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    row_id = tl.program_id(0)
    if row_id >= N:
        return
    row_offset = row_id * H

    # Pass 1: compute sum
    sum_ = 0.0
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
        sum_ += tl.sum(x, axis=0)
    mean = sum_ / H

    # Pass 2: compute sum of squares to get variance
    sumsq = 0.0
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        sumsq += tl.sum(diff * diff, axis=0)
    var = sumsq / H
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 3: normalize, apply affine, store bfloat16
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        y = y * gamma + beta
        tl.store(y_ptr + row_offset + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def linear_kernel(
    A_ptr,           # *ptr to A (M, K), bfloat16
    W_ptr,           # *ptr to W^T (N, K), bfloat16
    Bias_ptr,        # *ptr to bias (N), bfloat16
    C_ptr,           # *ptr to output (M, N), float32
    M,               # number of rows in A
    K,               # K dimension
    N,               # N dimension (output features)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch: pid_m over rows, pid_n over cols
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A tile: (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + (m_offsets[:, None] * K) + k_offsets[None, :]
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # W^T tile: (BLOCK_K, BLOCK_N)
        w_ptrs = W_ptr + (k_offsets[:, None] * N) + n_offsets[None, :]
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, w)

    # Add bias
    bias = tl.load(Bias_ptr + n_offsets, mask=(n_offsets < N), other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store to C (M, N)
    c_ptrs = C_ptr + (m_offsets[:, None] * N) + n_offsets[None, :]
    m_mask = m_offsets[:, None] < M
    n_mask = n_offsets[None, :] < N
    store_mask = m_mask & n_mask
    tl.store(c_ptrs, acc, mask=store_mask)


@triton.jit
def gelu_kernel(
    X_ptr,           # *ptr to input (M, K), float32
    Y_ptr,           # *ptr to output (M, K), float32
    M,               # number of rows
    K,               # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)  # one program per row
    if row_id >= M:
        return
    row_offset = row_id * K
    for off in range(0, K, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < K
        x = tl.load(X_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
        # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        gelu = 0.5 * x * (1.0 + tl.math.erf(x * 0.70710678))  # 1/sqrt(2)
        tl.store(Y_ptr + row_offset + cols, gelu, mask=mask)


# Optional: a Triton cat_rows for future use, though we keep torch.cat for concatenating shuffled patches
@triton.jit
def cat_rows_kernel(
    src1_ptr,        # *ptr to first source tensor
    src2_ptr,        # *ptr to second source tensor
    dst_ptr,         # *ptr to destination tensor
    size1,           # number of rows in src1
    size2,           # number of rows in src2
    stride1,         # stride between rows in src1 (in elements)
    stride2,         # stride between rows in src2 (in elements)
    dst_stride,      # stride between rows in dst (in elements)
    total_rows,      # size1 + size2
    K,               # number of columns
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= total_rows:
        return
    if row_id < size1:
        src_row = row_id
        row_base = src1_ptr + src_row * stride1
    else:
        src_row = row_id - size1
        row_base = src2_ptr + src_row * stride2
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < K
    vals = tl.load(row_base + col_offsets, mask=mask, other=0.0)
    dst_row = row_id * dst_stride
    tl.store(dst_ptr + dst_row, vals, mask=mask)


def triton_layer_norm(hidden: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float):
    """
    Triton LayerNorm over last dim for each row.
    hidden: (num_patches, 1536), bfloat16
    ln_weight, ln_bias: (1536), bfloat16
    Returns normalized tensor of same shape and dtype (bfloat16).
    """
    N, H = hidden.shape
    y = torch.empty_like(hidden)
    BLOCK_SIZE = 256  # chunk size; 1536/256 = 6 iterations
    grid = (N,)
    layer_norm_kernel_three_pass[grid](hidden, y, ln_weight, ln_bias, N, H, eps, BLOCK_SIZE, num_warps=4, num_stages=2)
    return y


def triton_linear(A_bf16: torch.Tensor, W_bf16: torch.Tensor, b_bf16: torch.Tensor):
    """
    Triton linear: C[M, N] = A[M, K] @ W^T[N, K] + b[N], compute in float32, return float32.
    A: (M, K), bfloat16; W: (K, N), bfloat16; b: (N), bfloat16.
    """
    M, K = A_bf16.shape
    N = W_bf16.shape[0]  # K dimension of W
    # Prepare transposed W (N, K) in bfloat16 for Triton
    Wt = W_bf16.transpose(0, 1).contiguous()
    C = torch.empty((M, N), dtype=torch.float32, device=A_bf16.device)
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    linear_kernel[grid](A_bf16, Wt, b_bf16, C, M, K, N, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4, num_stages=2)
    return C


def triton_gelu(X: torch.Tensor):
    """
    Triton GELU activation over entire tensor (flattened per row). X: (M, K), float32.
    Returns Y: (M, K), float32.
    """
    M, K = X.shape
    Y = torch.empty_like(X, dtype=torch.float32)
    BLOCK_SIZE = 256
    grid = (M,)
    gelu_kernel[grid](X, Y, M, K, BLOCK_SIZE, num_warps=4, num_stages=2)
    return Y


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect: hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps
        hidden = args[0]
        ln_weight = args[3]
        ln_bias = args[4]
        fc1_weight = args[5]  # (6144, 6144)
        fc1_bias = args[6]
        fc2_weight = args[7]  # (3584, 6144)
        fc2_bias = args[8]
        eps = args[9]

        # Ensure all on CUDA device
        assert hidden.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda and fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda

        # Step 1: LayerNorm on hidden (num_patches, 1536) using Triton
        hidden_norm = triton_layer_norm(hidden, ln_weight, ln_bias, eps)  # bfloat16

        # Step 2: Spatial shuffle to produce hidden_shuffled (num_merged_patches, 6144) using PyTorch
        # Reconstruct patches per grid from hidden_norm
        grid_thw = args[1]
        num_grids = grid_thw.shape[0]
        patches = []
        offset = 0
        for i in range(num_grids):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            num_patches_this = t * h * w
            patches.append(hidden_norm[offset: offset + num_patches_this])
            offset += num_patches_this

        # Build shuffled_patches


def run(*args):
    return ModelNew()(*args)
