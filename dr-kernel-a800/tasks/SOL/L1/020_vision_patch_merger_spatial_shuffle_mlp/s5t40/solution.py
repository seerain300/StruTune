import math
import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_rows_kernel(
    x_ptr,           # *ptr to input (N, H), float32 (we will feed float32, not bf16)
    y_ptr,           # *ptr to output (N, H), float32
    ln_weight_ptr,   # *ptr to ln_weight (H), float32
    ln_bias_ptr,     # *ptr to ln_bias (H), float32
    N,               # number of rows (num_merged_patches)
    H: tl.constexpr, # hidden_size_expanded (e.g., 6144)
    eps,             # epsilon (float32 scalar)
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return
    row_offset = row * H

    # Compute sum and sum of squares in float32
    sum_ = 0.0
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_ += tl.sum(x, axis=0)

    sumsq = 0.0
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    mean = sum_ / H
    var = sumsq / H - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize, affine, store (fp32 output)
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        y = y * gamma + beta
        tl.store(y_ptr + row_offset + cols, y, mask=mask)


@triton.jit
def linear_gemm_kernel(
    A_ptr,           # *ptr to A (M x K), float32
    WT_ptr,          # *ptr to W^T (K x N), float32
    B_ptr,           # *ptr to output (M x N), float32
    bias_ptr,        # *ptr to bias (N), float32
    M,               # number of rows in A
    K,               # hidden_size_expanded (e.g., 6144)
    N,               # out_dim (6144 or 3584)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        # Load A tile: (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + m_offsets[:, None] * K + k_offsets[None, :]
        mask_a = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)
        A_tile = A_tile.to(tl.float32)

        # Load WT tile: (BLOCK_K, BLOCK_N)
        wt_ptrs = WT_ptr + k_offsets[:, None] * N + n_offsets[None, :]
        mask_wt = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        WT_tile = tl.load(wt_ptrs, mask=mask_wt, other=0.0)
        WT_tile = WT_tile.to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, WT_tile)

    # Add bias
    bias = tl.load(bias_ptr + n_offsets, mask=n_offsets < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store
    B_ptrs = B_ptr + m_offsets[:, None] * N + n_offsets[None, :]
    mask_b = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(B_ptrs, acc, mask=mask_b)


@triton.jit
def gelu_kernel(
    x_ptr,           # *ptr to input (M x N), float32
    y_ptr,           # *ptr to output (M x N), float32
    M,               # number of rows
    N,               # number of cols
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)

    x_ptrs = x_ptr + m_offsets[:, None] * N + n_offsets[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)  # float32
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    y_ptrs = y_ptr + m_offsets[:, None] * N + n_offsets[None, :]
    tl.store(y_ptrs, y, mask=mask)


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.hidden_size = 1536
        self.hidden_size_expanded = 6144
        self.out_hidden_size = 3584
        self.merge_size = 2
        self.eps = 1e-6

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor,
                device: torch.device):
        # Compute grid configurations from grid_thw (pure host int math; no torch ops)
        num_patches = int(hidden.shape[0])
        num_merged_patches = int(grid_thw.shape[0]) * grid_thw[0, 0].item() * grid_thw[0, 1].item() * grid_thw[0, 2].item()
        # Build hidden_shuffled directly without using torch.cat/permute in host:
        # We need t,h,w per grid; compute per-grid elements via integer arithmetic.
        num_grids = int(grid_thw.shape[0])
        patches_per_grid = num_patches // num_grids
        # To construct hidden_shuffled without torch ops, we need to know the ordering. Since original uses
        # "merge_size=2" and specific reshapes, we can derive the mapping using integer arithmetic. However,
        # the heavy numerical ops are the LN and matmuls; the


def run(*args):
    return ModelNew()(*args)
