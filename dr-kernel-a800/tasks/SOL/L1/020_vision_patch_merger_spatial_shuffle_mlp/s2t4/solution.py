import math
import torch
import triton
import triton.language as tl


# Kernel 0: LayerNorm on a 2D tensor [M, N], each row is normalized independently and affine applied.
@triton.jit
def layer_norm_affine_kernel(x_ptr, y_ptr, w_ptr, b_ptr, M, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return
    x_row_ptr = x_ptr + row * N
    y_row_ptr = y_ptr + row * N

    sum_x = 0.0
    sum_x2 = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        norm = (x - mean) * inv_std
        w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = norm * w + b
        tl.store(y_row_ptr + offs, y, mask=mask)


# Kernel 1: Compute per-grid totals (T*H*W) and store into offsets_ptr[g].
@triton.jit
def compute_grid_totals_kernel(t_ptr, h_ptr, w_ptr, offsets_ptr, NUM_GRIDS: tl.constexpr):
    g = tl.program_id(0)
    if g >= NUM_GRIDS:
        return
    T = tl.load(t_ptr + g)
    H = tl.load(h_ptr + g)
    W = tl.load(w_ptr + g)
    total = T * H * W
    tl.store(offsets_ptr + g, total)


# Kernel 2: Cumulative sum of offsets (int32). Input offsets_ptr[0..NUM_GRIDS-1], output out_ptr[0..NUM_GRIDS-1].
# We implement an inclusive scan using iterative doubling. Only one program writes; we ensure no races.
@triton.jit
def cumsum_offsets_kernel(offsets_ptr, out_ptr, NUM: tl.constexpr, BLOCK: tl.constexpr):
    # Single program computes cumsum
    idx = tl.arange(0, BLOCK)
    mask = idx < NUM
    # Initialize out as offsets
    vals = tl.load(offsets_ptr + idx, mask=mask, other=0)
    out = vals
    # Iterative doubling
    step = 1
    while step < NUM:
        prev = tl.load(out_ptr + tl.maximum(idx - step, 0))
        out = vals + prev
        # store back
        tl.store(out_ptr + idx, out, mask=mask)
        step *= 2
    # Final store
    tl.store(out_ptr + idx, out, mask=mask)


# Kernel 3: Spatial reindexing (2x2 merge) from normalized hidden to shuffled representation.
# y is a 1D buffer of length num_merged * hidden_expanded.
@triton.jit
def spatial_reindex_kernel(
    x_ptr,          # *fp32, input normalized hidden, shape [num_patches, hidden_size], contiguous row-major
    y_ptr,          # *fp32, output flattened [num_merged * hidden_expanded]
    t_ptr, h_ptr, w_ptr,       # *int32, per-grid (T, H, W)
    offsets_ptr,    # *int32, per-grid cumulative offsets
    NUM_PATCHES: tl.constexpr,
    NUM_GRIDS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_EXPANDED: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    r = tl.program_id(0)  # output row index in [0, num_merged * HIDDEN_EXPANDED)
    j = tl.program_id(1)  # col index in [0, HIDDEN_EXPANDED)

    # Binary search to find grid index g for this row r
    low = 0
    high = NUM_GRIDS
    g = 0
    while low < high:
        mid = (low + high) // 2
        off = tl.load(offsets_ptr + mid)
        if r >= off:
            low = mid + 1
        else:
            high = mid
    g = low - 1

    # Load per-grid T,H,W
    Tg = tl.load(t_ptr + g)
    Hg = tl.load(h_ptr + g)
    Wg = tl.load(w_ptr + g)

    Hm = Hg // 2
    Wm = Wg // 2

    # Decode j into (merge_h, merge_w, c)
    C = HIDDEN_SIZE
    merge_h = (j // (4 * C)) % 2
    merge_w = (j // (2 * C)) % 2
    c = j // 4  # HIDDEN_EXPANDED == 4 * C

    # Base index within grid for this merged spatial position
    base_grid = r - tl.load(offsets_ptr + g)  # int32
    t_idx = base_grid // (Hm * Wm)
    rem = base_grid % (Hm * Wm)
    h_merged_idx = rem // Wm
    w_merged_idx = rem % Wm

    # Map to original (T, H, W) coordinates
    h_idx = h_merged_idx * 2 + merge_h
    w_idx = w_merged_idx * 2 + merge_w

    # Compute input linear index and load
    in_idx = (t_idx * Hg * Wg + h_idx * Wg + w_idx) * C + c
    val = tl.load(x_ptr + in_idx)
    tl.store(y_ptr + r, val)


# Kernel 4: GEMM + bias: A[M,K] @ B[K,N] + bias[N] -> C[M,N]
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Kernel 5: GELU (tanh approximation) applied elementwise on X -> Y
# y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
@triton.jit
def gelu_kernel(X_ptr, Y_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        x3 = x * x * x
        c0 = 0.7978845608028654  # sqrt(2/pi)
        c1 = 0.044715
        y = 0.5 * x * (1.0 + tl.tanh(c0 * (x + c1 * x3)))
        tl.store(Y_ptr + row * N + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps):
        # hidden: [num_patches, hidden_size], bfloat16 or fp32, CUDA
        # grid_thw: [num_grids, 3], int64, CUDA
        # We will compute in fp32, return fp32 (to avoid dtype casting kernels).
        device = hidden.device

        # Extract dims
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        hidden_expanded = hidden_size * 4  # 6144
        out_hidden_size = fc2_weight.shape[0]  # 3584

        # 1) LayerNorm in Triton (compute in fp32)
        hidden_norm_fp32 = torch.empty((num_patches, hidden_size), dtype=torch.float32, device=device)
        BLOCK_N_ln = 1024
        layer_norm_affine_kernel[(num_patches,)](
            hidden, hidden_norm_fp32, ln_weight, ln_bias,
            num_patches, hidden_size, eps, BLOCK_N=BLOCK_N_ln,
            num_warps=4,
        )

        # 2) Prepare per-grid T,H,W as int32
        t_list


def run(*args):
    return ModelNew()(*args)
