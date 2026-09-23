import math
import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr,           # *float32, [NUM_PATCHES, HIDDEN_SIZE]
    ln_w_ptr,        # *float32, [HIDDEN_SIZE]
    ln_b_ptr,        # *float32, [HIDDEN_SIZE]
    out_ptr,         # *float32, [NUM_PATCHES, HIDDEN_SIZE]
    HIDDEN_SIZE: tl.constexpr,  # int32
    NUM_PATCHES: tl.constexpr,  # int32
    X_stride_row, X_stride_col,
    OUT_stride_row, OUT_stride_col,
):
    pid = tl.program_id(axis=0)  # row index
    if pid >= NUM_PATCHES:
        return

    # Compute mean
    mean = 0.0
    for i in range(0, HIDDEN_SIZE):
        xi = tl.load(x_ptr + pid * X_stride_row + i * X_stride_col)
        mean += xi
    mean = mean / HIDDEN_SIZE

    # Compute variance
    var = 0.0
    for i in range(0, HIDDEN_SIZE):
        xi = tl.load(x_ptr + pid * X_stride_row + i * X_stride_col)
        var += (xi - mean) * (xi - mean)
    var = var / HIDDEN_SIZE

    inv_std = 1.0 / tl.sqrt(var + 1e-6)

    # Normalize and affine
    for i in range(0, HIDDEN_SIZE):
        xi = tl.load(x_ptr + pid * X_stride_row + i * X_stride_col)
        norm = (xi - mean) * inv_std
        wi = tl.load(ln_w_ptr + i)
        bi = tl.load(ln_b_ptr + i)
        y = norm * wi + bi
        tl.store(out_ptr + pid * OUT_stride_row + i * OUT_stride_col, y)


@triton.jit
def spatial_reindex_kernel(
    x_ptr,           # *float32, flattened normalized hidden: [NUM_PATCHES * HIDDEN_SIZE]
    out_ptr,         # *float32, flattened output: [NUM_MERGED * (4*HIDDEN_SIZE)]
    t_list, h_list, w_list,     # *int64, [NUM_GRIDS]
    offsets,                 # *int64, [NUM_GRIDS] cumulative offsets per grid
    NUM_PATCHES: tl.constexpr,  # int32
    NUM_GRIDS: tl.constexpr,    # int32
    HIDDEN_SIZE: tl.constexpr,  # int32
    HIDDEN_EXPANDED: tl.constexpr,  # int32, 4 * HIDDEN_SIZE
    X_stride_row, X_stride_col,    # strides for x (flattened), not used directly
):
    # 2D grid over (row r, col j)
    r = tl.program_id(axis=0)  # output row index in [0, NUM_MERGED * HIDDEN_EXPANDED)
    j = tl.program_id(axis=1)  # output col index in [0, HIDDEN_EXPANDED)

    # Find grid index g for this row via binary search on offsets
    low = 0
    high = NUM_GRIDS
    g = 0
    while low < high:
        mid = (low + high) // 2
        off = tl.load(offsets + mid)  # int64
        if r >= off:
            low = mid + 1
        else:
            high = mid
    g = low - 1  # when low == high, g is correct

    # Load per-grid T,H,W
    Tg = tl.load(t_list + g)
    Hg = tl.load(h_list + g)
    Wg = tl.load(w_list + g)

    Hm = Hg // 2
    Wm = Wg // 2

    # Decode j into (merge_h, merge_w, c)
    C = HIDDEN_SIZE
    merge_h = (j // (4 * C)) % 2
    merge_w = (j // (2 * C)) % 2
    c = j // 4

    # Base row within this grid after offsets
    base_grid = r - tl.load(offsets + g)  # int64
    t_idx = base_grid // (Hm * Wm)
    rem = base_grid % (Hm * Wm)
    h_merged_idx = rem // Wm
    w_merged_idx = rem % Wm

    # Map to original (T, H, W) coordinates (2x2 merge)
    h_idx = h_merged_idx * 2 + merge_h
    w_idx = w_merged_idx * 2 + merge_w

    # Linear index in x (normalized hidden) and load
    in_idx = (t_idx * Hg * Wg + h_idx * Wg + w_idx) * C + c
    val = tl.load(x_ptr + in_idx)

    # Store to output
    out_idx = r * HIDDEN_EXPANDED + j
    tl.store(out_ptr + out_idx, val)


@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    A_stride_row, A_stride_col,
    B_stride_row, B_stride_col,
    C_stride_row, C_stride_col,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tile coordinates
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Pointers for A and B tiles
        a_ptrs = A_ptr + (offs_m[:, None] * A_stride_row + offs_k[None, :] * A_stride_col)
        b_ptrs = B_ptr + (offs_k[:, None] * B_stride_row + offs_n[None, :] * B_stride_col)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc = acc + bias[None, :]

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * C_stride_row + offs_n[None, :] * C_stride_col)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def gelu_kernel(
    inp_ptr, out_ptr,
    SIZE: tl.constexpr,  # int32
    IN_stride, OUT_stride,
):
    pid = tl.program_id(axis=0)
    if pid >= SIZE:
        return
    x = tl.load(inp_ptr + pid * IN_stride)
    # tanh-based GELU approximation
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    k = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = k * (x + 0.044715 * x3)
    t = tl.tanh(inner)
    y = 0.5 * x * (1.0 + t)
    tl.store(out_ptr + pid * OUT_stride, y)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

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
        Triton-only forward:
        - LayerNorm on hidden
        - Spatial reindex into hidden_shuffled
        - fc1: GEMM + bias
        - GELU (tanh approximation)
        - fc2: GEMM + bias
        """
        device = hidden.device
        hidden_size = hidden.shape[1]
        hidden_norm_fp32 = torch.empty_like(hidden, dtype=torch.float32, device=device)
        # Kernel 1: LayerNorm + affine
        NUM_PATCHES = hidden.shape[0]
        grid = (NUM_PATCHES,)
        layer_norm_kernel[grid](
            hidden, ln_weight.to(torch.float32), ln_bias.to(torch.float32), hidden_norm_fp32,
            hidden_size, NUM_PATCHES,
            hidden.stride(0), hidden.stride(1),
            hidden_norm_fp32.stride(0), hidden_norm_fp32.stride(1),
            num_warps=4,
        )

        # Prepare per-grid T,H,W (int64 tensors) and offsets (int64 cumulative) on host
        num_grids = grid_thw.shape[0]
        t_list = torch.empty((num_grids,), dtype=torch.int64, device=device)
        h_list = torch.empty((num_grids,), dtype=torch.int64, device=device)
        w_list = torch.empty((num_grids,), dtype=torch.int64, device=device)
        for g in range(num_grids):
            t_list[g] = int(grid_thw[g, 0].item())
            h_list[g] = int(grid_thw[g, 1].item())
            w_list[g] = int(grid_thw[g, 2].item())

        # Compute offsets: cumulative sum of per-grid totals
        total_per_grid = t_list * h_list * w_list  # int64 tensor on device
        offsets = torch.empty((num_grids,), dtype=torch.int64, device=device)
        # offs[0] = total_per_grid[0]
        offsets[0] = total_per_grid[0]
        for g in range(1, num_grids):
            offsets[g] = offsets[g - 1] + total_per_grid[g]
        num_merged = int(offsets[-1].item())  # total number of patches across grids

        hidden_expanded = hidden_size * 4  # 6144

        # Allocate flattened hidden shuffled
        hidden_shuffled_fp32 = torch.empty((num_merged * hidden_expanded,), dtype=torch.float32, device=device)

        # Kernel 2: Spatial reindex (2D grid over rows and columns)
        BLOCK_N = 128
        grid_reindex = (num_merged * hidden_expanded, triton.cdiv(hidden_expanded, BLOCK_N))
        spatial_reindex_kernel[grid_reindex](
            hidden_norm_fp32, hidden_shuffled_fp32,
            t_list, h_list, w_list, offsets,
            NUM_PATCHES, num_grids, hidden_size, hidden_expanded,
            hidden_norm_fp32.stride(0), hidden_norm_fp32.stride(1),
            num_warps=4,
        )

        # Reshape to [num_merged, hidden_expanded]
        hidden_shuffled_fp32 = hidden_shuffled_fp32.view(num_merged, hidden_expanded)

        # 3) FC1: hidden_shuffled @ fc1_weight.T (+ fc1_bias) -> [num_merged, hidden_expanded]
        M = num_merged
        K1 = hidden_expanded  # 6144
        N1 = K1
        fc1_out_fp32 = torch.empty((M, N1), dtype=torch.float32, device=device)

        BLOCK_M_fc1 = 64
        BLOCK_N_fc1 = 64
        BLOCK_K_fc1 = 32
        grid_fc1 = (triton.cdiv(M, BLOCK_M_fc1), triton.cdiv(N1, BLOCK_N_fc1))
        matmul_bias_kernel[grid_fc1](
            hidden_shuffled_fp32, fc1_weight, fc1_bias, fc1_out_fp32,
            M, N1, K1,
            hidden_shuffled_fp32.stride(0), hidden_shuffled_fp32.stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            fc1_out_fp32.stride(0), fc1_out_fp32.stride(1),
            BLOCK_M=BLOCK_M_fc1, BLOCK_N=BLOCK_N_fc1, BLOCK_K=BLOCK_K_fc1,
            num_warps=4,
        )

        # 4) GELU (tanh approximation)
        fc1_out_flat = fc1_out_fp32.reshape(-1)  # [M*N1]
        fc1_out_flat = fc1_out_flat.contiguous()
        gelu_out_flat = torch.empty_like(fc1_out_flat, dtype=torch.float32, device=device)
        grid_gelu = (fc1_out_flat.shape[0],)
        gelu_kernel[grid_gelu](
            fc1_out_flat, gelu_out_flat,
            fc1_out_flat.shape[0], fc1_out_flat.stride(0), gelu_out_flat.stride(0),
            num_warps=4,
        )
        fc1_out_fp32 = gelu_out_flat.view(M, N1)

        # 5) FC2: fc1_out_fp32 @ fc2_weight.T (+ fc2_bias) -> [num_merged, out_hidden_size]
        out_hidden_size = fc2_weight.shape[0]  # 3584
        fc2_out_fp32 = torch.empty((M, out_hidden_size), dtype=torch.float32, device=device)

        BLOCK_M_fc2 = 64
        BLOCK_N_fc2 = 64
        BLOCK_K_fc2 = 32
        grid_fc2 = (triton.cdiv(M, BLOCK_M_fc2), triton.cdiv(out_hidden_size, BLOCK_N_fc2))
        matmul_bias_kernel[grid_fc2](
            fc1_out_fp32, fc2_weight, fc2_bias, fc2_out_fp32,
            M, out_hidden_size, N1,
            fc1_out_fp32.stride(0), fc1_out_fp32.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            fc2_out_fp32.stride(0), fc2_out_fp32.stride(1),
            BLOCK_M=BLOCK_M_fc2, BLOCK_N=BLOCK_N_fc2, BLOCK_K=BLOCK_K_fc2,
            num_warps=4,
        )

        # Return output (fp32). If you need bfloat16 to match original signature, cast here.
        # Note: original code returns fp32 tensors; we keep fp32 for numerical stability.
        return fc2_out_fp32


def run(*args):
    return ModelNew()(*args)
