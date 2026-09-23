import torch
import math
import triton
import triton.language as tl


# Triton kernel: LayerNorm over last dim (hidden_size) for each row, then affine
# Input: X [num_patches, hidden_size] (bf16), Output: Y [num_patches, hidden_size] (fp32)
@triton.jit
def layer_norm_affine_kernel(
    X_ptr,  # *bf16
    Y_ptr,  # *fp32
    LN_W_ptr,  # *bf16, shape [hidden_size]
    LN_B_ptr,  # *bf16, shape [hidden_size]
    NUM_PATCHES: tl.constexpr,  # int
    HIDDEN_SIZE: tl.constexpr,  # int
    eps,  # float
    X_stride_row, X_stride_col,
    Y_stride_row, Y_stride_col,
):
    row = tl.program_id(0)  # which patch row
    if row >= NUM_PATCHES:
        return
    # Loop over columns to compute mean and var
    sum_x = 0.0
    sum_x2 = 0.0
    # Iterate over hidden_size in chunks
    for c in range(0, HIDDEN_SIZE):
        x = tl.load(X_ptr + row * X_stride_row + c * X_stride_col)
        x = x.to(tl.float32)
        sum_x += x
        sum_x2 += x * x
    mean = sum_x / HIDDEN_SIZE
    var = sum_x2 / HIDDEN_SIZE - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, store to Y
    for c in range(0, HIDDEN_SIZE):
        x = tl.load(X_ptr + row * X_stride_row + c * X_stride_col)
        x = x.to(tl.float32)
        n = (x - mean) * inv_std
        w = tl.load(LN_W_ptr + c).to(tl.float32)
        b = tl.load(LN_B_ptr + c).to(tl.float32)
        y = n * w + b
        tl.store(Y_ptr + row * Y_stride_row + c * Y_stride_col, y)


# Triton kernel: Spatial reindex (merge 2x2) from normalized hidden into a flat [num_merged, hidden_expanded] buffer
# Grid: (M=num_merged_rows, N=hidden_expanded tiles)
# We decode output column j into (merge_h, merge_w, c) where c in [0, hidden_size)
@triton.jit
def spatial_reindex_kernel(
    X_ptr,   # *fp32, normalized hidden [num_patches, hidden_size], flattened for simple row stride
    Y_ptr,   # *fp32, output flattened [num_merged * hidden_expanded]
    t_list_ptr,   # *int64, [num_grids]
    h_list_ptr,   # *int64, [num_grids]
    w_list_ptr,   # *int64, [num_grids]
    offsets_ptr,   # *int64, [num_grids], cumulative starting indices per grid (in num_patches)
    NUM_PATCHES,   # int32
    NUM_GRIDS,     # int32
    HIDDEN_SIZE,   # int32
    HIDDEN_EXPANDED,  # int32, equals 4*HIDDEN_SIZE
    X_stride_row, X_stride_col,  # strides for X
    BLOCK_N: tl.constexpr,       # columns tile
):
    r = tl.program_id(0)  # output row index (across all grids)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col < HIDDEN_EXPANDED

    # Find grid index g via binary search
    low = 0
    high = NUM_GRIDS
    while low < high:
        mid = (low + high) // 2
        off = tl.load(offsets_ptr + mid)  # int64
        if r >= off:
            low = mid + 1
        else:
            high = mid
    g = low - 1

    Tg = tl.load(t_list_ptr + g)
    Hg = tl.load(h_list_ptr + g)
    Wg = tl.load(w_list_ptr + g)

    Hm = Hg // 2
    Wm = Wg // 2

    # Decode col into (merge_h, merge_w, c)
    C = HIDDEN_SIZE
    # hidden_expanded == 4*C
    merge_h = (col // (4 * C)) % 2
    merge_w = (col // (2 * C)) % 2
    c = col // 4

    # Base in this grid for this row
    base_in_grid = r - tl.load(offsets_ptr + g)  # int64
    t_idx = base_in_grid // (Hm * Wm)
    rem = base_in_grid % (Hm * Wm)
    h_merged_idx = rem // Wm
    w_merged_idx = rem % Wm

    # Map to original (T, H, W) coordinates
    h_idx = h_merged_idx * 2 + merge_h
    w_idx = w_merged_idx * 2 + merge_w

    # Compute input linear index
    in_idx = (t_idx * Hg * Wg + h_idx * Wg + w_idx) * C + c  # int64
    # Ensure in_idx within NUM_PATCHES * HIDDEN_SIZE (r < total_patches; Hm,Wm derived from grid sizes, so valid)
    # Load corresponding X element and store to Y[r * hidden_expanded + col]
    x_val = tl.load(X_ptr + in_idx, mask=mask, other=0.0)
    y_row = r * HIDDEN_EXPANDED
    y_col = col
    tl.store(Y_ptr + y_row + y_col, x_val, mask=mask)


# Triton kernel: GEMM with bias, A[M,K] @ B[K,N] + bias[N] -> C[M,N] (fp32)
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    A_stride_row, A_stride_col,
    B_stride_row, B_stride_col,
    C_stride_row, C_stride_col,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + offs_m[:, None] * A_stride_row + offs_k[None, :] * A_stride_col,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B_ptr + offs_k[:, None] * B_stride_row + offs_n[None, :] * B_stride_col,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    tl.store(
        C_ptr + offs_m[:, None] * C_stride_row + offs_n[None, :] * C_stride_col,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# Triton kernel: GELU (tanh approximation). Applied elementwise on fp32 buffer.
@triton.jit
def gelu_kernel(
    X_ptr, Y_ptr,
    SIZE,  # total number of elements
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < SIZE
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    # tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    t = tl.tanh(inner)
    y = 0.5 * x * (1.0 + t)
    tl.store(Y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor, eps: float):
        # All computation will be in Triton. We set up inputs and launch kernels.

        # 1) LayerNorm on hidden (bf16) -> hidden_norm_fp32 (fp32)
        # hidden: [num_patches, hidden_size] (bf16), grid_thw: [num_grids, 3] (int64)
        device = hidden.device
        hidden_size = hidden.shape[1]
        num_patches = hidden.shape[0]

        hidden_norm_fp32 = torch.empty((num_patches, hidden_size), dtype=torch.float32, device=device)

        # Launch LayerNorm + affine Triton kernel over rows
        grid_ln = (num_patches,)
        layer_norm_affine_kernel[grid_ln](
            hidden, hidden_norm_fp32, ln_weight, ln_bias,
            NUM_PATCHES=num_patches,
            HIDDEN_SIZE=hidden_size,
            eps=eps,
            X_stride_row=hidden.stride(0), X_stride_col=hidden.stride(1),
            Y_stride_row=hidden_norm_fp32.stride(0), Y_stride_col=hidden_norm_fp32.stride(1),
            num_warps=4,
        )

        # 2) Prepare per-grid metadata
        # t_list, h_list, w_list, and offsets for spatial reindex
        num_grids = grid_thw.shape[0]
        t_list = torch.empty((num_grids,), dtype=torch.int64, device=device)
        h_list = torch.empty((num_grids,), dtype=torch.int64, device=device)
        w_list = torch.empty((num_grids,), dtype=torch.int64, device=device)
        offsets = torch.empty((num_grids,), dtype=torch.int64, device=device)

        # Fill t_list, h_list, w_list
        for i in range(num_grids):
            t_list[i] = int(grid_thw[i, 0].item())
            h_list[i] = int(grid_thw[i, 1].item())
            w_list[i] = int(grid_thw[i, 2].item())

        # Compute offsets and total per grid on host, but avoid torch reductions in forward:
        # We need cumulative starting indices and totals for each grid. These are derived from how num_patches is distributed.
        # The reference uses a heuristic to set actual_patches_per_grid = t*h*w, but in practice each grid takes contiguous patches from front.
        # We emulate that by assuming the order: for each grid, take a chunk t*h*w patches sequentially.
        # Compute total patches per grid (redundant since t*h*w is known, but we keep consistent).
        total_per_grid = (t_list * h_list * w_list).tolist()
        running = 0
        for i in range(num_grids):
            offsets[i] = running
            running += total_per_grid[i]

        num_merged = int(sum(total_per_grid))
        hidden_expanded = hidden_size * 4  # 6144

        # 3) Spatial reindex: hidden_norm_fp32 -> hidden_shuffled_fp32 flattened [num_merged * hidden_expanded]
        hidden_shuffled_fp32 = torch.empty((num_merged * hidden_expanded,), dtype=torch.float32, device=device)

        BLOCK_N = 128
        grid_reindex = (num_merged, triton.cdiv(hidden_expanded, BLOCK_N))
        spatial_reindex_kernel[grid_reindex](
            hidden_norm_fp32, hidden_shuffled_fp32,
            t_list, h_list, w_list, offsets,
            NUM_PATCHES=num_patches,
            NUM_GRIDS=num_grids,
            HIDDEN_SIZE=hidden_size,
            HIDDEN_EXPANDED=hidden_expanded,
            X_stride_row=hidden_norm_fp32.stride(0), X_stride_col=hidden_norm_fp32.stride(1),
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )

        # Reshape to [num_merged, hidden_expanded]
        hidden_shuffled_fp32 = hidden_shuffled_fp32.view(num_merged, hidden_expanded)

        # 4) FC1: hidden_shuffled_fp32 @ fc1_weight.T (+ fc1_bias) -> [num_merged, hidden_expanded]
        M = num_merged
        K1 = hidden_expanded  # 6144
        N1 = K1

        fc1_out_fp32 = torch.empty((M, N1), dtype=torch.float32, device=device)

        BLOCK_M_fc1 = 64
        BLOCK_N_fc1 = 64
        BLOCK_K_fc1 = 32
        grid_fc1 = (triton.cdiv(M, BLOCK_M_fc1), triton.cdiv(N1, BLOCK_N_fc1))
        gemm_bias_kernel[grid_fc1](
            hidden_shuffled_fp32, fc1_weight, fc1_bias, fc1_out_fp32,
            M, N1, K1,
            hidden_shuffled_fp32.stride(0), hidden_shuffled_fp32.stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            fc1_out_fp32.stride(0), fc1_out_fp32.stride(1),
            BLOCK_M=BLOCK_M_fc1, BLOCK_N=BLOCK_N_fc1, BLOCK_K=BLOCK_K_fc1,
            num_warps=4,
        )

        # Optional: GELU activation (not called by original run, but kept for completeness; ensure Triton-only)
        # gelu_input = fc1_out_fp32  # [M, N1]
        # gelu_output = torch.empty_like(gelu_input)
        # size = M * N1
        # BLOCK_G = 1024
        # grid_gelu = (triton.cdiv(size, BLOCK_G),)
        # gelu_kernel[grid_gelu](gelu_input, gelu_output, size, BLOCK=BLOCK_G, num_warps=4)
        # fc1_out_fp32 = gelu_output.view(M, N1)

        # 5) FC2: fc1_out_fp32 @ fc2_weight.T (+ fc2_bias) -> [num_merged, out_hidden_size]
        out_hidden_size = fc2_weight.shape[0]  # 3584
        fc2_out_fp32 = torch.empty((M, out_hidden_size), dtype=torch.float32, device=device)

        BLOCK_M_fc2 = 64
        BLOCK_N_fc2 = 64
        BLOCK_K_fc2 = 32
        grid_fc2 = (triton.cdiv(M, BLOCK_M_fc2), triton.cdiv(out_hidden_size, BLOCK_N_fc2))
        gemm_bias_kernel[grid_fc2](
            fc1_out_fp32, fc2_weight, fc2_bias, fc2_out_fp32,
            M, out_hidden_size, K1,
            fc1_out_fp32.stride(0), fc1_out_fp32.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            fc2_out_fp32.stride(0), fc2_out_fp32.stride(1),
            BLOCK_M=BLOCK_M_fc2, BLOCK_N=BLOCK_N_fc2, BLOCK_K=BLOCK_K_fc2,
            num_warps=4,
        )

        # Return as bfloat16 (original run returns bfloat16)
        return fc2_out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
