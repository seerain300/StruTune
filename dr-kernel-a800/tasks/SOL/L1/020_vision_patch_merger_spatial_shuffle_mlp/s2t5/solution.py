import math
import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr,           # *float32, [num_patches, hidden_size]
    ln_w_ptr,        # *float32, [hidden_size]
    ln_b_ptr,        # *float32, [hidden_size]
    out_ptr,         # *float32, [num_patches, hidden_size]
    N,               # int32, hidden_size
    NUM_PATCHES,     # int32, num_patches
    X_stride_row, X_stride_col,
    OUT_stride_row, OUT_stride_col,
):
    # Each program handles one row (one patch)
    pid = tl.program_id(axis=0)
    if pid >= NUM_PATCHES:
        return

    # Compute mean
    mean = 0.0
    for i in range(0, N):
        val = tl.load(x_ptr + pid * X_stride_row + i * X_stride_col)
        mean += val
    mean = mean / N

    # Compute variance
    var = 0.0
    for i in range(0, N):
        val = tl.load(x_ptr + pid * X_stride_row + i * X_stride_col)
        var += (val - mean) * (val - mean)
    var = var / N

    inv_std = 1.0 / tl.sqrt(var + 1e-6)

    # Normalize and affine
    for i in range(0, N):
        x = tl.load(x_ptr + pid * X_stride_row + i * X_stride_col)
        norm = (x - mean) * inv_std
        w = tl.load(ln_w_ptr + i)
        b = tl.load(ln_b_ptr + i)
        y = norm * w + b
        tl.store(out_ptr + pid * OUT_stride_row + i * OUT_stride_col, y)


@triton.jit
def spatial_reindex_kernel(
    x_ptr,           # *float32, [num_patches, hidden_size] = normalized hidden
    out_ptr,         # *float32, [num_merged, hidden_expanded] flattened
    t_list_ptr,      # *int64, [num_grids]
    h_list_ptr,      # *int64, [num_grids]
    w_list_ptr,      # *int64, [num_grids]
    offsets_ptr,     # *int64, [num_grids] cumulative offsets per grid
    NUM_PATCHES,     # int32
    NUM_GRIDS,       # int32
    HIDDEN_SIZE,     # int32
    HIDDEN_EXPANDED, # int32, must be 4 * HIDDEN_SIZE
    X_stride_row, X_stride_col,
    BLOCK_N: tl.constexpr,
):
    # 2D launch: axis 0 over rows (num_merged), axis 1 over N tiles
    pid_r = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    col_start = pid_n * BLOCK_N
    for col in range(col_start, col_start + BLOCK_N):
        if col >= HIDDEN_EXPANDED:
            continue
        # Find grid g for row pid_r
        low = 0
        high = NUM_GRIDS
        g = 0
        while low < high:
            mid = (low + high) // 2
            off = tl.load(offsets_ptr + mid)
            if pid_r >= off:
                low = mid + 1
            else:
                high = mid
        g = low - 1  # if low == high, off >= pid_r

        Tg = tl.load(t_list_ptr + g)
        Hg = tl.load(h_list_ptr + g)
        Wg = tl.load(w_list_ptr + g)

        Hm = Hg // 2
        Wm = Wg // 2

        # Decode col into (merge_h, merge_w, c) where c in [0, HIDDEN_SIZE)
        C = HIDDEN_SIZE
        merge_h = (col // (4 * C)) % 2
        merge_w = (col // (2 * C)) % 2
        c = col // 4  # since HIDDEN_EXPANDED == 4*C, col % 4 == 0

        # Determine base index within grid for this merged spatial position
        # row pid_r corresponds to index in merged patches list
        # offsets[g] is the starting row of this grid
        base_grid = pid_r - tl.load(offsets_ptr + g)
        if base_grid < 0:
            # This should not happen since we computed offsets to include all patches
            continue
        t_idx = base_grid // (Hm * Wm)
        rem = base_grid % (Hm * Wm)
        h_merged_idx = rem // Wm
        w_merged_idx = rem % Wm

        # Map to original (T, H, W) coordinates accounting for 2x2 merge
        h_idx = h_merged_idx * 2 + merge_h
        w_idx = w_merged_idx * 2 + merge_w

        # Compute input linear index and load
        in_idx = (t_idx * Hg * Wg + h_idx * Wg + w_idx) * C + c
        val = tl.load(x_ptr + in_idx)

        # Store to output flattened
        out_idx = pid_r * HIDDEN_EXPANDED + col
        tl.store(out_ptr + out_idx, val)


@triton.jit
def matmul_bias_kernel(
    A_ptr,           # *float32, [M, K]
    B_ptr,           # *float32, [K, N] (weight, we'll treat as [K, N] via strides)
    Bias_ptr,        # *float32, [N] or None if no bias
    C_ptr,           # *float32, [M, N]
    M, N, K,         # int32 sizes
    A_stride_row, A_stride_col,
    B_stride_row, B_stride_col,
    C_stride_row, C_stride_col,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * A_stride_row + offs_k[None, :] * A_stride_col)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # Load B tile as [BLOCK_K, BLOCK_N] by reading [K, N]
        b_ptrs = B_ptr + (offs_k[:, None] * B_stride_row + offs_n[None, :] * B_stride_col)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Add bias if provided
    if Bias_ptr is not None:
        bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BLOCK_N]
        acc += bias[None, :]

    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * C_stride_row + offs_n[None, :] * C_stride_col)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def gelu_kernel(
    X_ptr, Y_ptr, SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    if pid >= SIZE:
        return
    x = tl.load(X_ptr + pid)
    # tanh-based GELU approximation
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    y = 0.5 * x * (1.0 + tl.tanh(c * (x + 0.044715 * x3)))
    tl.store(Y_ptr + pid, y)


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        # Ensure device and dtype
        device = hidden.device
        hidden_size = hidden.shape[1]
        hidden_norm_fp32 = torch.empty_like(hidden, dtype=torch.float32, device=device)

        # Kernel 1: LayerNorm + affine
        NUM_PATCHES = hidden.shape[0]
        ln_weight_fp32 = ln_weight.to(torch.float32).contiguous()
        ln_bias_fp32 = ln_bias.to(torch.float32).contiguous()
        grid = (NUM_PATCHES,)
        layer_norm_kernel[grid](
            hidden, ln_weight_fp32, ln_bias_fp32, hidden_norm_fp32,
            hidden_size,
            NUM_PATCHES,
            hidden.stride(0), hidden.stride(1),
            hidden_norm_fp32.stride(0), hidden_norm_fp32.stride(1),
            num_warps=4,
        )

        # Prepare per-grid T,H,W and offsets for spatial reindex
        num_grids = grid_thw.shape[0]
        t_list = torch.empty((num_grids,), dtype=torch.int64, device=device)
        h_list = torch.empty((num_grids,), dtype=torch.int64, device=device)
        w_list = torch.empty((num_grids,), dtype=torch.int64, device=device)

        offsets = torch.empty((num_grids,), dtype=torch.int64, device=device)

        # Compute per-grid T,H,W
        for g in range(num_grids):
            t_list[g] = int(grid_thw[g, 0].item())
            h_list[g] = int(grid_thw[g, 1].item())
            w_list[g] = int(grid_thw[g, 2].item())

        # Compute offsets: cumulative totals
        total_per_grid = torch.tensor([0], device=device, dtype=torch.int64)  # not used directly; compute with host loop
        total_so_far = 0
        for g in range(num_grids):
            Tg = int(t_list[g].item())
            Hg = int(h_list[g].item())
            Wg = int(w_list[g].item())
            patches_per_grid = Tg * Hg * Wg
            offsets[g] = total_so_far
            total_so_far += patches_per_grid

        num_merged = int(total_so_far.item())
        hidden_expanded = hidden_size * 4  # 6144

        # Allocate flattened hidden shuffled
        hidden_shuffled_fp32 = torch.empty((num_merged * hidden_expanded,), dtype=torch.float32, device=device)

        # Kernel 2: Spatial reindex
        BLOCK_N = 128
        grid_reindex = (num_merged, triton.cdiv(hidden_expanded, BLOCK_N))
        spatial_reindex_kernel[grid_reindex](
            hidden_norm_fp32, hidden_shuffled_fp32,
            t_list, h_list, w_list, offsets,
            NUM_PATCHES, num_grids, hidden_size, hidden_expanded,
            hidden_norm_fp32.stride(0), hidden_norm_fp32.stride(1),
            BLOCK_N=BLOCK_N, num_warps=4,
        )

        # Reshape to [num_merged, hidden_expanded]
        hidden_shuffled = hidden_shuffled_fp32.view(num_merged, hidden_expanded)

        # Kernel 3: FC1 GEMM + bias -> [num_merged, hidden_expanded]
        M = num_merged
        K1 = hidden_expanded  # 6144
        N1 = K1

        fc1_out_fp32 = torch.empty((M, N1), dtype=torch.float32, device=device)

        BLOCK_M_fc1 = 64
        BLOCK_N_fc1 = 64
        BLOCK_K_fc1 = 32
        grid_fc1 = (triton.cdiv(M, BLOCK_M_fc1), triton.cdiv(N1, BLOCK_N_fc1))
        matmul_bias_kernel[grid_fc1](
            hidden_shuffled, fc1_weight.to(torch.float32), fc1_bias.to(torch.float32), fc1_out_fp32,
            M, N1, K1,
            hidden_shuffled.stride(0), hidden_shuffled.stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            fc1_out_fp32.stride(0), fc1_out_fp32.stride(1),
            BLOCK_M=BLOCK_M_fc1, BLOCK_N=BLOCK_N_fc1, BLOCK_K=BLOCK_K_fc1,
            num_warps=4,
        )

        # Kernel 4: GELU (optional, but defined and launched for Triton-ONLY compliance)
        SIZE_gelu = M * N1
        fc1_out_gelu = torch.empty((SIZE_gelu,), dtype=torch.float32, device=device)
        gelu_kernel[(SIZE_gelu,)](
            fc1_out_fp32, fc1_out_gelu, SIZE_gelu
        )
        # Reshape back
        fc1_out_gelu = fc1_out_gelu.view(M, N1)

        # Kernel 5: FC2 GEMM + bias -> [num_merged, out_hidden_size]
        out_hidden_size = fc2_weight.shape[0]  # 3584
        fc2_out_fp32 = torch.empty((M, out_hidden_size), dtype=torch.float32, device=device)

        BLOCK_M_fc2 = 64
        BLOCK_N_fc2 = 64
        BLOCK_K_fc2 = 32
        grid_fc2 = (triton.cdiv(M, BLOCK_M_fc2), triton.cdiv(out_hidden_size, BLOCK_N_fc2))
        matmul_bias_kernel[grid_fc2](
            fc1_out_gelu, fc2_weight.to(torch.float32), fc2_bias.to(torch.float32), fc2_out_fp32,
            M, out_hidden_size, N1,
            fc1_out_gelu.stride(0), fc1_out_gelu.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            fc2_out_fp32.stride(0), fc2_out_fp32.stride(1),
            BLOCK_M=BLOCK_M_fc2, BLOCK_N=BLOCK_N_fc2, BLOCK_K=BLOCK_K_fc2,
            num_warps=4,
        )

        # Return in bfloat16 to match original signature (original returns bfloat16)
        return fc2_out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
