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
    hidden_size: tl.constexpr,  # int (constexpr for loop)
    NUM_PATCHES: tl.constexpr,  # int (constexpr for grid)
    X_stride_row, X_stride_col,
    OUT_stride_row, OUT_stride_col,
):
    pid = tl.program_id(axis=0)
    if pid >= NUM_PATCHES:
        return

    # Compute mean
    s1 = 0.0
    for i in range(0, hidden_size):
        s1 += tl.load(x_ptr + pid * X_stride_row + i * X_stride_col)
    mean = s1 / hidden_size

    # Compute variance
    s2 = 0.0
    for i in range(0, hidden_size):
        xi = tl.load(x_ptr + pid * X_stride_row + i * X_stride_col)
        s2 += (xi - mean) * (xi - mean)
    var = s2 / hidden_size
    inv_std = 1.0 / tl.sqrt(var + 1e-6)

    # Normalize and affine
    for i in range(0, hidden_size):
        xi = tl.load(x_ptr + pid * X_stride_row + i * X_stride_col)
        norm = (xi - mean) * inv_std
        wi = tl.load(ln_w_ptr + i)
        bi = tl.load(ln_b_ptr + i)
        out_val = norm * wi + bi
        tl.store(out_ptr + pid * OUT_stride_row + i * OUT_stride_col, out_val)


@triton.jit
def spatial_reindex_kernel(
    x_ptr,           # *float32, normalized hidden flattened [num_patches * hidden_size]
    out_ptr,         # *float32, flattened output [num_merged * hidden_expanded]
    t_ptr, h_ptr, w_ptr,     # *int64, per-grid T,H,W
    offsets_ptr,     # *int64, cumulative offsets per grid
    NUM_PATCHES: tl.constexpr,  # int
    NUM_GRIDS: tl.constexpr,    # int
    HIDDEN_SIZE: tl.constexpr,  # int (1536)
    HIDDEN_EXPANDED: tl.constexpr,  # int (6144)
    X_stride_row, X_stride_col,
    OUT_stride_row, OUT_stride_col,
):
    row = tl.program_id(axis=0)  # r in [0, num_merged)
    col = tl.program_id(axis=1)  # j in [0, hidden_expanded)

    # Binary search to find grid index g for this row via offsets
    low = 0
    high = NUM_GRIDS
    g = 0
    while low < high:
        mid = (low + high) // 2
        off = tl.load(offsets_ptr + mid)  # int64
        if row >= off:
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

    # Decode col into (merge_h, merge_w, c) where c in [0, hidden_size)
    C = HIDDEN_SIZE
    merge_h = (col // (4 * C)) % 2
    merge_w = (col // (2 * C)) % 2
    c = col // 4  # since hidden_expanded == 4*C, col % 4 == 0

    # Determine base index within grid for this merged spatial position
    # base = row - offsets[g]
    base = row - tl.load(offsets_ptr + g)
    t_idx = base // (Hm * Wm)
    rem = base % (Hm * Wm)
    h_merged_idx = rem // Wm
    w_merged_idx = rem % Wm

    # Map to original (T, H, W) coordinates accounting for 2x2 merge
    h_idx = h_merged_idx * 2 + merge_h
    w_idx = w_merged_idx * 2 + merge_w

    # Compute input linear index and load
    in_idx = (t_idx * Hg * Wg + h_idx * Wg + w_idx) * C + c
    val = tl.load(x_ptr + in_idx)

    # Compute output linear index and store
    out_idx = row * HIDDEN_EXPANDED + col
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
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + offs_m[:, None] * A_stride_row + offs_k[None, :] * A_stride_col,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        )
        b = tl.load(
            B_ptr + offs_n[None, :] * B_stride_row + offs_k[:, None] * B_stride_col,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
        )
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N)
    c = acc + bias[None, :]
    tl.store(
        C_ptr + offs_m[:, None] * C_stride_row + offs_n[None, :] * C_stride_col,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def gelu_kernel(
    x_ptr, y_ptr, N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return
    x = tl.load(x_ptr + pid)
    # GELU tanh approximation
    # y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    y = 0.5 * x * (1.0 + tl.tanh(c * (x + 0.044715 * x3)))
    tl.store(y_ptr + pid, y)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden: torch.Tensor,
        grid_thw: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
        eps: float,
    ):
        # Ensure CUDA and contiguous
        device = hidden.device
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()
        fc1_weight = fc1_weight.contiguous()
        fc1_bias = fc1_bias.contiguous()
        fc2_weight = fc2_weight.contiguous()
        fc2_bias = fc2_bias.contiguous()

        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        hidden_norm_fp32 = torch.empty_like(hidden, dtype=torch.float32, device=device)

        # Kernel 1: LayerNorm + affine
        grid_ln = (num_patches,)
        layer_norm_kernel[grid_ln](
            hidden, ln_weight.to(torch.float32), ln_bias.to(torch.float32), hidden_norm_fp32,
            hidden_size,
            num_patches,
            hidden.stride(0), hidden.stride(1),
            hidden_norm_fp32.stride(0), hidden_norm_fp32.stride(1),
            num_warps=4,
        )

        # Compute per-grid total and offsets without torch reductions in forward
        num_grids = grid_thw.shape[0]
        total_per_grid = torch.empty((num_grids,), dtype=torch.int64, device=device)
        offsets = torch.empty((num_grids,), dtype=torch.int64, device=device)

        for g in range(num_grids):
            Tg = int(grid_thw[g, 0].item())
            Hg = int(grid_thw[g, 1].item())
            Wg = int(grid_thw[g, 2].item())
            total_per_grid[g] = Tg * Hg * Wg
        # offsets[0] = 0; offsets[i] = offsets[i-1] + total_per_grid[i-1]
        offsets[0] = 0
        for i in range(1, num_grids):
            offsets[i] = offsets[i - 1] + total_per_grid[i - 1]
        num_merged = int(total_per_grid[-1].item())  # last equals total patches (sum is not used in kernel but needed for output)

        # Prepare per-grid tensors for Triton
        t_list = torch.empty((num_grids,), dtype=torch.int64, device=device)
        h_list = torch.empty((num_grids,), dtype=torch.int64, device=device)
        w_list = torch.empty((num_grids,), dtype=torch.int64, device=device)
        for g in range(num_grids):
            t_list[g] = grid_thw[g, 0].item()
            h_list[g] = grid_thw[g, 1].item()
            w_list[g] = grid_thw[g, 2].item()

        hidden_expanded = hidden_size * 4  # 6144

        # Allocate flattened hidden shuffled
        hidden_shuffled_fp32 = torch.empty((num_merged * hidden_expanded,), dtype=torch.float32, device=device)

        # Kernel 2: Spatial reindex
        BLOCK_N = 128
        grid_reindex = (num_merged, triton.cdiv(hidden_expanded, BLOCK_N))
        spatial_reindex_kernel[grid_reindex](
            hidden_norm_fp32, hidden_shuffled_fp32,
            t_list, h_list, w_list, offsets,
            num_patches,
            num_grids,
            hidden_size,
            hidden_expanded,
            hidden_norm_fp32.stride(0), hidden_norm_fp32.stride(1),
            hidden_shuffled_fp32.stride(0), hidden_shuffled_fp32.stride(1),
            num_warps=4,
        )

        # Reshape to [num_merged, hidden_expanded]
        hidden_shuffled_fp32 = hidden_shuffled_fp32.view(num_merged, hidden_expanded)

        # Kernel 3: FC1 GEMM + bias, output [num_merged, hidden_expanded]
        M = num_merged
        K1 = hidden_expanded  # 6144
        N1 = K1
        fc1_out_fp32 = torch.empty((M, N1), dtype=torch.float32, device=device)

        BLOCK_M_fc1 = 64
        BLOCK_N_fc1 = 64
        BLOCK_K_fc1 = 32
        grid_fc1 = (triton.cdiv(M, BLOCK_M_fc1), triton.cdiv(N1, BLOCK_N_fc1))
        matmul_bias_kernel[grid_fc1](
            hidden_shuffled_fp32, fc1_weight.to(torch.float32), fc1_bias.to(torch.float32), fc1_out_fp32,
            M, N1, K1,
            hidden_shuffled_fp32.stride(0), hidden_shuffled_fp32.stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            fc1_out_fp32.stride(0), fc1_out_fp32.stride(1),
            BLOCK_M=BLOCK_M_fc1, BLOCK_N=BLOCK_N_fc1, BLOCK_K=BLOCK_K_fc1,
            num_warps=4,
        )

        # GELU activation in Triton
        fc1_out_fp32_gelu = torch.empty_like(fc1_out_fp32, dtype=torch.float32, device=device)
        grid_gelu = (M * N1,)
        gelu_kernel[grid_gelu](
            fc1_out_fp32, fc1_out_fp32_gelu,
            M * N1,
            num_warps=4,
        )
        fc1_out_fp32 = fc1_out_fp32_gelu.view(M, N1)

        # Kernel 4: FC2 GEMM + bias, output [num_merged, out_hidden_size]
        out_hidden_size = fc2_weight.shape[0]  # 3584
        fc2_out_fp32 = torch.empty((M, out_hidden_size), dtype=torch.float32, device=device)

        BLOCK_M_fc2 = 64
        BLOCK_N_fc2 = 64
        BLOCK_K_fc2 = 32
        grid_fc2 = (triton.cdiv(M, BLOCK_M_fc2), triton.cdiv(out_hidden_size, BLOCK_N_fc2))
        matmul_bias_kernel[grid_fc2](
            fc1_out_fp32, fc2_weight.to(torch.float32), fc2_bias.to(torch.float32), fc2_out_fp32,
            M, out_hidden_size, N1,
            fc1_out_fp32.stride(0), fc1_out_fp32.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            fc2_out_fp32.stride(0), fc2_out_fp32.stride(1),
            BLOCK_M=BLOCK_M_fc2, BLOCK_N=BLOCK_N_fc2, BLOCK_K=BLOCK_K_fc2,
            num_warps=4,
        )

        # Return output as fp32 (original model returns fp32 for these ops). Cast to bfloat16 to match expected dtype if needed.
        output = fc2_out_fp32.to(torch.bfloat16)
        return output


def run(*args):
    return ModelNew()(*args)
