import torch
import triton
import triton.language as tl


# Kernel 1: LayerNorm + affine (fp32 compute), outputs fp32
@triton.jit
def layernorm_affine_kernel(
    x_ptr,          # *fp32, [num_patches, hidden_size]
    y_ptr,          # *fp32, [num_patches, hidden_size] output
    weight_ptr,     # *fp32, [hidden_size]
    bias_ptr,       # *fp32, [hidden_size]
    N,              # int32, hidden_size
    eps,            # fp32
    x_stride_row, x_stride_col,  # strides for x
    y_stride_row, y_stride_col,  # strides for y
    BLOCK_C: tl.constexpr
):
    row = tl.program_id(0)  # each program handles one row
    # compute mean and variance
    sum_val = 0.0
    sum_sq = 0.0
    for c in range(0, N, BLOCK_C):
        offs = c + tl.arange(0, BLOCK_C)
        mask = offs < N
        x = tl.load(x_ptr + row * x_stride_row + offs * x_stride_col, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / N
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # normalize and apply affine
    for c in range(0, N, BLOCK_C):
        offs = c + tl.arange(0, BLOCK_C)
        mask = offs < N
        x = tl.load(x_ptr + row * x_stride_row + offs * x_stride_col, mask=mask, other=0.0)
        y = (x - mean) * rstd
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        y = y * w + b
        tl.store(y_ptr + row * y_stride_row + offs * y_stride_col, y, mask=mask)


# Kernel 2: Spatial reindexing from normalized hidden to shuffled hidden (fp32)
@triton.jit
def spatial_reindex_kernel(
    x_ptr,           # *fp32, [num_patches, hidden_size] normalized input
    y_ptr,           # *fp32, [num_merged_patches, hidden_size_expanded] output
    offsets_ptr,     # *int64, [num_grids] cumulative counts per grid
    t_ptr, h_ptr, w_ptr,        # *int64, [num_grids] per-grid T,H,W
    num_patches,     # int32
    num_grids,       # int32
    hidden_size,             # int32
    hidden_expanded,         # int32, hidden_size_expanded = 4 * hidden_size
    # strides
    x_stride_row, x_stride_col,
    y_stride_row, y_stride_col,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)  # output row index in [0, num_merged_patches)
    col = tl.program_id(1)  # output col index in [0, hidden_expanded)

    # Find grid index g for this row via binary search on offsets
    low = 0
    high = num_grids
    g = 0
    while low < high:
        mid = (low + high) // 2
        off = tl.load(offsets_ptr + mid)  # int64
        if row >= off:
            low = mid + 1
        else:
            high = mid
    g = low - 1  # correct g when low == high

    # Load per-grid T,H,W
    Tg = tl.load(t_ptr + g)
    Hg = tl.load(h_ptr + g)
    Wg = tl.load(w_ptr + g)

    Hm = Hg // 2
    Wm = Wg // 2

    # Decode col into (merge_h, merge_w, c) where c in [0, hidden_size)
    C = hidden_size
    merge_h = (col // (4 * C)) % 2
    merge_w = (col // (2 * C)) % 2
    c = col // 4  # since hidden_expanded == 4*C, col % 4 == 0

    # Determine base index within grid for this merged spatial position
    base_grid = row - tl.load(offsets_ptr + g)  # int64
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

    # Store to output
    out_idx = row * hidden_expanded + col
    tl.store(y_ptr + out_idx, val)


# Kernel 3: GEMM + bias: A[M,K] @ B[K,N] + bias[N] -> C[M,N]
@triton.jit
def matmul_linear_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    eps,  # unused, but kept for signature symmetry
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + rm[:, None] * A_stride_m + rk[None, :] * A_stride_k
        b_ptrs = B_ptr + rk[:, None] * B_stride_k + rn[None, :] * B_stride_n
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + rn, mask=(rn < N), other=0.0)
    acc = acc + bias[None, :]
    c_ptrs = C_ptr + rm[:, None] * C_stride_m + rn[None, :] * C_stride_n
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Kernel 4: GELU elementwise using tanh approximation (for MLP output)
@triton.jit
def gelu_tanh_kernel(x_ptr, y_ptr, M, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    y = 0.5 * x * (1.0 + tl.tanh(c * (x + 0.044715 * x3)))
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps):
        # Ensure CUDA and contiguous
        device = hidden.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors"
        hidden = hidden.contiguous()
        grid_thw = grid_thw.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()
        fc1_weight = fc1_weight.contiguous()
        fc1_bias = fc1_bias.contiguous()
        fc2_weight = fc2_weight.contiguous()
        fc2_bias = fc2_bias.contiguous()

        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]  # 1536
        hidden_expanded = hidden_size * 4  # 6144
        # Prepare inputs for Triton kernels in fp32
        hidden_fp32 = hidden.to(torch.float32)
        ln_weight_fp32 = ln_weight.to(torch.float32)
        ln_bias_fp32 = ln_bias.to(torch.float32)

        # 1) LayerNorm + affine
        hidden_norm_fp32 = torch.empty((num_patches, hidden_size), dtype=torch.float32, device=device)
        BLOCK_C = 128
        grid_ln = (num_patches,)
        layernorm_affine_kernel[grid_ln](
            hidden_fp32,
            hidden_norm_fp32,
            ln_weight_fp32,
            ln_bias_fp32,
            hidden_size,
            eps,
            hidden_fp32.stride(0), hidden_fp32.stride(1),
            hidden_norm_fp32.stride(0), hidden_norm_fp32.stride(1),
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        # 2) Spatial reindexing: build hidden_shuffled_fp32 [num_merged_patches, hidden_expanded]
        # We need per-grid (T,H,W) and offsets. Extract num_grids from grid_thw shape.
        num_grids = grid_thw.shape[0]
        # Build t, h, w arrays [num_grids]
        t_list = grid_thw[:, 0].to(torch.int64)
        h_list = grid_thw[:, 1].to(torch.int64)
        w_list = grid_thw[:, 2].to(torch.int64)
        # Compute offsets (cumulative sums): offsets[i] = sum_{j<i} t[j]*h[j]*w[j]
        # But we need total_per_grid = t*h*w per grid; offsets is inclusive sum of previous grids.
        # We'll compute total per grid and then prefix sums.
        # total_per_grid = t*g*h*g*w*g
        total_per_grid = (t_list * h_list * w_list).to(torch.int64)
        offsets = torch.zeros(num_grids, dtype=torch.int64, device=device)
        if num_grids > 0:
            offsets[0] = 0
            for i in range(1, num_grids):
                offsets[i] = offsets[i - 1] + total_per_grid[i - 1]
        # For the last grid, offsets[-1] already covers all num_patches (by construction in get_inputs).

        # Allocate output
        # We don't know num_merged_patches a priori; however, the value is passed by get_inputs (not available here).
        # To work around, we can compute num_merged_patches by summing total_per_grid: total_patches = sum(total_per_grid)
        num_merged = int(total_per_grid.sum().item())
        hidden_shuffled_fp32 = torch.empty((num_merged, hidden_expanded), dtype=torch.float32, device=device)

        # Launch spatial reindex kernel: grid over (M=num_merged, N=hidden_expanded)
        BLOCK_N = 128
        grid_mm = (num_merged, triton.cdiv(hidden_expanded, BLOCK_N))
        spatial_reindex_kernel[grid_mm](
            hidden_norm_fp32,
            hidden_shuffled_fp32,
            offsets,
            t_list, h_list, w_list,
            num_patches,
            num_grids,
            hidden_size,
            hidden_expanded,
            hidden_norm_fp32.stride(0), hidden_norm_fp32.stride(1),
            hidden_shuffled_fp32.stride(0), hidden_shuffled_fp32.stride(1),
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )

        # 3) FC1: GEMM + bias, output [num_merged, hidden_expanded]
        M = num_merged
        K1 = hidden_expanded
        N1 = K1  # 6144
        fc1_out_fp32 = torch.empty((M, N1), dtype=torch.float32, device=device)

        BLOCK_M_fc1 = 64
        BLOCK_N_fc1 = 64
        BLOCK_K_fc1 = 32
        grid_fc1 = (triton.cdiv(M, BLOCK_M_fc1), triton.cdiv(N1, BLOCK_N_fc1))
        matmul_linear_bias_kernel[grid_fc1](
            hidden_shuffled_fp32, fc1_weight.to(torch.float32), fc1_bias.to(torch.float32), fc1_out_fp32,
            M, N1, K1,
            hidden_shuffled_fp32.stride(0), hidden_shuffled_fp32.stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            fc1_out_fp32.stride(0), fc1_out_fp32.stride(1),
            0.0,
            BLOCK_M=BLOCK_M_fc1, BLOCK_N=BLOCK_N_fc1, BLOCK_K=BLOCK_K_fc1,
            num_warps=4,
        )

        # 4) GELU activation
        fc1_out_tanh_fp32 = torch.empty_like(fc1_out_fp32, dtype=torch.float32, device=device)
        BLOCK_B = 1024
        gelu_tanh_kernel[(triton.cdiv(M * N1, BLOCK_B),)](
            fc1_out_fp32,
            fc1_out_tanh_fp32,
            M * N1,
            BLOCK=BLOCK_B,
            num_warps=4,
        )

        # 5) FC2: GEMM + bias, output [num_merged, out_hidden_size]
        out_hidden_size = fc2_weight.shape[0]  # 3584
        fc2_out_fp32 = torch.empty((M, out_hidden_size), dtype=torch.float32, device=device)

        BLOCK_M_fc2 = 64
        BLOCK_N_fc2 = 64
        BLOCK_K_fc2 = 32
        grid_fc2 = (triton.cdiv(M, BLOCK_M_fc2), triton.cdiv(out_hidden_size, BLOCK_N_fc2))
        matmul_linear_bias_kernel[grid_fc2](
            fc1_out_tanh_fp32, fc2_weight.to(torch.float32), fc2_bias.to(torch.float32), fc2_out_fp32,
            M, out_hidden_size, K1,
            fc1_out_tanh_fp32.stride(0), fc1_out_tanh_fp32.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            fc2_out_fp32.stride(0), fc2_out_fp32.stride(1),
            0.0,
            BLOCK_M=BLOCK_M_fc2, BLOCK_N=BLOCK_N_fc2, BLOCK_K=BLOCK_K_fc2,
            num_warps=4,
        )

        # Return result in bfloat16 to match the original pipeline's final dtype
        return fc2_out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
