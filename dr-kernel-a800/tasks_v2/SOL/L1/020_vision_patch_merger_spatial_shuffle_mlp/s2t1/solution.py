import torch
import triton
import triton.language as tl

# 1) LayerNorm kernel: per-row mean/var, affine, store fp32 result
@triton.jit
def layernorm_affine_kernel(
    x_ptr,          # *fp32, [num_patches, hidden_size]
    y_ptr,          # *fp32, [num_patches, hidden_size] output
    weight_ptr,     # *fp32, [hidden_size]
    bias_ptr,       # *fp32, [hidden_size]
    N,              # int, hidden_size
    eps,            # fp32
    x_stride_row, x_stride_col,  # strides for x
    y_stride_row, y_stride_col,  # strides for y
    BLOCK_C: tl.constexpr
):
    row = tl.program_id(0)  # each program handles one row
    # compute mean
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

    # compute normalized + affine and store
    for c in range(0, N, BLOCK_C):
        offs = c + tl.arange(0, BLOCK_C)
        mask = offs < N
        x = tl.load(x_ptr + row * x_stride_row + offs * x_stride_col, mask=mask, other=0.0)
        y = (x - mean) * rstd
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        y = y * w + b
        tl.store(y_ptr + row * y_stride_row + offs * y_stride_col, y, mask=mask)


# 2) Spatial reindexing kernel: hidden_norm_fp32 -> hidden_shuffled_fp32
@triton.jit
def spatial_reindex_kernel(
    x_ptr,           # *fp32, [num_patches, hidden_size] normalized input
    y_ptr,           # *fp32, [num_merged_patches, hidden_size_expanded] output
    offsets_ptr,     # *int64, [num_grids] cumulative counts per grid
    num_grids,       # int
    num_patches,     # int
    t_ptr, h_ptr, w_ptr,        # *int64, [num_grids] per-grid T,H,W
    hidden_size,             # int
    hidden_expanded,         # int, hidden_size_expanded = 6144
    # strides
    x_stride_row, x_stride_col,
    y_stride_row, y_stride_col,
):
    row = tl.program_id(0)  # merged patch index
    col = tl.program_id(1)  # output column in [0, hidden_expanded)

    # Find grid index g for this row via binary search on offsets
    # offsets[g] <= row < offsets[g+1]
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
    g = low - 1  # when low == high, g is correct

    # Load T,H,W for this grid
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
    # Each grid has total patches = Tg * Hm * Wm
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


# 3) GEMM + bias kernel: A[M,K] @ B[K,N] + bias[N] -> C[M,N]
@triton.jit
def matmul_linear_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * A_stride_m + offs_k[None, :] * A_stride_k, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * B_stride_k + offs_n[None, :] * B_stride_n, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    # add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc += bias[None, :]
    # store
    tl.store(C_ptr + offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 4) GELU (tanh approximation) kernel: elementwise on fp32
@triton.jit
def gelu_tanh_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; everything is computed in kernels

    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        """
        Compute the entire pipeline using Triton kernels:
        - LayerNorm on hidden (fp32 compute, store fp32)
        - Spatial reindexing to produce hidden_shuffled (fp32)
        - FC1: GEMM + bias (fp32)
        - GELU activation (fp32, tanh-approx)
        - FC2: GEMM + bias (fp32)
        - Cast final output to bfloat16 and return
        """
        assert hidden.is_cuda and grid_thw.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda \
               and fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, \
            "All tensors must be on CUDA device."

        # Ensure contiguity
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
        hidden_size_expanded = 6144  # 4 * hidden_size
        out_hidden_size = fc2_weight.shape[0]  # 3584

        # 1) LayerNorm (fp32 compute), output in fp32
        hidden_fp32 = hidden.to(torch.float32)
        hidden_norm_fp32 = torch.empty((num_patches, hidden_size), dtype=torch.float32, device=hidden.device)

        # Prepare strides
        x_stride_row = hidden_fp32.stride(0)
        x_stride_col = hidden_fp32.stride(1)
        y_stride_row = hidden_norm_fp32.stride(0)
        y_stride_col = hidden_norm_fp32.stride(1)

        BLOCK_C = 128
        layernorm_affine_kernel[(num_patches,)](
            hidden_fp32, hidden_norm_fp32,
            ln_weight.to(torch.float32), ln_bias.to(torch.float32),
            hidden_size, eps,
            x_stride_row, x_stride_col,
            y_stride_row, y_stride_col,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        # 2) Spatial reindexing: compute offsets and launch kernel
        num_grids = grid_thw.shape[0]
        # Prepare t,h,w as tensors for kernel
        t_list = grid_thw[:, 0].to(torch.int64).contiguous()
        h_list = grid_thw[:, 1].to(torch.int64).contiguous()
        w_list = grid_thw[:, 2].to(torch.int64).contiguous()

        # Compute offsets: cumulative sum of patches per grid
        patches_per_grid = (t_list * h_list * w_list).tolist()
        offsets_list = []
        total_prev = 0
        for p in patches_per_grid:
            offsets_list.append(total_prev)
            total_prev += p
        # Total patches should equal num_patches
        assert total_prev == num_patches, "Sum of per-grid patches must equal num_patches."
        offsets = torch.tensor(offsets_list, dtype=torch.int64, device=hidden.device)

        # Allocate output for shuffled hidden
        hidden_shuffled_fp32 = torch.empty((num_patches, hidden_size_expanded), dtype=torch.float32, device=hidden.device)

        # Launch 2D grid: rows=M=num_patches, cols=N=hidden_size_expanded
        M_merged = num_patches  # we don't know exact M_merged ahead; we can't precompute it from grid_thw without splitting.
        # But we can compute M_merged by summing t*(H//2)*(W//2) per grid. Let's do it in a loop to build sizes.
        # However, since we don't have t,h,w per grid in the normal sense, we rely on the fact that grid_thw is per-grid.
        # We need to know M_merged; the original code constructs it. We can infer it from the fact that hidden_shuffled
        # comes from per-grid patches. Since we cannot predict M_merged without knowing how hidden_norm is split, we
        # instead build hidden_shuffled by directly mapping rows. We'll compute the total number of rows as the sum
        # of t * (H//2) * (W//2) for each grid. Let's do that in Python, and then launch the kernel with grid=(M_merged, hidden_size_expanded).

        # Compute M_merged (num_merged_patches) as sum over grids
        M_merged = 0
        for t, h, w in zip(t_list, h_list, w_list):
            M_merged += t * (h // 2) * (w // 2)

        # Launch spatial reindex kernel with grid (M_merged, hidden_size_expanded)
        # Note: We need to decide how to map output rows to input rows because we don't have a prebuilt ordering.
        # The original logic assigns rows in order of grids; since we don't have the exact ordering, we recompute M_merged
        # and rely on the kernel using offsets and per-grid t,h,w to map each row properly. We'll just run the kernel
        # with grid (M_merged, hidden_size_expanded). Triton kernel will pick the correct g via offsets binary search
        # for each row index.

        spatial_reindex_kernel[(M_merged, hidden_size_expanded)](
            hidden_norm_fp32, hidden_shuffled_fp32,
            offsets, num_grids, num_patches, t_list, h_list, w_list,
            hidden_size, hidden_size_expanded,
            hidden_norm_fp32.stride(0), hidden_norm_fp32.stride(1),
            hidden_shuffled_fp32.stride(0), hidden_shuffled_fp32.stride(1),
            num_warps=4,
        )

        # 3) FC1: GEMM + bias (fp32)
        M = hidden_shuffled_fp32.shape[0]
        K = hidden_shuffled_fp32.shape[1]  # 6144
        N1 = fc1_weight.shape[1]  # 6144
        fc1_out_fp32 = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)

        # Tile sizes
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_fc1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        matmul_linear_kernel[grid_fc1](
            hidden_shuffled_fp32, fc1_weight.to(torch.float32), fc1_bias.to(torch.float32), fc1_out_fp32,
            M, N1, K,
            hidden_shuffled_fp32.stride(0), hidden_shuffled_fp32.stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            fc1_out_fp32.stride(0), fc1_out_fp32.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # 4) GELU activation (tanh approximation) fp32
        fc1_gelu_fp32 = torch.empty_like(fc1_out_fp32)
        BLOCK_GELU = 1024
        grid_gelu = (triton.cdiv(N1, BLOCK_GELU),)
        gelu_tanh_kernel[grid_gelu](fc1_out_fp32, fc1_gelu_fp32, N1, BLOCK=BLOCK_GELU, num_warps=4)

        # 5) FC2: GEMM + bias (fp32)
        N2 = fc2_weight.shape[1]  # 6144
        output_fp32 = torch.empty((M, N2), dtype=torch.float32, device=hidden.device)
        grid_fc2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N2, BLOCK_N))
        matmul_linear_kernel[grid_fc2](
            fc1_gelu_fp32, fc2_weight.to(torch.float32), fc2_bias.to(torch.float32), output_fp32,
            M, N2, N1,  # here N1 is the input dimension for fc2 which is K=6144
            fc1_gelu_fp32.stride(0), fc1_gelu_fp32.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            output_fp32.stride(0), output_fp32.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # Cast final output to bfloat16 and return (to match original pipeline's dtype)
        return output_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
