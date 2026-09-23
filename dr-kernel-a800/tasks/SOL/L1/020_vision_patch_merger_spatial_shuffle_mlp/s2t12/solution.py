import math
import triton
import triton.language as tl

# Kernel 1: LayerNorm per-row with affine ln_weight, ln_bias. Input X [rows, hidden], output Y [rows, hidden].
@triton.jit
def layernorm_affine_kernel(X_ptr, Y_ptr, LN_W_ptr, LN_B_ptr,
                             rows, hidden, eps,
                             X_stride_row, X_stride_col,
                             Y_stride_row, Y_stride_col,
                             BLOCK: tl.constexpr):
    row = tl.program_id(0)
    # Accumulate sum and sum of squares over hidden dimension
    sum_ = 0.0
    sumsq_ = 0.0
    for k in range(0, hidden, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = (row < rows) & (offs < hidden)
        x_ptrs = X_ptr + row * X_stride_row + offs * X_stride_col
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_ += tl.sum(x, axis=0)
        sumsq_ += tl.sum(x * x, axis=0)
    mean = sum_ / hidden
    var = sumsq_ / hidden - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine
    for k in range(0, hidden, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        mask = (row < rows) & (offs < hidden)
        x_ptrs = X_ptr + row * X_stride_row + offs * X_stride_col
        y_ptrs = Y_ptr + row * Y_stride_row + offs * Y_stride_col
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        w = tl.load(LN_W_ptr + offs, mask=offs < hidden, other=1.0).to(tl.float32)
        b = tl.load(LN_B_ptr + offs, mask=offs < hidden, other=0.0).to(tl.float32)
        y = y * w + b
        tl.store(y_ptrs, y, mask=mask)

# Kernel 2: Spatial reindexing (2x2 merge) using per-grid parameters total_per_grid and offsets.
# Inputs:
#   - Xn: normalized hidden of shape [num_patches, hidden_size].
#   - grid_thw: [num_grids, 3] int64 with (T, H, W).
#   - total_per_grid: int64 vector of length num_grids (total patches per grid).
#   - offsets: int64 vector of length num_grids (cumulative starting index per grid).
# Output:
#   - Y: [num_merged_patches, hidden_size_expanded=4*hidden_size] with 2x2 spatial merges applied per grid.
@triton.jit
def spatial_shuffle_kernel(
    Xn_ptr,           # normalized hidden [num_patches, hidden_size]
    grid_thw_ptr,     # [num_grids, 3] int64
    total_pg_ptr,     # [num_grids] int64
    offsets_ptr,      # [num_grids] int64
    Y_ptr,            # [M, hidden_expanded] where M = num_merged_patches and hidden_expanded = 4 * hidden_size
    num_grids,        # int32
    hidden,           # int32
    merge_size: tl.constexpr,  # must be 2
    hidden_expanded: tl.constexpr,  # 4 * hidden
    BLOCK: tl.constexpr
):
    # 2D launch: (row r in [0, M), col j in [0, hidden_expanded))
    r = tl.program_id(0)
    j = tl.program_id(1)
    # Decode j into (merge_h, merge_w, c)
    merge_h = j // (hidden * merge_size * merge_size)      # j // (hidden * 4)
    rem = j % (hidden * merge_size * merge_size)
    merge_w = rem // (hidden * merge_size)                 # rem // (hidden * 2)
    c = rem % (hidden * merge_size)                        # within [0, 2*hidden)
    c_half = c % hidden
    c2 = c // hidden
    # Determine grid index g for row r based on total_per_grid and offsets
    # Iterate grids and find which r belongs to
    g = 0
    start = tl.load(offsets_ptr + 0)
    while g < num_grids:
        t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)  # T
        h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)  # H
        w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)  # W
        total = tl.load(total_pg_ptr + g).to(tl.int32)
        if (start <= r) & (r < start + total):
            # Compute base row index in Xn for this grid:
            # r within grid corresponds to t * (h//2) * (w//2) rows; output row index r' = r - start
            r_grid = r - start
            t_merged = t // merge_size
            h_merged = h // merge_size
            w_merged = w // merge_size
            # map r_grid to (t_merged_idx, h_merged_idx, w_merged_idx)
            num_rows = t_merged * h_merged * w_merged
            t_merged_idx = r_grid // num_rows
            rem1 = r_grid % num_rows
            h_merged_idx = rem1 // w_merged
            w_merged_idx = rem1 % w_merged
            # original patch indices
            t_idx = t_merged_idx
            h_idx = h_merged_idx * merge_size + merge_h
            w_idx = w_merged_idx * merge_size + merge_w
            patch_row = t_idx * h * w + h_idx * w + w_idx
            # load and store
            # Input index in Xn is patch_row * hidden + c2 * hidden + c_half
            x_ptrs = Xn_ptr + patch_row * hidden + c2 * hidden + c_half
            # output index in Y is r * hidden_expanded + j
            y_ptrs = Y_ptr + r * hidden_expanded + j
            val = tl.load(x_ptrs)
            tl.store(y_ptrs, val)
            break
        start = start + tl.load(total_pg_ptr + g + 1) if (g + 1) < num_grids else start
        g += 1

# Kernel 3: GEMM + bias: A[M, K] @ B[K, N] (+ Bias[N]) -> C[M, N], fp32 accumulation
@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:, None] * A_stride_m + offs_k[None, :] * A_stride_k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = B_ptr + offs_k[:, None] * B_stride_k + offs_n[None, :] * B_stride_n
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # add bias
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # store
    c_ptrs = C_ptr + offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

# Kernel 4: Elementwise GELU (tanh approximation) on X[M], output Y[M]
@triton.jit
def gelu_kernel(X_ptr, Y_ptr, M, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    # tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + c * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(Y_ptr + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float,
                total_per_grid: torch.Tensor,  # [num_grids] int64
                offsets: torch.Tensor):       # [num_grids] int64
        """
        hidden: [num_patches, hidden_size] (bfloat16 or float32), requires_contiguous
        grid_thw: [num_grids, 3] int64 (T,H,W)
        ln_weight/bias: [hidden_size] bfloat16
        fc1_weight: [hidden_size_expanded, hidden_size_expanded] bfloat16 (6144x6144)
        fc1_bias: [hidden_size_expanded] bfloat16
        fc2_weight: [out_hidden_size, hidden_size_expanded] bfloat16 (3584x6144)
        fc2_bias: [out_hidden_size] bfloat16
        eps: float
        total_per_grid: [num_grids] int64
        offsets: [num_grids] int64
        """
        assert hidden.is_cuda and grid_thw.is_cuda and fc1_weight.is_cuda and fc2_weight.is_cuda, "All tensors must be on CUDA"
        # 1) LayerNorm in Triton (fp32 accumulate, affine in fp32, write back)
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        # Ensure contiguous
        hidden = hidden.contiguous()
        ln_weight = ln_weight.to(torch.float32).contiguous()
        ln_bias = ln_bias.to(torch.float32).contiguous()

        hidden_norm = torch.empty_like(hidden, dtype=torch.float32)
        # Launch layernorm kernel
        BLOCK = 1024  # works for hidden_size=1536
        grid_layernorm = (num_patches,)
        layernorm_affine_kernel[grid_layernorm](
            hidden, hidden_norm, ln_weight, ln_bias,
            num_patches, hidden_size, eps,
            hidden.stride(0), hidden.stride(1),
            hidden_norm.stride(0), hidden_norm.stride(1),
            BLOCK=BLOCK
        )

        # 2) Spatial shuffle using Triton
        num_grids = grid_thw.shape[0]
        # Compute num_merged_patches from offsets (last offset gives total)
        last_offset = offsets[-1].item()
        num_merged_patches = last_offset
        hidden_expanded = 4 * hidden_size  # merge_size = 2
        hidden_shuffled = torch.empty((num_merged_patches, hidden_expanded), device=hidden.device, dtype=torch.float32)
        grid_spatial = (num_merged_patches, hidden_expanded)
        # merge_size is constexpr 2 in kernel signature
        spatial_shuffle_kernel[grid_spatial](
            hidden_norm, grid_thw, total_per_grid, offsets, hidden_shuffled,
            num_grids, hidden_size,
            merge_size=2, hidden_expanded=hidden_expanded,
            BLOCK=1024
        )

        # 3) FC1: Triton GEMM + bias
        M = hidden_shuffled.shape[0]  # num_merged_patches
        K = hidden_shuffled.shape[1]  # 6144
        N1 = fc1_weight.shape[0]      # 6144
        # Ensure matrices are contiguous
        A = hidden_shuffled.contiguous()
        B = fc1_weight.contiguous()
        Bias1 = fc1_bias.to(torch.float32).contiguous()

        C1 = torch.empty((M, N1), device=hidden.device, dtype=torch.float32)
        # Tile sizes tuned for these dimensions
        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 64
        grid_fc1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        matmul_bias_kernel[grid_fc1](
            A, B, Bias1, C1,
            M, N1, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C1.stride(0), C1.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        # 4) GELU (Triton)
        C1_gelu = torch.empty_like(C1, dtype=torch.float32)
        BLOCK_GELU = 1024
        grid_gelu = (triton.cdiv(M * N1, BLOCK_GELU),)
        gelu_kernel[grid_gelu](C1_gelu, C1_gelu, M * N1, BLOCK=BLOCK_GELU)

        # 5) FC2: Triton GEMM + bias
        N2 = fc2_weight.shape[0]      # 3584
        C2 = torch.empty((M, N2), device=hidden.device, dtype=torch.float32)

        A2 = C1_gelu.reshape(M * N1)  # flatten for gelu kernel launch; but we need A as [M, N1] for matmul
        # Reuse C1_gelu as [M, N1] (we applied GELU to C1 and stored back to C1_gelu)
        # Here C1_gelu is still [M, N1] from matmul output; apply GELU to that [M, N1]
        # But we already applied GELU to C1_gelu above; correct approach: write GELU output to a new tensor and then use it for matmul.
        # Fix: recompute GELU using torch.nn.functional.gelu(C1, approximate='tanh')? However, we must keep Triton-only.
        # To keep Triton-only, we can either:
        #   a) compute GELU with Triton by launching gelu_kernel on C1 row-wise, but that would require row-wise tiling and launching a 1D grid which is possible, but not already defined; to keep code compact, we can implement GELU via PyTorch since forward must be correct, or
        #   b) note that we already applied GELU to C1_gelu in previous step; C1_gelu is [M, N1] with GELU; we can use it directly.
        # Correction: We applied GELU to C1_gelu in the previous step, so C1_gelu is the post-GELU tensor. Now we need A for fc2 to be the post-GELU tensor. Let's create A2 = C1_gelu (we'll launch gelu per row directly).
        # However, we only have gelu_kernel defined for 1D. To keep strict Triton-only, we implement GELU inside matmul by keeping intermediate in fp32 and recompute. Easiest: recompute GELU in Triton by launching over rows and columns in chunks; since we need elementwise over M*N1, we can launch a 1D grid.

        # Recompute GELU via Triton elementwise on C1 (previously we wrote GELU to C1_gelu; to avoid confusion, we will launch gelu_kernel on C1 directly and write result into a new tensor G. Since we don't have G allocated yet, we will perform GELU in-place on C1 by using a temporary tensor. For simplicity, we will do GELU using torch for correctness, but the requirement is Triton-only. To adhere, we will implement GELU via Triton by flattening and launching a 1D kernel.
        # Define a helper to apply GELU Triton elementwise: we can do Y = gelu(C1) in a separate kernel launch. We'll create GELU output tensor and launch.

        # Launch GELU Triton elementwise kernel on C1 (post-FC1) into a new tensor GELU_out
        # Allocate GELU_out
        gelu_out = torch.empty_like(C1, dtype=torch.float32)
        # Flatten for elementwise kernel
        total_elems = M * N1
        grid_gelu_rows = (triton.cdiv(total_elems, BLOCK_GELU),)
        gelu_kernel[grid_gelu_rows](C1, gelu_out, total_elems, BLOCK=BLOCK_GELU)

        # Now gelu_out contains the GELU of C1 (post-FC1). Use it as input for fc2
        A_for_fc2 = gelu_out.contiguous()

        # Prepare fc2 bias as fp32
        Bias2 = fc2_bias.to(torch.float32).contiguous()

        # FC2 matmul + bias
        grid_fc2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N2, BLOCK_N))
        matmul_bias_kernel[grid_fc2](
            A_for_fc2, fc2_weight, Bias2, C2,
            M, N2, N1,
            A_for_fc2.stride(0), A_for_fc2.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            C2.stride(0), C2.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        # Output should be fp32. If original hidden was bfloat16, the reference returns fp32 after GEMMs; our result is fp32. Cast to bfloat16 only if required by environment; here we keep fp32 for accuracy.
        return C2


def run(*args):
    return ModelNew()(*args)
