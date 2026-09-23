import torch
import math
import triton
import triton.language as tl

# Triton kernel: LayerNorm per row. One program per row.
# x_ptr: *bf16, shape [N, C], row-major
# y_ptr: *bf16, shape [N, C]
# ln_weight_ptr, ln_bias_ptr: *bf16, shape [C]
# eps: float32
@triton.jit
def _layer_norm_rows_kernel(x_ptr, y_ptr, ln_weight_ptr, ln_bias_ptr,
                             N, C, eps,
                             BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= N:
        return

    x_row_ptr = x_ptr + row * C
    y_row_ptr = y_ptr + row * C

    # Pass 1: compute mean and variance in fp32
    sum_val = 0.0
    sum_sq = 0.0
    col = 0
    while col < C:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        col += BLOCK_SIZE

    mean = sum_val / C
    var = sum_sq / C
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine
    col = 0
    while col < C:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


# Triton kernel: Exact 2x2 spatial shuffle per grid, writing to [num_merged_patches, 4*C]
# hidden_ptr: *bf16, shape [total_patches, C], contiguous
# grid_thw_ptr: *int64, shape [NUM_GRIDS, 3], rows [t, h, w]
# out_ptr: *bf16, shape [total_merged_rows, 4*C], contiguous
@triton.jit
def _shuffle_2x2_per_grid_vec_kernel(hidden_ptr, grid_thw_ptr, out_ptr,
                                     total_patches, C, NUM_GRIDS,
                                     BLOCK_ROWS: tl.constexpr):
    """
    For each grid g, we compute all merged rows out_row in [0, t*h_merged*w_merged),
    where h_merged = h // 2, w_merged = w // 2, and write the four 2x2 positions
    into out[4*C] contiguous columns.
    """
    g = tl.program_id(0)
    if g >= NUM_GRIDS:
        return

    # Load grid dims
    t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)

    h_merged = h // 2
    w_merged = w // 2
    num_merged_rows = t * h_merged * w_merged

    # Vector of output rows for this grid
    out_rows = tl.arange(0, BLOCK_ROWS)
    mask_out = out_rows < num_merged_rows

    # Decode out_rows -> (t_index, h2, w2)
    t_index = out_rows // (h_merged * w_merged)
    hw_rem = out_rows % (h_merged * w_merged)
    h2 = hw_rem // w_merged
    w2 = hw_rem % w_merged

    # Corresponding original patch indices m
    # m = t_index * (h*w) + (h2*2) * w + (w2*2)
    m = t_index * (h * w) + (h2 * 2) * w + (w2 * 2)

    # Source offsets into hidden: since hidden is [total_patches, C], row idx is m, columns 0..C-1
    src_row_base = m * C  # each original patch has C columns; we need to copy 4 values per patch

    # Indices for the four 2x2 positions: (r=0,c=0), (0,1), (1,0), (1,1)
    # For each, compute source column index = cdim (0..C-1), and source row = m
    # Store into out at rows 'out_rows', columns:
    #   0: (h2*2)*w*C + (w2*2)*C
    #   1: (h2*2)*w*C + (w2*2+1)*C
    #   2: (h2*2+1)*w*C + (w2*2)*C
    #   3: (h2*2+1)*w*C + (w2*2+1)*C
    for cdim in range(4 * C):  # we only use cdim < C; since C is 1536, we keep 4*C columns per row
        # Compute which 2x2 idx we are writing (0..3)
        idx2x2 = cdim // C  # should be 0..3
        if idx2x2 < 4:
            # src_h_offset and src_w_offset depending on idx2x2
            if idx2x2 == 0:
                src_h_offset = 0
                src_w_offset = 0
            elif idx2x2 == 1:
                src_h_offset = 0
                src_w_offset = 1
            elif idx2x2 == 2:
                src_h_offset = 1
                src_w_offset = 0
            else:
                src_h_offset = 1
                src_w_offset = 1

            # Source row (original patch) is m
            # Source column in that patch: src_h_offset * w + src_w_offset
            src_col = src_h_offset * w + src_w_offset  # integer
            # Compute absolute source index in hidden: row = m, col = src_col * C + cdim
            src_abs = src_row_base + src_col * C + (cdim % C)

            # Destination row = out_rows, col = idx2x2 * C + cdim % C
            dst_col = idx2x2 * C + (cdim % C)

            val = tl.load(hidden_ptr + src_abs, mask=mask_out, other=0.0).to(tl.bfloat16)
            tl.store(out_ptr + out_rows * (4 * C) + dst_col, val, mask=mask_out)


# Triton kernel: elementwise GELU on x, write to y. x,y are [N*C] flattened.
@triton.jit
def _gelu_kernel(x_ptr, y_ptr, N, C, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= N:
        return
    x_row_ptr = x_ptr + row * C
    y_row_ptr = y_ptr + row * C
    col = 0
    while col < C:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
        z = x * inv_sqrt2
        # GELU: 0.5*x*(1 + erf(z))
        # Triton may not have erf directly; implement approximation via tl.math.erf if available,
        # otherwise use tanh approximation.
        # Using tanh approximation for robustness:
        # gelu(x) ≈ 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x^3)))
        c = 0.044715
        sqrt_2_over_pi = 0.7978845608028654
        x3 = x * x * x
        t = sqrt_2_over_pi * (x + c * x3)
        gelu = 0.5 * x * (1.0 + tl.math.tanh(t))
        tl.store(y_row_ptr + offs, gelu.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


# Triton GEMM kernel: each program computes one output row and reduces across K.
# x_ptr: *bf16, shape [N, K_in], row-major
# w_ptr: *bf16, shape [K_out, K_in], row-major (note: PyTorch weight is [K_out, K_in])
# b_ptr: *bf16, shape [K_out]
# y_ptr: *bf16, shape [N, K_out], row-major
@triton.jit
def _gemm_row_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                     N, K_in, K_out, eps,  # eps unused, kept for signature compatibility
                     BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    if row >= N:
        return
    x_row_ptr = x_ptr + row * K_in
    y_row_ptr = y_ptr + row * K_out

    acc = tl.zeros([K_out], dtype=tl.float32)

    k = 0
    while k < K_in:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_in
        x = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]

        # For each output column (we iterate K_out outside per program, but here we accumulate)
        # We will do a simple loop over k block and accumulate per output element.
        # However Triton requires vectorized ops; better approach: loop over k block and multiply
        # into a scalar accumulator. To vectorize, we'd need a 2D tiling which is more complex.
        # Here, we keep the accumulation per output element inside the kernel by looping k in steps
        # of BLOCK_K. But Triton does not support dynamic loops over K_out here easily. Instead,
        # we implement a specialized kernel per output dimension. Given the provided sizes, we can
        # compile with constexpr for K_out as needed. For clarity and correctness, we'll launch
        # separate kernels for K_out=6144 and K_out=3584, passing K_out as constexpr.

        # The above comment is a reminder: for robustness, we'll implement two kernels specialized
        # for K_out values used in the problem: 6144 and 3584. We cannot pass runtime K_out into
        # this kernel as a constexpr in general. Therefore, we provide two @triton.jit variants below.


# We'll implement two variants for Linear1 and Linear2. To keep the file self-contained, we define
# the two specialized kernels below, using Python to select the appropriate signature at call time.

# Specialized Triton kernel for Linear1: K_in=6144, K_out=6144
@triton.jit
def _gemm_row_linear1_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                             N, K_in, K_out=6144, BLOCK_K: tl.constexpr=128):
    """
    Compute y[row, :] = x[row, :] @ w[:, :]^T + b
    where x: [N, 6144], w: [6144, 6144], y: [N, 6144]
    """
    row = tl.program_id(0)
    if row >= N:
        return
    x_row_ptr = x_ptr + row * K_in
    y_row_ptr = y_ptr + row * K_out

    acc = tl.zeros([K_out], dtype=tl.float32)

    k = 0
    while k < K_in:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_in
        x = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load w block as [BLOCK_K, K_out]: w[k+offs_k, :]
        # Pointer arithmetic: w_ptr has shape [K_out, K_in], row-major. Index for w is (row_w, col_w).
        # We want rows k+offs_k and columns all K_out. We'll iterate over output columns in chunks
        # and accumulate. But per this kernel design, we compute acc by multiplying x with each
        # output column. Triton does not support dynamic vectorization over K_out here; instead,
        # we perform a scalar accumulation per output column via Python loop when launching.
        # Therefore, to keep it correct, we will call this kernel in a loop over output columns
        # using PyTorch-like indexing, but since Triton kernels run in parallel, we cannot loop here.
        # Hence, we provide a second specialized kernel below that uses vectorized accumulation.
        # This comment emphasizes the design: we need a vectorized kernel to compute all outputs
        # per row at once. Since Triton does not let us vectorize over K_out inside a single
        # program easily, we use a Python wrapper to launch multiple programs per output column.
        # For simplicity and correctness, we restructure the forward to call a higher-level wrapper
        # that uses two kernels: one to compute intermediate and one to produce output. Here, we
        # implement the linear as two-phase: we can't; so we instead compute via torch.matmul in host
        # for correctness and speed. But the requirement is Triton-only. Therefore, we implement
        # a 2D tiling kernel below.

# Better approach: implement 2D tiling GEMM in Triton for these sizes. Below is the 2D tiling kernel.

@triton.jit
def _gemm_2d_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                    N, K_in, K_out,
                    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Compute y[M, N] = x[M, K] @ w[K, N] + b[N]
    We'll launch grid=(ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N)).
    Each program handles a BLOCK_M x BLOCK_N tile.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Accumulator for this tile
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    k_start = 0
    while k_start < K_in:
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K_in

        # Load x tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + (m_start + tl.arange(0, BLOCK_M))[:, None] * K_in + k_offsets[None, :]
        x_mask = (m_start + tl.arange(0, BLOCK_M))[:, None] < N
        x_mask = x_mask & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # Load w tile [BLOCK_K, BLOCK_N]
        w_ptrs = w_ptr + k_offsets[:, None] * K_out + (n_start + tl.arange(0, BLOCK_N))[None, :]
        w_mask = mask_k[:, None] & (n_start + tl.arange(0, BLOCK_N))[None, :] < K_out
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        # Triton supports tl.dot for these shapes
        acc += tl.dot(x_tile, w_tile)

        k_start += BLOCK_K

    # Add bias b[N] to each column
    b_cols = tl.load(b_ptr + (n_start + tl.arange(0, BLOCK_N)), mask=(n_start + tl.arange(0, BLOCK_N)) < K_out, other=0.0).to(tl.float32)  # [BLOCK_N]
    acc += b_cols[None, :]

    # Store results to y[M, N]
    y_ptrs = y_ptr + (m_start + tl.arange(0, BLOCK_M))[:, None] * K_out + (n_start + tl.arange(0, BLOCK_N))[None, :]
    y_mask = (m_start + tl.arange(0, BLOCK_M))[:, None] < N & (n_start + tl.arange(0, BLOCK_N))[None, :] < K_out
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=y_mask)


# Now, ModelNew will use Triton for LayerNorm, SpatialShuffle, GELU, and GEMM.
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we rely on inputs passed at forward time.

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
        hidden: [num_patches, hidden_size] (bfloat16), row-major
        grid_thw: [num_grids, 3] (int64), rows [t, h, w]
        ln_weight, ln_bias: [hidden_size] (bfloat16)
        fc1_weight: [hidden_size_expanded, hidden_size_expanded] (bfloat16) = [6144, 6144]
        fc1_bias: [hidden_size_expanded] (bfloat16)
        fc2_weight: [out_hidden_size, hidden_size_expanded] (bfloat1536), we'll compute
        hidden_size = 1536, hidden_size_expanded = 6144, out_hidden_size = 3584
        eps: float
        """
        device = hidden.device
        dtype = hidden.dtype

        # 1) LayerNorm (per-row) in Triton
        num_patches, hidden_size = hidden.shape
        hidden_norm = torch.empty_like(hidden)
        # Ensure contiguous
        hidden_c = hidden.contiguous()
        ln_w = ln_weight.contiguous()
        ln_b = ln_bias.contiguous()

        # Choose BLOCK_SIZE as constexpr: 256 works well for hidden_size=1536
        BLOCK_SIZE = 256
        grid_ln = (num_patches,)
        _layer_norm_rows_kernel[grid_ln](hidden_c, hidden_norm, ln_w, ln_b,
                                         num_patches, hidden_size, eps,
                                         BLOCK_SIZE=BLOCK_SIZE,
                                         num_warps=4, num_stages=2)

        # 2) Spatial shuffle: exact 2x2 merge per grid, write to [total_merged_rows, 4*C]
        # First compute total_num_merged_patches for output allocation
        # We need to iterate grids to compute total
        total_merged_rows = 0
        for g in range(grid_thw.shape[0]):
            t = int(grid_thw[g, 0].item())
            h = int(grid_thw[g, 1].item())
            w = int(grid_thw[g, 2].item())
            total_merged_rows += t * (h // 2) * (w // 2)

        hidden_norm_c = hidden_norm.contiguous()
        grid_thw_c = grid_thw.to(torch.int64).contiguous()
        out_cols = 4 * hidden_size
        hidden_shuffled = torch.empty((total_merged_rows, out_cols), dtype=torch.bfloat16, device=device)

        # Launch Triton kernel once; it computes for all grids. To be generic, we can iterate
        # or simply pass total_patches and let it decode per grid. Here we pass total_patches but
        # kernel doesn't need it; we set NUM_GRIDS and let it read grid_thw rows.
        NUM_GRIDS = grid_thw.shape[0]
        # Choose BLOCK_ROWS as constexpr, e.g., 2048; we mask beyond actual rows
        BLOCK_ROWS = 2048
        grid_shuffle = (NUM_GRIDS,)
        _shuffle_2x2_per_grid_vec_kernel[grid_shuffle](hidden_norm_c, grid_thw_c, hidden_shuffled,
                                                       num_patches, hidden_size, NUM_GRIDS,
                                                       BLOCK_ROWS=BLOCK_ROWS,
                                                       num_warps=4, num_stages=2)

        # 3) GELU activation (elementwise) on hidden_shuffled
        gelu_out = torch.empty_like(hidden_shuffled)
        N = hidden_shuffled.shape[0]
        C_out = hidden_shuffled.shape[1]
        # Launch one program per row
        grid_gelu = (N,)
        _gelu_kernel[grid_gelu](hidden_shuffled, gelu_out, N, C_out, BLOCK_SIZE=256,
                                num_warps=4, num_stages=2)

        # 4) Linear1: gelu_out @ fc1_weight.T + fc1_bias
        # Triton GEMM via 2D tiling kernel. Shapes:
        # gelu_out: [N, 4*hidden_size] = [num_merged_patches, 6144]
        # fc1_weight: [6144, 6144], fc1_bias: [6144]
        # Output: [N, 6144]
        N = gelu_out.shape[0]
        K_in = 6144
        K_out1 = 6144
        lin1_out = torch.empty((N, K_out1), dtype=torch.bfloat16, device=device)

        # Make sure inputs are contiguous and dtype is bf16
        gelu_c = gelu_out.contiguous()
        w1 = fc1_weight.contiguous()
        b1 = fc1_bias.contiguous()

        # Launch 2D GEMM kernel with tiling. Choose BLOCK sizes suitable for 6144
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 128
        grid_m = triton.cdiv(N, BLOCK_M)
        grid_n = triton.cdiv(K_out1, BLOCK_N)
        _gemm_2d_kernel[(grid_m, grid_n)](gelu_c, w1, b1, lin1_out,
                                          N, K_in, K_out1,
                                          BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
                                          num_warps=4, num_stages=3)

        # 5) GELU activation on lin1_out
        gelu_lin1 = torch.empty_like(lin1_out)
        N1 = lin1_out.shape[0]
        C1 = lin1_out.shape[1]
        grid_gelu2 = (N1,)
        _gelu_kernel[grid_gelu2](lin1_out, gelu_lin1, N1, C1, BLOCK_SIZE=256,
                                 num_warps=4, num_stages=2)

        # 6) Linear2: gelu_lin1 @ fc2_weight.T + fc2_bias
        # gelu_lin1: [N1, 6144], fc2_weight: [3584, 6144], fc2_bias: [3584]
        N2 = gelu_lin1.shape[0]
        K_in2 = 6144
        K_out2 = 3584
        out = torch.empty((N2, K_out2), dtype=torch.bfloat16, device=device)

        gelu_lin1_c = gelu_lin1.contiguous()
        w2 = fc2_weight.contiguous()
        b2 = fc2_bias.contiguous()

        # Launch 2D GEMM for this case
        BLOCK_M2 = 128
        BLOCK_N2 = 128
        BLOCK_K2 = 128
        grid_m2 = triton.cdiv(N2, BLOCK_M2)
        grid_n2 = triton.cdiv(K_out2, BLOCK_N2)
        _gemm_2d_kernel[(grid_m2, grid_n2)](gelu_lin1_c, w2, b2, out,
                                            N2, K_in2, K_out2,
                                            BLOCK_M=BLOCK_M2, BLOCK_K=BLOCK_K2, BLOCK_N=BLOCK_N2,
                                            num_warps=4, num_stages=3)

        return out


def run(*args):
    return ModelNew()(*args)
