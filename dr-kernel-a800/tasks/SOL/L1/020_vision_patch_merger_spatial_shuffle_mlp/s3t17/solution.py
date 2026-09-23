import math
import torch
import triton
import triton.language as tl


# Triton kernel: LayerNorm per row. One program per row.
@triton.jit
def _layer_norm_kernel(x_ptr, y_ptr, ln_weight_ptr, ln_bias_ptr,
                        N, C, eps,
                        BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    y_ptr: *bf16, shape [N, C], output
    ln_weight_ptr, ln_bias_ptr: *bf16, shape [C]
    eps: float32
    Each program handles one row: i in [0, N).
    """
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

    # Pass 2: normalize and apply affine, then store bfloat16
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


# Triton kernel: per-grid 2x2 patch merge producing [num_merged_rows, 4*C] for given grid dims.
@triton.jit
def _shuffle_2x2_per_grid_kernel(hidden_ptr, grid_thw_ptr, out_grid_ptr,
                                 total_patches, C, NUM_GRIDS,
                                 g, NUM_OUT_ROWS_G,
                                 BLOCK_M: tl.constexpr):
    """
    hidden_ptr: *bf16, flattened [total_patches, C]
    grid_thw_ptr: *int64, shape [NUM_GRIDS, 3], each row is [t, h, w]
    out_grid_ptr: *bf16, flattened [NUM_OUT_ROWS_G, 4*C] (output for this grid)
    We launch one program per grid and write its output into out_grid_ptr.
    """
    # Load grid dimensions
    t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)

    h_merged = h // 2
    w_merged = w // 2
    num_merged_rows = t * h_merged * w_merged

    # For each original patch m in [0, t*h*w):
    # Decode (t_index, h2, w2), then out_row = t_index * (h_merged * w_merged) + (h2//2) * w_merged + (w2//2)
    # Copy four 2x2 positions into the four contiguous columns of out.
    m = 0
    while m < t * h * w:
        t_index = m // (h * w)
        rem = m % (h * w)
        h2 = rem // w
        w2 = rem % w

        out_row = t_index * (h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)
        base_out = out_grid_ptr + out_row * (4 * C)

        # idx=0: (r=0,c=0) -> hidden[t_index, h2, w2]
        idx0 = 0
        col0 = (h2 * 2 + 0) * w_merged * C + (w2 * 2 + 0) * C  # = 0 * C
        src_offset0 = t_index * (h * w) + h2 * w + w2
        val0 = tl.load(hidden_ptr + src_offset0 * C + idx0 * C, mask=True, other=0.0).to(tl.bfloat16)
        tl.store(base_out + idx0 * C, val0)

        # idx=1: (r=0,c=1) -> hidden[t_index, h2, w2+1]
        idx1 = 1
        col1 = (h2 * 2 + 0) * w_merged * C + (w2 * 2 + 1) * C
        if w2 + 1 < w:
            src_offset1 = t_index * (h * w) + h2 * w + (w2 + 1)
            val1 = tl.load(hidden_ptr + src_offset1 * C + idx1 * C, mask=True, other=0.0).to(tl.bfloat16)
        else:
            val1 = tl.zeros((), dtype=tl.bfloat16)
        tl.store(base_out + idx1 * C, val1)

        # idx=2: (r=1,c=0) -> hidden[t_index, h2+1, w2]
        idx2 = 2
        col2 = (h2 * 2 + 1) * w_merged * C + (w2 * 2 + 0) * C
        if h2 + 1 < h and w2 < w:
            src_offset2 = t_index * (h * w) + (h2 + 1) * w + w2
            val2 = tl.load(hidden_ptr + src_offset2 * C + idx2 * C, mask=True, other=0.0).to(tl.bfloat16)
        else:
            val2 = tl.zeros((), dtype=tl.bfloat16)
        tl.store(base_out + idx2 * C, val2)

        # idx=3: (r=1,c=1) -> hidden[t_index, h2+1, w2+1]
        idx3 = 3
        col3 = (h2 * 2 + 1) * w_merged * C + (w2 * 2 + 1) * C
        if h2 + 1 < h and w2 + 1 < w:
            src_offset3 = t_index * (h * w) + (h2 + 1) * w + (w2 + 1)
            val3 = tl.load(hidden_ptr + src_offset3 * C + idx3 * C, mask=True, other=0.0).to(tl.bfloat16)
        else:
            val3 = tl.zeros((), dtype=tl.bfloat16)
        tl.store(base_out + idx3 * C, val3)

        m += 1


# Triton kernel: Row-wise Linear (GEMM) y = x @ W.T + b
# Specialized to K=6144 and output dim C_OUT. Each program computes one output row.
@triton.jit
def _row_linear_kernel(x_ptr, W_ptr, b_ptr, y_ptr,
                        N, C_OUT, K,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, K], row-major (i.e., x[i, j] at offset i*K + j)
    W_ptr: *bf16, shape [K, C_OUT], row-major for W[k, j] -> base at k*C_OUT + j
    b_ptr: *bf16, shape [C_OUT]
    y_ptr: *bf16, shape [N, C_OUT], row-major
    Each program computes one row i_out.
    """
    i_out = tl.program_id(0)
    if i_out >= N:
        return

    y_row_ptr = y_ptr + i_out * C_OUT

    k_start = 0
    acc = tl.zeros((C_OUT,), dtype=tl.float32)
    while k_start < K:
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_vec = tl.load(x_ptr + i_out * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]
        W_vec = tl.load(W_ptr + offs_k * C_OUT + 0, mask=mask_k, other=0.0).to(tl.float32)  # we need j=0..C_OUT-1, but load W[k, j] across j
        # To get acc += sum_j x_vec[j] * W[j, :] for j in C_OUT, we need W for all j. Triton doesn't support broadcasting of W across j here in one vector.
        # Instead, loop over j in tiles of BLOCK_N:
        j_start = 0
        while j_start < C_OUT:
            offs_j = j_start + tl.arange(0, BLOCK_N)
            mask_j = offs_j < C_OUT
            # Build a [BLOCK_K, BLOCK_N] tile for x and W:
            x_tile = tl.load(x_ptr + i_out * K + offs_k[:, None], mask=mask_k[:, None], other=0.0).to(tl.float32)  # [BLOCK_K, 1], but we want per j: we need to iterate j
            # Better approach: for each j, load W[:, j] and accumulate:
            # We'll recompute per j:
            for jj in range(BLOCK_N):
                j_j = j_start + jj
                if j_j < C_OUT:
                    W_col = tl.load(W_ptr + offs_k * C_OUT + j_j, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]
                    acc[j_j] += tl.sum(x_vec * W_col, axis=0)
            j_start += BLOCK_N

        # Now, add bias
        b = tl.load(b_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        acc += b

    # Store acc as bfloat16
    j = 0
    while j < C_OUT:
        tl.store(y_row_ptr + j, acc[j].to(tl.bfloat16))
        j += 1


# Triton kernel: Elementwise GELU on y_ptr [N, C], one program per row.
@triton.jit
def _gelu_kernel(x_ptr, y_ptr, N, C, BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, y_ptr: *bf16, shapes [N, C]
    GELU using tanh approximation.
    """
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
        k = 0.7978845608028654  # sqrt(2/pi)
        z3 = z * z * z
        t = k * (z + 0.044715 * z3)
        gelu = 0.5 * x * (1.0 + tl.tanh(t))
        tl.store(y_row_ptr + offs, gelu.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants per get_inputs
        self.hidden_size = 1536
        self.hidden_size_expanded = 6144
        self.out_hidden_size = 3584
        self.eps = 1e-6
        # Triton meta-parameters
        self.block_size_ln = 1024  # for LayerNorm reduction
        self.block_k = 1024        # for GEMM reduction chunks
        self.block_linear_n1 = 6144  # match K_out=6144 for Linear1 (single pass)
        self.block_linear_n2 = 3584  # match K_out=3584 for Linear2
        # Ensure divisibility for merge_size=2
        # We will not assume compile-time C values here; use loops.

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor):
        """
        hidden: [num_patches, hidden_size], bfloat16
        grid_thw: [num_grids, 3] int64, each row [t, h, w]
        ln_weight, ln_bias: [hidden_size], bfloat16
        fc1_weight: [hidden_size_expanded, hidden_size_expanded], bfloat16
        fc1_bias: [hidden_size_expanded], bfloat16
        fc2_weight: [out_hidden_size, hidden_size_expanded], bfloat16
        fc2_bias: [out_hidden_size], bfloat16
        eps: float
        Returns output: [num_merged_patches, out_hidden_size], bfloat16
        """
        device = hidden.device
        N = hidden.shape[0]
        C = hidden.shape[1]
        num_grids = grid_thw.shape[0]
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)

        # 1) Triton LayerNorm
        grid = (N,)
        _layer_norm_kernel[grid](hidden, hidden_norm, ln_weight, ln_bias,
                                 N, C, self.eps,
                                 BLOCK_SIZE=self.block_size_ln)

        # 2) SpatialShuffle per grid using Triton; we produce per-grid outputs and concatenate
        # We need to know total_num_merged_patches to allocate final output; since we don't have it upfront,
        # we allocate a list of outputs per grid, compute their sizes, and then copy into a final tensor.
        # First pass: compute per-grid row counts (sum over grids of t * (h//2) * (w//2)) and allocate final.
        total_num_merged = 0
        for g in range(num_grids):
            t = int(grid_thw[g, 0].item())
            h = int(grid_thw[g, 1].item())
            w = int(grid_thw[g, 2].item())
            h_merged = h // 2
            w_merged = w // 2
            total_num_merged += t * h_merged * w_merged

        hidden_shuffled = torch.empty((total_num_merged, 4 * C), dtype=torch.bfloat16, device=device)
        out_grid_ptrs = [torch.empty(t * (h//2) * (w//2), 4 * C, dtype=torch.bfloat16, device=device) for _ in range(num_grids)]

        # Launch per-grid shuffle kernels; we need their outputs to copy into hidden_shuffled by rows.
        row_start = 0
        for g in range(num_grids):
            t = int(grid_thw[g, 0].item())
            h = int(grid_thw[g, 1].item())
            w = int(grid_thw[g, 2].item())
            h_merged = h // 2
            w_merged = w // 2
            num_merged_rows_g = t * h_merged * w_merged

            _shuffle_2x2_per_grid_kernel[(1,)](hidden_norm, grid_thw, out_grid_ptrs[g],
                                               N, C, num_grids,
                                               g, num_merged_rows_g,
                                               BLOCK_M=1)  # not used, but signature requires
            # Copy out_grid_ptrs[g] into hidden_shuffled[row_start:row_start+num_merged_rows_g, :]
            # Use torch.copy_ for assembly (not math), which is acceptable per requirement.
            hidden_shuffled[row_start:row_start + num_merged_rows_g, :] = out_grid_ptrs[g]
            row_start += num_merged_rows_g

        # 3) Linear1: y1 = hidden_shuffled @ fc1_weight.T + fc1_bias
        # Shapes: [total_num_merged, 6144] @ [6144, 6144] + [6144]
        y1 = torch.empty_like(hidden_shuffled, dtype=torch.bfloat16, device=device)
        grid1 = (hidden_shuffled.shape[0],)
        _row_linear_kernel[grid1](hidden_shuffled, fc1_weight, fc1_bias, y1,
                                  hidden_shuffled.shape[0], self.hidden_size_expanded, self.hidden_size_expanded,
                                  BLOCK_N=self.block_linear_n1, BLOCK_K=self.block_k)

        # 4) GELU activation
        y1_gelu = torch.empty_like(y1, dtype=torch.bfloat16, device=device)
        grid_gelu = (hidden_shuffled.shape[0],)
        _gelu_kernel[grid_gelu](y1, y1_gelu, hidden_shuffled.shape[0], self.hidden_size_expanded,
                                BLOCK_SIZE=1024)

        # 5) Linear2: y = y1_gelu @ fc2_weight.T + fc2_bias
        # Shapes: [total_num_merged, 3584] = [total_num_merged, 6144] @ [6144, 3584] + [3584]
        y2 = torch.empty((hidden_shuffled.shape[0], self.out_hidden_size), dtype=torch.bfloat16, device=device)
        grid2 = (hidden_shuffled.shape[0],)
        _row_linear_kernel[grid2](y1_gelu, fc2_weight, fc2_bias, y2,
                                  hidden_shuffled.shape[0], self.out_hidden_size, self.hidden_size_expanded,
                                  BLOCK_N=self.block_linear_n2, BLOCK_K=self.block_k)

        return y2


def run(*args):
    return ModelNew()(*args)
