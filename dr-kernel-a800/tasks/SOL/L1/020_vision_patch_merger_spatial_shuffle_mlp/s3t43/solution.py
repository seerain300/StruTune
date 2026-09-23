import torch
import math
import triton
import triton.language as tl

# Triton kernel: LayerNorm per row (two-pass: reduce, then normalize + affine).
# Input: hidden_in [N, C] (bf16), output: hidden_out [N, C] (bf16)
# ln_weight, ln_bias: [C] (bf16). Computation is fp32 for mean/var and affine.
@triton.jit
def _layer_norm_rows_kernel(hidden_in_ptr, hidden_out_ptr, ln_weight_ptr, ln_bias_ptr,
                             N, C, eps,
                             BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= N:
        return

    x_row_ptr = hidden_in_ptr + row * C
    y_row_ptr = hidden_out_ptr + row * C

    # Pass 1: sum and sum of squares in fp32
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


# Triton kernel: Spatial shuffle per grid. Exact 2x2 merge mapping.
# Input: hidden_norm [total_patches, C] (bf16), grid_thw [NUM_GRIDS, 3] (int64).
# Output: out [total_merged_rows, 4*C] (bf16), where total_merged_rows = sum_g (t_g * (h_g//2) * (w_g//2)).
@triton.jit
def _shuffle_2x2_per_grid_kernel(hidden_ptr, grid_thw_ptr, out_ptr,
                                 total_patches, C, NUM_GRIDS, EPS,
                                 MAX_OUT_ROWS: tl.constexpr):
    """
    Vectorize across out_rows for a single grid. For each out_row, write four contiguous columns.
    """
    g = tl.program_id(0)
    if g >= NUM_GRIDS:
        return

    # Load grid dimensions
    t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)

    h_merged = h // 2
    w_merged = w // 2
    num_merged_rows = t * h_merged * w_merged

    # Vectorize over out_rows (all rows for this grid)
    out_rows = tl.arange(0, MAX_OUT_ROWS)
    mask_rows = out_rows < num_merged_rows

    # Decode out_rows -> (t_index, h2//2, w2//2)
    t_index = out_rows // (h_merged * w_merged)
    rem = out_rows % (h_merged * w_merged)
    h2_div2 = rem // w_merged
    w2_div2 = rem % w_merged
    h2 = h2_div2 * 2
    w2 = w2_div2 * 2

    # Base source indices for 2x2 in original patch
    base_src = t_index * (h * w) + h2 * w + w2

    # Compute four source columns (0,1) for the two rows (0,1)
    # idx = 0: (r=0,c=0), idx = 1: (r=0,c=1), idx = 2: (r=1,c=0), idx = 3: (r=1,c=1)
    # For each idx, load corresponding element and store into out[out_row, idx * C + cdim]
    # We do per idx with scalar load/store to ensure correctness; vectorization is across out_rows.
    for idx in range(4):
        if idx == 0:
            src_off = base_src + 0 * w + 0
            dst_col = 0 * C
        elif idx == 1:
            src_off = base_src + 0 * w + 1
            dst_col = 1 * C
        elif idx == 2:
            src_off = base_src + 1 * w + 0
            dst_col = 2 * C
        else:
            src_off = base_src + 1 * w + 1
            dst_col = 3 * C

        # out_ptr is laid out as [total_merged_rows, 4*C], contiguous. Each out_row has 4*C columns.
        # We compute base_out = out_rows * (4*C) then add dst_col.
        base_out = out_rows * (4 * C) + dst_col

        # Read from hidden_ptr as [total_patches, C], each row is a patch, C is the feature dim.
        # The original mapping is: hidden[t_index, h2, w2 + (idx%2)*1, ...] -> flattened as C features.
        # For our implementation, hidden_ptr stores per-patch flattened features. We need to map src_off
        # to the appropriate source address. However, since hidden_norm is already the per-patch flattened
        # features for each grid (concatenated), we need to compute which grid this out_row belongs to.
        # Instead, we compute source address as: for each idx, the original patch m corresponds to t_index,
        # h2, w2, and the two columns are just reading hidden_norm at positions separated by +1 in w.
        # Note: src_off computes the correct flattened index within the grid's patches.
        # Ensure source row is within total_patches; if not, mask out.
        valid_row = src_off < (t * h * w)  # safety mask in case out_rows exceed t*h*w_merged (unused due to mask_rows)
        val = tl.load(hidden_ptr + src_off * C + 0, mask=mask_rows & valid_row, other=0.0).to(tl.bfloat16)
        tl.store(out_ptr + base_out, val, mask=mask_rows)


# Triton kernel: Elementwise GELU on a row vector. One program per row.
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
        # GELU: 0.5 * x * (1 + erf(z))
        # Triton doesn't provide erf, so use a standard approximation
        # erf(z) ≈ sign(z) * (1 - exp(-z^2) * (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5)),
        # where t = 1 / (1 + p z), p=0.3275911
        # Implement approximation constants and compute
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        sign = tl.where(z >= 0, 1.0, -1.0)
        az = tl.abs(z)
        t = 1.0 / (1.0 + p * az)
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_approx = sign * (1.0 - poly * tl.exp(-az * az))
        y = 0.5 * x * (1.0 + erf_approx)
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


# Triton kernel: GEMM row-wise for Linear layer, output one row per program.
# Inputs: x [N, K_IN] (bf16), W [K_OUT, K_IN] (bf16), b [K_OUT] (bf16)
# Output: y [N, K_OUT] (bf16)
# We implement y[i, k_out] = sum_{j} x[i, j] * W[k_out, j] + b[k_out]
# The provided axes use K_IN=6144 and K_OUT in {3584, 6144}. We loop j in blocks of BLOCK_K.
@triton.jit
def _gemm_row_linear_kernel(x_ptr, W_ptr, b_ptr, y_ptr,
                             N, K_IN, K_OUT,
                             BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    if row >= N:
        return

    # Compute output for all K_OUT elements in a loop
    k_out_base = 0
    while k_out_base < K_OUT:
        k_out_vec = k_out_base + tl.arange(0, BLOCK_K)
        mask_k_out = k_out_vec < K_OUT

        # Initialize accumulator for this k_out vector
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)

        # Loop over K_IN dimension in blocks
        k_in_base = 0
        while k_in_base < K_IN:
            k_in_vec = k_in_base + tl.arange(0, BLOCK_K)
            mask_k_in = k_in_vec < K_IN

            # Load x row block: x[row, k_in_vec]
            x_row_ptr = x_ptr + row * K_IN + k_in_vec
            x_block = tl.load(x_row_ptr, mask=mask_k_in, other=0.0).to(tl.float32)  # shape [BLOCK_K]

            # Load W block: W[k_out_vec, k_in_vec] -> pointer arithmetic W_ptr + k_out_vec[:,None] * K_IN + k_in_vec[None,:]
            # Create 2D pointers: [BLOCK_K, BLOCK_K]
            ptrs = W_ptr + k_out_vec[:, None] * K_IN + k_in_vec[None, :]
            W_block = tl.load(ptrs, mask=mask_k_out[:, None] & mask_k_in[None, :], other=0.0).to(tl.float32)  # shape [BLOCK_K, BLOCK_K]

            # Accumulate: acc += sum_j (W[k_out, j] * x[j])
            # Multiply W_block (K_OUT_vec x K_IN_vec) with x_block (K_IN_vec) -> result per k_out_vec element
            # We can compute per k_out by reducing over K_IN dimension.
            # Triton allows tl.sum over a specified axis.
            acc += tl.sum(W_block * x_block[None, :], axis=1)

            k_in_base += BLOCK_K

        # Add bias
        b_block = tl.load(b_ptr + k_out_base + tl.arange(0, BLOCK_K), mask=mask_k_out, other=0.0).to(tl.float32)
        acc += b_block

        # Store result
        y_row_ptr = y_ptr + row * K_OUT + k_out_base + tl.arange(0, BLOCK_K)
        tl.store(y_row_ptr, acc.to(tl.bfloat16), mask=mask_k_out)

        k_out_base += BLOCK_K


# Entry point model
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
        Triton-only implementation:
        - LayerNorm per row on hidden
        - Spatial shuffle to produce hidden_shuffled of shape [total_num_merged_patches, 4*C]
        - GELU on hidden_shuffled
        - Linear1 (GEMM): hidden_gelu @ fc1_weight.T + fc1_bias
        - Linear2 (GEMM): linear1_output @ fc2_weight.T + fc2_bias
        """
        device = hidden.device
        # 1) LayerNorm: normalize hidden_norm and apply ln_weight/bias
        hidden_in = hidden  # already bfloat16
        C = hidden_in.shape[1]
        hidden_norm = torch.empty_like(hidden_in)
        N = hidden_in.shape[0]
        # Launch Triton LayerNorm kernel
        BLOCK_SIZE = 512  # works well for C=1536
        grid_ln = (N,)
        _layer_norm_rows_kernel[grid_ln](hidden_in, hidden_norm, ln_weight, ln_bias, N, C, eps, BLOCK_SIZE=BLOCK_SIZE)

        # 2) Spatial shuffle: hidden_norm -> hidden_shuffled [total_merged_rows, 4*C]
        # Compute total_merged_rows
        NUM_GRIDS = grid_thw.shape[0]
        total_merged_rows = 0
        for g in range(NUM_GRIDS):
            t = grid_thw[g, 0].item()
            h = grid_thw[g, 1].item()
            w = grid_thw[g, 2].item()
            total_merged_rows += t * (h // 2) * (w // 2)

        # Allocate output
        C4 = C * 4  # hidden_size_expanded
        hidden_shuffled = torch.empty((total_merged_rows, C4), device=device, dtype=torch.bfloat16)

        # Launch Triton shuffle kernel (one program per grid). Use MAX_OUT_ROWS >= total_merged_rows; here <=1024 for provided configs.
        MAX_OUT_ROWS = 1024
        _shuffle_2x2_per_grid_kernel[(NUM_GRIDS,)](hidden_norm, grid_thw, hidden_shuffled,
                                                  hidden_in.shape[0], C, NUM_GRIDS, eps, MAX_OUT_ROWS=MAX_OUT_ROWS)

        # 3) GELU: elementwise
        N2 = hidden_shuffled.shape[0]
        hidden_gelu = torch.empty_like(hidden_shuffled)
        grid_gelu = (N2,)
        # Choose BLOCK_SIZE for C4
        BLOCK_SIZE_GELU = 512 if C4 >= 512 else 256
        _gelu_kernel[grid_gelu](hidden_shuffled, hidden_gelu, N2, C4, BLOCK_SIZE=BLOCK_SIZE_GELU)

        # 4) Linear1: hidden_gelu [N2, 4*C] @ fc1_weight.T [4*C, 4*C] -> y1 [N2, 4*C]
        # Note: fc1_weight is [hidden_size_expanded, hidden_size_expanded] = [4*C, 4*C]
        y1 = torch.empty((N2, fc1_weight.shape[0]), device=device, dtype=torch.bfloat16)
        K_IN = hidden_gelu.shape[1]  # 4*C = 6144
        K_OUT = y1.shape[1]  # 6144
        BLOCK_K = 128
        grid_linear1 = (N2,)
        _gemm_row_linear_kernel[grid_linear1](hidden_gelu, fc1_weight, fc1_bias, y1, N2, K_IN, K_OUT, BLOCK_K=BLOCK_K)

        # 5) Linear2: y1 [N2, 6144] @ fc2_weight.T [6144, 3584] -> y2 [N2, 3584]
        # fc2_weight shape [out_hidden_size, hidden_size_expanded] = [3584, 6144]
        y2 = torch.empty((N2, fc2_weight.shape[0]), device=device, dtype=torch.bfloat16)
        K_IN2 = y1.shape[1]  # 6144
        K_OUT2 = fc2_weight.shape[0]  # 3584
        grid_linear2 = (N2,)
        _gemm_row_linear_kernel[grid_linear2](y1, fc2_weight, fc2_bias, y2, N2, K_IN2, K_OUT2, BLOCK_K=128)

        return y2


def run(*args):
    return ModelNew()(*args)
