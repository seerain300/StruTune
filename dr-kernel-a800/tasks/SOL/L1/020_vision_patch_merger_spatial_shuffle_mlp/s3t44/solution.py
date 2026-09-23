import torch
import math
import triton
import triton.language as tl

# Triton kernel: LayerNorm per row (two-pass: reduce, then normalize + affine).
# Input: hidden_in [N, C] (bf16), output: hidden_out [N, C] (bf16)
# ln_weight, ln_bias: [C] (bf16). Computations done in fp32.
@triton.jit
def _layer_norm_rows_kernel(hidden_in_ptr, hidden_out_ptr, ln_weight_ptr, ln_bias_ptr,
                             N, C, eps, BLOCK_SIZE: tl.constexpr):
    """
    hidden_in_ptr: *bf16, shape [N, C]
    hidden_out_ptr: *bf16, shape [N, C]
    ln_weight_ptr, ln_bias_ptr: *bf16, shape [C]
    eps: float32
    """
    row = tl.program_id(0)
    if row >= N:
        return

    x_row_ptr = hidden_in_ptr + row * C
    y_row_ptr = hidden_out_ptr + row * C

    # Pass 1: compute sum and sum of squares over C (fp32)
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

    # Pass 2: normalize, apply affine, store bfloat16
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


# Triton kernel: Spatial shuffle per-grid and concatenate into a single output
# Output is [total_num_merged_patches, 4*C], where total_num_merged_patches = sum_{g} t_g * (h_g//2) * (w_g//2).
# hidden_norm_full: *bf16, shape [total_patches, C] (row-major)
# grid_thw_ptr: *int64, shape [NUM_GRIDS, 3], each row is [t, h, w]
# out_ptr: *bf16, shape [total_merged_rows, 4*C], row-major
@triton.jit
def _shuffle_2x2_concat_kernel(hidden_norm_full_ptr, grid_thw_ptr, out_ptr,
                               total_patches, C, NUM_GRIDS, MERGE_SIZE: tl.constexpr,
                               MAX_OUT_ROWS: tl.constexpr):
    """
    MERGE_SIZE should be 2 in this task. MAX_OUT_ROWS is an upper bound; we iterate m and compute out_row safely.
    """
    g = tl.program_id(0)
    if g >= NUM_GRIDS:
        return

    # Load grid dimensions
    t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)

    h_merged = h // MERGE_SIZE
    w_merged = w // MERGE_SIZE
    num_patches_in_grid = t * h * w

    # Compute the base offset in out_ptr for this grid's section
    grid_rows = t * h_merged * w_merged
    base_grid_offset = g * (grid_rows * (4 * C))

    # For each original patch m in the grid:
    # Decode (t_index, h2, w2), then out_row = t_index * (h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)
    # Write four 2x2 positions into four contiguous columns of out.
    m = 0
    while m < num_patches_in_grid:
        t_index = m // (h * w)
        rem = m % (h * w)
        h2 = rem // w
        w2 = rem % w

        out_row = t_index * (h_merged * w_merged) + (h2 // MERGE_SIZE) * w_merged + (w2 // MERGE_SIZE)
        if out_row >= MAX_OUT_ROWS:
            break  # safety, though for typical configs out_row < grid_rows <= MAX_OUT_ROWS

        base_out = out_ptr + base_grid_offset + out_row * (4 * C)

        # idx=0: (r=0,c=0) -> hidden[t_index, h2, w2]
        col0 = (h2 * 2 + 0) * w + (w2 * 2 + 0) * C
        src_offset0 = t_index * (h * w) + h2 * w + w2
        val0 = tl.load(hidden_norm_full_ptr + src_offset0 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
        tl.store(base_out + 0 * C, val0)

        # idx=1: (r=0,c=1) -> hidden[t_index, h2, w2+1]
        col1 = (h2 * 2 + 0) * w + (w2 * 2 + 1) * C
        if (w2 + 1) < w:
            val1 = tl.load(hidden_norm_full_ptr + src_offset0 * C + 1, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(base_out + 1 * C, val1)

        # idx=2: (r=1,c=0) -> hidden[t_index, h2+1, w2]
        col2 = (h2 * 2 + 1) * w + (w2 * 2 + 0) * C
        if (h2 + 1) < h and (w2) < w:
            val2 = tl.load(hidden_norm_full_ptr + src_offset0 * C + 2, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(base_out + 2 * C, val2)

        # idx=3: (r=1,c=1) -> hidden[t_index, h2+1, w2+1]
        col3 = (h2 * 2 + 1) * w + (w2 * 2 + 1) * C
        if (h2 + 1) < h and (w2 + 1) < w:
            val3 = tl.load(hidden_norm_full_ptr + src_offset0 * C + 3, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(base_out + 3 * C, val3)

        m += 1


# Triton kernel: Elementwise GELU on input x (bf16), output y (bf16)
# GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
@triton.jit
def _gelu_kernel(x_ptr, y_ptr, N, C, BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C]
    y_ptr: *bf16, shape [N, C]
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
        # erf using Triton's math
        erf_val = tl.math.erf(x * inv_sqrt2)
        y = 0.5 * x * (1.0 + erf_val)
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


# Triton kernel: Row-wise GEMM for Linear1: computes y[n, j] = sum_k x[n, k] * W1[j, k] + bias1[j]
# x: [N, K_IN] (bf16), W1: [K_OUT, K_IN] (bf16), bias1: [K_OUT] (bf16), y: [N, K_OUT] (bf16)
@triton.jit
def _gemm_row_linear1_kernel(x_ptr, W1_ptr, bias1_ptr, y_ptr,
                              N, K_IN, K_OUT, eps, BLOCK_K: tl.constexpr):
    """
    One program per output row n. Accumulate across K_IN in blocks of BLOCK_K.
    x_ptr: *bf16, shape [N*K_IN] flattened row-major
    W1_ptr: *bf16, shape [K_OUT, K_IN] row-major (num_rows=K_OUT, num_cols=K_IN)
    bias1_ptr: *bf16, shape [K_OUT]
    y_ptr: *bf16, shape [N*K_OUT] flattened
    """
    n = tl.program_id(0)
    if n >= N:
        return
    # Initialize accumulator for this row
    acc = tl.zeros((K_OUT,), dtype=tl.float32)
    # Loop over K_IN in blocks
    k0 = 0
    while k0 < K_IN:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_IN
        # x_row is a block of K_IN; we need x[n, k0:k0+BLOCK_K]
        # Since x_ptr is flattened, index = n*K_IN + k. We load a vector of BLOCK_K.
        x_vec = tl.load(x_ptr + n * K_IN + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        # Load corresponding slice of W1: W1[j, k] for j in [0..K_OUT-1], k in offs_k
        # Build acc by dot product: for each k in offs_k, add W1[:, k] * x_vec[k]
        # We do this elementwise accumulation: for each j, acc[j] += sum_k W1[j, k] * x_vec[k]
        # Implement nested loop over j and k.
        j = 0
        while j < K_OUT:
            w_j = tl.load(W1_ptr + j * K_IN + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            acc[j] += tl.sum(w_j * x_vec, axis=0)
            j += 1
        k0 += BLOCK_K
    # Add bias
    j = 0
    while j < K_OUT:
        acc[j] += tl.load(bias1_ptr + j).to(tl.float32)
        j += 1
    # Store result row y[n, :]
    y_row_ptr = y_ptr + n * K_OUT
    j = 0
    while j < K_OUT:
        tl.store(y_row_ptr + j, acc[j].to(tl.bfloat16))
        j += 1


# Triton kernel: Row-wise GEMM for Linear2: computes y[n, j] = sum_k x[n, k] * W2[j, k] + bias2[j]
# x: [N, K_IN] (bf16), W2: [K_OUT, K_IN] (bf16), bias2: [K_OUT] (bf16), y: [N, K_OUT] (bf16)
# Here x is the output of GELU (from Linear1).
@triton.jit
def _gemm_row_linear2_kernel(x_ptr, W2_ptr, bias2_ptr, y_ptr,
                              N, K_IN, K_OUT, eps, BLOCK_K: tl.constexpr):
    """
    One program per output row n. Accumulate across K_IN in blocks of BLOCK_K.
    x_ptr: *bf16, shape [N*K_IN] flattened
    W2_ptr: *bf16, shape [K_OUT, K_IN] row-major
    bias2_ptr: *bf16, shape [K_OUT]
    y_ptr: *bf16, shape [N*K_OUT] flattened
    """
    n = tl.program_id(0)
    if n >= N:
        return
    acc = tl.zeros((K_OUT,), dtype=tl.float32)
    k0 = 0
    while k0 < K_IN:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_IN
        x_vec = tl.load(x_ptr + n * K_IN + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        j = 0
        while j < K_OUT:
            w_j = tl.load(W2_ptr + j * K_IN + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            acc[j] += tl.sum(w_j * x_vec, axis=0)
            j += 1
        k0 += BLOCK_K
    j = 0
    while j < K_OUT:
        acc[j] += tl.load(bias2_ptr + j).to(tl.float32)
        j += 1
    y_row_ptr = y_ptr + n * K_OUT
    j = 0
    while j < K_OUT:
        tl.store(y_row_ptr + j, acc[j].to(tl.bfloat16))
        j += 1


class ModelNew(torch.nn.Module):
    def __init__(self, axes_and_scalars: dict, device: torch.device):
        super().__init__()
        # No parameters; we will generate inputs in forward to match original signature
        pass

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor,
                eps: float):
        # Ensure device/dtype
        device = hidden.device
        hidden = hidden.to(torch.bfloat16)
        ln_weight = ln_weight.to(torch.bfloat16)
        ln_bias = ln_bias.to(torch.bfloat16)
        fc1_weight = fc1_weight.to(torch.bfloat16)
        fc1_bias = fc1_bias.to(torch.bfloat16)
        fc2_weight = fc2_weight.to(torch.bfloat16)
        fc2_bias = fc2_bias.to(torch.bfloat16)

        # 1) LayerNorm: hidden_norm = LN(hidden)
        N, C = hidden.shape
        hidden_norm = torch.empty((N, C), device=device, dtype=torch.bfloat16)
        _layer_norm_rows_kernel[(N,)](hidden, hidden_norm, ln_weight, ln_bias, N, C, eps, BLOCK_SIZE=256)

        # 2) Spatial shuffle: hidden_shuffled = concatenate of per-grid shuffles
        total_patches = N  # original num_patches
        num_grids = grid_thw.shape[0]
        h = int(math.sqrt(C))  # C=1536 -> h=39, w=39, but original code uses get_inputs which sets H=W, irrelevant here.
        w = int(math.sqrt(C))
        # Derive h, w from the provided grid_thw. We can't infer h,w directly from C here; however,
        # the original code guarantees hidden_norm is [num_patches, C], and it uses grid_thw to rearrange.
        # We proceed with the Triton shuffle using the provided grid_thw. We must compute total merged rows.
        # The original code computes patches_per_grid = num_patches // num_grids. Here total_patches == num_patches.
        # We allocate output [total_merged_rows, 4*C].
        # Compute total_merged_rows = sum over grids of t_g * (h_g//2) * (w_g//2).
        total_merged_rows = 0
        for g in range(num_grids):
            t = int(grid_thw[g, 0].item())
            h_g = int(grid_thw[g, 1].item())
            w_g = int(grid_thw[g, 2].item())
            total_merged_rows += t * (h_g // 2) * (w_g // 2)

        hidden_shuffled = torch.empty((total_merged_rows, 4 * C), device=device, dtype=torch.bfloat16)

        # Launch shuffle kernel: one program per grid
        _shuffle_2x2_concat_kernel[(num_grids,)](hidden_norm, grid_thw, hidden_shuffled,
                                                 total_patches, C, num_grids, MERGE_SIZE=2, MAX_OUT_ROWS=1024)

        # 3) Linear1: y1 = hidden_shuffled @ fc1_weight.T + fc1_bias
        N2, K_IN = hidden_shuffled.shape  # N2 = total_merged_rows
        K_OUT = fc1_weight.shape[0]      # 6144
        y1 = torch.empty((N2, K_OUT), device=device, dtype=torch.bfloat16)
        _gemm_row_linear1_kernel[(N2,)](hidden_shuffled, fc1_weight, fc1_bias, y1, N2, K_IN, K_OUT, eps, BLOCK_K=128)

        # 4) GELU
        y1_gelu = torch.empty((N2, K_OUT), device=device, dtype=torch.bfloat16)
        _gelu_kernel[(N2,)](y1, y1_gelu, N2, K_OUT, BLOCK_SIZE=256)

        # 5) Linear2: output = y1_gelu @ fc2_weight.T + fc2_bias
        K_IN2 = fc2_weight.shape[1]  # 6144
        K_OUT2 = fc2_weight.shape[0] # 3584
        output = torch.empty((N2, K_OUT2), device=device, dtype=torch.bfloat16)
        _gemm_row_linear2_kernel[(N2,)](y1_gelu, fc2_weight, fc2_bias, output, N2, K_IN2, K_OUT2, eps, BLOCK_K=128)

        return output


def run(*args):
    return ModelNew()(*args)
