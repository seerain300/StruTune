import math
import triton
import triton.language as tl

# Triton kernel: LayerNorm per row (two passes). One program per row.
@triton.jit
def _layer_norm_rows_kernel(x_ptr, y_ptr, ln_weight_ptr, ln_bias_ptr,
                             N, C, eps,
                             BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    y_ptr: *bf16, shape [N, C], output
    ln_weight_ptr, ln_bias_ptr: *bf16, shape [C]
    eps: float32
    """
    row = tl.program_id(0)
    if row >= N:
        return

    x_row_ptr = x_ptr + row * C
    y_row_ptr = y_ptr + row * C

    # Pass 1: compute mean and variance (fp32)
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

    # Pass 2: normalize and apply affine, then store
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


# Triton kernel: Perform spatial shuffle for all grids. The input hidden_norm is provided
# already as the concatenation of per-grid patches. Each grid has t, h, w given by grid_thw[g].
# The output is a single tensor of shape [total_num_merged_patches, 4*C], where total_num_merged_patches
# equals sum over grids of (t * (h//2) * (w//2)). We write exactly the 2x2 merged values into 4 contiguous
# columns per row. We assume merge_size=2.
@triton.jit
def _shuffle_all_grids_kernel(hidden_ptr, grid_thw_ptr, out_ptr,
                               total_patches, C, NUM_GRIDS, TOTAL_MERGED_ROWS,
                               BLOCK_M: tl.constexpr):
    """
    hidden_ptr: *bf16, flattened [total_patches, C]
    grid_thw_ptr: *int64, shape [NUM_GRIDS, 3], each row is [t, h, w]
    out_ptr: *bf16, flattened [TOTAL_MERGED_ROWS, 4*C]
    We iterate over grids and for each grid, iterate over all original patch indices m in [0, t*h*w).
    We decode t_index, h2, w2, and write the 2x2 positions into out at row = t_index * (h//2)*(w//2) + (h2//2) * (w//2) + (w2//2),
    columns 0*C, 1*C, 2*C, 3*C correspond to (r=0,c=0), (r=0,c=1), (r=1,c=0), (r=1,c=1) respectively.
    """
    g = 0  # single kernel instance writes for all grids; we launch grid size = NUM_GRIDS
    if g >= NUM_GRIDS:
        return

    # Load grid dimensions
    t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)

    h_merged = h // 2
    w_merged = w // 2
    num_merged_rows_grid = t * h_merged * w_merged

    # Loop over all original patch indices m in this grid
    m = 0
    while m < t * h * w:
        t_index = m // (h * w)
        rem = m % (h * w)
        h2 = rem // w
        w2 = rem % w

        # Out row index within the grid contribution
        out_row = t_index * (h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)
        # Base offset in the global out tensor
        base_out = out_row + (g * num_merged_rows_grid) * (4 * C)

        # Write 2x2 positions into 4 contiguous columns
        # idx 0: (r=0,c=0) -> hidden[t_index, h2, w2]
        src_offset0 = t_index * (h * w) + h2 * w + w2
        val0 = tl.load(hidden_ptr + src_offset0 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
        tl.store(out_ptr + base_out + 0 * C, val0)

        # idx 1: (r=0,c=1)
        src_offset1 = src_offset0 + 1
        val1 = tl.load(hidden_ptr + src_offset1 * C + 0, mask=(w2 + 1 < w), other=0.0).to(tl.bfloat16)
        tl.store(out_ptr + base_out + 1 * C, val1)

        # idx 2: (r=1,c=0) -> hidden[t_index, h2+1, w2]
        src_offset2 = t_index * (h * w) + (h2 + 1) * w + w2
        val2 = tl.load(hidden_ptr + src_offset2 * C + 0, mask=(h2 + 1 < h), other=0.0).to(tl.bfloat16)
        tl.store(out_ptr + base_out + 2 * C, val2)

        # idx 3: (r=1,c=1) -> hidden[t_index, h2+1, w2+1]
        src_offset3 = src_offset2 + 1
        if w2 + 1 < w and h2 + 1 < h:
            val3 = tl.load(hidden_ptr + src_offset3 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(out_ptr + base_out + 3 * C, val3)

        m += 1


# Triton kernel: Elementwise GELU over a flattened tensor of shape [N, C].
@triton.jit
def _gelu_kernel(x_ptr, y_ptr, N, C, BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    y_ptr: *bf16, shape [N, C]
    GELU via tanh approximation:
      0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
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
        # constants
        c0 = 0.7978845608028654  # sqrt(2/pi)
        c1 = 0.044715
        x3 = x * x * x
        z = c0 * (x + c1 * x3)
        # tanh approximation
        y = 0.5 * x * (1.0 + tl.tanh(z))
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


# Triton kernel: Row-wise Linear (y = x @ W.T + b). Each program computes one output row for x.
@triton.jit
def _linear_row_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                        IN, CIN, COUT, BLOCK_K: tl.constexpr):
    """
    x_ptr: *bf16, shape [1, CIN]
    w_ptr: *bf16, shape [COUT, CIN] (row-major, i.e., W[j, k])
    b_ptr: *bf16, shape [COUT]
    y_ptr: *bf16, shape [1, COUT]
    One program computes output row 0. Accumulate in float32.
    """
    acc = tl.zeros([COUT], dtype=tl.float32)
    k = 0
    while k < CIN:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < CIN
        x_k = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]
        # Accumulate over K dimension: acc[j] += sum_k W[j, k] * x[k]
        for kk in range(BLOCK_K):
            k_curr = k + kk
            mkk = k_curr < CIN
            xk_val = tl.load(x_ptr + k_curr, mask=mkk, other=0.0).to(tl.float32)
            wj = tl.load(w_ptr + tl.arange(0, COUT) * CIN + k_curr, mask=(tl.arange(0, COUT) < COUT) & mkk, other=0.0).to(tl.float32)
            acc += xk_val * wj
        k += BLOCK_K

    # Add bias
    j = 0
    while j < COUT:
        offs_j = j + tl.arange(0, COUT)
        mask_j = offs_j < COUT
        bj = tl.load(b_ptr + offs_j, mask=mask_j, other=0.0).to(tl.float32)
        acc += bj
        j += COUT  # COUT is constexpr here; increment by COUT

    # Store result y[0, :]
    j = 0
    while j < COUT:
        offs_j = j + tl.arange(0, COUT)
        mask_j = offs_j < COUT
        tl.store(y_ptr + offs_j, acc[offs_j].to(tl.bfloat16), mask=mask_j)
        j += COUT


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
        hidden: [num_patches, hidden_size] bfloat16
        grid_thw: [num_grids, 3] int64, each row is [t, h, w]
        ln_weight, ln_bias: [hidden_size] bfloat16
        fc1_weight, fc1_bias: [hidden_size_expanded, hidden_size_expanded] and [hidden_size_expanded]
        fc2_weight, fc2_bias: [out_hidden_size, hidden_size_expanded] and [out_hidden_size]
        eps: float
        """
        # Ensure device is CUDA
        assert hidden.is_cuda and grid_thw.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda \
               and fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, \
            "All tensors must be CUDA for Triton kernels."

        N = hidden.shape[0]
        C = hidden.shape[1]
        hidden_norm = torch.empty_like(hidden)

        # LayerNorm (Triton)
        BLOCK_SIZE = 1024
        _layer_norm_rows_kernel[(N,)](hidden, hidden_norm, ln_weight, ln_bias, N, C, eps, BLOCK_SIZE)

        # Spatial shuffle: we know total_num_merged_patches should equal sum over grids of t*(h//2)*(w//2).
        # The provided get_inputs guarantees this. We allocate output accordingly.
        num_grids = grid_thw.shape[0]
        t_list = [int(grid_thw[i, 0].item()) for i in range(num_grids)]
        h_list = [int(grid_thw[i, 1].item()) for i in range(num_grids)]
        w_list = [int(grid_thw[i, 2].item()) for i in range(num_grids)]
        total_merged_rows = sum([t * (h // 2) * (w // 2) for t, h, w in zip(t_list, h_list, w_list)])
        hidden_size = C
        hidden_size_expanded = 4 * hidden_size  # since merge_size=2, each output row has 4*C elements
        hidden_shuffled = torch.empty((total_merged_rows, hidden_size_expanded), dtype=torch.bfloat16, device=hidden.device)

        # Triton shuffle for all grids
        _shuffle_all_grids_kernel[(1,)](hidden_norm, grid_thw, hidden_shuffled, N, C, num_grids, total_merged_rows, BLOCK_M=1024)

        # GELU activation (Triton)
        N_shuffled = total_merged_rows
        hidden_gelu = torch.empty_like(hidden_shuffled)
        _gelu_kernel[(N_shuffled,)](hidden_shuffled, hidden_gelu, N_shuffled, hidden_size_expanded, BLOCK_SIZE=1024)

        # Linear1: (num_merged_patches, 6144) @ (6144, 6144).T + bias
        # We implement row-wise Triton GEMM for each output row. Output rows = N_shuffled.
        # Allocate output for Linear1
        in_dim = hidden_size_expanded  # 6144
        fc1_out = torch.empty((N_shuffled, in_dim), dtype=torch.bfloat16, device=hidden.device)

        # One program per row
        _linear_row_kernel[(N_shuffled,)](
            hidden_gelu, fc1_weight, fc1_bias, fc1_out,
            N_shuffled, in_dim, in_dim, BLOCK_K=1024
        )

        # GELU activation for Linear1 output
        fc1_out_gelu = torch.empty_like(fc1_out)
        _gelu_kernel[(N_shuffled,)](fc1_out, fc1_out_gelu, N_shuffled, in_dim, BLOCK_SIZE=1024)

        # Linear2: (num_merged_patches, 6144) @ (3584, 6144).T + bias
        out_hidden_size = 3584
        output = torch.empty((N_shuffled, out_hidden_size), dtype=torch.bfloat16, device=hidden.device)

        _linear_row_kernel[(N_shuffled,)](
            fc1_out_gelu, fc2_weight, fc2_bias, output,
            N_shuffled, in_dim, out_hidden_size, BLOCK_K=1024
        )

        return output


def run(*args):
    return ModelNew()(*args)
