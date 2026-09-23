import torch
import math
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


# Triton kernel: Exact spatial shuffle per grid producing hidden_shuffled
# of shape [total_num_merged_patches, 4*C]. One program per grid.
@triton.jit
def _shuffle_2x2_per_grid_kernel(hidden_ptr, grid_thw_ptr, out_ptr,
                                 total_patches, C, NUM_GRIDS,
                                 BLOCK_M: tl.constexpr):
    """
    hidden_ptr: *bf16, flattened [total_patches, C]
    grid_thw_ptr: *int64, shape [NUM_GRIDS, 3], each row is [t, h, w]
    out_ptr: *bf16, flattened [total_merged_rows, 4*C]
    Launch one program per grid.
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

    # Iterate over all original patches in this grid: m in [0, t*h*w)
    m = 0
    while m < t * h * w:
        t_index = m // (h * w)
        rem = m % (h * w)
        h2 = rem // w
        w2 = rem % w

        out_row = t_index * (h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)
        base_out = out_ptr + out_row * (4 * C)

        # idx=0: (r=0,c=0) -> hidden[t_index, h2, w2]
        idx0 = 0
        col0 = (h2 * 2 + 0) * w_merged * C + (w2 * 2 + 0) * C
        src_offset0 = t_index * (h * w) + h2 * w + w2
        val0 = tl.load(hidden_ptr + src_offset0 * C + 0, mask=(h2 < h) & (w2 < w), other=0.0).to(tl.bfloat16)
        tl.store(base_out + idx0 * C, val0)

        # idx=1: (r=0,c=1) -> hidden[t_index, h2, w2+1]
        idx1 = 1
        col1 = (h2 * 2 + 0) * w_merged * C + (w2 * 2 + 1) * C
        if w2 + 1 < w:
            src_offset1 = t_index * (h * w) + h2 * w + (w2 + 1)
            val1 = tl.load(hidden_ptr + src_offset1 * C + 0, mask=(h2 < h) & (w2 + 1 < w), other=0.0).to(tl.bfloat16)
            tl.store(base_out + idx1 * C, val1)

        # idx=2: (r=1,c=0) -> hidden[t_index, h2+1, w2]
        idx2 = 2
        col2 = (h2 * 2 + 1) * w_merged * C + (w2 * 2 + 0) * C
        if h2 + 1 < h:
            src_offset2 = t_index * (h * w) + (h2 + 1) * w + w2
            val2 = tl.load(hidden_ptr + src_offset2 * C + 0, mask=(h2 + 1 < h) & (w2 < w), other=0.0).to(tl.bfloat16)
            tl.store(base_out + idx2 * C, val2)

        # idx=3: (r=1,c=1) -> hidden[t_index, h2+1, w2+1]
        idx3 = 3
        col3 = (h2 * 2 + 1) * w_merged * C + (w2 * 2 + 1) * C
        if h2 + 1 < h and w2 + 1 < w:
            src_offset3 = t_index * (h * w) + (h2 + 1) * w + (w2 + 1)
            val3 = tl.load(hidden_ptr + src_offset3 * C + 0, mask=(h2 + 1 < h) & (w2 + 1 < w), other=0.0).to(tl.bfloat16)
            tl.store(base_out + idx3 * C, val3)

        m += 1


# Triton kernel: Elementwise GELU on a row vector. One program per row.
@triton.jit
def _gelu_row_kernel(x_ptr, y_ptr, N, C, BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    y_ptr: *bf16, shape [N, C]
    GELU: y = 0.5 * x * (1 + erf(x / sqrt(2)))
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
        y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


# Triton kernel: Row-wise GEMM for Linear1 (K_in=6144, K_out=6144). One program per output row.
@triton.jit
def _linear1_row_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                         NUM_ROWS, K_IN, K_OUT,
                         BLOCK_K_IN: tl.constexpr, BLOCK_K_OUT: tl.constexpr):
    """
    x_ptr: *bf16, shape [NUM_ROWS, K_IN], row-major
    w_ptr: *bf16, shape [K_OUT, K_IN] (note: we pass W^T as [K_IN, K_OUT] from host)
    b_ptr: *bf16, shape [K_OUT]
    y_ptr: *bf16, shape [NUM_ROWS, K_OUT]
    """
    row = tl.program_id(0)
    if row >= NUM_ROWS:
        return
    y_row_ptr = y_ptr + row * K_OUT

    # Accumulator for this output row in fp32
    acc = tl.zeros((K_OUT,), dtype=tl.float32)

    k_out = 0
    while k_out < K_OUT:
        offs_kout = k_out + tl.arange(0, BLOCK_K_OUT)
        mask_kout = offs_kout < K_OUT
        acc[offs_kout] = tl.load(b_ptr + offs_kout, mask=mask_kout, other=0.0).to(tl.float32)

        k_in = 0
        while k_in < K_IN:
            offs_kin = k_in + tl.arange(0, BLOCK_K_IN)
            mask_kin = offs_kin < K_IN
            x_vec = tl.load(x_ptr + row * K_IN + offs_kin, mask=mask_kin, other=0.0).to(tl.float32)  # [BLOCK_K_IN]
            w_vec = tl.load(w_ptr + offs_kin * K_OUT + offs_kout, mask=mask_kin & mask_kout, other=0.0).to(tl.float32)  # [BLOCK_K_IN]
            acc[offs_kout] += tl.sum(x_vec * w_vec, axis=0)
            k_in += BLOCK_K_IN
        k_out += BLOCK_K_OUT

    tl.store(y_row_ptr + tl.arange(0, K_OUT), acc.to(tl.bfloat16))


# Triton kernel: Row-wise GEMM for Linear2 (K_in=6144, K_out=3584). One program per output row.
@triton.jit
def _linear2_row_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                         NUM_ROWS, K_IN, K_OUT,
                         BLOCK_K_IN: tl.constexpr, BLOCK_K_OUT: tl.constexpr):
    """
    x_ptr: *bf16, shape [NUM_ROWS, K_IN], row-major
    w_ptr: *bf16, shape [K_OUT, K_IN] (we pass W2^T as [K_IN, K_OUT] from host)
    b_ptr: *bf16, shape [K_OUT]
    y_ptr: *bf16, shape [NUM_ROWS, K_OUT]
    """
    row = tl.program_id(0)
    if row >= NUM_ROWS:
        return
    y_row_ptr = y_ptr + row * K_OUT

    acc = tl.zeros((K_OUT,), dtype=tl.float32)

    k_out = 0
    while k_out < K_OUT:
        offs_kout = k_out + tl.arange(0, BLOCK_K_OUT)
        mask_kout = offs_kout < K_OUT
        acc[offs_kout] = tl.load(b_ptr + offs_kout, mask=mask_kout, other=0.0).to(tl.float32)

        k_in = 0
        while k_in < K_IN:
            offs_kin = k_in + tl.arange(0, BLOCK_K_IN)
            mask_kin = offs_kin < K_IN
            x_vec = tl.load(x_ptr + row * K_IN + offs_kin, mask=mask_kin, other=0.0).to(tl.float32)
            w_vec = tl.load(w_ptr + offs_kin * K_OUT + offs_kout, mask=mask_kin & mask_kout, other=0.0).to(tl.float32)
            acc[offs_kout] += tl.sum(x_vec * w_vec, axis=0)
            k_in += BLOCK_K_IN
        k_out += BLOCK_K_OUT

    tl.store(y_row_ptr + tl.arange(0, K_OUT), acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters needed; all math is done in Triton

    def forward(self, hidden, grid_thw, ln_weight, ln_bias,
                fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps):
        """
        hidden: (num_patches, hidden_size), bfloat16
        grid_thw: (num_grids, 3), int64 (t, h, w)
        ln_weight, ln_bias: (hidden_size), bfloat16
        fc1_weight: (hidden_size_expanded, hidden_size_expanded) = (6144, 6144), bfloat16
        fc1_bias: (6144), bfloat16
        fc2_weight: (out_hidden_size, hidden_size_expanded) = (3584, 6144), bfloat16
        fc2_bias: (3584), bfloat16
        eps: float
        """
        # Step 1: LayerNorm per row
        num_patches = hidden.shape[0]
        hidden_norm = torch.empty_like(hidden)
        N = num_patches
        C = hidden.shape[1]
        # Choose BLOCK_SIZE as the nearest power-of-two up to C; for simplicity, use 1024
        BLOCK_SIZE = 1024
        _layer_norm_kernel[(N,)](hidden, hidden_norm, ln_weight, ln_bias, N, C, eps, BLOCK_SIZE=BLOCK_SIZE)

        # Step 2: Spatial shuffle (exact 2x2 merge) into a single tensor
        # Compute total_num_merged_patches = sum_{g} t_g * (h_g//2) * (w_g//2)
        num_grids = grid_thw.shape[0]
        # We need to accumulate num_merged_patches across all grids
        total_num_merged = 0
        for g in range(num_grids):
            t = int(grid_thw[g, 0].item())
            h = int(grid_thw[g, 1].item())
            w = int(grid_thw[g, 2].item())
            h_merged = h // 2
            w_merged = w // 2
            total_num_merged += t * h_merged * w_merged

        hidden_shuffled = torch.empty((total_num_merged, 4 * C), dtype=torch.bfloat16, device=hidden.device)

        # Launch shuffle kernel: one program per grid
        # For Triton launch, we need the total_patches; it's just N=num_patches
        _shuffle_2x2_per_grid_kernel[(num_grids,)](hidden_norm, grid_thw, hidden_shuffled,
                                                  num_patches, C, num_grids,
                                                  BLOCK_M=1)

        # Step 3: GELU on shuffled
        hidden_gelu = torch.empty_like(hidden_shuffled)
        _gelu_row_kernel[(hidden_gelu.shape[0],)](hidden_shuffled, hidden_gelu, hidden_gelu.shape[0], hidden_gelu.shape[1], BLOCK_SIZE=256)

        # Step 4: Linear1 in Triton (row-wise GEMM), output y1 [num_merged, 6144]
        y1 = torch.empty((hidden_gelu.shape[0], 6144), dtype=torch.bfloat16, device=hidden.device)
        # Note: fc1_weight is [6144, 6144]; we need W1^T as [6144, 6144] which is already correct for our kernel.
        _linear1_row_kernel[(hidden_gelu.shape[0],)](hidden_gelu, fc1_weight, fc1_bias, y1,
                                                     hidden_gelu.shape[0], 6144, 6144,
                                                     BLOCK_K_IN=128, BLOCK_K_OUT=128)

        # GELU on y1
        y1_gelu = torch.empty_like(y1)
        _gelu_row_kernel[(y1.shape[0],)](y1, y1_gelu, y1.shape[0], y1.shape[1], BLOCK_SIZE=256)

        # Step 5: Linear2 in Triton, output [num_merged, 3584]
        output = torch.empty((y1_gelu.shape[0], 3584), dtype=torch.bfloat16, device=hidden.device)
        _linear2_row_kernel[(y1_gelu.shape[0],)](y1_gelu, fc2_weight, fc2_bias, output,
                                                 y1_gelu.shape[0], 6144, 3584,
                                                 BLOCK_K_IN=128, BLOCK_K_OUT=128)

        return output


def run(*args):
    return ModelNew()(*args)
