import torch
import math
import triton
import triton.language as tl


# Triton kernel: LayerNorm per row (reduce then apply). One program per row.
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


# Triton kernel: Exact spatial shuffle per grid. Produces hidden_shuffled of shape
# [total_num_merged_patches, 4*C], where total_num_merged_patches = sum_{g} t_g * (h_g//2) * (w_g//2).
@triton.jit
def _shuffle_2x2_per_grid_kernel(hidden_ptr, grid_thw_ptr, out_ptr,
                                 total_patches, C, NUM_GRIDS,
                                 BLOCK_M: tl.constexpr):
    """
    hidden_ptr: *bf16, flattened [total_patches, C]
    grid_thw_ptr: *int64, shape [NUM_GRIDS, 3], each row is [t, h, w]
    out_ptr: *bf16, flattened [total_merged_rows, 4*C]
    We launch one program per grid.
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

    # Process all original patches in this grid
    m = 0
    while m < t * h * w:
        # Decode (t_index, h2, w2) from linear index m
        t_index = m // (h * w)
        rem = m % (h * w)
        h2 = rem // w
        w2 = rem % w

        # Compute output row index in the merged tensor
        out_row = t_index * (h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)
        base_out = out_ptr + out_row * (4 * C)

        # Column indices correspond to the 2x2 merge positions
        # idx=0: (r=0,c=0), idx=1:(0,1), idx=2:(1,0), idx=3:(1,1)
        idx = 0
        row2 = h2 * 2 + 0
        col2 = w2 * 2 + 0
        col0 = row2 * w + col2 * C  # scalar index in [0, 4*C)
        src_offset0 = t_index * (h * w) + h2 * w + w2
        val0 = tl.load(hidden_ptr + src_offset0 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
        tl.store(base_out + idx * C, val0)

        idx = 1
        row2 = h2 * 2 + 0
        col2 = w2 * 2 + 1
        col1 = row2 * w + col2 * C
        src_offset1 = src_offset0
        val1 = tl.load(hidden_ptr + src_offset1 * C + 1, mask=True, other=0.0).to(tl.bfloat16)
        tl.store(base_out + idx * C, val1)

        idx = 2
        row2 = h2 * 2 + 1
        col2 = w2 * 2 + 0
        col2_idx = row2 * w + col2 * C
        src_offset2 = (t_index * (h * w)) + (h2 + 1) * w + w2
        val2 = tl.load(hidden_ptr + src_offset2 * C + 2, mask=True, other=0.0).to(tl.bfloat16)
        tl.store(base_out + idx * C, val2)

        idx = 3
        row2 = h2 * 2 + 1
        col2 = w2 * 2 + 1
        col3 = row2 * w + col2 * C
        src_offset3 = src_offset2
        val3 = tl.load(hidden_ptr + src_offset3 * C + 3, mask=True, other=0.0).to(tl.bfloat16)
        tl.store(base_out + idx * C, val3)

        m += 1


# Triton kernel: GELU activation, elementwise on a vector. One program per row.
@triton.jit
def _gelu_kernel(x_ptr, y_ptr, N, C, BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    y_ptr: *bf16, shape [N, C]
    GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
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
        # erf approximation (Abramowitz & Stegun 7.1.26)
        # erf(z) ≈ sign(z) * (1 - t * exp(-z^2) * (a1 + a2 t + a3 t^2 + a4 t^3 + a5 t^4)),
        # where t = 1 / (1 + p z), p=0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = tl.where(z >= 0.0, 1.0, -1.0)
        z_abs = tl.abs(z)
        t = 1.0 / (1.0 + p * z_abs)
        # Horner’s method for polynomial
        poly = a5
        poly = poly * t + a4
        poly = poly * t + a3
        poly = poly * t + a2
        poly = poly * t + a1
        poly = poly * t
        erf_z = sign * (1.0 - poly * tl.exp(-z_abs * z_abs))
        y = 0.5 * x * (1.0 + erf_z)
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


# Triton kernel: Linear layer for dim=6144, K=6144, output=6144. Row-wise computation.
@triton.jit
def _linear6144x6144_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                            N, K_IN, K_OUT,
                            BLOCK_K: tl.constexpr):
    # Each program computes one output row of y
    row = tl.program_id(0)
    if row >= N:
        return

    x_row_ptr = x_ptr + row * K_IN
    y_row_ptr = y_ptr + row * K_OUT

    # Initialize accumulator for this row
    acc = tl.zeros((K_OUT,), dtype=tl.float32)

    # Reduce across K_IN in blocks
    k0 = 0
    while k0 < K_IN:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_IN

        # Load x_block: shape [BLOCK_K]
        x_block = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        # Load W_block: shape [BLOCK_K, K_OUT] (row-major: contiguous along K_OUT)
        # We'll load W[k, :] for each k in the block and accumulate
        k_out_total = 0
        while k_out_total < K_OUT:
            offs_k_out = k_out_total + tl.arange(0, BLOCK_K)
            mask_k_out = offs_k_out < K_OUT
            # Build 2D pointer for W_block
            w_ptrs = w_ptr + offs_k[:, None] * K_OUT + offs_k_out[None, :]
            w_block = tl.load(w_ptrs, mask=mask_k[:, None] & mask_k_out[None, :], other=0.0).to(tl.float32)
            # Accumulate: acc += sum_k x_block[k] * w_block[k, :]
            # Do per-k accumulation
            # acc += sum over k of x_block[k] * w_block[k, :]
            # Triton doesn't support dynamic sum of vectors easily; we'll loop in k manually.
            for kk in range(BLOCK_K):
                # If kk >= actual K_IN - k0, x_block[kk] is masked and zero; mask ensures safety
                x_val = tl.where(mask_k[kk], x_block[kk], 0.0)
                # Load w_vec for this kk across K_OUT
                w_vec = tl.load(w_ptr + (k0 + kk) * K_OUT + offs_k_out, mask=mask_k_out, other=0.0).to(tl.float32)
                acc += x_val * w_vec
            k_out_total += BLOCK_K

        k0 += BLOCK_K

    # Add bias
    b = tl.load(b_ptr + tl.arange(0, BLOCK_K), mask=tl.arange(0, BLOCK_K) < K_OUT, other=0.0).to(tl.float32)
    acc += b
    tl.store(y_row_ptr, acc.to(tl.bfloat16))  # store the whole row at once


# Triton kernel: Linear layer for input 6144, K_in=6144, output=3584. Row-wise.
@triton.jit
def _linear6144x3584_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                            N, K_IN, K_OUT,
                            BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    if row >= N:
        return

    x_row_ptr = x_ptr + row * K_IN
    y_row_ptr = y_ptr + row * K_OUT

    acc = tl.zeros((K_OUT,), dtype=tl.float32)

    k0 = 0
    while k0 < K_IN:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_IN

        x_block = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        k_out_total = 0
        while k_out_total < K_OUT:
            offs_k_out = k_out_total + tl.arange(0, BLOCK_K)
            mask_k_out = offs_k_out < K_OUT

            w_ptrs = w_ptr + offs_k[:, None] * K_OUT + offs_k_out[None, :]
            w_block = tl.load(w_ptrs, mask=mask_k[:, None] & mask_k_out[None, :], other=0.0).to(tl.float32)

            for kk in range(BLOCK_K):
                x_val = tl.where(mask_k[kk], x_block[kk], 0.0)
                w_vec = tl.load(w_ptr + (k0 + kk) * K_OUT + offs_k_out, mask=mask_k_out, other=0.0).to(tl.float32)
                acc += x_val * w_vec

            k_out_total += BLOCK_K

        k0 += BLOCK_K

    b = tl.load(b_ptr + tl.arange(0, BLOCK_K), mask=tl.arange(0, BLOCK_K) < K_OUT, other=0.0).to(tl.float32)
    acc += b
    tl.store(y_row_ptr, acc.to(tl.bfloat16))


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
        hidden: [num_patches, hidden_size], bfloat16
        grid_thw: [num_grids, 3], int64, each row is [t, h, w]
        ln_weight, ln_bias: [hidden_size], bfloat16
        fc1_weight, fc1_bias: [hidden_size_expanded, hidden_size_expanded], bfloat16
        fc2_weight, fc2_bias: [out_hidden_size, hidden_size_expanded], bfloat16
        eps: float
        """
        device = hidden.device
        num_patches = hidden.shape[0]
        C = hidden.shape[1]
        hidden_size_expanded = fc1_weight.shape[1]  # 6144
        out_hidden_size = fc2_weight.shape[0]       # 3584

        # 1) LayerNorm (pre-shuffle)
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        # Choose BLOCK_SIZE as next power of two for C up to 16384, but here C=1536 -> 1024 is fine
        # We use 1024 here as constexpr
        _layer_norm_kernel[(num_patches,)](hidden, hidden_norm, ln_weight, ln_bias,
                                           num_patches, C, eps, BLOCK_SIZE=1024)

        # 2) Spatial shuffle (2x2 merge) into a single tensor
        # Compute total number of merged rows
        total_merged_rows = 0
        for i in range(grid_thw.shape[0]):
            t = grid_thw[i, 0].item()
            h = grid_thw[i, 1].item()
            w = grid_thw[i, 2].item()
            h_merged = h // 2
            w_merged = w // 2
            total_merged_rows += t * h_merged * w_merged

        hidden_shuffled = torch.empty((total_merged_rows, 4 * C), dtype=torch.bfloat16, device=device)

        _shuffle_2x2_per_grid_kernel[(grid_thw.shape[0],)](
            hidden_norm, grid_thw, hidden_shuffled,
            num_patches, C, grid_thw.shape[0],
            BLOCK_M=1  # one program per grid, process all m in while-loop
        )

        N_merged = hidden_shuffled.shape[0]  # should equal total_merged_rows per configuration

        # 3) Linear1: y1 = x @ fc1_weight.T + fc1_bias, where x has 4*C features (hidden_size_expanded)
        # Shapes: x [N_merged, 6144], W [6144, 6144], b [6144]
        y1 = torch.empty((N_merged, 6144), dtype=torch.bfloat16, device=device)
        # Note: We will launch one program per row; since N_merged can be large, this kernel runs over all rows.
        # Using BLOCK_K=128 for K reduction chunks.
        grid = (N_merged,)
        _linear6144x6144_kernel[grid](hidden_shuffled, fc1_weight, fc1_bias, y1,
                                      N_merged, 6144, 6144, BLOCK_K=128)

        # 4) GELU activation
        y1_gelu = torch.empty_like(y1, dtype=torch.bfloat16, device=device)
        _gelu_kernel[(N_merged,)](y1, y1_gelu, N_merged, 6144, BLOCK_SIZE=256)

        # 5) Linear2: y2 = y1_gelu @ fc2_weight.T + fc2_bias, where output is 3584
        y2 = torch.empty((N_merged, 3584), dtype=torch.bfloat16, device=device)
        _linear6144x3584_kernel[(N_merged,)](
            y1_gelu, fc2_weight, fc2_bias, y2,
            N_merged, 6144, 3584, BLOCK_K=128
        )

        return y2


def run(*args):
    return ModelNew()(*args)
