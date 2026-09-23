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


# Triton kernel: Exact 2x2 spatial shuffle per grid. Produces hidden_shuffled of shape
# [total_num_merged_patches, 4*C], where total_num_merged_patches = sum_{g} t_g * (h_g//2) * (w_g//2).
@triton.jit
def _shuffle_2x2_per_grid_kernel(hidden_ptr, grid_thw_ptr, out_ptr,
                                 total_patches, C, NUM_GRIDS,
                                 BLOCK_ROWS: tl.constexpr):
    """
    hidden_ptr: *bf16, flattened per grid. For grid g, hidden entries start at base = g * total_patches * C.
                 The kernel treats hidden_ptr as a single contiguous array and computes base per iteration.
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
    total_per_grid = t * h * w

    base_hidden = g * total_patches * C

    # For each original patch m in [0, t*h*w):
    m = 0
    while m < total_per_grid:
        t_index = m // (h * w)
        rem = m % (h * w)
        h2 = rem // w
        w2 = rem % w

        out_row = t_index * (h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)
        base_out = out_row * (4 * C)

        # idx=0: (r=0,c=0) -> hidden[t_index, h2, w2]
        src_offset0 = t_index * (h * w) + h2 * w + w2
        val0 = tl.load(hidden_ptr + base_hidden + src_offset0 * C + 0, mask=1, other=0.0).to(tl.bfloat16)
        tl.store(out_ptr + base_out + 0 * C, val0)

        # idx=1: (r=0,c=1) -> hidden[t_index, h2, w2+1]
        src_offset1 = t_index * (h * w) + h2 * w + (w2 + 1)
        val1 = tl.load(hidden_ptr + base_hidden + src_offset1 * C + 0, mask=(w2 + 1) < w, other=0.0).to(tl.bfloat16)
        tl.store(out_ptr + base_out + 1 * C, val1)

        # idx=2: (r=1,c=0) -> hidden[t_index, h2+1, w2]
        src_offset2 = t_index * (h * w) + (h2 + 1) * w + w2
        val2 = tl.load(hidden_ptr + base_hidden + src_offset2 * C + 0, mask=(h2 + 1) < h and (w2) < w, other=0.0).to(tl.bfloat16)
        tl.store(out_ptr + base_out + 2 * C, val2)

        # idx=3: (r=1,c=1) -> hidden[t_index, h2+1, w2+1]
        src_offset3 = t_index * (h * w) + (h2 + 1) * w + (w2 + 1)
        val3 = tl.load(hidden_ptr + base_hidden + src_offset3 * C + 0, mask=(h2 + 1) < h and (w2 + 1) < w, other=0.0).to(tl.bfloat16)
        tl.store(out_ptr + base_out + 3 * C, val3)

        m += 1


# Triton kernel: GELU activation, elementwise on a vector of length N*C. One program per row.
@triton.jit
def _gelu_kernel(x_ptr, y_ptr, N, C, BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    y_ptr: *bf16, shape [N, C]
    GELU implementation: 0.5 * x * (1 + erf(x / sqrt(2)))
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


class ModelNew(torch.nn.Module):
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
        Triton-only implementation of the original forward:
        1) LayerNorm (pre-shuffle) on hidden [num_patches, hidden_size] using Triton
        2) Spatial shuffle to merge 2x2 patches using Triton per-grid kernel
        3) GELU activation using Triton elementwise kernel
        4) MLP layers using torch.nn.functional.linear (to keep correctness)
        Returns: output [num_merged_patches, out_hidden_size]
        """
        device = hidden.device
        dtype = hidden.dtype

        # 1) LayerNorm: y0 = LN(hidden) using Triton
        num_patches, hidden_size = hidden.shape
        hidden_ln = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        # Launch Triton kernel one program per row
        grid = (num_patches,)
        _layer_norm_kernel[grid](
            hidden, hidden_ln, ln_weight, ln_bias,
            num_patches, hidden_size, eps,
            BLOCK_SIZE=128,
        )

        # 2) Spatial shuffle per grid: hidden_shuffled of shape
        #    total_num_merged_patches x 4*hidden_size
        num_grids = grid_thw.shape[0]
        total_per_grid = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).to(torch.int64).tolist()
        total_num_merged_patches = sum(total_per_grid)
        hidden_expanded = 4 * hidden_size  # hidden_size_expanded

        # Allocate output shuffled tensor
        hidden_shuffled = torch.empty((total_num_merged_patches, hidden_expanded), dtype=torch.bfloat16, device=device)

        # Launch Triton shuffle per-grid kernel (one program per grid)
        grid2 = (num_grids,)
        _shuffle_2x2_per_grid_kernel[grid2](
            hidden_ln, grid_thw, hidden_shuffled,
            num_patches, hidden_size, num_grids,
            BLOCK_ROWS=1,  # iterate rows with while in kernel
        )

        # 3) GELU activation using Triton kernel on shuffled hidden
        num_merged = hidden_shuffled.shape[0]
        hidden_gelu = torch.empty_like(hidden_shuffled, dtype=torch.bfloat16, device=device)
        grid3 = (num_merged,)
        _gelu_kernel[grid3](
            hidden_shuffled, hidden_gelu,
            num_merged, hidden_expanded,
            BLOCK_SIZE=256,
        )

        # 4) MLP layers: use torch for linear layers
        # Linear1: hidden_gelu @ fc1_weight.T + fc1_bias
        y1 = torch.nn.functional.linear(hidden_gelu.to(torch.float32), fc1_weight.to(torch.float32), fc1_bias.to(torch.float32))
        y1 = y1.to(torch.bfloat16)

        # GELU after Linear1: original code applies GELU before fc2, so we apply GELU here
        y1_g = torch.empty_like(y1, dtype=torch.bfloat16, device=device)
        grid4 = (num_merged,)
        _gelu_kernel[grid4](
            y1, y1_g,
            num_merged, hidden_expanded,
            BLOCK_SIZE=256,
        )

        # Linear2: y1_g @ fc2_weight.T + fc2_bias
        output = torch.nn.functional.linear(y1_g.to(torch.float32), fc2_weight.to(torch.float32), fc2_bias.to(torch.float32))
        output = output.to(torch.bfloat16)

        return output


def run(*args):
    return ModelNew()(*args)
