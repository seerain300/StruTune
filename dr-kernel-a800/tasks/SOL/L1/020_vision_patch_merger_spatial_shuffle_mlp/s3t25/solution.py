import torch
import math
import triton
import triton.language as tl

# Triton kernel: LayerNorm per row. One program per row.
@triton.jit
def _layer_norm_rows_kernel(x_ptr, y_ptr, ln_weight_ptr, ln_bias_ptr,
                             N, C, eps,
                             BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    y_ptr: *bf16, shape [N, C]
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


# Triton kernel: Exact spatial shuffle across all grids. Produces hidden_shuffled
# of shape [total_num_merged_patches, 4*C], where total_num_merged_patches = sum_{g} t_g * (h_g//2) * (w_g//2).
@triton.jit
def _shuffle_2x2_all_grids_kernel(hidden_ptr, grid_thw_ptr, out_ptr,
                                  total_patches, C, NUM_GRIDS, NUM_OUT_ROWS,
                                  BLOCK_M: tl.constexpr):
    """
    hidden_ptr: *bf16, flattened [total_patches, C]
    grid_thw_ptr: *int64, shape [NUM_GRIDS, 3], each row is [t, h, w]
    out_ptr: *bf16, flattened [NUM_OUT_ROWS, 4*C]
    We launch one program to process all grids sequentially. It computes total NUM_OUT_ROWS
    and fills out_ptr accordingly. We pass NUM_OUT_ROWS as meta-parameter for correct sizing.
    """
    # This kernel runs as a single program to compute all merged rows.
    # We iterate over grids, then over all original patches within each grid.
    # Since we don't have per-grid parallelism here, we keep it simple and sequential.
    # However, Triton encourages 1D grid; thus we implement a single-program loop over grids.
    g = 0
    while g < NUM_GRIDS:
        # Load grid dimensions
        t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
        h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
        w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)

        h_merged = h // 2
        w_merged = w // 2
        num_merged_rows_g = t * h_merged * w_merged

        # For each original patch m in this grid
        m = 0
        while m < t * h * w:
            t_index = m // (h * w)
            rem = m % (h * w)
            h2 = rem // w
            w2 = rem % w

            out_row = t_index * (h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)
            # Each out_row has 4*C columns corresponding to the 2x2 merge positions flattened in order:
            # 0..C-1 for (r=0,c=0), C..2*C-1 for (r=0,c=1), 2*C..3*C-1 for (r=1,c=0), 3*C..4*C-1 for (r=1,c=1)

            # idx0: (r=0,c=0) -> hidden[t_index, h2, w2]
            idx0 = 0
            col0 = idx0 * C + 0
            src_offset0 = t_index * (h * w) + h2 * w + w2
            val0 = tl.load(hidden_ptr + src_offset0 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(out_ptr + out_row * (4 * C) + col0, val0)

            # idx1: (r=0,c=1) -> hidden[t_index, h2, w2+1]
            idx1 = 1
            col1 = idx1 * C + 0
            if w2 + 1 < w:
                src_offset1 = t_index * (h * w) + h2 * w + (w2 + 1)
                val1 = tl.load(hidden_ptr + src_offset1 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
                tl.store(out_ptr + out_row * (4 * C) + col1, val1)

            # idx2: (r=1,c=0) -> hidden[t_index, h2+1, w2]
            idx2 = 2
            col2 = idx2 * C + 0
            if h2 + 1 < h and w2 < w:
                src_offset2 = t_index * (h * w) + (h2 + 1) * w + w2
                val2 = tl.load(hidden_ptr + src_offset2 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
                tl.store(out_ptr + out_row * (4 * C) + col2, val2)

            # idx3: (r=1,c=1) -> hidden[t_index, h2+1, w2+1]
            idx3 = 3
            col3 = idx3 * C + 0
            if h2 + 1 < h and w2 + 1 < w:
                src_offset3 = t_index * (h * w) + (h2 + 1) * w + (w2 + 1)
                val3 = tl.load(hidden_ptr + src_offset3 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
                tl.store(out_ptr + out_row * (4 * C) + col3, val3)

            m += 1
        g += 1


# Triton kernel: Elementwise GELU activation over a flattened tensor [N, C].
@triton.jit
def _gelu_kernel(x_ptr, y_ptr, N, C, BLOCK_SIZE: tl.constexpr):
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
        # Triton does not provide erf directly; we approximate:
        # erf(z) ≈ sign(z) * (1 - exp(-z^2) * (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5)),
        # where t = 1 / (1 + p z), p = 0.3275911
        z = x * inv_sqrt2
        # constants for approximation
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        sign = tl.where(z >= 0.0, 1.0, -1.0)
        az = tl.abs(z)
        t = 1.0 / (1.0 + p * az)
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_approx = sign * (1.0 - poly * tl.exp(-az * az))
        y = 0.5 * x * (1.0 + erf_approx)
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


# Triton kernel: Row-wise GEMM for Linear1: y = x @ W1.T + b1
# x: [N, K], W1: [K, K], y: [N, K], but here K=6144, out_dim=6144 (same as input)
@triton.jit
def _linear1_rows_kernel(x_ptr, w1_ptr, b1_ptr, y_ptr,
                          N, K, BLOCK_K: tl.constexpr):
    """
    Compute y[i, :] = sum_{k=0}^{K-1} x[i, k] * w1[k, :] + b1
    We launch one program per output row i, reduce across K in blocks.
    """
    i = tl.program_id(0)
    if i >= N:
        return
    # x_row_ptr = x_ptr + i * K
    # y_row_ptr = y_ptr + i * K
    k = 0
    acc = tl.zeros([K], dtype=tl.float32)
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        # Load x[i, offs_k]
        x_vec = tl.load(x_ptr + i * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        # Load W1[offs_k, :] which is a row of length K
        w_row_ptr = w1_ptr + offs_k * K
        w_vec = tl.load(w_row_ptr + tl.arange(0, K), mask=mask_k, other=0.0).to(tl.float32)
        # Multiply and reduce
        prod = x_vec * w_vec
        acc += tl.sum(prod, axis=0)
        k += BLOCK_K
    # Add bias
    b = tl.load(b1_ptr + tl.arange(0, K), mask=True, other=0.0).to(tl.float32)
    acc += b
    # Store result
    tl.store(y_ptr + i * K + tl.arange(0, K), acc.to(tl.bfloat16))


# Triton kernel: Row-wise GEMM for Linear2: z = y @ W2.T + b2
# y: [N, 6144], W2: [3584, 6144], z: [N, 3584]
@triton.jit
def _linear2_rows_kernel(y_ptr, w2_ptr, b2_ptr, z_ptr,
                          N, K_IN, OUT_DIM, BLOCK_K: tl.constexpr):
    """
    Compute z[i, :] = sum_{k=0}^{K_IN-1} y[i, k] * w2[:, k] + b2
    One program per output row i.
    """
    i = tl.program_id(0)
    if i >= N:
        return
    acc = tl.zeros([OUT_DIM], dtype=tl.float32)
    k = 0
    while k < K_IN:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_IN
        # y[i, offs_k]
        y_vec = tl.load(y_ptr + i * K_IN + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        # Load column vector w2[:, offs_k] which has length OUT_DIM
        w_col_ptr = w2_ptr + offs_k * OUT_DIM
        w_vec = tl.load(w_col_ptr + tl.arange(0, OUT_DIM), mask=mask_k, other=0.0).to(tl.float32)
        # Multiply and reduce across k into acc
        prod = y_vec * w_vec
        acc += tl.sum(prod, axis=0)
        k += BLOCK_K
    # Add bias b2
    b = tl.load(b2_ptr + tl.arange(0, OUT_DIM), mask=True, other=0.0).to(tl.float32)
    acc += b
    # Store result
    tl.store(z_ptr + i * OUT_DIM + tl.arange(0, OUT_DIM), acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from the original code
        self.hidden_size = 1536
        self.hidden_size_expanded = 4 * self.hidden_size  # 6144
        self.out_hidden_size = 3584
        self.eps = 1e-6

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor):
        """
        hidden: [num_patches, hidden_size] (bfloat16)
        grid_thw: [num_grids, 3], int64
        ln_weight, ln_bias: [hidden_size], bfloat16
        fc1_weight: [hidden_size_expanded, hidden_size_expanded] (6144x6144), bfloat16
        fc1_bias: [hidden_size_expanded], bfloat16
        fc2_weight: [out_hidden_size, hidden_size_expanded] (3584x6144), bfloat16
        fc2_bias: [out_hidden_size], bfloat16
        """
        # 1) LayerNorm per row (fp32 reduction + affine in fp32, store bf16)
        hidden_norm = torch.empty_like(hidden)  # [num_patches, hidden_size], bf16
        N = hidden.shape[0]
        C = hidden.shape[1]
        # Launch one program per row
        grid0 = (N,)
        _layer_norm_rows_kernel[grid0](
            hidden, hidden_norm, ln_weight, ln_bias,
            N, C, self.eps,
            BLOCK_SIZE=1024,  # reduction block
        )

        # 2) Spatial shuffle across all grids to produce hidden_shuffled
        # We need total_num_merged_patches = sum_{g} t_g * (h_g//2) * (w_g//2)
        total_merged = 0
        num_grids = grid_thw.shape[0]
        for g in range(num_grids):
            t = int(grid_thw[g, 0].item())
            h = int(grid_thw[g, 1].item())
            w = int(grid_thw[g, 2].item())
            total_merged += t * (h // 2) * (w // 2)

        hidden_expanded = torch.empty((total_merged, self.hidden_size_expanded), dtype=torch.bfloat16, device=hidden.device)

        # Launch single-program kernel to fill hidden_expanded
        # We pass NUM_OUT_ROWS and NUM_GRIDS as meta-parameters. But Triton requires constexpr;
        # Triton accepts passing Python ints as constexpr for small loops.
        _shuffle_2x2_all_grids_kernel[(1,)](
            hidden_norm, grid_thw, hidden_expanded,
            hidden_norm.numel(), self.hidden_size, num_grids, total_merged,
            BLOCK_M=1024,  # not used in inner loops but kept for signature
        )

        # 3) GELU activation (elementwise)
        hidden_gelu = torch.empty_like(hidden_expanded)
        N_g = hidden_expanded.shape[0]
        C_exp = hidden_expanded.shape[1]
        grid_gelu = (N_g,)
        _gelu_kernel[grid_gelu](
            hidden_expanded, hidden_gelu,
            N_g, C_exp,
            BLOCK_SIZE=1024,
        )

        # 4) Linear1: hidden_gelu @ fc1_weight.T + fc1_bias, output [N_g, 6144]
        y = torch.empty((N_g, self.hidden_size_expanded), dtype=torch.bfloat16, device=hidden.device)
        K = self.hidden_size_expanded  # 6144
        # Launch one program per row
        grid_lin1 = (N_g,)
        _linear1_rows_kernel[grid_lin1](
            hidden_gelu, fc1_weight, fc1_bias, y,
            N_g, K,
            BLOCK_K=1024,
        )

        # 5) Linear2: y @ fc2_weight.T + fc2_bias, output [N_g, 3584]
        z = torch.empty((N_g, self.out_hidden_size), dtype=torch.bfloat16, device=hidden.device)
        OUT_DIM = self.out_hidden_size  # 3584
        grid_lin2 = (N_g,)
        _linear2_rows_kernel[grid_lin2](
            y, fc2_weight, fc2_bias, z,
            N_g, K, OUT_DIM,
            BLOCK_K=1024,
        )

        return z


def run(*args):
    return ModelNew()(*args)
