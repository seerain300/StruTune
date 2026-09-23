import torch
import math
import triton
import triton.language as tl

# Triton kernel: LayerNorm per row (reduce then apply). One program per row.
@triton.jit
def _layer_norm_rows(x_ptr, y_ptr, ln_weight_ptr, ln_bias_ptr,
                     N, C, eps, BLOCK_SIZE: tl.constexpr):
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

    # Pass 1: compute mean and variance (fp32) using a block vector
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


# Triton kernel: Build hidden_shuffled exactly per grid, writing into a single output tensor.
# Original logic: for each grid g, given t,h,w, create patches as hidden[t,h,w], then shuffle
# 2x2 into contiguous columns: for each original patch (t_index, h2, w2), out_row maps to
# t_index * (h//2) * (w//2) + (h2//2) * (w//2) + (w2//2), and columns are four positions in
# the 2x2 patch at (h2*2, w2*2) corresponding to C features.
@triton.jit
def _shuffle_per_grid_kernel(hidden_norm_ptr, grid_thw_ptr, out_ptr,
                             total_patches, C, NUM_GRIDS,
                             BLOCK_M: tl.constexpr):
    """
    hidden_norm_ptr: *bf16, flattened [total_patches, C]
    grid_thw_ptr: *int64, shape [NUM_GRIDS, 3], each row is [t, h, w]
    out_ptr: *bf16, flattened [total_merged_rows, 4*C]
    One program per grid.
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

    # Vectorize over out rows; we write one out_row per loop iteration for correctness
    # but Triton likes static BLOCK_M for vector operations. We'll iterate scalar m and
    # compute out_row and four columns. This avoids dynamic while-loops and ensures
    # exact mapping.
    m = 0
    while m < t * h * w:
        t_index = m // (h * w)
        rem = m % (h * w)
        h2 = rem // w
        w2 = rem % w

        out_row = t_index * (h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)
        base_out = out_ptr + out_row * (4 * C)

        # idx 0: (r=0,c=0) -> hidden[t_index, h2, w2]
        col0 = (h2 * 2 + 0) * w * C + (w2 * 2 + 0) * C
        src0 = t_index * (h * w) + h2 * w + w2
        val0 = tl.load(hidden_norm_ptr + src0 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
        tl.store(base_out + 0 * C, val0)

        # idx 1: (r=0,c=1) -> hidden[t_index, h2, w2+1]
        if w2 + 1 < w:
            col1 = col0 + C
            val1 = tl.load(hidden_norm_ptr + src0 * C + C, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(base_out + 1 * C, val1)

        # idx 2: (r=1,c=0) -> hidden[t_index, h2+1, w2]
        if h2 + 1 < h:
            src2 = t_index * (h * w) + (h2 + 1) * w + w2
            col2 = (h2 * 2 + 1) * w * C + (w2 * 2 + 0) * C
            val2 = tl.load(hidden_norm_ptr + src2 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(base_out + 2 * C, val2)

        # idx 3: (r=1,c=1) -> hidden[t_index, h2+1, w2+1]
        if (h2 + 1 < h) and (w2 + 1 < w):
            src3 = t_index * (h * w) + (h2 + 1) * w + (w2 + 1)
            col3 = (h2 * 2 + 1) * w * C + (w2 * 2 + 1) * C
            val3 = tl.load(hidden_norm_ptr + src3 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(base_out + 3 * C, val3)

        m += 1


# Triton kernel: Row-wise matmul + bias (y = x @ W.T + b), one program per output row.
# x: *bf16 [1, K_in] (we pass single row vector), W: *bf16 [K_out, K_in], b: *bf16 [K_out]
@triton.jit
def _row_matmul_bias_kernel(x_ptr, W_ptr, b_ptr, y_ptr,
                             K_in, K_out, eps, BLOCK_K: tl.constexpr):
    """
    Compute y[i] = sum_{k=0}^{K_in-1} x[k] * W[k, i] + b[i]
    This kernel expects x_ptr to point to a single row vector of length K_in (we pass it
    by flattening and indexing row 0), W_ptr is [K_out, K_in], b_ptr [K_out], y_ptr [K_out].
    All computation done in fp32, store as bfloat16.
    """
    row = tl.program_id(0)  # since we launch grid=(K_out,), row==0, but we keep general
    # NOTE: this kernel is launched with grid=(K_out,) and we pass x as a single row vector.
    # We will only process the first row (row=0) here. Triton requires single program index,
    # so we implement a scalar row. This is fine for our use: we will launch it with grid=(N_out,)
    # where N_out = total_num_merged_patches.
    # We need to load x row vector: x_row_ptr = x_ptr + row * K_in, but we launch with grid=(K_out,).
    # Instead, we pass x as a flattened tensor of length K_in and set row=0. To make it general,
    # we pass x as a pointer to the first row and rely on host code to pass a single-row tensor.
    # Here we assume x_ptr points to the single row (which the host sets). If row != 0, return.
    if row != 0:
        return

    x_row_ptr = x_ptr
    W_row_ptr = W_ptr  # pointer to W; we will iterate k in blocks
    y_row_ptr = y_ptr + row * K_out

    acc = tl.zeros((K_out,), dtype=tl.float32)

    k = 0
    while k < K_in:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_in
        x = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        Wk = tl.load(W_row_ptr + offs_k * K_out, mask=mask_k, other=0.0).to(tl.float32)
        acc += x * Wk
        k += BLOCK_K

    # Add bias
    b = tl.load(b_ptr + tl.arange(0, K_out), mask=True, other=0.0).to(tl.float32)
    acc += b

    # Store result
    tl.store(y_row_ptr + tl.arange(0, K_out), acc.to(tl.bfloat16))


# Triton kernel: Elementwise GELU over a row vector. One program per row.
@triton.jit
def _gelu_row_kernel(x_ptr, y_ptr, C, BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    y_ptr: *bf16, shape [N, C]
    GELU: y = 0.5 * x * (1 + erf(x / sqrt(2)))
    Use a standard approximation for erf to avoid relying on tl.erf.
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
        # erf approximation: erf(z) ≈ sign(z) * (1 - exp(-z^2) * (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)),
        # t = 1 / (1 + p*z), p=0.3275911
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        sign = tl.where(z >= 0, 1.0, -1.0)
        az = tl.abs(z)
        t = 1.0 / (1.0 + p * az)
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_z = sign * (1.0 - poly * tl.exp(-az * az))
        y = 0.5 * x * (1.0 + erf_z)
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        # Fixed sizes per original code
        self.hidden_size = 1536
        self.hidden_size_expanded = 6144
        self.out_hidden_size = 3584
        self.merge_size = 2

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor):
        """
        hidden: [num_patches, hidden_size] bfloat16
        grid_thw: [num_grids, 3] int64 (t,h,w per grid)
        ln_weight, ln_bias: [hidden_size] bfloat16
        fc1_weight: [hidden_size_expanded, hidden_size_expanded] bfloat16
        fc1_bias: [hidden_size_expanded] bfloat16
        fc2_weight: [out_hidden_size, hidden_size_expanded] bfloat16
        fc2_bias: [out_hidden_size] bfloat16
        """
        device = hidden.device
        N = hidden.shape[0]
        C = hidden.shape[1]

        # 1) LayerNorm on hidden (per-row)
        hidden_norm = torch.empty_like(hidden)
        # launch Triton LayerNorm: one program per row
        grid = (N,)
        _layer_norm_rows[grid](
            hidden, hidden_norm, ln_weight, ln_bias,
            N, C, self.eps,
            BLOCK_SIZE=256
        )

        # 2) Spatial shuffle per grid: construct hidden_shuffled [total_num_merged_patches, 4*C]
        total_patches = N  # original code uses all patches concatenated
        # We need total_num_merged_patches = sum_g t_g * (h_g//2) * (w_g//2)
        total_merged = 0
        for g in range(grid_thw.shape[0]):
            t = int(grid_thw[g, 0].item())
            h = int(grid_thw[g, 1].item())
            w = int(grid_thw[g, 2].item())
            total_merged += t * (h // 2) * (w // 2)

        hidden_shuffled = torch.empty((total_merged, 4 * C), dtype=torch.bfloat16, device=device)

        # launch Triton shuffle per grid: one program per grid
        _shuffle_per_grid_kernel[(grid_thw.shape[0],)](
            hidden_norm, grid_thw, hidden_shuffled,
            total_patches, C, grid_thw.shape[0],
            BLOCK_M=1
        )

        # 3) Linear1: hidden_shuffled @ fc1_weight.T + fc1_bias
        M1 = hidden_shuffled.shape[0]
        K_in1 = hidden_shuffled.shape[1]  # 4*C
        K_out1 = fc1_weight.shape[0]      # 6144
        output1 = torch.empty((M1, K_out1), dtype=torch.bfloat16, device=device)

        # Row-wise matmul for Linear1: one program per output row
        # We need to pass x as a single row vector for each iteration. Triton grid is (M1,)
        # and we'll launch once per row. However, Triton kernels are static at compile-time,
        # so we launch grid=(M1,) with row index corresponding to program_id(0).
        # But _row_matmul_bias_kernel expects x as pointer to a single row vector, not 2D.
        # Implement a loop over rows in host, or call the kernel per row via grid (not supported directly).
        # Instead, we will launch a 1D grid where each program computes its own row.
        # For correctness, we will loop over rows in host and call the kernel for each row:
        # This ensures Triton-only computation and avoids torch ops.
        for i in range(M1):
            # Slice x_row as a 1D tensor of length K_in1
            x_row = hidden_shuffled[i, :].contiguous()
            _row_matmul_bias_kernel[(1,)](
                x_row, fc1_weight, fc1_bias, output1[i, :],
                K_in1, K_out1, self.eps,
                BLOCK_K=256
            )

        # 4) GELU activation
        # Implement GELU as Triton kernel: one program per row
        output1_gelu = torch.empty_like(output1)
        N_gelu = M1
        # One program per row
        _gelu_row_kernel[(N_gelu,)](
            output1, output1_gelu,
            output1.shape[1],
            BLOCK_SIZE=256
        )

        # 5) Linear2: output1_gelu @ fc2_weight.T + fc2_bias
        M2 = output1_gelu.shape[0]
        K_in2 = output1_gelu.shape[1]  # 6144
        K_out2 = fc2_weight.shape[0]   # 3584
        output2 = torch.empty((M2, K_out2), dtype=torch.bfloat16, device=device)

        # Row-wise matmul for Linear2: one program per output row
        for i in range(M2):
            x_row = output1_gelu[i, :].contiguous()
            _row_matmul_bias_kernel[(1,)](
                x_row, fc2_weight, fc2_bias, output2[i, :],
                K_in2, K_out2, self.eps,
                BLOCK_K=256
            )

        return output2


def run(*args):
    return ModelNew()(*args)
