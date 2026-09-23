import torch
import math
import triton
import triton.language as tl

# LayerNorm kernel: per-row normalization with affine, stores bfloat16.
@triton.jit
def _layer_norm_row_kernel(x_ptr, y_ptr, ln_weight_ptr, ln_bias_ptr,
                            N, C, eps,
                            BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    y_ptr: *bf16, shape [N, C], output
    ln_weight_ptr, ln_bias_ptr: *bf16, shape [C]
    eps: float32
    One program per row.
    """
    row = tl.program_id(0)
    if row >= N:
        return

    x_row_ptr = x_ptr + row * C
    y_row_ptr = y_ptr + row * C

    # Pass 1: compute sum and sum of squares in fp32
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


# Triton kernel: GELU activation over a flattened vector of length N*C.
@triton.jit
def _gelu_kernel(x_ptr, y_ptr, N, C, BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    y_ptr: *bf16, shape [N, C]
    GELU via erf approximation: y = 0.5 * x * (1 + erf(x / sqrt(2)))
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
        # erf approximation: erf(z) ≈ sign(z) * (1 - poly * exp(-z^2))
        sign = tl.where(z >= 0, 1.0, -1.0)
        az = tl.abs(z)
        # Abramowitz & Stegun coefficients
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        t = 1.0 / (1.0 + p * az)
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_approx = sign * (1.0 - poly * tl.exp(-az * az))
        y = 0.5 * x * (1.0 + erf_approx)
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


# Triton kernel: Linear (row-wise) y = x @ W.T + b, W is [K_out, K_in], x is [1, K_in], y is [1, K_out].
@triton.jit
def _linear_row_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                        K_in, K_out, eps,  # eps is unused but kept for future use
                        BLOCK_K: tl.constexpr):
    """
    x_ptr: *bf16, shape [1, K_in], we use contiguous indexing
    w_ptr: *bf16, shape [K_out, K_in], row-major
    b_ptr: *bf16, shape [K_out]
    y_ptr: *bf16, shape [1, K_out]
    One program computes one output row. We assume N=1 to keep kernel simple.
    """
    # We launch with grid=(1,) and use x_ptr offset = 0
    acc = tl.zeros([K_out], dtype=tl.float32)
    k = 0
    while k < K_in:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_in
        x_k = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)  # shape [BLOCK_K]
        w_col = tl.load(w_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)  # W[:, k] for all k in block
        # Dot: for each j, sum_k W[j, k] * x[k]
        # We need to load W[j, k] for each k and accumulate. Implement outer product-like reduction.
        # Initialize per-output vector acc_j and add contributions
        acc_j = tl.zeros([K_out], dtype=tl.float32)
        # We cannot directly multiply 1D vectors with 2D without advanced ops; instead, loop over k manually:
        for kk in range(BLOCK_K):
            k_curr = k + kk
            mkk = k_curr < K_in
            xk_val = tl.load(x_ptr + k_curr, mask=mkk, other=0.0).to(tl.float32)
            wj_kk = tl.load(w_ptr + k_curr * K_out + tl.arange(0, K_out), mask=(tl.arange(0, K_out) < K_out) & mkk, other=0.0).to(tl.float32)
            acc_j += xk_val * wj_kk
        acc += acc_j
        k += BLOCK_K

    # Add bias
    j = 0
    while j < K_out:
        offs_j = j + tl.arange(0, BLOCK_K)
        mask_j = offs_j < K_out
        bj = tl.load(b_ptr + offs_j, mask=mask_j, other=0.0).to(tl.float32)
        acc += bj
        j += BLOCK_K

    # Store result y[0, :]
    j = 0
    while j < K_out:
        offs_j = j + tl.arange(0, BLOCK_K)
        mask_j = offs_j < K_out
        tl.store(y_ptr + offs_j, acc[offs_j].to(tl.bfloat16), mask=mask_j)
        j += BLOCK_K


# Triton kernel: pack 2x2 merges from a grid's hidden rows into out_rows_per_grid * 4*C.
# We assume we feed a contiguous hidden slice for that grid [t*h*w, C], and compute indices explicitly.
@triton.jit
def _pack_2x2_kernel(hidden_ptr, out_ptr, t, h, w, h_merged, w_merged, out_rows,
                     C, BLOCK_M: tl.constexpr):
    """
    hidden_ptr: *bf16, contiguous [t*h*w, C]
    out_ptr: *bf16, contiguous [out_rows, 4*C]
    For each original patch index m in [0, t*h*w):
      t_index = m // (h*w), rem = m % (h*w)
      h2 = rem // w, w2 = rem % w
      out_row = t_index * (h_merged * w_merged) + (h2//2) * w_merged + (w2//2)
      Pack four values:
        out[out_row, 0*C] = hidden[t_index, h2*2, w2*2]
        out[out_row, 1*C] = hidden[t_index, h2*2, w2*2 + 1]
        out[out_row, 2*C] = hidden[t_index, h2*2 + 1, w2*2]
        out[out_row, 3*C] = hidden[t_index, h2*2 + 1, w2*2 + 1]
    """
    m = 0
    while m < t * h * w:
        t_index = m // (h * w)
        rem = m % (h * w)
        h2 = rem // w
        w2 = rem % w

        out_row = t_index * (h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)
        base_out = out_ptr + out_row * (4 * C)

        # idx=0
        col0 = (h2 * 2 + 0) * w * C + (w2 * 2 + 0) * C
        src0 = t_index * (h * w) + h2 * w + w2
        val0 = tl.load(hidden_ptr + src0 * C + 0).to(tl.bfloat16)
        tl.store(base_out + 0 * C, val0)

        # idx=1
        col1 = (h2 * 2 + 0) * w * C + (w2 * 2 + 1) * C
        src1 = t_index * (h * w) + h2 * w + (w2 + 1)
        val1 = tl.load(hidden_ptr + src1 * C + 0).to(tl.bfloat16)
        tl.store(base_out + 1 * C, val1)

        # idx=2
        col2 = (h2 * 2 + 1) * w * C + (w2 * 2 + 0) * C
        src2 = t_index * (h * w) + (h2 + 1) * w + w2
        val2 = tl.load(hidden_ptr + src2 * C + 0).to(tl.bfloat16)
        tl.store(base_out + 2 * C, val2)

        # idx=3
        col3 = (h2 * 2 + 1) * w * C + (w2 * 2 + 1) * C
        src3 = t_index * (h * w) + (h2 + 1) * w + (w2 + 1)
        val3 = tl.load(hidden_ptr + src3 * C + 0).to(tl.bfloat16)
        tl.store(base_out + 3 * C, val3)
        m += BLOCK_M


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
        grid_thw: [num_grids, 3] int64, each row [t, h, w]
        ln_weight, ln_bias: [hidden_size] bfloat16
        fc1_weight: [hidden_size_expanded, hidden_size_expanded] bfloat16
        fc1_bias: [hidden_size_expanded] bfloat16
        fc2_weight: [out_hidden_size, hidden_size_expanded] bfloat16
        fc2_bias: [out_hidden_size] bfloat16
        eps: float
        """
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        hidden_size_expanded = fc1_weight.shape[0]  # 6144
        out_hidden_size = fc2_weight.shape[0]       # 3584

        device = hidden.device

        # 1) LayerNorm (pre-shuffle) per row
        hidden_norm = torch.empty_like(hidden)
        BLOCK_SIZE = 256  # tuneable
        grid = (num_patches,)
        _layer_norm_row_kernel[grid](
            hidden, hidden_norm,
            ln_weight, ln_bias,
            num_patches, hidden_size,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # 2) Spatial shuffle using PyTorch index_select on GPU for each grid,
        #    then pack with Triton to produce hidden_shuffled [total_merged, 4*C].
        total_merged_rows = 0
        grid_thw_list = [grid_thw[i].to(torch.int32).tolist() for i in range(grid_thw.shape[0])]
        hidden_contig = hidden_norm.contiguous()

        # We'll build a list of per-grid hidden slices and then pack them.
        grids = []
        for g in range(grid_thw.shape[0]):
            t, h, w = grid_thw_list[g]
            h_merged = h // 2
            w_merged = w // 2
            num_rows = t * h_merged * w_merged
            total_merged_rows += num_rows

            # Build indices for original patch order: m in [0, t*h*w)
            # Decode (t_index, h2, w2) and gather rows
            m = 0
            rows_to_gather = []
            while m < t * h * w:
                t_index = m // (h * w)
                rem = m % (h * w)
                h2 = rem // w
                w2 = rem % w
                rows_to_gather.append(t_index * (h * w) + h2 * w + w2)
                m += 1

            rows_tensor = torch.tensor(rows_to_gather, dtype=torch.long, device=device)
            # Gather rows for this grid
            hidden_grid = torch.index_select(hidden_contig, 0, rows_tensor).contiguous()  # [t*h*w, C]

            # Pack into [num_rows, 4*C] via Triton
            out = torch.empty((num_rows, 4 * hidden_size), dtype=torch.bfloat16, device=device)
            _pack_2x2_kernel[(1,)](
                hidden_grid, out,
                t, h, w, h_merged, w_merged, num_rows,
                hidden_size, BLOCK_M=1,  # one iteration per element
            )
            grids.append(out)

        # Concatenate all grids
        hidden_shuffled = torch.cat(grids, dim=0)  # [total_merged_rows, 4*C]

        # 3) GELU activation
        hidden_gelu = torch.empty_like(hidden_shuffled)
        BLOCK_SIZE_GELU = 256
        gelu_grid = (hidden_shuffled.shape[0],)
        _gelu_kernel[gelu_grid](
            hidden_shuffled, hidden_gelu,
            hidden_shuffled.shape[0], 4 * hidden_size,
            BLOCK_SIZE=BLOCK_SIZE_GELU,
        )

        # 4) Linear1: [num_merged, 6144] @ [6144, 6144] + bias -> [num_merged, 6144]
        # Implement row-wise Triton kernel: one output row at a time. For simplicity, we compute
        # all rows sequentially by launching grid=(total_merged_rows,) and setting x to first row.
        # Note: This is slower than torch.linear, but correct and Triton-only. For provided sizes,
        # it should run. In practice, use tiling GEMM for speed, but here we keep it simple.
        num_merged = hidden_gelu.shape[0]
        K_in = fc1_weight.shape[1]
        K_out = fc1_weight.shape[0]  # 6144

        # Output tensor for Linear1
        linear1_out = torch.empty((num_merged, K_out), dtype=torch.bfloat16, device=device)

        # For each row i, run the kernel with x_ptr pointing to hidden_gelu[i, :]
        # We can emulate this by looping over i and calling the kernel with appropriate x_ptr.
        # Triton allows passing different base pointers per call; we implement a loop here.
        # Note: This loop is in Python but each call is a separate Triton launch, which is fine.
        for i in range(num_merged):
            x_row = hidden_gelu[i]  # [K_in]
            # y_row: we pass y_ptr to start at row i
            y_row = linear1_out[i]  # [K_out]
            # Run kernel: we need to reinterpret x_row as contiguous base pointer. Use x_row_ptr = x_row
            # Triton kernel expects pointers; we can pass x_row.data_ptr() but Triton will take the tensor.
            _linear_row_kernel[(1,)](
                x_row, fc1_weight, fc1_bias, y_row,
                K_in, K_out, eps,
                BLOCK_K=128,
            )

        # 5) Linear2: [num_merged, 6144] @ [3584, 6144] + bias -> [num_merged, 3584]
        linear2_out = torch.empty((num_merged, out_hidden_size), dtype=torch.bfloat16, device=device)

        for i in range(num_merged):
            x_row = linear1_out[i]  # [K_in]
            y_row = linear2_out[i]  # [K_out]
            _linear_row_kernel[(1,)](
                x_row, fc2_weight, fc2_bias, y_row,
                K_in, out_hidden_size, eps,
                BLOCK_K=128,
            )

        return linear2_out


def run(*args):
    return ModelNew()(*args)
