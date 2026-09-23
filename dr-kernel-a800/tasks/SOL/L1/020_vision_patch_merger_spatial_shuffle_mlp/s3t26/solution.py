import torch
import math
import triton
import triton.language as tl


# Triton kernel: LayerNorm per row. Inputs: x [N, C], outputs: y [N, C]
# x_ptr and y_ptr are pointers to bfloat16 data, but we compute in fp32.
@triton.jit
def _layer_norm_kernel(x_ptr, y_ptr, ln_weight_ptr, ln_bias_ptr,
                        N, C, eps,
                        BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    y_ptr: *bf16, shape [N, C], output
    ln_weight_ptr, ln_bias_ptr: *bf16, shape [C]
    eps: float32 scalar
    """
    row = tl.program_id(0)
    if row >= N:
        return

    x_row_ptr = x_ptr + row * C
    y_row_ptr = y_ptr + row * C

    # Pass 1: compute mean and variance
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


# Triton kernel: exact spatial shuffle across all grids. Produces hidden_shuffled
# of shape [num_merged_patches, 4*C] as a flattened tensor of length NUM_MERGED_ROWS * 4*C.
# For grid g: t = grid_thw[g,0], h = grid_thw[g,1], w = grid_thw[g,2]
# For each original patch m in [0, t*h*w):
#   t_index = m // (h*w), rem = m % (h*w), h2 = rem // w, w2 = rem % w
#   out_row_in_grid = t_index * (h//2) * (w//2) + (h2//2) * (w//2) + (w2//2)
#   out_row_global = g * (t*(h//2)*(w//2)) + out_row_in_grid
#   For 2x2 merge positions r in {0,1}, c in {0,1}:
#     src_idx = t_index*(h*w) + h2*w + w2 + c
#     val = hidden_norm[src_idx]
#     write to out_ptr[ out_row_global * 4*C + idx * C : (idx+1)*C ] where idx in {0,1,2,3}
@triton.jit
def _shuffle_2x2_all_grids_kernel(hidden_ptr, grid_thw_ptr, out_ptr,
                                  NUM_PATCHES, C, NUM_GRIDS, H_MAX, W_MAX,
                                  BLOCK_M: tl.constexpr):
    """
    hidden_ptr: *bf16, flattened [NUM_PATCHES, C] row-major (i.e., index i*C + j)
    grid_thw_ptr: *int64, shape [NUM_GRIDS, 3], rows are [t, h, w]
    out_ptr: *bf16, flattened [NUM_MERGED_ROWS, 4*C]
    We loop over grids and for each grid, we iterate m over 0..t*h*w-1 and write to out.
    """
    # Single program; loop over grids and m
    g = 0
    while g < NUM_GRIDS:
        t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
        h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
        w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)

        h_merged = h // 2
        w_merged = w // 2
        num_merged_rows_g = t * h_merged * w_merged
        base_out_row = g * num_merged_rows_g

        m = 0
        while m < t * h * w:
            t_index = m // (h * w)
            rem = m % (h * w)
            h2 = rem // w
            w2 = rem % w

            out_row = base_out_row + (t_index * h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)

            # 4*C columns for the 4 positions
            # idx0: (r=0,c=0)
            idx0 = 0
            col0 = idx0 * C + 0
            src_idx0 = t_index * (h * w) + h2 * w + w2 + 0
            val0 = tl.load(hidden_ptr + src_idx0 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(out_ptr + out_row * (4 * C) + col0, val0)

            # idx1: (r=0,c=1)
            idx1 = 1
            col1 = idx1 * C + 0
            if (w2 + 1) < w:
                src_idx1 = t_index * (h * w) + h2 * w + (w2 + 1) + 0
                val1 = tl.load(hidden_ptr + src_idx1 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
                tl.store(out_ptr + out_row * (4 * C) + col1, val1)
            else:
                tl.store(out_ptr + out_row * (4 * C) + col1, tl.zeros((C,), dtype=tl.bfloat16))

            # idx2: (r=1,c=0)
            idx2 = 2
            col2 = idx2 * C + 0
            if (h2 + 1) < h and (w2) < w:
                src_idx2 = t_index * (h * w) + (h2 + 1) * w + w2 + 0
                val2 = tl.load(hidden_ptr + src_idx2 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
                tl.store(out_ptr + out_row * (4 * C) + col2, val2)
            else:
                tl.store(out_ptr + out_row * (4 * C) + col2, tl.zeros((C,), dtype=tl.bfloat16))

            # idx3: (r=1,c=1)
            idx3 = 3
            col3 = idx3 * C + 0
            if (h2 + 1) < h and (w2 + 1) < w:
                src_idx3 = t_index * (h * w) + (h2 + 1) * w + (w2 + 1) + 0
                val3 = tl.load(hidden_ptr + src_idx3 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
                tl.store(out_ptr + out_row * (4 * C) + col3, val3)
            else:
                tl.store(out_ptr + out_row * (4 * C) + col3, tl.zeros((C,), dtype=tl.bfloat16))

            m += 1
        g += 1


# Triton kernel: Linear y = x @ W.T + b, row-wise. x: [N, K], W: [K_out, K], y: [N, K_out]
# Specialized for K=6144, K_out=6144 (Linear1), and K=6144, K_out=3584 (Linear2).
@triton.jit
def _row_gemm_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                     N, K, K_OUT,
                     BLOCK_K: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, K], row-major
    w_ptr: *bf16, shape [K_OUT, K], row-major
    b_ptr: *bf16, shape [K_OUT]
    y_ptr: *bf16, shape [N, K_OUT], row-major
    Each program computes one output row i.
    """
    i = tl.program_id(0)
    if i >= N:
        return

    # Accumulator vector for output row i
    acc = tl.zeros((K_OUT,), dtype=tl.float32)

    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load x[i, k:k+BLOCK_K] as fp32
        x_vec = tl.load(x_ptr + i * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        # Load W[k:k+BLOCK_K, :] as fp32, then reduce over K dimension
        # w_ptr is [K_OUT, K], row-major -> w[offs_w, offs_k] = w_ptr + offs_w * K + offs_k
        # We need dot = sum_k W[:, k] * x_vec[k], implemented by iterating over offs_w
        dot = tl.zeros((BLOCK_K,), dtype=tl.float32)
        kk = 0
        while kk < BLOCK_K:
            k_curr = k + kk
            mask_k_curr = k_curr < K
            # For each output row jj, sum W[jj, k_curr] * x[i, k_curr]
            jj = 0
            col_sum = tl.zeros((BLOCK_K,), dtype=tl.float32)
            while jj < K_OUT:
                # W[jj, k_curr] if k_curr < K else 0
                w_val = tl.load(w_ptr + jj * K + k_curr, mask=mask_k_curr, other=0.0).to(tl.float32)
                col_sum += w_val * x_vec[kk]
                jj += 1
            dot += col_sum
            kk += 1

        # Accumulate into acc for all jj positions
        jj = 0
        while jj < K_OUT:
            acc[jj] += tl.sum(dot, axis=0)
            jj += 1

        k += BLOCK_K

    # Add bias and store result
    jj = 0
    while jj < K_OUT:
        y_val = acc[jj] + tl.load(b_ptr + jj).to(tl.float32)
        tl.store(y_ptr + i * K_OUT + jj, y_val.to(tl.bfloat16))
        jj += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                hidden: torch.Tensor,
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
        grid_thw: [num_grids, 3] int64
        ln_weight, ln_bias: [hidden_size] bfloat16
        fc1_weight: [hidden_size_expanded, hidden_size_expanded] bfloat16 (6144x6144)
        fc1_bias: [hidden_size_expanded] bfloat16
        fc2_weight: [out_hidden_size, hidden_size_expanded] bfloat16 (3584x6144)
        fc2_bias: [out_hidden_size] bfloat16
        eps: float
        """
        device = hidden.device
        N = hidden.shape[0]
        C = hidden.shape[1]
        hidden_norm = torch.empty_like(hidden)  # output of LayerNorm

        # Launch LayerNorm Triton kernel
        BLOCK_SIZE = 256
        _layer_norm_kernel[(N,)](
            hidden, hidden_norm, ln_weight, ln_bias,
            N, C, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Prepare output for shuffled patches: shape [num_merged_patches, 4*C]
        num_patches = hidden_norm.shape[0]
        hidden_size_expanded = 4 * C
        NUM_MERGED_ROWS = 0
        NUM_GRIDS = grid_thw.shape[0]
        for g in range(NUM_GRIDS):
            t = int(grid_thw[g, 0].item())
            h = int(grid_thw[g, 1].item())
            w = int(grid_thw[g, 2].item())
            NUM_MERGED_ROWS += t * (h // 2) * (w // 2)

        hidden_shuffled = torch.empty((NUM_MERGED_ROWS, hidden_size_expanded), dtype=torch.bfloat16, device=device)

        # Run Triton spatial shuffle kernel. Single program loops over grids and m.
        _shuffle_2x2_all_grids_kernel[(1,)](
            hidden_norm, grid_thw, hidden_shuffled,
            num_patches, C, NUM_GRIDS, 65535, 65535,
            BLOCK_M=1,  # dummy, not used in whiles; Triton allows loops with runtime bounds
        )

        # Now perform the MLP in Triton


def run(*args):
    return ModelNew()(*args)
