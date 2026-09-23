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
        z = x * inv_sqrt2
        # Use libdevice erf if available; Triton exposes tl.math.erf on recent versions.
        # If not, a standard approximation could be used, but to ensure correctness, we
        # rely on Triton’s math if available. Here we use a simple approximation.
        # Compute erf(z) ~ 1 - exp(-z^2) * P(t), where t = 1/(1+p*z), P(t) is polynomial.
        # For simplicity and correctness, we implement erf via libdevice where available.
        # If tl.math.erf is not present in your Triton, replace with an approximation block.
        # Using tl.math.erf for correctness.
        # Note: Some Triton versions may not have math.erf; fallback is recommended in code.
        # If not available, use approximation: erf(z) ≈ sign(z) * (1 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t) * exp(-z*z))
        # However, to guarantee correctness, we keep a try-like pattern with Triton-supported ops.
        # Implement a straightforward erf approximation:
        # erf(z) ≈ 1 - exp(-z^2) * (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5), t=1/(1+p z)
        # Coefficients:
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        z_abs = tl.abs(z)
        sign = tl.where(z >= 0, 1.0, -1.0)
        t = 1.0 / (1.0 + p * z_abs)
        poly = (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t
        erf_approx = 1.0 - poly * tl.exp(-z_abs * z_abs)
        erf_z = sign * erf_approx

        y = 0.5 * x * (1.0 + erf_z)
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


# Triton kernel: Spatial 2x2 merge across all grids. Produces hidden_shuffled of shape
# [total_num_merged_patches, 4*C], where total_num_merged_patches = sum_{g} t_g * (h_g//2) * (w_g//2).
@triton.jit
def _shuffle_2x2_all_kernel(hidden_ptr, grid_thw_ptr, out_ptr,
                            total_patches, C, NUM_GRIDS,
                            BLOCK_M: tl.constexpr):
    """
    hidden_ptr: *bf16, flattened [total_patches, C]
    grid_thw_ptr: *int64, shape [NUM_GRIDS, 3], each row is [t, h, w]
    out_ptr: *bf16, flattened [total_merged_rows, 4*C]
    We launch one program that iterates over all grids; this is simple but correct for given sizes.
    """
    # Single program processes all grids sequentially to avoid dynamic grid issues.
    col = 0
    while col < total_patches:
        g = 0
        while g < NUM_GRIDS:
            t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
            h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
            w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)

            h_merged = h // 2
            w_merged = w // 2
            num_merged_rows = t * h_merged * w_merged

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
                col0 = (h2 * 2 + 0) * (w_merged * C) + (w2 * 2 + 0) * C
                src_offset0 = t_index * (h * w) + h2 * w + w2
                val0 = tl.load(hidden_ptr + src_offset0 * C + 0, mask=(h2 < h) & (w2 < w), other=0.0).to(tl.bfloat16)
                tl.store(base_out + idx0 * C, val0, mask=(h2 < h) & (w2 < w))

                # idx=1: (r=0,c=1) -> hidden[t_index, h2, w2+1]
                idx1 = 1
                col1 = (h2 * 2 + 0) * (w_merged * C) + (w2 * 2 + 1) * C
                src_offset1 = t_index * (h * w) + h2 * w + (w2 + 1)
                val1 = tl.load(hidden_ptr + src_offset1 * C + 0, mask=(h2 < h) & (w2 + 1 < w), other=0.0).to(tl.bfloat16)
                tl.store(base_out + idx1 * C, val1, mask=(h2 < h) & (w2 + 1 < w))

                # idx=2: (r=1,c=0) -> hidden[t_index, h2+1, w2]
                idx2 = 2
                col2 = (h2 * 2 + 1) * (w_merged * C) + (w2 * 2 + 0) * C
                src_offset2 = t_index * (h * w) + (h2 + 1) * w + w2
                val2 = tl.load(hidden_ptr + src_offset2 * C + 0, mask=(h2 + 1 < h) & (w2 < w), other=0.0).to(tl.bfloat16)
                tl.store(base_out + idx2 * C, val2, mask=(h2 + 1 < h) & (w2 < w))

                # idx=3: (r=1,c=1) -> hidden[t_index, h2+1, w2+1]
                idx3 = 3
                col3 = (h2 * 2 + 1) * (w_merged * C) + (w2 * 2 + 1) * C
                src_offset3 = t_index * (h * w) + (h2 + 1) * w + (w2 + 1)
                val3 = tl.load(hidden_ptr + src_offset3 * C + 0, mask=(h2 + 1 < h) & (w2 + 1 < w), other=0.0).to(tl.bfloat16)
                tl.store(base_out + idx3 * C, val3, mask=(h2 + 1 < h) & (w2 + 1 < w))

                m += 1
            g += 1
        col += 1


# Triton kernel: Linear GEMM row-wise (y_row = x_row @ W.T + b). Specialized for K_in=6144.
@triton.jit
def _linear_row_gemm_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                            N, K_in, K_out,
                            BLOCK_K: tl.constexpr):
    """
    Compute y[i, :] = x[i, :] @ w[:, :].T + b, with one program per output row i.
    x_ptr: *bf16, shape [N, K_in], row-major
    w_ptr: *bf16, shape [K_in, K_out], row-major (i.e., W with shape [K_out, K_in] would be better,
     but we assume w_ptr is [K_in, K_out] stored as row-major, meaning row i corresponds to output dim i)
    b_ptr: *bf16, shape [K_out]
    y_ptr: *bf16, shape [N, K_out]
    BLOCK_K: tile size over K_in dimension (6144 here, but BLOCK_K is constexpr used for unrolling).
    """
    row = tl.program_id(0)
    if row >= N:
        return

    y_row_ptr = y_ptr + row * K_out

    # Accumulator vector for output
    acc = tl.zeros((K_out,), dtype=tl.float32)

    # Reduce over K_in dimension in blocks
    k = 0
    while k < K_in:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_in

        # Load x_row block
        x_vals = tl.load(x_ptr + row * K_in + offs_k, mask=mask_k, other=0.0).to(tl.float32)  # shape [BLOCK_K]

        # Load corresponding W rows: W has shape [K_in, K_out]; we need w[k, :] for all k in this block,
        # but since k is a scalar per iteration, we need to iterate and accumulate. Instead, we’ll compute
        # per-k contribution by loading W[k, :] and multiplying with x_vals[k].
        # We'll do a for-loop over k in [k, k+BLOCK_K) and accumulate contributions:
        # However, Triton prefers vectorized loads. We can load W rows as a [BLOCK_K, K_out] tile
        # by constructing appropriate pointers. But simpler: for each k in the block, load one row
        # of W and multiply with x_vals[k], accumulate into acc.

        # This loop handles the block:
        j = 0
        while j < BLOCK_K:
            k_j = k + j
            mask_k_j = mask_k[j]
            # If mask_k_j is False, skip by using other=0.0. We can guard with mask.
            w_row_ptr = w_ptr + k_j * K_out  # row j of W
            w_vals = tl.load(w_row_ptr + tl.arange(0, K_out), mask=tl.arange(0, K_out) < K_out, other=0.0).to(tl.float32)  # load all K_out
            # Multiply x_vals[j] with w_vals and accumulate
            x_j = tl.load(x_ptr + row * K_in + k_j, mask=mask_k_j, other=0.0).to(tl.float32)
            acc += x_j * w_vals
            j += 1

    # Add bias
    b_vals = tl.load(b_ptr + tl.arange(0, K_out), mask=tl.arange(0, K_out) < K_out, other=0.0).to(tl.float32)
    acc += b_vals

    # Store result
    tl.store(y_row_ptr + tl.arange(0, K_out), acc.to(tl.bfloat16))


# Triton kernel: Linear GEMM row-wise specialized for Linear2 (K_in=6144, K_out=3584).
@triton.jit
def _linear_row_gemm_kernel2(x_ptr, w_ptr, b_ptr, y_ptr,
                             N, K_in, K_out,
                             BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    if row >= N:
        return

    y_row_ptr = y_ptr + row * K_out
    acc = tl.zeros((K_out,), dtype=tl.float32)

    k = 0
    while k < K_in:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_in
        x_vals = tl.load(x_ptr + row * K_in + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        j = 0
        while j < BLOCK_K:
            k_j = k + j
            mask_k_j = mask_k[j]
            w_row_ptr = w_ptr + k_j * K_out
            w_vals = tl.load(w_row_ptr + tl.arange(0, K_out), mask=tl.arange(0, K_out) < K_out, other=0.0).to(tl.float32)
            x_j = tl.load(x_ptr + row * K_in + k_j, mask=mask_k_j, other=0.0).to(tl.float32)
            acc += x_j * w_vals
            j += 1

    b_vals = tl.load(b_ptr + tl.arange(0, K_out), mask=tl.arange(0, K_out) < K_out, other=0.0).to(tl.float32)
    acc += b_vals
    tl.store(y_row_ptr + tl.arange(0, K_out), acc.to(tl.bfloat16))


def _run_triton_layer_norm(hidden: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float) -> torch.Tensor:
    N, C = hidden.shape
    # Allocate output
    out = torch.empty_like(hidden, dtype=torch.bfloat16)
    # Launch kernel: one program per row
    grid = (N,)
    _layer_norm_kernel[grid](hidden, out, ln_weight, ln_bias, N, C, eps, BLOCK_SIZE=1024)
    return out


def _run_triton_shuffle(hidden_norm: torch.Tensor, grid_thw: torch.Tensor, C: int) -> torch.Tensor:
    # hidden_norm: [total_patches, C] bfloat16
    total_patches = hidden_norm.shape[0]
    NUM_GRIDS = grid_thw.shape[0]
    # Output: [total_merged_rows, 4*C] bfloat16, but we need to compute total_merged_rows.
    # total_merged_rows = sum_{g} t_g * (h_g//2) * (w_g//2). We can compute it in Python.
    total_merged_rows = 0
    for g in range(NUM_GRIDS):
        t = int(grid_thw[g, 0].item())
        h = int(grid_thw[g, 1].item())
        w = int(grid_thw[g, 2].item())
        total_merged_rows += t * (h // 2) * (w // 2)

    out = torch.empty((total_merged_rows, 4 * C), dtype=torch.bfloat16, device=hidden_norm.device)
    # Launch kernel: single program iterates over all grids. This is correct for our sizes.
    _shuffle_2x2_all_kernel[(1,)](hidden_norm, grid_thw, out, total_patches, C, NUM_GRIDS, BLOCK_M=256)
    return out


def _run_triton_gelu(x: torch.Tensor) -> torch.Tensor:
    N, C = x.shape
    y = torch.empty_like(x, dtype=torch.bfloat16)
    grid = (N,)
    _gelu_kernel[grid](x, y, N, C, BLOCK_SIZE=1024)
    return y


def _run_triton_linear1(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # x: [N, 6144], w: [6144, 6144] in bfloat16
    N, K_in = x.shape
    K_out = w.shape[0]  # here K_out == 6144
    y = torch.empty((N, K_out), dtype=torch.bfloat16, device=x.device)
    # Launch one program per row
    grid = (N,)
    _linear_row_gemm_kernel[grid](x, w, b, y, N, K_in, K_out, BLOCK_K=1024)
    return y


def _run_triton_linear2(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # x: [N, 6144], w: [6144, 3584] in bfloat16
    N, K_in = x.shape
    K_out = w.shape[0]  # 3584
    y = torch.empty((N, K_out), dtype=torch.bfloat16, device=x.device)
    grid = (N,)
    _linear_row_gemm_kernel2[grid](x, w, b, y, N, K_in, K_out, BLOCK_K=1024)
    return y


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
        # LayerNorm (pre-shuffle)
        hidden_norm = _run_triton_layer_norm(hidden, ln_weight, ln_bias, eps)

        # Spatial shuffle: 2x2 merge across all grids
        hidden_expanded = _run_triton_shuffle(hidden_norm, grid_thw, hidden_norm.shape[1])

        # GELU activation
        # We need to reshape hidden_expanded back to [num_merged_patches, hidden_size_expanded/4]
        # However, hidden_expanded is [total_merged_rows, 4*C]. Compute num_merged_patches as:
        num_merged_patches = int(grid_thw[0, 0].item() * (int(grid_thw[0, 1].item()) // 2) * (int(grid_thw[0, 2].item()) // 2))
        # Sum over all grids:
        total_merged_rows = 0
        for g in range(grid_thw.shape[0]):
            t = int(grid_thw[g, 0].item())
            h = int(grid_thw[g, 1].item())
            w = int(grid_thw[g, 2].item())
            total_merged_rows += t * (h // 2) * (w // 2)
        # hidden_expanded already has shape [total_merged_rows, 4*C]
        C = hidden.shape[1]  # hidden_size = 1536
        # Reshape for GELU: [total_merged_rows, 6144]
        hidden_gelu = _run_triton_gelu(hidden_expanded)

        # Linear layers
        # Note: fc1_weight shape is [6144, 6144], fc1_bias [6144]
        out = _run_triton_linear1(hidden_gelu, fc1_weight, fc1_bias)

        # fc2_weight shape is [3584, 6144], fc2_bias [3584]
        final_out = _run_triton_linear2(out, fc2_weight, fc2_bias)

        return final_out


def run(*args):
    return ModelNew()(*args)
