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
    x_ptr: *bf16, shape [N, C], row-major (flattened addressing: row start = row * C)
    y_ptr: *bf16, shape [N, C], output
    ln_weight_ptr, ln_bias_ptr: *bf16, shape [C]
    eps: float32
    """
    row = tl.program_id(0)
    if row >= N:
        return

    x_row_ptr = x_ptr + row * C
    y_row_ptr = y_ptr + row * C

    # Pass 1: compute mean and variance (fp32) over C
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


# Triton kernel: Linear GEMM row-wise (y_row = x_row @ W.T + b).
# We use one program per output row. x_ptr is [N, K_in] flattened as [N*K_in].
# w_ptr is [K_out, K_in] flattened as [K_out*K_in]. We iterate over K_in in blocks.
@triton.jit
def _linear_row_gemm_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                            N, K_in, K_out,
                            BLOCK_K: tl.constexpr):
    """
    Compute y[i, :] = x[i, :] @ w[:, :].T + b, with one program per output row i.
    x_ptr: *bf16, shape [N, K_in], row-major (flattened addressing: i*K_in + k)
    w_ptr: *bf16, shape [K_out, K_in], row-major (flattened addressing: j*K_in + k)
    b_ptr: *bf16, shape [K_out]
    y_ptr: *bf16, shape [N, K_out], row-major (flattened addressing: i*K_out + j)
    """
    i = tl.program_id(0)
    if i >= N:
        return

    # y[i, :]
    y_row_ptr = y_ptr + i * K_out

    # Accumulator in fp32
    acc = tl.zeros([K_out], dtype=tl.float32)

    # Reduce over K_in in blocks
    k = 0
    while k < K_in:
        offs_k = k + tl.arange(0, BLOCK_K)  # vector of length BLOCK_K
        mask_k = offs_k < K_in

        # Load x[i, offs_k] as a vector (bf16 -> fp32)
        x_vec = tl.load(x_ptr + i * K_in + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        # For each element j in offs_k, load W[j, :] and dot with x_vec
        # Note: we compute dot in fp32 and accumulate into acc.
        j = 0
        while j < BLOCK_K:
            idx_j = k + j
            mask_j = idx_j < K_in
            # W[idx_j, 0..K_in-1] flattened: address idx_j*K_in + offs_k
            w_vec_j = tl.load(w_ptr + idx_j * K_in + offs_k, mask=mask_j & mask_k, other=0.0).to(tl.float32)
            # Elementwise multiply and reduce (sum) over offs_k
            acc += tl.sum(w_vec_j * x_vec, axis=0)
            j += 1

        k += BLOCK_K

    # Add bias and store
    col = 0
    while col < K_out:
        offs = col + tl.arange(0, BLOCK_K)
        mask = offs < K_out
        b_vec = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc_vec = acc[offs]
        y = acc_vec + b_vec
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_K


def _triton_linear(x_row_bf16, w_bf16, b_bf16):
    """
    Launch Triton _linear_row_gemm_kernel for a single row vector x_row_bf16 of shape [K_in], 
    weight w_bf16 of shape [K_out, K_in], and bias b_bf16 of shape [K_out].
    Returns y_row_bf16 of shape [K_out].
    """
    assert x_row_bf16.dtype == torch.bfloat16 and w_bf16.dtype == torch.bfloat16 and b_bf16.dtype == torch.bfloat16
    N = 1  # single row
    K_in = x_row_bf16.shape[0]
    K_out = w_bf16.shape[0]
    x_flat = x_row_bf16.view(-1)
    w_flat = w_bf16.view(-1)  # [K_out*K_in]
    b_flat = b_bf16.view(-1)
    y_row = torch.empty(K_out, dtype=torch.bfloat16, device=x_row_bf16.device)
    # Choose BLOCK_K: 1024 is reasonable for these sizes
    BLOCK_K = 1024
    grid = (N,)
    _linear_row_gemm_kernel[grid](x_flat, w_flat, b_flat, y_row, N, K_in, K_out, BLOCK_K)
    return y_row


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
        hidden: [num_patches, 1536], bfloat16
        grid_thw: [num_grids, 3], int64 (t, h, w per grid)
        ln_weight, ln_bias: [1536], bfloat16
        fc1_weight: [6144, 6144], bfloat16
        fc1_bias: [6144], bfloat16
        fc2_weight: [3584, 6144], bfloat16
        fc2_bias: [3584], bfloat16
        eps: float
        """
        device = hidden.device

        # Step 1: LayerNorm (per row) on hidden in Triton
        hidden_fp32 = hidden.to(torch.float32)
        hidden_norm_fp32 = torch.empty_like(hidden_fp32, dtype=torch.float32, device=device)
        ln_w_fp32 = ln_weight.to(torch.float32)
        ln_b_fp32 = ln_bias.to(torch.float32)

        N = hidden_norm_fp32.shape[0]
        C = hidden_norm_fp32.shape[1]
        BLOCK_SIZE = 1024
        grid_ln = (N,)
        _layer_norm_kernel[grid_ln](hidden_fp32, hidden_norm_fp32, ln_w_fp32, ln_b_fp32, N, C, float(eps), BLOCK_SIZE)

        # Cast back to bfloat16 for subsequent ops
        hidden_norm = hidden_norm_fp32.to(torch.bfloat16)

        # Step 2: GELU activation using PyTorch (exact erf-based GELU) to ensure correctness
        # The original code uses default GELU, which is erf-based. We match that.
        hidden_gelu = torch.nn.functional.gelu(hidden_norm, approximate='none')

        # Step 3: Linear1: y1 = hidden_gelu @ fc1_weight.T + fc1_bias
        num_merged = hidden_gelu.shape[0]  # num_merged_patches
        K_in_linear1 = hidden_gelu.shape[1]  # 6144
        K_out_linear1 = fc1_weight.shape[0]  # 6144

        y1 = torch.empty((num_merged, K_out_linear1), dtype=torch.bfloat16, device=device)
        for i in range(num_merged):
            y1[i] = _triton_linear(hidden_gelu[i], fc1_weight, fc1_bias)

        # GELU on y1 (ensure exact match with original)
        y1_gelu = torch.nn.functional.gelu(y1, approximate='none')

        # Step 4: Linear2: output = y1_gelu @ fc2_weight.T + fc2_bias
        num_out = y1_gelu.shape[0]
        K_in_linear2 = y1_gelu.shape[1]  # 6144
        K_out_linear2 = fc2_weight.shape[0]  # 3584

        output = torch.empty((num_out, K_out_linear2), dtype=torch.bfloat16, device=device)
        for i in range(num_out):
            output[i] = _triton_linear(y1_gelu[i], fc2_weight, fc2_bias)

        return output


def run(*args):
    return ModelNew()(*args)
