import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: LayerNorm over last dim (size K) with affine, 2D tiled across rows and K.
# Input: X[M, K], ln_weight[K], ln_bias[K], Output: Y[M, K] (store as BF16, compute in FP32).
if TRITON_AVAILABLE:
    @triton.jit
    def _layer_norm_affine_kernel(
        X_ptr, W_ptr, B_ptr, Y_ptr,
        M: tl.constexpr, K: tl.constexpr, eps: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        # 2D grid: over row tiles and feature tiles
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        row_start = pid_m * BLOCK_M
        cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_rows = row_start + tl.arange(0, BLOCK_M) < M
        mask_cols = cols < K

        # We iterate over K dimension in chunks of BLOCK_K and process BLOCK_M rows at a time.
        # Pass 1: compute mean and variance
        sum_row = tl.zeros((BLOCK_M,), dtype=tl.float32)
        sumsq_row = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Loop over feature tiles
        for k0 in range(0, K, BLOCK_K):
            cols = k0 + tl.arange(0, BLOCK_K)
            mask_cols = cols < K
            # Pointer arithmetic: load X[row, cols]
            # X layout is row-major [M, K], strides: row_stride = K, col_stride = 1
            x_ptrs = X_ptr + row_start * K + cols
            x_tile = tl.load(x_ptrs, mask=mask_rows[:, None] & mask_cols[None, :], other=0.0)
            x_fp32 = x_tile.to(tl.float32)
            # Sum and sumsq per row
            sum_row += tl.sum(x_fp32, axis=1)
            sumsq_row += tl.sum(x_fp32 * x_fp32, axis=1)

        mean = sum_row / K
        var = sumsq_row / K - mean * mean
        inv_std = 1.0 / tl.sqrt(var + eps)

        # Pass 2: normalize and apply affine
        for k0 in range(0, K, BLOCK_K):
            cols = k0 + tl.arange(0, BLOCK_K)
            mask_cols = cols < K
            x_ptrs = X_ptr + row_start * K + cols
            x_tile = tl.load(x_ptrs, mask=mask_rows[:, None] & mask_cols[None, :], other=0.0)
            x_fp32 = x_tile.to(tl.float32)

            w_ptrs = W_ptr + cols
            b_ptrs = B_ptr + cols
            w_tile = tl.load(w_ptrs, mask=mask_cols, other=1.0).to(tl.float32)
            b_tile = tl.load(b_ptrs, mask=mask_cols, other=0.0).to(tl.float32)

            y_fp32 = (x_fp32 - mean[:, None]) * inv_std[:, None]
            y_fp32 = y_fp32 * w_tile[None, :] + b_tile[None, :]

            # Store as BF16
            y_bf16 = y_fp32.to(tl.bfloat16)
            y_ptrs = Y_ptr + row_start * K + cols
            tl.store(y_ptrs, y_bf16, mask=mask_rows[:, None] & mask_cols[None, :])


def _layer_norm_affine_triton(x: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Triton implementation of LayerNorm over last dimension + affine.
    - Input x: [M, K], dtype bfloat16 or float32, on CUDA
    - ln_weight, ln_bias: [K], dtype bfloat16, on CUDA
    Returns: y [M, K] in bfloat16
    """
    assert TRITON_AVAILABLE and x.is_cuda, "Triton LayerNorm requires CUDA and Triton."
    M, K = x.shape
    # Output buffer
    y = torch.empty_like(x, dtype=torch.bfloat16, device=x.device)

    # Choose tiles. BLOCK_M and BLOCK_K are constexpr at launch time.
    BLOCK_M = 128
    BLOCK_K = 256
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

    _layer_norm_affine_kernel[grid](
        x, ln_weight, ln_bias, y,
        M, K, eps,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )
    return y


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        """
        Triton LayerNorm + affine on hidden (M=num_patches, K=hidden_size),
        then packing via torch.view (num_patches % 4 == 0), then PyTorch linears + GELU.
        All math is on GPU, no host tensor operations for compute.
        """
        device = hidden.device
        # Ensure ln params are on device
        ln_weight = ln_weight.to(device)
        ln_bias = ln_bias.to(device)

        # Triton LayerNorm + affine
        hidden_norm = _layer_norm_affine_triton(hidden, ln_weight, ln_bias, eps)

        # Packing: T=1, merge 2x2 -> expanded features (4*K)
        K = hidden_norm.shape[1]
        K_expanded = 4 * K
        M_out = hidden_norm.shape[0]  # num_patches
        # Given get_inputs ensures num_patches % 4 == 0
        assert (M_out % 4) == 0, "num_patches must be divisible by 4 for 2x2 packing."
        hidden_packed = hidden_norm.view(M_out // 4, K_expanded)

        # MLP fc1: (M_merged, 6144) @ (6144, 6144)^T + bias
        M_merged = hidden_packed.shape[0]
        fc1_out = torch.nn.functional.linear(hidden_packed, fc1_weight, fc1_bias)  # [M_merged, 6144]

        # GELU activation
        fc1_out = torch.nn.functional.gelu(fc1_out)

        # MLP fc2: (M_merged, 3584) @ (3584, 6144)^T + bias
        output = torch.nn.functional.linear(fc1_out, fc2_weight, fc2_bias)  # [M_merged, 3584]

        return output


def run(*args):
    return ModelNew()(*args)
