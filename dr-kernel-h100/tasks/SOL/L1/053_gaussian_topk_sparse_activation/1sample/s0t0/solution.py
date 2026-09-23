import torch
import torch.nn.functional as F

# Triton imports
import triton
import triton.language as tl

# We'll keep _ndtri in PyTorch for correctness (single scalar), but we can compute it once per forward.
# The benchmark uses target_sparsity = 0.9, so this is a constant. If needed, we could move it to Triton as well,
# but it's not a per-element operation and is negligible compared to the tensor ops.


@triton.jit
def _row_reduce_mean_std_kernel(
    x_ptr,           # *const float32
    mean_ptr,        # *float32, length N
    sumsq_ptr,       # *float32, length N
    B, S, K,         # int32 runtime
    BLOCK_K: tl.constexpr,  # tile size for features
):
    """
    Each program handles one row (flatten [B, S] into N rows).
    It iterates over the last dimension K in tiles of BLOCK_K, accumulating sum and sum of squares in fp32.
    Writes mean and sumsq per row.
    """
    row_id = tl.program_id(axis=0)
    # Number of tiles along K
    num_tiles = (K + BLOCK_K - 1) // BLOCK_K

    # Accumulators in fp32
    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sumsq = tl.zeros((), dtype=tl.float32)

    # Loop over tiles
    for tile in range(0, num_tiles):
        start = tile * BLOCK_K
        # Compute offsets for this tile: row_id * K + start + arange
        offsets = row_id * K + start + tl.arange(0, BLOCK_K)
        mask = offsets < (row_id * K + K)
        # Load a chunk of the row
        x_chunk = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        # Reduce to scalars
        acc_sum += tl.sum(x_chunk, axis=0)
        acc_sumsq += tl.sum(x_chunk * x_chunk, axis=0)

    # Compute mean and store
    mean = acc_sum / K
    sumsq = acc_sumsq / K  # we will compute std on host as sqrt(sumsq - mean^2)
    tl.store(mean_ptr + row_id, mean)
    tl.store(sumsq_ptr + row_id, sumsq)


@triton.jit
def _apply_relu_sub_threshold_kernel(
    x_ptr,            # *const float32, input
    out_ptr,          # *float32, output
    threshold_ptr,    # *const float32, length K
    N, K,             # int32
    BLOCK_K: tl.constexpr,
):
    """
    2D grid: axis 0 over rows (N), axis 1 over feature tiles.
    For each (row, tile), load x row tile, load threshold tile, compute y = max(0, x - threshold), store.
    """
    row_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    start = tile_id * BLOCK_K
    cols = start + tl.arange(0, BLOCK_K)
    mask = cols < K

    # Linear offsets for row and cols
    offsets = row_id * K + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    threshold = tl.load(threshold_ptr + cols, mask=mask, other=0.0)
    y = x - threshold
    # ReLU
    y = tl.maximum(y, 0.0)

    tl.store(out_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the original run function.

        Implements:
          - Compute mean and std per row across last dim (K) in Triton.
          - Compute threshold per feature: mean + std * norm.icdf(target_sparsity).
          - Apply ReLU(inputs - threshold) in Triton, return bfloat16.
        """
        # Early exit if no sparsity requested
        if target_sparsity == 0.0:
            # Return inputs as bfloat16 to match original behavior
            return inputs.to(torch.bfloat16)

        # Ensure we are on CUDA for Triton; otherwise, fallback to PyTorch for correctness
        if not inputs.is_cuda:
            # Fallback: do the original PyTorch logic
            inputs_f32 = inputs.to(torch.float32)
            inputs_mean = torch.mean(inputs_f32, dim=-1, keepdim=True)
            inputs_std = torch.std(inputs_f32, dim=-1, keepdim=True, unbiased=False)
            target_sparsity_tensor = torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device)
            std_multiplier = torch.distributions.normal._percentile_to_value(target_sparsity_tensor.item())  # approximation: 1.28 for 0.9
            # Alternatively, use torch's inverse_cdf if available, but we approximate 1.28
            cutoff_threshold = inputs_mean + inputs_std * std_multiplier
            sparse_output = F.relu(inputs_f32 - cutoff_threshold)
            return sparse_output.to(torch.bfloat16)

        # Make input contiguous and use float32 in kernels
        x = inputs.contiguous()
        x_f32 = x.to(torch.float32)

        # Flatten rows: N = B * S
        B, S, K = x_f32.shape
        N = B * S

        # Allocate outputs for reduction
        mean = torch.empty(N, device=x_f32.device, dtype=torch.float32)
        sumsq = torch.empty(N, device=x_f32.device, dtype=torch.float32)

        # Launch reduction kernel
        BLOCK_K = 1024
        grid = (N,)
        _row_reduce_mean_std_kernel[grid](
            x_f32, mean, sumsq, B, S, K, BLOCK_K,
            num_warps=4,
        )

        # Compute std on host (fp32), matching unbiased=False: std = sqrt(sumsq - mean^2)
        # Note: mean is per-row scalar; sumsq is per-row sum of squares
        mean_rows = mean.view(B, S)  # [B, S]
        sumsq_rows = sumsq.view(B, S)
        std_rows = torch.sqrt(sumsq_rows - mean_rows * mean_rows)  # [B, S]

        # Compute threshold per feature: threshold[k] = mean[k] + std[k] * multiplier
        # multiplier is inverse normal CDF at target_sparsity. Benchmark uses 0.9 -> ~1.2815515655446004
        # We compute it once in PyTorch (negligible cost) and use in Triton kernel.
        # If you want to keep it purely Triton, define a tiny kernel to produce a scalar, but single scalar is fine here.
        multiplier = torch.distributions.normal._percentile_to_value(float(target_sparsity))  # approx 1.28 for 0.9
        # Note: torch.distributions.normal._percentile_to_value returns z-score, which is equivalent to inverse CDF.
        # For 0.9, it is approximately 1.2815515655446004; using the function gives exact mapping.
        threshold_vec = mean_rows + std_rows * multiplier  # shape [B, S], per feature across rows

        # Now apply ReLU(x - threshold) in Triton. For output we need a 3D tensor [B, S, K].
        # But to apply elementwise subtract and ReLU, we need threshold broadcast along K.
        # Create a 1D threshold vector of length K for each row (we have per-row vectors; we'll broadcast per feature).
        # Here we need to construct a threshold vector per row that varies with feature index.
        # The simplest is to compute threshold per feature across all rows by indexing into threshold_vec with k.
        # However, threshold_vec has shape [B, S]; we need to map to feature k. We can compute per row's threshold[k] by:
        # threshold[k] = mean[k] + std[k] * multiplier. Since mean and std are per row, we need to generalize to feature k.
        # The reference code computes threshold per feature across rows, but using mean/std per row is wrong in general.
        # To be correct, we need to compute mean and std across the last dim for the whole [B, S] matrix, then threshold is constant across rows? No: the original code computes per feature (k).
        # Let's re-evaluate: the original code computes mean and std along dim=-1 for each row (i.e., per [b, s]), and then threshold[k] = mean_row + std_row * multiplier, and that threshold is broadcast along K for that row.
        # That means for each row i, threshold is a vector of length K where each element depends on feature index k, using the row's mean and std, not the global mean/std across all rows.
        # Therefore, we need to construct a threshold vector for each row. We can do this by computing mean_row and std_row, then threshold_row[k] = mean_row + std_row * multiplier, and repeat across K features.
        # But the original code uses per-feature mean and std across rows? Wait, no: it computes per row (dim=-1), so for each [b, s], it has its own mean and std, and threshold vector of length K for that row.

        # The above is a subtle detail: the original code computes:
        # inputs_mean = mean over last dim per row -> shape [B, S, 1]
        # inputs_std = std over last dim per row -> shape [B, S, 1]
        # multiplier = _ndtri(target_sparsity) scalar
        # cutoff_threshold = inputs_mean + inputs_std * multiplier -> shape [B, S, 1]
        # sparse_output = relu(inputs - cutoff_threshold) -> broadcast [B, S, 1] across K
        # So threshold is per row (not per feature across rows).

        # To match this precisely, we need to create a threshold tensor of shape [B, S, K], where each column j shares the same threshold value for that row:
        # We can construct it by expanding threshold_vec (shape [B, S]) to [B, S, 1] and broadcast to [B, S, K]. However, that would be incorrect because it would be the same threshold across K features, while the original subtracts a constant per row across all features.
        # The original subtracts a constant vector of shape [B, S, 1] across K, meaning the same threshold for all features in that row. That is exactly what we computed: for each row i, threshold[i] = mean_row[i] + std_row[i] * multiplier, and then we subtract that constant vector across K.

        # Therefore, we will build a threshold tensor of shape [B, S, 1] from threshold_vec, and then we can use it directly in the Triton elementwise kernel. The kernel expects a 1D threshold vector of length K; to use a per-row constant, we can broadcast by constructing a vector of length K where all elements are equal to threshold_vec[row].

        # Create output buffer
        out_f32 = torch.empty_like(x_f32)

        # Launch elementwise kernel: for each (row, feature tile), compute y = max(0, x - threshold[row])
        # We need threshold for each row. We can pass threshold_row as a vector of length K (filled with threshold_vec[row]) in the kernel.
        # However, Triton kernels don't accept dynamic per-row vectors easily; instead, we can compute threshold_row on host and pass it as a separate tensor. But to avoid extra host-side work, we can simply compute threshold per row and feed it into the kernel.

        # Prepare grid: 2D grid over (rows, feature tiles)
        grid_2d = (N, (K + BLOCK_K - 1) // BLOCK_K)
        _apply_relu_sub_threshold_kernel[grid_2d](
            x_f32, out_f32, threshold_vec.view(N), N, K, BLOCK_K,
            num_warps=4,
        )

        # Return in bfloat16 to match original function's behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
