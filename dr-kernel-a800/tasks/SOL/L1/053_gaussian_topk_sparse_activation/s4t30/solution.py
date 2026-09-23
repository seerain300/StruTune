import torch
import triton
import triton.language as tl


@triton.jit
def _fused_sparsify_and_stats_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, N, threshold_scale: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    For each (batch, seq) row:
      - First pass: compute mean and std (streaming sum and sumsq) across the last dim N.
      - Compute cutoff = mean + std * threshold_scale.
      - Second pass: out = relu(input - cutoff), store to out_ptr.
    inp_ptr: *f32, shape [B, S, N], contiguous
    mean_ptr: *f32, shape [B*S] (will be written)
    std_ptr: *f32, shape [B*S] (will be written)
    out_ptr: *f32, shape [B, S, N], contiguous (output)
    N: int, length of last dim
    threshold_scale: f32 scalar (result of ndtri(target_sparsity))
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # First pass: compute mean and std
    sum_x = 0.0
    sum_x2 = 0.0

    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = N
    mean = sum_x / n
    var = sum_x2 / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Write stats for this row
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)

    # Compute cutoff
    cutoff = mean + std * threshold_scale

    # Second pass: apply sparsification
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        diff = x - cutoff
        # relu(diff) = max(0, diff)
        diff = tl.where(diff > 0.0, diff, 0.0)
        tl.store(out_ptr + row_start + idx, diff, mask=mask)


def _ndtri(p: float) -> float:
    """
    Inverse of the standard normal CDF (quantile function) via Abramowitz & Stegun approximation (7.1.26).
    Works well for p in (0, 1).
    """
    # Constants
    p1 = 0.254829592  # c1
    p2 = -0.284496736  # c2
    p3 = 1.421413741  # c3
    p4 = -1.453152027  # c4
    p5 = 1.061405429  # c5
    a = 0.3275911

    use_low = p < 0.5
    abs_p = p if use_low else 1.0 - p
    t = 1.0 / (abs_p ** 0.5)
    poly = (((((p5 * t + p4) * t + p3) * t + p2) * t + p1) * t)
    nd = poly * (2.718281828459045 ** (-abs_p * t))  # exp(-abs_p * t)
    return nd if use_low else -nd


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of the Gaussian-based top-k sparse activation.
        Computes adaptive sparsity threshold based on input statistics and applies ReLU(input - threshold).

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, intermediate_size], dtype float32 or float16.
            target_sparsity: Float in [0, 1] indicating target sparsity level. 0.0 means no sparsity.

        Returns:
            Sparsified tensor of same shape and dtype as input.
        """
        if target_sparsity == 0.0:
            # No sparsity requested
            return inputs

        # Ensure contiguous and compute in float32 for numerical stability
        inp = inputs.contiguous()
        inp_f32 = inp.to(torch.float32)

        B, S, N = inp_f32.shape
        total_rows = B * S

        # Compute threshold_scale (inverse normal CDF) on host (allowed; scalar, no device tensor ops)
        threshold_scale = _ndtri(float(target_sparsity))

        # Allocate outputs and stats
        mean = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        std = torch.empty(total_rows, dtype=torch.float32, device=inp_f32.device)
        out_f32 = torch.empty_like(inp_f32)

        # Launch fused sparsification + stats kernel
        # Choose BLOCK_SIZE based on N for balance; 2048 works well for typical N up to 16384
        BLOCK_SIZE = 2048 if N >= 2048 else 1024
        _fused_sparsify_and_stats_kernel[(total_rows,)](
            inp_f32, mean, std, out_f32, N, threshold_scale, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4
        )

        # Cast back to original dtype
        return out_f32.to(inp.dtype)


def run(*args):
    return ModelNew()(*args)
