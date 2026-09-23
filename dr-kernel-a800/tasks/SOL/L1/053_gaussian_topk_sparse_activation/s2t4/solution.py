import math
import torch
import triton
import triton.language as tl


@triton.jit
def mean_std_kernel(x_ptr, mean_ptr, std_ptr, B, S, N, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row (batch, seq) mean and std across the last dimension (N).
    One program per row. Writes mean and std to mean_ptr[row] and std_ptr[row].
    """
    row_id = tl.program_id(0)  # spans rows in [0, B*S)
    if row_id >= B * S:
        return
    b = row_id // S
    s = row_id % S

    base = b * S * N + s * N  # For contiguous [B, S, N], row base = b*S*N + s*N

    # Accumulators in float32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Loop over N in chunks of BLOCK_SIZE
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n_f = tl.float32(N)
    mean = sum_val / n_f
    var = sum_sq / n_f - mean * mean
    var = tl.maximum(var, 0.0)  # guard against tiny negative due to roundoff
    std = tl.sqrt(var)

    # Write mean and std to output arrays indexed by (b, s)
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def gate_kernel(x_ptr, out_ptr, mean_ptr, std_ptr, B, S, N, B_MULTIPLIER: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Apply ReLU gating per row: out = max(0, x - (mean + std * B_MULTIPLIER)).
    One program per row. Reads mean and std from mean_ptr/std_ptr[row], computes threshold,
    and writes gated output to out.
    """
    row_id = tl.program_id(0)
    if row_id >= B * S:
        return
    b = row_id // S
    s = row_id % S

    base = b * S * N + s * N

    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    threshold = mean + std * B_MULTIPLIER  # B_MULTIPLIER is std_multiplier (inv-Phi(target_sparsity))

    # Process the entire row in chunks
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only forward:
        - If target_sparsity == 0.0, return inputs.
        - Else, compute per-row mean and std, threshold = mean + std * inv-Phi(target_sparsity),
          and apply ReLU gating: max(0, inputs - threshold). Return in bfloat16.
        """
        if target_sparsity == 0.0:
            return inputs

        if inputs.ndim != 3:
            raise ValueError("ModelNew expects input of shape [batch_size, seq_len, intermediate_size].")

        B, S, N = inputs.shape

        # Ensure contiguous input
        x = inputs.contiguous()

        # Compute inv-Phi(target_sparsity) on host using Abramowitz & Stegun 26.2.23 approximation.
        # Single scalar; avoid torch.tensor() in host code.
        p = float(target_sparsity)
        if p < 0.0 or p > 1.0:
            std_multiplier = 0.0  # fallback for invalid probabilities
        else:
            # Constants for lower/upper region approximations
            c1 = -7.784894002430293e-03
            c2 = -3.223964580411365e-01
            c3 = -2.400758277161838e+00
            c4 = -2.549732539343734e+00
            c5 = 4.374664141464968e+00
            c6 = 2.938163982698783e+00
            d1 = 7.784695709041462e-03
            d2 = 3.224671290700398e-01
            d3 = 2.445134137142996e+00
            d4 = 3.754408661907416e+00
            p_low = 0.


def run(*args):
    return ModelNew()(*args)
