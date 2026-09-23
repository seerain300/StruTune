import torch
import triton
import triton.language as tl


@triton.jit
def _row_reduce_mean_std_kernel(
    inp_ptr,            # *const float, input tensor base (flattened per row)
    mean_ptr,           # *float, output means, length B*S
    std_ptr,            # *float, output stds, length B*S
    N,                  # int: feature dimension size
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    # Accumulate sum and sum of squares across the N features for this row
    sum_val = 0.0
    sum_sq = 0.0

    # Iterate over N in chunks of BLOCK_SIZE
    for offs in range(0, N, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        vals = tl.load(inp_ptr + row_id * N + idx, mask=mask, other=0.0)
        # Reduce within the vector to scalars and accumulate
        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    var = tl.maximum(var, 0.0)  # guard against tiny negative due to FP
    std = tl.sqrt(var)

    # Write results
    tl.store(mean_ptr + row_id, mean)
    tl.store(std_ptr + row_id, std)


@triton.jit
def _ndtri_scalar_kernel(
    p_ptr,              # *const float, 1-element tensor with target_sparsity
    out_ptr,            # *float, 1-element tensor to store inv std norm
):
    # Load scalar p
    p = tl.load(p_ptr)

    # Abramowitz & Stegun 7.1.26 approximation for inverse standard normal CDF
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Polynomial coefficients
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

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

    # Compute branch-specific values
    q_low = tl.sqrt(-2.0 * tl.log(p))
    y_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))

    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) / \
               (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    y_mid = poly_mid

    # For the high branch, compute q_high = sqrt(-2*log(1-p)) and its polynomial
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    y_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))

    # Select result based on p
    # Triton supports scalar tl.where for selection
    y = tl.where(p < p_low, y_low, tl.where(p > p_high, y_high, y_mid))
    tl.store(out_ptr, y)


@triton.jit
def _row_sparsify_kernel(
    inp_ptr,            # *const float, input base
    out_ptr,            # *float, output base
    mean_ptr,           # *const float, means per row
    std_ptr,            # *const float, stds per row
    N,                  # int: feature dimension size
    std_multiplier,     # float: inv std norm from host
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    cutoff = mean + std * std_multiplier

    # Iterate over N in chunks, compute relu(inp - cutoff)
    for offs in range(0, N, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        vals = tl.load(inp_ptr + row_id * N + idx, mask=mask, other=0.0)
        diff = vals - cutoff  # cutoff is scalar, broadcast
        diff = tl.maximum(diff, 0.0)  # ReLU
        tl.store(out_ptr + row_id * N + idx, diff, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return inputs unchanged
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and compute in float32
        inp = inputs.contiguous().to(torch.float32)
        B, S, N = inp.shape

        # Buffers for mean and std per row
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=inp.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=inp.device)

        # Launch reduction kernel: one program per (batch, seq) row
        BLOCK_SIZE = 1024
        grid = (B * S,)
        _row_reduce_mean_std_kernel[grid](inp, mean_buf, std_buf, N, BLOCK_SIZE)

        # Compute inv std norm for target_sparsity via Triton scalar kernel
        p_dev = torch.tensor(float(target_sparsity), dtype=torch.float32, device=inp.device)
        std_multiplier_buf = torch.empty(1, dtype=torch.float32, device=inp.device)
        _ndtri_scalar_kernel[(1,)](p_dev, std_multiplier_buf)
        std_multiplier_scalar = float(std_multiplier_buf.item())

        # Output buffer
        out = torch.empty_like(inp)

        # Launch sparsification kernel
        _row_sparsify_kernel[grid](inp, out, mean_buf, std_buf, N, std_multiplier_scalar, BLOCK_SIZE)

        # Cast back to original dtype
        return out.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
