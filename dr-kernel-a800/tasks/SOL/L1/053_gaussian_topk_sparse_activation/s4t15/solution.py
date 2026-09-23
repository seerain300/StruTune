import torch
import triton
import triton.language as tl


@triton.jit
def _row_reduce_sum_and_sumsq_kernel(inp_ptr, sum_ptr, sumsq_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    For each (batch, seq) row, compute sum and sum of squares across the last dimension of length N.
    inp_ptr: *f32, shape [B, S, N], contiguous
    sum_ptr: *f32, shape [B*S]
    sumsq_ptr: *f32, shape [B*S]
    N: int, length of last dim
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Accumulators in f32
    sum_x = 0.0
    sum_x2 = 0.0

    # Number of chunks and loop
    num_chunks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    for chunk in range(0, num_chunks):
        off = chunk * BLOCK_SIZE
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = N
    # Write partial results (sum and sum of squares)
    tl.store(sum_ptr + pid, sum_x)
    tl.store(sumsq_ptr + pid, sum_x2)


@triton.jit
def _ndtri_scalar_kernel(p_dev_ptr, out_ptr):
    """
    Compute inverse standard normal CDF for scalar p via A&S approximation (7.1.26).
    Stores the result to out_ptr[0].
    p_dev_ptr: *f32, shape [1] (device scalar tensor; kernel loads it)
    out_ptr: *f32, shape [1] (output buffer for scalar result)
    """
    # Load p
    p = tl.load(p_dev_ptr)

    # Constants for A&S approximation
    # Lower region approximation for p in [0, 0.5)
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

    # Polynomial c(z), d(z) for z = sqrt(-2 * log(p))
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

    p_low = 0.02425
    p_high = 1.0 - p_low

    # Use lower region approximation when p < p_low
    use_low = p < p_low

    # Compute z = sqrt(-2 * log(p)) for lower region, else use central or upper
    if use_low:
        z = tl.sqrt(-2.0 * tl.log(p))
        t = z
        poly_c = (((((c1 * t + c2) * t + c3) * t + c4) * t + c5) * t + c6)
        poly_d = (((((d1 * t + d2) * t + d3) * t + d4) * t + 1.0))
        nd = poly_c / poly_d
        result = -z - nd
    else:
        # Central region: p in [p_low, p_high]
        # Map p>0.5 to 1-p to use central formula
        if p > 0.5:
            p_cent = 1.0 - p
        else:
            p_cent = p
        z = tl.sqrt(-2.0 * tl.log(p_cent))
        t = z
        poly_a = (((((a1 * t * t + a2) * t * t + a3) * t * t + a4) * t * t + a5) * t * t + a6) * z
        poly_b = (((((b1 * t * t + b2) * t * t + b3) * t * t + b4) * t * t + b5) * t * t + 1.0)
        nd = poly_a / poly_b
        # If p > 0.5, the original CDF value was for p>0.5; inverse should yield negative for p>0.5.
        result = nd if p <= 0.5 else -nd

    # Store the scalar result
    tl.store(out_ptr, result)


@triton.jit
def _sparsify_row_kernel(inp_ptr, sum_ptr, sumsq_ptr, threshold_scale_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sparsification per row:
      mean = sum / N
      std = sqrt(sumsq / N - mean^2)  (population std, unbiased=False)
      cutoff = mean + std * threshold_scale
      out[i] = max(0, inp[i] - cutoff)
    inp_ptr: *f32, shape [B, S, N]
    sum_ptr: *f32, shape [B*S]
    sumsq_ptr: *f32, shape [B*S]
    threshold_scale_ptr: *f32, shape [1] (device scalar with inv-std-normal for target_sparsity)
    out_ptr: *f32, shape [B, S, N]
    N: int, length of last dim
    """
    pid = tl.program_id(0)
    row_start = pid * N
    sum_x = tl.load(sum_ptr + pid)
    sum_x2 = tl.load(sumsq_ptr + pid)
    n = N
    mean = sum_x / n
    var = sum_x2 / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)
    scale = tl.load(threshold_scale_ptr)  # scalar
    cutoff = mean + std * scale

    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
          - If target_sparsity == 0.0: return inputs.
          - Else: compute per-row sum and sum of squares across last dim, compute inv-std-normal multiplier via Triton,
                  then apply adaptive threshold ReLU in Triton.
        """
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous float32 for computation
        inputs = inputs.contiguous()
        inputs_f32 = inputs.to(torch.float32)

        B, S, N = inputs_f32.shape
        # Allocate per-row stats (sum and sumsq)
        sum_x = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        sum_x2 = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Launch reduction kernel to compute sum and sumsq
        BLOCK_SIZE = 2048  # tuned for typical sizes up to 16384
        grid = (B * S,)
        _row_reduce_sum_and_sumsq_kernel[grid](inputs_f32, sum_x, sum_x2, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)

        # Compute inv-std-normal multiplier for target_sparsity using Triton (scalar)
        p_dev = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=inputs.device)
        std_multiplier = torch.empty([1], dtype=torch.float32, device=inputs.device)
        _ndtri_scalar


def run(*args):
    return ModelNew()(*args)
