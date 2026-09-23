import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Original _ndtri approximation (Abramowitz & Stegun 7.1.26) for inverse of std normal CDF.
# We'll use it on the host to get a scalar multiplier. We could implement in Triton, but
# it's a scalar and doing it in Python is fine since it's not part of the Triton hot path.
def _ndtri(p: torch.Tensor) -> torch.Tensor:
    """Inverse of the standard normal CDF (quantile function) using A&S approximation."""
    # Constants
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
    c6 = 2.549732539343734e+00  # using provided c6 (original had a typo in c5/c6)

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    # We only ever call with a 0-dim tensor scalar, so just compute directly
    p0 = float(p.item())
    if p0 < p_low:
        q = math.sqrt(-2.0 * math.log(p0))
        result = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                 ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
        return torch.tensor(result, dtype=torch.float32, device=p.device)
    elif p0 > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p0))
        result = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                 ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
        return torch.tensor(result, dtype=torch.float32, device=p.device)
    else:
        q = p0 - 0.5
        r = q * q
        result = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q / \
                 (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
        return torch.tensor(result, dtype=torch.float32, device=p.device)


@triton.jit
def _row_reduce_mean_std_kernel(inp_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    For each row (pid = program_id(0)), compute mean and std over the last dimension of length N.
    inp_ptr: pointer to input tensor of shape [batch, seq, N]
    mean_ptr: pointer to output mean of shape [batch*seq]
    std_ptr: pointer to output std of shape [batch*seq]
    """
    pid = tl.program_id(0)
    # Compute base offset for this row. We assume the input is laid out as [B, S, N] contiguous:
    # row base = pid * N, since for each (b,s), the feature slice is contiguous of length N.
    # We need to map pid to (b, s). Using B and S from host: pid = b*S + s.
    # But we don't have B, S here, so we instead ensure the caller launches with grid=(B*S,).
    # So base = pid * N.
    base = pid * N

    sum_val = 0.0
    sum_sq = 0.0

    offset = 0
    while offset < N:
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        offset += BLOCK_SIZE

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    std = tl.sqrt(var)  # population std (unbiased=False), matches PyTorch default in code

    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def _row_sparsify_kernel(inp_ptr, mean_ptr, std_ptr, out_ptr, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    For each row (pid = program_id(0)):
      Load mean and std, compute cutoff = mean + std * std_multiplier
      Subtract cutoff, apply ReLU, store result
    inp_ptr: pointer to input [B, S, N] contiguous
    mean_ptr/std_ptr: pointers to mean/std of shape [B*S]
    out_ptr: pointer to output [B, S, N] contiguous
    N: length of feature dimension
    std_multiplier: scalar float32
    """
    pid = tl.program_id(0)
    base = pid * N

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    cutoff = mean + std * std_multiplier  # scalar

    offset = 0
    while offset < N:
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = x - cutoff
        # ReLU: y = max(y, 0)
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + idx, y, mask=mask)
        offset += BLOCK_SIZE


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward:
        - Compute per-(batch, seq) mean and std across feature dim using Triton reduction kernel.
        - Compute cutoff threshold = mean + std * inv_norm(target_sparsity) on host (scalar).
        - Apply elementwise ReLU(inputs - cutoff) using Triton kernel.
        Returns: tensor of same shape as inputs, cast back to bfloat16 like original.
        """
        # Early exit: no sparsity requested
        if target_sparsity == 0.0:
            # Return a new tensor to avoid autograd issues if needed
            return inputs.clone()

        # Ensure inputs are on CUDA for Triton. If not, fall back to PyTorch (not preferred in this benchmark).
        if not inputs.is_cuda or not TRITON_AVAILABLE:
            # Fallback: pure PyTorch path (to maintain correctness if Triton not available).
            inputs_f32 = inputs.to(torch.float32)
            inputs_mean = inputs_f32.mean(dim=-1, keepdim=True)
            inputs_std = inputs_f32.std(dim=-1, keepdim=True, unbiased=False)
            std_multiplier = _ndtri(torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device))
            cutoff = inputs_mean + inputs_std * std_multiplier
            sparse_output = torch.relu(inputs_f32 - cutoff)
            return sparse_output.to(torch.bfloat16)

        # We need shape info
        B, S, N = inputs.shape
        # For best performance, make input contiguous along last dim
        inp = inputs.contiguous()

        # Allocate buffers for mean and std (float32 for numerical stability)
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Launch reduction kernel: one program per (b, s) row
        grid = (B * S,)
        # Choose BLOCK_SIZE; 1024 works well for typical N up to 16k
        BLOCK_SIZE = 1024
        _row_reduce_mean_std_kernel[grid](inp, mean_buf, std_buf, N, BLOCK_SIZE)

        # Compute std_multiplier as scalar (host) using the same A&S approximation
        std_multiplier = _ndtri(torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device))
        # Ensure std_multiplier is a Python float for Triton scalar arg
        std_multiplier_scalar = float(std_multiplier.item())

        # Allocate output (compute in float32, then cast to bfloat16 to match original)
        out = torch.empty_like(inp, dtype=torch.float32, device=inputs.device)

        # Launch sparsify kernel
        _row_sparsify_kernel[grid](inp, mean_buf, std_buf, out, N, std_multiplier_scalar, BLOCK_SIZE)

        # Cast back to bfloat16 to match original Model behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
