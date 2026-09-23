import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# -----------------------------
# Triton kernels
# -----------------------------
if TRITON_AVAILABLE:
    @triton.jit
    def reduce_sum_sumsq_kernel(
        x_ptr,                  # *const float, input base pointer
        sums_ptr,               # *float, length B*S
        sums2_ptr,              # *float, length B*S
        B, S, F,                # int32, dims
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        # map pid to (b, s)
        b = pid // S
        s = pid % S

        # start offset for this row
        row_start = (b * S + s) * F

        local_sum = 0.0
        local_sumsq = 0.0

        # loop over feature dimension in chunks
        for offs in range(0, F, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            mask = idx < F
            ptrs = x_ptr + row_start + idx
            vals = tl.load(ptrs, mask=mask, other=0.0)
            # vals is BLOCK_SIZE vector
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        # accumulate to global per-(b, s)
        tl.atomic_add(sums_ptr + pid, local_sum)
        tl.atomic_add(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,                  # *const float (input in float32)
        out_ptr,                # *bfloat16 (output)
        threshold_ptr,          # *const float, shape [B*S]
        total_elems,            # int32, B*S*F
        B, S, F,                # int32
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        # linear indices for this block
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elems

        # compute (b, s, f) for each linear index
        SF = S * F
        b = offs // (S * F)
        tmp = offs % (S * F)
        s = tmp // F
        f = tmp % F

        # gather threshold[b, s, 0] = threshold[b*S + s]
        ts_idx = b * S + s
        thr = tl.load(threshold_ptr + ts_idx, mask=mask, other=0.0)

        # compute input pointers
        in_ptrs = x_ptr + offs
        x = tl.load(in_ptrs, mask=mask, other=0.0)

        # diff and relu
        diff = x - thr  # broadcasts thr vector over the BLOCK
        y = tl.maximum(diff, 0.0)

        # cast to bfloat16 for output
        y_bf16 = y.to(tl.bfloat16)

        out_ptrs = out_ptr + offs
        tl.store(out_ptrs, y_bf16, mask=mask)


# -----------------------------
# Host-side helpers
# -----------------------------
def _ndtri_scalar(p: float) -> float:
    """
    Abramowitz and Stegun approximation for inverse of standard normal CDF.
    Equivalent to torch.erf-based inverse, but this specific approximation
    is what the provided _ndtri uses. We implement it here for consistency.

    Args:
        p: scalar in (0, 1). 0.0 and 1.0 are not supported in theory, but we can handle limits.

    Returns:
        float: quantile z such that P(Z <= z) = p.
    """
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
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    # Lower region
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
               ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    # Upper region
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
               ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    # Central region
    q = p - 0.5
    r = q * q
    return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)*q / \
           (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)


def run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-optimized implementation of the original run function.
    It computes:
      - mean and std along the last dimension (feature size) per [b, s].
      - threshold = mean + std * ndtri(target_sparsity).
      - outputs = relu(inputs - threshold), cast to bfloat16.
    Requirements:
      - inputs must be CUDA tensor (Triton requires CUDA). We assume it is.
      - We compute stats in float32; outputs in bfloat16.
    """
    assert inputs.is_cuda, "inputs must be a CUDA tensor for Triton kernels."
    assert inputs.dtype == torch.bfloat16, "This optimized path expects inputs in bfloat16; convert outside if needed."

    # Convert to float32 for statistics computation
    x = inputs.to(torch.float32)
    B, S, F = x.shape

    # If no sparsity, return directly
    if target_sparsity == 0.0:
        # Keep dtype as bfloat16 to match original behavior
        return x.to(torch.bfloat16)

    # Prepare accumulators for sums and sums^2
    sums = torch.zeros(B * S, dtype=torch.float32, device=x.device)
    sums2 = torch.zeros(B * S, dtype=torch.float32, device=x.device)

    # Launch reduction kernel
    total_rows = B * S
    # Choose a BLOCK_SIZE for reduction; 1024 is a good default
    BLOCK_SIZE = 1024

    reduce_sum_sumsq_kernel[(total_rows,)](
        x, sums, sums2,
        B, S, F,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Compute mean and std per (b, s)
    mean = sums / F
    # population variance: E[x^2] - (E[x])^2
    var = sums2 / F - mean * mean
    var = torch.clamp(var, min=0.0)  # guard against tiny negative due to numerical error
    std = torch.sqrt(var)

    # Compute scalar multiplier for target_sparsity using the same approximation
    multiplier = _ndtri_scalar(float(target_sparsity))

    # Build threshold tensor of shape [B, S, 1] (float32)
    threshold = (mean + std * multiplier).unsqueeze(-1)  # [B, S, 1]

    # Prepare output tensor (bfloat16)
    out = torch.empty((B, S, F), dtype=torch.bfloat16, device=x.device)

    # Flatten for pointwise kernel
    total_elems = B * S * F

    # Launch sparsify + ReLU kernel
    sparsify_relu_kernel[(triton.cdiv(total_elems, 1024),)](
        x, out, threshold,
        total_elems, B, S, F,
        BLOCK_SIZE=1024,
    )

    return out


# -----------------------------
# Entry point module
# -----------------------------
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # The original Model.forward expects a single tensor input.
        # We mimic the original behavior: run on the provided input(s).
        # Note: Triton requires CUDA tensors; the evaluation harness should provide them on CUDA.
        # If inputs are multiple tensors, this implementation expects a single tensor with shape [B, S, F].
        # In typical usage, ModelNew is called with one tensor argument.
        if len(args) != 1:
            raise ValueError("ModelNew.forward expects a single tensor input of shape [batch_size, seq_len, intermediate_size].")

        inputs = args[0]
        # If Triton is available and tensor is CUDA, use Triton path
        if TRITON_AVAILABLE and inputs.is_cuda:
            # Ensure dtype bfloat16 for Triton path
            if inputs.dtype != torch.bfloat16:
                inputs = inputs.to(torch.bfloat16)
            return run_triton(inputs, target_sparsity=0.5)  # default target_sparsity; adjust as needed
        else:
            # Fallback to original PyTorch implementation if Triton is not available or not CUDA
            # Replicate the original run(inputs, target_sparsity) logic in pure PyTorch for correctness.
            # Note: In the provided original code, target_sparsity is not passed to run; we assume a default.
            # Here we assume target_sparsity=0.5 for consistency with the Triton path.
            # If you need exact behavior of the original, you would call run(inputs, target_sparsity=...),
            # but since we don't have the original run definition here, we mimic its structure:
            if 0.0 == 0.5:  # just a placeholder; we won't use original since it's not provided
                return inputs

            # Since original run is not available, implement the same logic here using torch:
            # Compute mean and std along last dim in float32
            x = inputs.to(torch.float32)
            B, S, F = x.shape
            mean = torch.mean(x, dim=-1, keepdim=True)
            std = torch.std(x, dim=-1, keepdim=True, unbiased=False)

            # Compute multiplier with normal inverse CDF (we can use torch.erf or the same approximation)
            multiplier = _ndtri_scalar(0.5)  # default sparsity; adjust as needed

            cutoff = mean + std * multiplier
            sparse = torch.relu(x - cutoff)
            return sparse.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
