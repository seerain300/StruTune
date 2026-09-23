import torch
import triton
import triton.language as tl


# Triton kernel: compute per-row mean and population std along the last dimension.
# Assumes X is 2D [rows, N], with rows = batch*seq. Outputs mean and std (unbiased=False).
@triton.jit
def row_stats_kernel(
    X_ptr,           # *f32
    MEAN_ptr,        # *f32, shape [rows]
    STD_ptr,         # *f32, shape [rows]
    rows,            # int
    N,               # int
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Each program handles one row
    # Accumulate sum and sum of squares in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # Iterate over columns in chunks of BLOCK
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X_ptr + row_id * N + cols, mask=mask, other=0.0)
        # x is fp32
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / N
    # Population std: sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / N - mean * mean
    # Clamp variance to non-negative to avoid tiny negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store results
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Triton kernel: compute ndtri(p) for a single p (Abramowitz & Stegun approximation).
# p is a scalar in (0, 1). Output is the quantile z of the standard normal at p.
# We launch this kernel once with grid=(1,) to compute the multiplier for all rows.
@triton.jit
def ndtri_kernel(P_scalar, OUT_ptr):
    # Single program, compute scalar
    p = P_scalar  # f32 scalar
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
    c3 = -2.506628277459239e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    # Determine region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        # Horner for numerator
        num = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        z = num / den
    elif p > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        num = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        z = num / den
    else:
        q = p - 0.5
        r = q * q
        num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        z = num / den

    tl.store(OUT_ptr, z)


# Triton kernel: apply ReLU(x - threshold) elementwise for each row.
# X: [rows, N] fp32, THRESH: [rows] fp32, OUT: [rows, N] fp32
@triton.jit
def relu_threshold_kernel(
    X_ptr,            # *f32
    THRESH_ptr,       # *f32, length = rows
    OUT_ptr,          # *f32
    rows,             # int
    N,                # int
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    # Loop over columns in chunks
    for start in range(0, N, BLOCK):
        cols = start + offs
        mask = cols < N
        x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0)
        thr = tl.load(THRESH_ptr + row)
        y = x - thr
        y = tl.maximum(y, 0.0)
        tl.store(OUT_ptr + row * N + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float):
        super().__init__()
        # Keep target_sparsity as a Python float; no torch ops in forward
        self.target_sparsity = float(target_sparsity)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expect inputs of shape [batch_size, seq_len, intermediate_size]
        assert inputs.dim() == 3, "inputs must be 3D: [batch, seq, features]"
        assert inputs.is_cuda, "inputs must be on CUDA for Triton execution"
        assert inputs.dtype in (torch.bfloat16, torch.float16, torch.float32), "unsupported dtype"

        B, S, N = inputs.shape
        rows = B * S

        # Ensure contiguous
        x = inputs.contiguous()

        # Compute statistics in fp32 for numerical stability
        # We’ll create a 2D fp32 view of the input to feed row_stats_kernel
        x_f32 = x.to(torch.float32)

        # Allocate outputs for mean and std
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        std = torch.empty(rows, dtype=torch.float32, device=x.device)

        # Launch row_stats_kernel
        BLOCK = 256
        grid = (rows,)
        row_stats_kernel[grid](
            x_f32,
            mean,
            std,
            rows,
            N,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # Compute inverse CDF for target_sparsity using Triton
        p = torch.tensor(self.target_sparsity, dtype=torch.float32, device=x.device)
        multiplier = torch.empty(1, dtype=torch.float32, device=x.device)
        ndtri_kernel[(1,)](
            p,
            multiplier,
            num_warps=1,
        )
        multiplier = float(multiplier.item())  # keep as Python float for host-side math

        # Compute per-row thresholds: mean[row] + std[row] * multiplier
        # We don’t create a large THRESH tensor; instead we’ll build per-row pointers in the next kernel.
        # However, Triton kernels can’t index per-row scalars dynamically across threads; so we precompute a
        # threshold vector on device using torch, which is only used to drive the Triton elementwise kernel.
        # This avoids host torch ops after this point.
        # Note: creating this vector is fine, since it’s small O(rows).
        threshold = mean + std * multiplier  # shape [rows], fp32, on device

        # Apply ReLU(inputs - threshold) in Triton
        out = torch.empty_like(x_f32)  # fp32 output
        grid_relu = (rows,)
        relu_threshold_kernel[grid_relu](
            x_f32,
            threshold,      # fp32 device tensor
            out,
            rows,
            N,
            BLOCK=256,
            num_warps=4,
        )

        # Match original behavior: return bf16
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
