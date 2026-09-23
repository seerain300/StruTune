import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(X_ptr, Sums_ptr, S: tl.int32, H: tl.int32, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    row_start = row_id * H
    sum_val = 0.0
    col = 0
    while col < H:
        offsets = col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H
        vals = tl.load(X_ptr + row_start + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
        col += BLOCK_SIZE
    tl.store(Sums_ptr + row_id, sum_val)


@triton.jit
def _row_sumsq_kernel(X_ptr, Sumsq_ptr, S: tl.int32, H: tl.int32, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    row_start = row_id * H
    sumsq_val = 0.0
    col = 0
    while col < H:
        offsets = col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H
        vals = tl.load(X_ptr + row_start + offsets, mask=mask, other=0.0)
        sumsq_val += tl.sum(vals * vals, axis=0)
        col += BLOCK_SIZE
    tl.store(Sumsq_ptr + row_id, sumsq_val)


@triton.jit
def _mean_std_kernel(Sums_ptr, Sumsq_ptr, Mean_ptr, Var_ptr, S: tl.int32, H: tl.int32):
    row_id = tl.program_id(0)
    sum_val = tl.load(Sums_ptr + row_id)
    sumsq_val = tl.load(Sumsq_ptr + row_id)
    mean = sum_val / H
    var = sumsq_val / H - mean * mean
    var = tl.maximum(var, 0.0)  # avoid negative due to rounding
    std = tl.sqrt(var)
    tl.store(Mean_ptr + row_id, mean)
    tl.store(Var_ptr + row_id, var)


@triton.jit
def _gate_kernel(X_ptr, Y_ptr, Mean_ptr, Var_ptr, z_ptr, S: tl.int32, H: tl.int32, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    row_start = row_id * H
    mean = tl.load(Mean_ptr + row_id)
    var = tl.load(Var_ptr + row_id)
    std = tl.sqrt(var)  # recompute std inside kernel
    threshold = mean + std * tl.load(z_ptr)
    # Initialize output row to zeros
    col = 0
    while col < H:
        offsets = col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < H
        x = tl.load(X_ptr + row_start + offsets, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(Y_ptr + row_id * H + offsets, y, mask=mask)
        col += BLOCK_SIZE


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-row mean and std across the last dim, then applies y = max(0, x - (mean + std * _ndtri(target_sparsity)))
    and returns bfloat16.
    """
    # Ensure input is contiguous and on CUDA
    inputs = inputs.contiguous()
    if not inputs.is_cuda:
        inputs = inputs.cuda()
    B, L, H = inputs.shape
    S = B * L

    # Flatten to [S, H] and cast to float32 for stable math
    X = inputs.to(torch.float32).view(S, H)

    # Allocate outputs
    Sums = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Sumsq = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Mean = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Var = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Y = torch.empty((S, H), dtype=torch.float32, device=inputs.device)

    # Compute z_scalar via pure Python (Abramowitz-Stegun 7.1.26 approximation) to avoid torch.log/tensor in host
    p = float(target_sparsity)
    p_low = 0.02425
    p_high = 1.0 - p_low
    pi = 3.141592653589793

    # a coefficients for lower and upper regions
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

    # c and d coefficients for lower/upper polynomials
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

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    q = p - 0.5
    r = q * q
    z_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
              ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Select piecewise: for p, use lower, upper, or mid
    # Triton requires vectorization; we create a small vector to mimic behavior
    # But since we only need scalar z, we take the lower or upper depending on p:
    use_low = p < p_low
    use_high = p > p_high
    z_scalar = tl.where(use_low, z_low, 0.0)
    z_scalar = tl.where(use_high, z_high, z_scalar)
    z_scalar = tl.where(~use_low & ~use_high, z_mid, z_scalar)

    # Launch kernels
    BLOCK_SIZE = 1024
    _row_sum_kernel[(S,)](X, Sums, S=S, H=H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    _row_sumsq_kernel[(S,)](X, Sumsq, S=S, H=H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    _mean_std_kernel[(S,)](Sums, Sumsq, Mean, Var, S=S, H=H)
    _gate_kernel[(S,)](X, Y, Mean, Var, z_scalar, S=S, H=H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16 to match original behavior
    out = Y.view(B, L, H).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        return _run_triton(inputs, target_sparsity=0.1)  # default sparsity; configurable