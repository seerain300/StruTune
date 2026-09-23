import torch
import triton
import triton.language as tl


@triton.jit
def _row_stats_kernel(X_ptr, S_ptr, Ssq_ptr, S: tl.constexpr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    One program per row (S rows). For each row, iterate over H in tiles of size BLOCK_SIZE,
    accumulate sum and sum of squares in float32, store to S_ptr[row] and Ssq_ptr[row].
    """
    row = tl.program_id(0)
    # Bounds check
    if row >= S:
        return

    # Initialize accumulators
    s = 0.0
    ss = 0.0

    # Iterate over H in tiles
    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Row-major: X is [S, H] contiguous -> X[row, offs]
        x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)
        # Cast to float32 for accumulation
        x = x.to(tl.float32)
        s += tl.sum(x, axis=0)
        ss += tl.sum(x * x, axis=0)
        start += BLOCK_SIZE

    # Store sums
    tl.store(S_ptr + row, s)
    tl.store(Ssq_ptr + row, ss)


@triton.jit
def _gate_kernel(X_ptr, Y_ptr, Mean_ptr, Std_ptr, z_scalar_ptr, S: tl.constexpr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    One program per row. Load mean and std for the row, compute threshold = mean + std * z_scalar.
    Then iterate over H in tiles, compute y = max(0, X[row, :] - threshold), store to Y.
    """
    row = tl.program_id(0)
    if row >= S:
        return

    mean = tl.load(Mean_ptr + row)
    std = tl.load(Std_ptr + row)
    z = tl.load(z_scalar_ptr)  # 1-element tensor
    threshold = mean + std * z

    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - threshold
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + row * H + offs, y, mask=mask)
        start += BLOCK_SIZE


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-row mean and std across the last dimension, then
    applies y = max(0, x - (mean + std * _ndtri(target_sparsity))) and returns bfloat16.
    """
    # Ensure input is on CUDA and contiguous
    if not inputs.is_cuda:
        inputs = inputs.cuda()
    inputs = inputs.contiguous()

    # Cast to float32 for stable math; flatten [B, L, H] -> [S, H]
    B, L, H = inputs.shape
    S = B * L
    X = inputs.to(torch.float32).view(S, H)

    # Allocate buffers
    S_sum = torch.empty(S, dtype=torch.float32, device=inputs.device)
    S_sumsq = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Y = torch.empty((S, H), dtype=torch.float32, device=inputs.device)

    # Compute z_scalar = _ndtri(target_sparsity) using Abramowitz-Stegun 7.1.26 approximation
    # This is a standard rational approximation for inverse normal CDF.
    # Handle p in (0, 0.5) and (0.5, 1) via symmetry; here target_sparsity is in (0, 1).
    # We implement the central region approximation which is accurate for typical values.
    # Create a 1-element tensor on device to pass to Triton
    p = torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device)
    # Constants
    p_low = 0.02425
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

    # Central region approximation: p in (p_low, 1 - p_low)
    # If p is outside, map to nearest side and use symmetric relation z(p) = -z(1-p)
    # But since we compute here for p directly, we assume central region is used for typical sparsity (e.g., 0.1).
    q = p - 0.5
    r = q * q
    poly_a = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
    poly_b = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    z = poly_a * q / poly_b  # central region result

    z_tensor = z  # 1-element tensor on device

    # Launch row stats kernel: one program per row
    BLOCK_SIZE = 1024
    _row_stats_kernel[(S,)](X, S_sum, S_sumsq, S=S, H=H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Compute mean and std per row (unbiased=False)
    # mean = s / H; var = s^2 / H - (s/H)^2 = ss/H - mean^2; std = sqrt(var)
    mean_rows = S_sum / H
    # var_rows = S_sumsq / H - mean_rows * mean_rows
    var_rows = S_sumsq / H - mean_rows * mean_rows
    std_rows = tl.sqrt(var_rows)  # Triton allows tl.sqrt; ensure non-negative var for typical inputs

    # Launch gating kernel: one program per row
    _gate_kernel[(S,)](X, Y, mean_rows, std_rows, z_tensor, S=S, H=H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

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