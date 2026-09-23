import torch
import torch.nn.functional as F
import math

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def sparse_gate_rows_kernel(X_ptr, C_ptr, Y_ptr,
                             rows, D,
                             BLOCK_SIZE: tl.constexpr):
    # Each program handles one row
    row_id = tl.program_id(0)
    # If grid is larger than rows (shouldn't happen), return
    if row_id >= rows:
        return

    # Compute base offset for this row in the 2D view
    base = row_id * D

    # Loop over the feature dimension in chunks of BLOCK_SIZE
    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D

        # Load input row slice and per-row cutoff
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
        c = tl.load(C_ptr + base + offs, mask=mask, other=0.0)

        # Apply ReLU gating with per-row threshold: y = max(x - c, 0)
        y = x - c
        y = tl.maximum(y, 0.0)

        # Store result
        tl.store(Y_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, just return inputs
        if target_sparsity == 0.0:
            return inputs

        # Ensure we have a tensor input
        assert isinstance(inputs, torch.Tensor), "inputs must be a torch.Tensor"
        # Compute in float32 for stability
        inputs_f32 = inputs.to(torch.float32)

        # Compute mean and std along last dimension (feature dim), keepdim=True
        # These are per (batch, seq) row scalars
        inputs_mean = torch.mean(inputs_f32, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs_f32, dim=-1, keepdim=True, unbiased=False)

        # Compute the standard deviation multiplier using the provided inverse CDF approximation
        # target_sparsity is a float scalar in (0, 1)
        # We convert it to a 0-dim float32 tensor on the right device
        p = torch.tensor(target_sparsity, dtype=torch.float32, device=inputs.device)
        std_multiplier = _ndtri(p)  # shape: []

        # Compute per-row cutoff: mean + std * multiplier
        # Broadcast multiplier to match shapes
        cutoff = inputs_mean + inputs_std * std_multiplier

        # Prepare 2D contiguous views for Triton: [rows, D]
        rows = inputs_f32.shape[0] * inputs_f32.shape[1]
        D = inputs_f32.shape[2]

        # Flatten inputs and cutoff to 2D views
        X_2d = inputs_f32.reshape(rows, D).contiguous()
        C_2d = cutoff.reshape(rows, D).contiguous()

        # Output buffer (float32)
        Y_2d = torch.empty((rows, D), dtype=torch.float32, device=inputs.device)

        # Launch Triton kernel
        if TRITON_AVAILABLE:
            # Choose a block size; 1024 works well for typical D like 12288
            BLOCK_SIZE = 1024
            grid = (rows,)
            # num_warps can be tuned; 4 or 8 are typical for 1024
            sparse_gate_rows_kernel[grid](
                X_2d, C_2d, Y_2d,
                rows, D,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4
            )
        else:
            # Fallback: pure PyTorch elementwise gating (still correct)
            Y_2d = torch.maximum(X_2d - C_2d, torch.tensor(0.0, dtype=torch.float32, device=inputs.device))

        # Reshape back to original [batch, seq, D] and cast to original dtype (match original behavior: bfloat16 output)
        Y = Y_2d.reshape(inputs_f32.shape).to(inputs.dtype)
        return Y


def _ndtri(p: torch.Tensor) -> torch.Tensor:
    """Inverse of the standard normal CDF (quantile function).
    
    Uses Abramowitz and Stegun approximation (formula 26.2.23).
    This is a rational approximation that works well for p in (0, 1).
    """
    # Constants for the approximation
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

    result = torch.zeros_like(p)

    # Lower region approximation
    mask_low = p < p_low
    q = torch.sqrt(-2.0 * torch.log(p[mask_low]))
    result[mask_low] = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                       (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))

    # Central region approximation
    mask_mid = (p >= p_low) & (p <= p_high)
    q = p[mask_mid] - 0.5
    r = q * q
    result[mask_mid] = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                       (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region approximation
    mask_high = p > p_high
    q = torch.sqrt(-2.0 * torch.log(1.0 - p[mask_high]))
    result[mask_high] = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                        ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    return result