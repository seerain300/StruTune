import torch
import torch.nn.functional as F
import math

# Triton is required. Uncomment if not installed:
# import triton
# import triton.language as tl

# We will implement the elementwise transform in Triton, and use torch for stats.

# Optional: if triton import fails, we could try to import; but we assume Triton is available here.

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

    # Lower region
    mask_low = p < p_low
    q = torch.sqrt(-2.0 * torch.log(p[mask_low]))
    result[mask_low] = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                       ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q = p[mask_mid] - 0.5
    r = q * q
    result[mask_mid] = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)*q / \
                       (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)

    # Upper region
    mask_high = p > p_high
    q = torch.sqrt(-2.0 * torch.log(1.0 - p[mask_high]))
    result[mask_high] = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                        ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)

    return result


@triton.jit
def _relu_sub_scalar_kernel(
    X_ptr,          # *float32, input of shape [B, S, F] flattened as [rows, F]
    C_ptr,          # *float32, cutoff per row, shape [rows] where rows = B*S
    Out_ptr,        # *float32, output of shape [B, S, F] flattened as [rows, F]
    F: tl.constexpr,       # feature size (intermediate_size)
    rows: tl.constexpr,    # number of rows = B*S
    BLOCK_SIZE: tl.constexpr,
):
    # program ids
    pid_row = tl.program_id(0)  # which row (over B*S)
    pid_col = tl.program_id(1)  # which block of features

    # Feature offsets for this block
    cols = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < F

    # Compute pointers
    x_row_ptr = X_ptr + pid_row * F + cols
    out_row_ptr = Out_ptr + pid_row * F + cols

    # Load input and cutoff
    x = tl.load(x_row_ptr, mask=mask, other=0.0)
    cutoff = tl.load(C_ptr + pid_row)  # scalar

    # Compute y = max(0, x - cutoff)
    y = x - cutoff
    y = tl.maximum(y, 0.0)

    # Store
    tl.store(out_row_ptr, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We expect a single 3D tensor input: [batch_size, seq_len, intermediate_size]
        assert len(args) == 1, "ModelNew expects exactly one input tensor"
        input = args[0]
        assert input.dim() == 3, "Input must be a 3D tensor [batch_size, seq_len, intermediate_size]"
        B, S, F = input.shape

        # Convert to float32 for numerical stability in stats
        inputs_f32 = input.to(torch.float32)

        # Compute mean and std along the feature dimension (last dim), keepdim=True
        # Note: unbiased=False matches PyTorch default behavior when computing population std
        inputs_mean = inputs_f32.mean(dim=-1, keepdim=True)
        inputs_std = inputs_f32.std(dim=-1, keepdim=True, unbiased=False)

        # Compute std multiplier using inverse CDF for target sparsity.
        # target_sparsity is a scalar argument expected by the original run function.
        # If not provided, default to 0.0 to avoid error; original code also checks.
        # Here we assume target_sparsity is provided at call site; otherwise, set to 0.0.
        # The original signature of Model.forward is run(*args), and the provided run
        # takes (inputs, target_sparsity). We mirror that behavior.
        # We'll treat target_sparsity as the second argument if present; otherwise default 0.0.
        if len(args) > 1 and isinstance(args[1], (float, torch.Tensor)):
            target_sparsity = float(args[1])
        else:
            target_sparsity = 0.0

        # Create a device scalar tensor for the sparsity
        sparsity_tensor = torch.tensor(target_sparsity, dtype=torch.float32, device=inputs_f32.device)

        std_multiplier = _ndtri(sparsity_tensor)  # 0-d tensor on device
        # Make sure it is a 1-element tensor for broadcasting
        std_multiplier = std_multiplier.view(1)

        # Compute per-row cutoff: mean + std * multiplier, shape [B, S, 1]
        cutoff = inputs_mean + inputs_std * std_multiplier  # broadcast along last dim

        # Flatten rows for kernel launch
        rows = B * S
        # We need to provide C_ptr of shape [rows], i.e., flatten per-row cutoffs
        # We can extract per-row cutoffs by taking last dim = 1
        # Build a vector of cutoffs for each row
        # Note: cutoff has shape [B, S, 1]; we can reshape to [rows]
        cutoff_vec = cutoff.reshape(rows).contiguous()  # [B*S]

        # Allocate output tensor in float32
        out_f32 = torch.empty((B, S, F), dtype=torch.float32, device=inputs_f32.device)

        # Flatten input and output to [rows, F]
        X_flat = inputs_f32.reshape(rows, F)
        Out_flat = out_f32.reshape(rows, F)

        # Launch Triton kernel
        BLOCK_SIZE = 128
        grid = (rows, triton.cdiv(F, BLOCK_SIZE))
        _relu_sub_scalar_kernel[grid](
            X_flat, cutoff_vec, Out_flat,
            F, rows,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4, num_stages=2
        )

        # Cast back to bfloat16 to match original behavior
        out_bf16 = out_f32.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
