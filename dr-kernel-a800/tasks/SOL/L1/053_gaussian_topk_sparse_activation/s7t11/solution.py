import torch
import triton
import triton.language as tl


@triton.jit
def apply_threshold_relu_to_bf16(in_ptr, out_ptr, mean_ptr, std_ptr, z_val, B, S, H, BLOCK: tl.constexpr):
    # One program per (b, s) row
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Load per-row mean and std (fp32 scalars)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)

    # Compute threshold (fp32) using scalar z passed as Python float
    thr = mean + std * z_val

    # Base linear index for this row
    base = b * S * H + s * H

    # Iterate over H in chunks of BLOCK
    offsets = tl.arange(0, BLOCK)
    i = 0
    while i < H:
        idx = i + offsets
        mask = idx < H
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)  # x is fp32
        # Apply threshold: y = max(x - thr, 0)
        y = x - thr
        y = tl.maximum(y, 0.0)
        # Store as bfloat16
        tl.store(out_ptr + base + idx, y.to(tl.bfloat16), mask=mask)
        i += BLOCK


def _ndtri(p: float) -> float:
    """Inverse of the standard normal CDF (quantile function).
    Uses Abramowitz and Stegun approximation (formula 26.2.23) for 0 < p < 1.
    Returns float."""
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

    if p <= p_low:
        q = (-2.0 * tl.log(p))  # note: Triton expects tl.log, but we supply Python float; math computed here
        # Since we return float, emulate A&S
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        nd = poly / den
        return nd
    elif p >= p_high:
        q = (-2.0 * tl.log(1.0 - p))
        poly = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        nd = poly / den
        return nd
    else:
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5))
        nd = poly * q / den
        return nd


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return inputs
        if target_sparsity == 0.0:
            return inputs

        # Ensure 3D [B, S, H] and contiguous
        assert inputs.ndim == 3, "inputs must be [batch_size, seq_len, intermediate_size]"
        inputs = inputs.contiguous()
        B, S, H = inputs.shape

        # Compute statistics using PyTorch to match original exactly (unbiased=False for std)
        in_fp32 = inputs.to(torch.float32)
        # mean across last dim, keepdim=False => shape [B, S]
        inputs_mean = in_fp32.mean(dim=-1)  # [B, S]
        # std across last dim, unbiased=False => shape [B, S]
        inputs_std = in_fp32.std(dim=-1, unbiased=False)  # default unbiased=False in PyTorch is True, but original uses unbiased=False

        # Cast to fp32 and flatten to [B*S] for per-row kernel access
        mean_vec = inputs_mean.to(torch.float32)  # [B, S]
        std_vec = inputs_std.to(torch.float32)    # [B, S]
        mean_ptr = mean_vec.view(-1)  # length B*S
        std_ptr = std_vec.view(-1)    # length B*S

        # Compute inverse normal CDF (z) using original A&S approximation implemented as a helper function
        # Note: target_sparsity is a float in [0, 1]. We avoid creating torch.tensor for this scalar.
        z_val = _ndtri(target_sparsity)  # scalar float

        # Allocate output tensor in bfloat16
        out_bf16 = torch.empty((B, S, H), dtype=torch.bfloat16, device=inputs.device)

        # Launch Triton kernel: one program per row
        grid = (B * S,)
        apply_threshold_relu_to_bf16[grid](
            in_fp32, out_bf16, mean_ptr, std_ptr, z_val,
            B, S, H,
            BLOCK=1024,  # tuneable
            num_warps=4,  # tuneable
            num_stages=2  # tuneable
        )

        return out_bf16


def run(*args):
    return ModelNew()(*args)
