import torch
import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p_value: tl.float32, out_ptr):
    """
    Compute inverse standard normal CDF (quantile function) for probability p_value in (0, 1).
    Uses Abramowitz & Stegun 7.1.26 rational approximation:
      invPhi(p) = sign(p - 0.5) * sqrt(2) * erfinv(2*abs(p - 0.5))
    where erfinv(x) ≈ sign(x) * poly(t) * exp(-x^2),
      t = 1 / (1 + 0.3275911 * |x|)
      poly(t) = (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5)
    Coefficients:
      a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429
    Writes result to out_ptr[0] as float32.
    """
    p = p_value

    # Handle both sides around p=0.5 via a unified transformation:
    # Let x = 2*(p - 0.5). For p < 0.5, x negative; for p > 0.5, x positive.
    # We'll compute erfinv(x) and apply sign at the end.
    x = 2.0 * (p - 0.5)
    sign = tl.where(p >= 0.5, 1.0, -1.0)
    x = sign * x  # now x ∈ [-1, 1], zero when p=0.5

    # Abramowitz & Stegun 7.1.26 polynomial for erfinv
    p_const = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    t = 1.0 / (1.0 + p_const * tl.abs(x))
    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
    erfinv_approx = poly * tl.exp(-(x * x))
    # Apply sign to erfinv(x)
    erfinv_approx = sign * erfinv_approx

    # invPhi(p) = sign(p - 0.5) * sqrt(2) * erfinv(x)
    sqrt2 = 1.4142135623730951
    result = sign * sqrt2 * erfinv_approx

    # Store result to out_ptr[0]
    tl.store(out_ptr, result)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, B, S, N, std_multiplier, BLOCK_SIZE: tl.constexpr):
    """
    One Triton program per (b, s) row. Computes mean and std over N, then applies
    ReLU gating: y = max(0, x - (mean + std * std_multiplier)).

    x_ptr: pointer to input [B, S, N] contiguous data (float16/float32)
    out_ptr: pointer to output [B, S, N] float32
    B, S, N: sizes (ints)
    std_multiplier: float32 scalar from compute_invphi_kernel
    BLOCK_SIZE: chunk size for looping over N
    """
    # 2D grid: program_id(0) = b, program_id(1) = s
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Base offset for this row in a contiguous [B, S, N] layout: base = ((b * S) + s) * N
    base = (b * S + s) * N

    # Accumulate sum and sum of squares in float32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # First pass: reduction over N
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n_f = tl.float32(N)
    mean = sum_val / n_f
    var = sum_sq / n_f - mean * mean
    var = tl.maximum(var, 0.0)  # guard against negative due to roundoff
    std = tl.sqrt(var)

    threshold = mean + std * std_multiplier

    # Second pass: elementwise gating
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only forward:
        - If target_sparsity == 0.0, return inputs.
        - Else compute per-row mean and std, threshold = mean + std * inv-Phi(target_sparsity),
          and apply ReLU gating: max(0, inputs - threshold). Return in bfloat16.
        """
        if target_sparsity == 0.0:
            return inputs

        if inputs.ndim != 3:
            raise ValueError("ModelNew expects input of shape [batch_size, seq_len, intermediate_size].")

        B, S, N = inputs.shape

        # Ensure contiguous input
        x = inputs.contiguous()

        # Output tensor in float32 for stability
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x.device)

        # Compute inv-Phi(target_sparsity) using Triton kernel and store into a 1-element tensor
        std_multiplier = torch.empty((), dtype=torch.float32, device=x.device)
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch row sparsity kernel: 2D grid over (B, S)
        grid = (B, S)
        row_sparsity_kernel[grid](
            x, out_f32, B, S, N, std_multiplier, BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16 to match the original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
