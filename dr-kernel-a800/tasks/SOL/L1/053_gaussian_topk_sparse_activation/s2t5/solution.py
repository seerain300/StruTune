import torch
import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(prob: tl.float32, out_ptr):
    """
    Compute inverse standard normal CDF for p = prob using bisection.
    Writes result to out_ptr[0] as float32.
    """
    low = -6.0
    high = 6.0
    # Bisection iterations: 30 provides good accuracy for p in (0, 1)
    for _ in range(30):
        mid = 0.5 * (low + high)
        # erf approximation (Abramowitz & Stegun 7.1.26)
        sqrt2 = 1.4142135623730951
        z = mid / sqrt2
        az = tl.abs(z)
        p_const = 0.3275911
        t = 1.0 / (1.0 + p_const * az)
        # Polynomial in t (ascending powers)
        poly = t
        poly = poly - 0.254829592 * t * t
        poly = poly + 0.284496736 * t * t * t
        poly = poly - 0.161512557 * t * t * t * t
        poly = poly + 0.027643881 * t * t * t * t * t
        poly = poly - 0.003840873 * t * t * t * t * t * t
        erf_approx = 1.0 - poly * tl.exp(-z * z)
        # erf(z/sqrt(2)) is odd: place in [-1, 1] with sign
        sign = tl.where(z >= 0.0, 1.0, -1.0)
        erf_approx = sign * erf_approx

        cdf = 0.5 * (1.0 + erf_approx)
        # bisection update: mid moves toward low if cdf > prob, else toward high
        direction = tl.where(cdf > prob, -1.0, 1.0)
        low = low + direction * (mid - low)
        high = mid + (1.0 - direction) * (high - mid)
    tl.store(out_ptr, mid)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, B: tl.int32, S: tl.int32, N: tl.int32, invphi_ptr, BLOCK_SIZE: tl.constexpr):
    """
    One program per (b, s) row in [B, S, N].
    Computes:
      - mean and std across N in float32
      - threshold = mean + std * invphi
      - y = max(0, x - threshold) elementwise
    """
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S
    row_index = b * S + s
    base = row_index * N  # contiguous linear index for the row

    # Accumulators (float32)
    sum_val = 0.0
    sum_sq = 0.0

    # First pass: reduce sum and sum of squares across the row
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n_f = tl.float32(N)
    mean = sum_val / n_f
    var = sum_sq / n_f - mean * mean
    var = tl.maximum(var, 0.0)  # guard against tiny negatives
    std = tl.sqrt(var)

    # Load inv-Phi scalar
    invphi = tl.load(invphi_ptr)  # float32 scalar
    threshold = mean + std * invphi

    # Second pass: elementwise ReLU gating
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

        x = inputs.contiguous()
        B, S, N = x.shape

        # Output in float32 for numerical stability
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x.device)

        # Compute inv-Phi(target_sparsity) using Triton (store to 1-element tensor)
        std_multiplier = torch.empty((), dtype=torch.float32, device=x.device)
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch row sparsity kernel: one program per (b, s)
        grid = (B * S,)
        row_sparsity_kernel[grid](x, out_f32, B, S, N, std_multiplier, BLOCK_SIZE=1024, num_warps=8)

        # Return in bfloat16 to match the original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
