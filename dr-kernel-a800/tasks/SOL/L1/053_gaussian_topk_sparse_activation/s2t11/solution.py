import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p, out_ptr):
    """
    Compute inverse standard normal CDF for probability p (0 < p < 1).
    Writes the result to out_ptr (1-element device tensor of float32).
    Uses bisection over z in [-6, 6] and an erf approximation for CDF.
    p: Python float
    out_ptr: pointer to 1-element device tensor of float32
    """
    low = -6.0
    high = 6.0
    # Bisection with 30 iterations for good precision
    for _ in range(30):
        mid = (low + high) * 0.5
        inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)
        t = mid * inv_sqrt2
        # Abramowitz & Stegun erf approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        e = poly * tl.exp(-t * t)
        erf_t = 1.0 - e
        # erf is odd; use sign(t)
        sign_t = tl.where(t >= 0.0, 1.0, -1.0)
        erf_t = sign_t * erf_t
        cdf_mid = 0.5 * (1.0 + erf_t)
        if cdf_mid > p:
            high = mid
        else:
            low = mid
    # Store result
    tl.store(out_ptr, mid)


@triton.jit
def row_sparsity_kernel(x_ptr, out_ptr, B, S, N, std_multiplier_ptr, BLOCK_SIZE: tl.constexpr):
    """
    One Triton program per row (b, s).
    Two passes over N:
      - Pass 1: accumulate sum and sum of squares to compute mean and std.
      - Pass 2: compute threshold and write output = max(0, x - threshold).
    x_ptr: *fp16/bf16 pointer to input tensor [B, S, N] (assumed contiguous).
    out_ptr: *fp32 pointer to output tensor [B, S, N] (float32).
    std_multiplier_ptr: pointer to 1-element device tensor with inv-Phi(target_sparsity) in float32.
    """
    row_id = tl.program_id(0)  # 0 .. B*S-1
    b = row_id // S
    s = row_id % S
    base = b * S * N + s * N  # row start offset in contiguous layout

    # Pass 1: compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    n0 = 0
    while n0 < N:
        offs = n0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        n0 += BLOCK_SIZE

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    std = tl.sqrt(var)

    # Load std_multiplier (inv-Phi(target_sparsity))
    std_multiplier = tl.load(std_multiplier_ptr)
    threshold = mean + std * std_multiplier

    # Pass 2: write output
    n0 = 0
    while n0 < N:
        offs = n0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)
        tl.store(out_ptr + base + offs, y, mask=mask)
        n0 += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        """
        x: input tensor of shape [B, S, N], any floating dtype.
        target_sparsity: float in (0,1), e.g., 0.01. If 0.0, return x unchanged.
        Returns: tensor of shape [B, S, N], dtype bfloat16.
        """
        if target_sparsity == 0.0:
            # No sparsity: return original, cast to bfloat16 to match output behavior
            return x.to(torch.bfloat16)

        x_contig = x.contiguous()
        B, S, N = x_contig.shape

        # Output in float32 (compute dtype), will cast to bfloat16 at end
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_contig.device)

        # Prepare 1-element device tensor for std_multiplier
        std_multiplier = torch.empty((), dtype=torch.float32, device=x_contig.device)

        # Compute inv-Phi(target_sparsity) in Triton (no torch in forward)
        compute_invphi_kernel[(1,)](float(target_sparsity), std_multiplier)

        # Launch sparsity kernel: one program per row (B*S)
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_contig, out_f32, B, S, N, std_multiplier, BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16 to match typical model behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
