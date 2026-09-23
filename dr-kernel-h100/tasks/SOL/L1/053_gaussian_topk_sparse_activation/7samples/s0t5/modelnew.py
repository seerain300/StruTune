import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(
    x_ptr,                # *float32, shape [B*S*K] contiguous
    mean_ptr,             # *float32, shape [B*S] (viewed as [rows])
    std_ptr,              # *float32, shape [B*S] (viewed as [rows])
    B: tl.constexpr,      # batch size
    S: tl.constexpr,      # seq_len
    K: tl.constexpr,      # intermediate_size (last dim)
    BLOCK_SIZE: tl.constexpr
):
    # One program per row (i in [0, B*S))
    i = tl.program_id(0)
    # Map linear row index to (batch, seq)
    b = i // S
    s = i % S
    # Base offset for this row in flattened [B*S*K]
    row_base = i * K
    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0
    # Loop over columns in chunks
    for col in range(0, K, BLOCK_SIZE):
        offs = row_base + col + tl.arange(0, BLOCK_SIZE)
        mask = (col + tl.arange(0, BLOCK_SIZE)) < K
        x_chunk = tl.load(x_ptr + offs, mask=mask, other=0.0)
        # tl.sum reduces the vector to a scalar
        sum_val += tl.sum(x_chunk)
        sum_sq += tl.sum(x_chunk * x_chunk)
    # Compute mean and std (population std: unbiased=False)
    mean = sum_val / K
    var = sum_sq / K - mean * mean
    # Numerical safety: clamp variance to non-negative
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    # Store mean and std to [B*S] buffers (viewed as [rows])
    tl.store(mean_ptr + i, mean)
    tl.store(std_ptr + i, std)


@triton.jit
def compute_ndtri_kernel(
    z_ptr,                # *float32, length 1
    sparsity: tl.constexpr,  # float32 scalar
    # Constants for Abramowitz & Stegun 5.2.23
    a1: tl.constexpr, a2: tl.constexpr, a3: tl.constexpr, a4: tl.constexpr, a5: tl.constexpr, a6: tl.constexpr,
    b1: tl.constexpr, b2: tl.constexpr, b3: tl.constexpr, b4: tl.constexpr, b5: tl.constexpr,
    c1: tl.constexpr, c2: tl.constexpr, c3: tl.constexpr, c4: tl.constexpr, c5: tl.constexpr, c6: tl.constexpr,
    d1: tl.constexpr, d2: tl.constexpr, d3: tl.constexpr, d4: tl.constexpr,
    p_low: tl.constexpr, p_high: tl.constexpr
):
    # Compute inverse normal CDF z for given sparsity
    # We treat sparsity as p in (0, 1). We implement piecewise approximation.
    # Default initialization: z = 0
    z = 0.0
    # Lower region: p < p_low
    # q = sqrt(-2*log(p)), approx via log(p)
    # Note: Triton allows tl.log and tl.sqrt.
    if sparsity < p_low:
        # For p very close to 0, avoid log(0). Here sparsity is > 0.
        q = tl.sqrt(-2.0 * tl.log(sparsity))
        poly = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
        poly2 = (((((d1*q + d2)*q + d3)*q + d4)*q + 1.0))
        z = poly / poly2
    else:
        # Central region: p_low <= p <= p_high
        # Use p directly (since p_high = 1 - p_low, and p >= p_low).
        q = sparsity - 0.5
        r = q * q
        poly = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)
        poly2 = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
        z = poly * q / poly2
    # Upper region: p > p_high
    if sparsity > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - sparsity))
        poly = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
        poly2 = (((((d1*q + d2)*q + d3)*q + d4)*q + 1.0))
        z = -poly / poly2
    # Store result to z_ptr
    tl.store(z_ptr, z)


@triton.jit
def apply_gating_kernel(
    x_ptr,                # *float32, shape [B*S*K] contiguous
    mean_ptr,             # *float32, shape [B*S]
    std_ptr,              # *float32, shape [B*S]
    z_ptr,                # *float32, length 1 (scalar z)
    out_ptr,              # *float32, shape [B*S*K] contiguous
    total,                # int: B*S*K
    K: tl.constexpr,      # last dim size
    BLOCK_SIZE: tl.constexpr
):
    # 1D grid over total elements
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    # Load input chunk
    x_chunk = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # Compute row index for each offset
    row_idx = offs // K  # vector of rows
    # Load per-row mean and std
    mean_row = tl.load(mean_ptr + row_idx, mask=mask, other=0.0)
    std_row = tl.load(std_ptr + row_idx, mask=mask, other=0.0)
    # Load scalar z
    z = tl.load(z_ptr)  # scalar
    # Compute threshold and apply gating
    thr = mean_row + std_row * z
    gated = x_chunk - thr
    out_chunk = tl.maximum(gated, 0.0)  # ReLU
    tl.store(out_ptr + offs, out_chunk, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 1024, num_warps: int = 4, num_stages: int = 2):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity requested, return as is (original returns same dtype)
        if target_sparsity == 0.0:
            return x

        # Ensure contiguous and compute in fp32
        B, S, K = x.shape
        x_f32 = x.contiguous().to(torch.float32)

        # Prepare output buffers
        out_f32 = torch.empty_like(x_f32)

        # Buffers for per-row stats [B*S] (viewed as [rows])
        mean = torch.empty(B * S, dtype=torch.float32, device=x.device)
        std = torch.empty(B * S, dtype=torch.float32, device=x.device)

        # Launch reduction kernel: one program per row
        grid_rows = (B * S,)
        compute_row_stats_kernel[grid_rows](
            x_f32.view(-1),          # flatten [B*S*K]
            mean, std,
            B, S, K,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages
        )

        # Allocate a 1-element tensor for z and compute it via Triton
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        # Constants for A&S 5.2.23 approximation
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425; p_high = 1.0 - p_low

        compute_ndtri_kernel[(1,)](
            z_buf,
            target_sparsity,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, p_high,
            num_warps=1,
            num_stages=1
        )

        # Launch gating kernel over all elements
        total = B * S * K
        grid_gate = (triton.cdiv(total, self.block_size),)
        apply_gating_kernel[grid_gate](
            x_f32.view(-1), mean, std, z_buf, out_f32.view(-1),
            total, K,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)