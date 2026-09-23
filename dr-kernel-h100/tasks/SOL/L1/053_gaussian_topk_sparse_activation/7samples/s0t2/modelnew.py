import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr,
                              B, S, K,
                              stride_x0, stride_x1, stride_x2,
                              stride_mean0, stride_mean1, stride_mean2,
                              stride_std0, stride_std1, stride_std2,
                              BLOCK_SIZE: tl.constexpr):
    # One program per row: pid selects the [b, s] pair
    # Note: We assume x is 3D [B, S, K] and mean/std are [B, S, 1]
    # Triton program id 0 covers all rows: total rows = B * S
    row = tl.program_id(0)
    b = row // S
    s = row % S

    # Accumulators
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Loop over columns in chunks of BLOCK_SIZE
    for col_start in range(0, K, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < K

        # Compute the linear index for x[b, s, cols]
        x_offs = b * stride_x0 + s * stride_x1 + cols * stride_x2
        # Load values (masked), cast to float32
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)

        # Reduce over this chunk
        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    # Compute mean and std for this row
    K_f = tl.float32(K)  # ensure scalar float32
    mean = sum_val / K_f
    # population std: std = sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / K_f - mean * mean
    # Clamp variance to avoid negative due to rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Store mean and std to [B, S, 1] tensors
    out_off_mean = b * stride_mean0 + s * stride_mean1 + 0 * stride_mean2
    out_off_std = b * stride_std0 + s * stride_std1 + 0 * stride_std2
    tl.store(mean_ptr + out_off_mean, mean)
    tl.store(std_ptr + out_off_std, std)


@triton.jit
def ndtri_kernel(p, out):
    # Compute inverse of standard normal CDF using A&S 26.2.23
    # p: scalar float32
    p_low = 0.02425
    p_high = 1.0 - p_low

    # lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    c1 = -7.784695709041462e-03
    c2 = 3.224671290700398e-01
    c3 = 2.445134137142996e+00
    c4 = 3.754408661907416e+00
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    poly = (((((c1 * q + c2) * q + c3) * q + d4) * q + d3) * q + d2) * q + d1
    result = poly / q

    # central region
    q_mid = p - 0.5
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

    poly_mid = (((((a1 * q_mid * q_mid + a2) * q_mid + a3) * q_mid + a4) * q_mid + a5) * q_mid + a6) * q_mid
    poly_denom = (((((b1 * q_mid * q_mid + b2) * q_mid + b3) * q_mid + b4) * q_mid + b5) * q_mid + 1.0)
    result_mid = poly_mid / poly_denom

    # upper region
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    c1u = -7.784894002430293e-03
    c2u = -3.223964580411365e-01
    c3u = -2.400758277161838e+00
    c4u = -2.549732539343734e+00
    c5u = 4.374664141464968e+00
    c6u = 2.938163982698783e+00
    d1u = 7.784695709041462e-03
    d2u = 3.224671290700398e-01
    d3u = 2.445134137142996e+00
    d4u = 3.754408661907416e+00

    poly_u = (((((c1u * q_up + c2u) * q_up + c3u) * q_up + c4u) * q_up + c5u) * q_up + c6u)
    result_u = -poly_u / ((((d1u * q_up + d2u) * q_up + d3u) * q_up + d4u) * q_up + 1.0)

    # Select result based on p region
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_up = p > p_high

    # out[0] is a scalar; Triton allows scalar loads/stores
    # We can perform elementwise selection by constructing a vector of p and choosing
    # But here p is scalar; Triton will broadcast result vector to scalar store via out[0]
    # The kernel writer chooses the first region branch; better construct a vector
    # We can use tl.where with scalar conditions to choose. Triton supports scalar branching.
    res = result
    if mask_low:
        res = result
    elif mask_mid:
        res = result_mid
    else:
        res = result_u

    tl.store(out, res)


@triton.jit
def apply_gating_kernel(x_flat_ptr, mean_ptr, std_ptr, out_flat_ptr,
                         M, K,
                         stride_mean0, stride_mean1, stride_mean2,
                         stride_std0, stride_std1, stride_std2,
                         z,
                         BLOCK_SIZE: tl.constexpr):
    # Linearized over total elements: M = B*S rows, each of length K
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = M * K

    # Mask for valid elements
    mask = offs < total

    # Compute per-element row index: row = offs // K
    row = offs // K

    # Compute offsets for mean/std loads: mean_ptr[row], std_ptr[row]
    mean_offs = row * stride_mean0 + 0 * stride_mean1 + 0 * stride_mean2
    std_offs = row * stride_std0 + 0 * stride_std1 + 0 * stride_std2

    # Load per-row mean and std
    mean = tl.load(mean_ptr + mean_offs, mask=mask, other=0.0).to(tl.float32)
    std = tl.load(std_ptr + std_offs, mask=mask, other=0.0).to(tl.float32)

    # Load input
    x_vals = tl.load(x_flat_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute threshold and gated output
    threshold = mean + std * z
    y = x_vals - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    # Store
    tl.store(out_flat_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.1,
                 block_size: int = 4096, num_warps: int = 4, num_stages: int = 2):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If no sparsity, return directly (matches original behavior)
        if self.target_sparsity == 0.0:
            return x

        # Ensure 3D input: [B, S, K]
        assert x.dim() == 3, "Input must be 3D [batch_size, seq_len, intermediate_size]"
        B, S, K = x.shape

        # Work in float32 inside Triton kernels for numerical stability
        x32 = x.to(torch.float32).contiguous()
        device = x32.device

        # 1) Compute per-row mean and std via Triton reduction
        mean = torch.empty((B, S, 1), dtype=torch.float32, device=device)
        std = torch.empty((B, S, 1), dtype=torch.float32, device=device)

        grid_stats = (B * S,)
        compute_row_stats_kernel[grid_stats](
            x32, mean, std,
            B, S, K,
            x32.stride(0), x32.stride(1), x32.stride(2),
            mean.stride(0), mean.stride(1), mean.stride(2),
            std.stride(0), std.stride(1), std.stride(2),
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages
        )

        # 2) Compute z = _ndtri(target_sparsity) using Triton
        z = torch.empty(1, dtype=torch.float32, device=device)
        ndtri_kernel[(1,)](self.target_sparsity, z)

        # 3) Apply gating via Triton: out = max(input - (mean + std * z), 0)
        x_flat = x32.view(-1)
        out_flat = torch.empty_like(x_flat, dtype=torch.float32, device=device)

        total = B * S * K
        grid_gate = (triton.cdiv(total, self.block_size),)
        apply_gating_kernel[grid_gate](
            x_flat, mean, std, out_flat,
            B * S, K,
            mean.stride(0), mean.stride(1), mean.stride(2),
            std.stride(0), std.stride(1), std.stride(2),
            z.item(),  # pass scalar z
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages
        )

        # Reshape back and cast to original dtype (original returns bfloat16)
        out = out_flat.view(B, S, K)
        return out.to(torch.bfloat16)