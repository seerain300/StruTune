import torch
import triton
import triton.language as tl


# Triton kernel: compute per-row mean and population std (unbiased=False)
# x_flat: [rows, N], out_mean: [rows], out_std: [rows]
@triton.jit
def reduce_mean_std(x_ptr, out_mean_ptr, out_std_ptr, N: tl.int32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Compute base pointer for this row
    row_start = pid * N

    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over N in chunks of BLOCK_SIZE
    for offs in range(0, N, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        # Accumulate in float32
        x32 = x.to(tl.float32)
        sum_val += tl.sum(x32, axis=0)
        sum_sq += tl.sum(x32 * x32, axis=0)

    n_float = tl.float32(N)
    mean = sum_val / n_float
    var = sum_sq / n_float - mean * mean
    std = tl.sqrt(var)
    # Store as 1-element tensors per row
    tl.store(out_mean_ptr + pid, mean)
    tl.store(out_std_ptr + pid, std)


# Triton kernel: compute inv_norm_cdf(target_sparsity) via A&S approximation
# p: 1-element tensor with target_sparsity, out: 1-element tensor with inv_norm_cdf
@triton.jit
def compute_inv_ndtri(p_ptr, out_ptr, p_low: tl.float32, BLOCK_SIZE: tl.constexpr):
    p = tl.load(p_ptr).to(tl.float32)
    # Constants
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

    p_low = p_low
    p_high = 1.0 - p_low

    # Masks
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high

    inv = 0.0

    if mask_low.any():
        q = tl.sqrt(-2.0 * tl.log(p))
        inv_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                  ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        inv = tl.where(mask_low, inv_low, inv)
    if mask_mid.any():
        q = p - 0.5
        r = q * q
        inv_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                  (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        inv = tl.where(mask_mid, inv_mid, inv)
    if mask_high.any():
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        inv_high = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        inv = tl.where(mask_high, inv_high, inv)

    tl.store(out_ptr, inv)


# Triton kernel: per-row gating
# x_flat: [rows, N], mean: [rows], std: [rows], inv_scale: scalar, out_flat: [rows, N]
@triton.jit
def gate_rows(x_ptr, mean_ptr, std_ptr, inv_scale: tl.float32, out_ptr, N: tl.int32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid * N

    # Load per-row mean and std
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    threshold = mean + std * inv_scale

    for offs in range(0, N, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        y = tl.maximum(x32 - threshold, 0.0)
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure inputs are float32 for computation
        x = inputs.contiguous().to(torch.float32)

        # Shapes
        B, S, N = x.shape
        rows = B * S

        # Flatten to [rows, N]
        x_flat = x.view(rows, N)
        out_flat = torch.empty_like(x_flat, dtype=torch.float32)

        # 1) Reduce to per-row mean and std (float32)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)
        BLOCK_SIZE = 1024
        grid = (rows,)
        reduce_mean_std[grid](x_flat, mean, std, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # 2) Compute inv_norm_cdf(target_sparsity) in Triton (scalar)
        p_tensor = torch.tensor(target_sparsity, device=x.device, dtype=torch.float32)
        inv_cdf_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        p_low = 0.02425
        compute_inv_ndtri[(1,)](p_tensor, inv_cdf_buf, p_low, BLOCK_SIZE=1, num_warps=1)
        inv_cdf_scalar = float(inv_cdf_buf.item())
        inv_scale = inv_cdf_scalar * target_sparsity  # threshold = mean + std * inv_cdf

        # 3) Gate rows: y = max(0, x - (mean + std * inv_scale))
        gate_rows[grid](x_flat, mean, std, inv_scale, out_flat, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)

        # Reshape and cast to bfloat16 to match original behavior
        out = out_flat.view(B, S, N).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
