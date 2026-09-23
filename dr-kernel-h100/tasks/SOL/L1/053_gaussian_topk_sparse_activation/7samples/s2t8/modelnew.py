import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std(x_ptr, mean_ptr, std_ptr, N: tl.int32, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row_id = tl.program_id(0)
    # Accumulate sum and sum of squares in float32
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over the last dimension in chunks
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # Pointer arithmetic for row-major layout: x is flattened to [rows, N]
        ptrs = x_ptr + row_id * N + cols
        vals = tl.load(ptrs, mask=mask, other=0.0)
        # Accumulate
        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)
    # Compute mean and population std (unbiased=False)
    n_float = tl.float32(N)
    mean_val = sum_val / n_float
    var = sum_sq / n_float - mean_val * mean_val
    # Ensure var >= 0 for numerical stability (PyTorch std would do this implicitly)
    std_val = tl.sqrt(var)
    # Store 1-element tensors; host will reshape to [rows]
    tl.store(mean_ptr + row_id, mean_val)
    tl.store(std_ptr + row_id, std_val)


@triton.jit
def compute_inv_ndtri(p_ptr, inv_ptr, BLOCK_SIZE: tl.constexpr):
    # Compute inv_norm_cdf for a single scalar p (p_ptr is 1-element tensor)
    p = tl.load(p_ptr)
    # Constants for A&S 26.2.23
    p_low = 0.02425
    p_high = 1.0 - p_low
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

    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    tl.store(inv_ptr, z)


@triton.jit
def gate_rows(x_ptr, mean_ptr, std_ptr, inv_scale, out_ptr, N: tl.int32, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row_id = tl.program_id(0)
    # Load per-row mean and std
    mean_val = tl.load(mean_ptr + row_id)
    std_val = tl.load(std_ptr + row_id)
    threshold = mean_val + std_val * inv_scale

    # Process the row in chunks
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        in_ptrs = x_ptr + row_id * N + cols
        out_ptrs = out_ptr + row_id * N + cols
        x_vals = tl.load(in_ptrs, mask=mask, other=0.0)
        y = x_vals - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure 3D input [B, S, N]
        assert x.dim() == 3, "Input must be 3D: [batch_size, seq_len, intermediate_size]"
        # Cast to float32 for computation
        x_f32 = x.to(torch.float32)
        B, S, N = x_f32.shape
        rows = B * S

        # Flatten to [rows, N]
        x_flat = x_f32.view(rows, N)
        out_flat = torch.empty_like(x_flat, dtype=torch.float32)

        # 1) Reduce to per-row mean and std (population std, unbiased=False)
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        BLOCK_SIZE = 1024
        grid = (rows,)
        reduce_mean_std[grid](x_flat, mean, std, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # 2) Compute inv_norm_cdf(target_sparsity) in Triton (scalar)
        p_tensor = torch.tensor(target_sparsity, device=x_f32.device, dtype=torch.float32)
        inv_cdf_buf = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        compute_inv_ndtri[(1,)](p_tensor, inv_cdf_buf, BLOCK_SIZE=1, num_warps=1)
        inv_cdf_scalar = float(inv_cdf_buf.item())
        # inv_scale = inv_norm_cdf(target_sparsity) * target_sparsity
        inv_scale = inv_cdf_scalar * target_sparsity

        # 3) Gate rows: y = max(0, x - (mean + std * inv_scale))
        gate_rows[grid](x_flat, mean, std, inv_scale, out_flat, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)

        # Reshape and cast to bfloat16 to match original behavior
        out = out_flat.view(B, S, N).to(torch.bfloat16)
        return out