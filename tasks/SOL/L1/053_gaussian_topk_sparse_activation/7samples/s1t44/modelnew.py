import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and population std over last dim N.
# X: [rows, N] where rows = batch_size * seq_len, N = intermediate_size.
# MEAN: fp32 [rows], STD: fp32 [rows]
@triton.jit
def row_stats_kernel(
    X_ptr,           # *f32, contiguous [rows, N]
    MEAN_ptr,        # *f32, [rows]
    STD_ptr,         # *f32, [rows]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over columns in chunks of BLOCK
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        # Accumulate in fp32
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    n = N  # population std
    mean = sum_val / n
    # population variance: E[x^2] - (E[x])^2
    var = sum_sq / n - mean * mean
    # Clamp variance to [0, eps] to avoid tiny negative due to round-off
    eps = 1e-20
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    # Store
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: scalar ndtri using Abramowitz & Stegun 5.2.23 approximation
@triton.jit
def ndtri_scalar_kernel(P_ptr, Q_ptr):
    p = tl.load(P_ptr)  # scalar
    # Constants for A&S approximation
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

    # Lower region
    if p < p_low:
        # Use q = sqrt(-2*log(p)) for small p
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        poly2 = (((((d1 * q + d2) * q + d3) * q + d4) * q) + 1.0)
        q2 = poly / poly2
        q_out = q2
    else:
        # Central region
        z = p - 0.5
        r = z * z
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        poly2 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        q_out = poly * z / poly2
    # Upper region: if p > p_high, we can reuse symmetry with z = 1 - p
    # but here we only need one branch; q_out is valid for p in [p_low, p_high].
    tl.store(Q_ptr, q_out)


# Kernel 3: compute threshold per row in fp32: mean[row] + std[row] * multiplier
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,  # *f32 [rows]
    STD_ptr,   # *f32 [rows]
    q,         # scalar f32
    OUT_ptr,   # *f32 [rows]
    rows: tl.constexpr,
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    threshold = mean + std * q
    tl.store(OUT_ptr + row_id, threshold)


# Kernel 4: elementwise ReLU(x - threshold[row]) over [rows, N]
# X is laid out as contiguous [rows, N]; linearized as [rows*N]
@triton.jit
def relu_threshold_kernel(
    X_ptr,          # *f32, contiguous [rows*N]
    threshold_ptr,  # *f32, [rows]
    OUT_ptr,        # *f32, [rows*N]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    base = row_id * N
    thresh = tl.load(threshold_ptr + row_id)
    # Process columns in chunks
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = base + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = x - thresh
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Handle empty batch quickly
        if target_sparsity == 0.0:
            return x

        # Flatten shape [B, S, N] to process as [rows, N]
        assert x.dim() == 3, "Input must be 3D: [batch, seq_len, intermediate_size]"
        B, S, N = x.shape
        rows = B * S

        # Compute in fp32
        x_f32 = x.to(torch.float32).contiguous()

        # 1) Compute mean and std per row
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)
        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x_f32,
            mean,
            std,
            rows=rows,
            N=N,
            BLOCK=1024,
            num_warps=16,  # more warps to improve throughput for large N
        )

        # 2) Compute multiplier = ndtri(target_sparsity) in Triton (scalar)
        p = torch.full((1,), float(target_sparsity), device=x.device, dtype=torch.float32)
        q = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_scalar_kernel[(1,)](p, q)

        # 3) Compute threshold per row (fp32)
        threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

        # 4) Apply ReLU(x - threshold[row]) elementwise, write to fp32 OUT
        X_lin = x_f32.view(rows * N)
        OUT = torch.empty(rows * N, device=x.device, dtype=torch.float32)
        relu_threshold_kernel[grid_stats](
            X_lin,
            threshold,
            OUT,
            rows=rows,
            N=N,
            BLOCK=2048,
            num_warps=8,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)