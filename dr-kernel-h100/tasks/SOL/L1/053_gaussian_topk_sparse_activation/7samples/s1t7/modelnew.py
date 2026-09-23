import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and population std over last dim N.
# X: [rows, N] linearized as [rows*N]. Outputs MEAN[row], STD[row] (fp32).
@triton.jit
def row_stats_kernel(
    X_ptr,           # *f16/f32/bf16 (we load and promote to f32 inside)
    MEAN_ptr,        # *f32, shape [rows]
    STD_ptr,         # *f32, shape [rows]
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0
    # Loop over N in chunks
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        # Pointer to this row's data: linear index = row_id * N + offs
        x = tl.load(X_ptr + row_id * N + offs, mask=mask, other=0.0)
        # Promote to fp32 for numeric stability
        x = x.to(tl.float32)
        # Reduce
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance
    std = tl.sqrt(var)
    # Store results
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Kernel 2: scalar inverse-normal CDF (Abramowitz & Stegun 5.2.23).
# Input p_ptr: 1-element device tensor (probability), output q_ptr: 1-element device tensor (quantile).
@triton.jit
def ndtri_kernel(p_ptr, q_ptr):
    p = tl.load(p_ptr)  # scalar
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

    p_low = 0.02425
    p_high = 1.0 - p_low

    # Piecewise computation
    # Lower tail
    q_low = tl.sqrt(-2.0 * tl.log(p))
    y_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    num_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    den_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    y_mid = num_mid / den_mid

    # Upper tail
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    y_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
           ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

    # Select piece
    is_low = p < p_low
    is_mid = (p >= p_low) & (p <= p_high)
    is_up = p > p_high
    y = tl.where(is_low, y_low, 0.0)
    y = tl.where(is_mid, y_mid, y)
    y = tl.where(is_up, y_up, y)

    tl.store(q_ptr, y)


# Kernel 3: compute per-row threshold = mean[row] + std[row] * multiplier.
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,     # *f32, [rows]
    STD_ptr,      # *f32, [rows]
    MULTIPLIER_ptr,  # *f32, [1]
    THRESH_ptr,   # *f32, [rows]
    rows: tl.constexpr,
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    m = tl.load(MULTIPLIER_ptr)  # scalar
    thr = mean + std * m
    tl.store(THRESH_ptr + row_id, thr)


# Kernel 4: elementwise ReLU(x - threshold[row]) for each row, write to OUT (fp32), linearized.
@triton.jit
def relu_threshold_kernel(
    X_ptr,         # *f16/f32/bf16 (we promote to f32 in-kernel)
    THRESH_ptr,    # *f32, [rows]
    OUT_ptr,       # *f32, [rows*N] linearized
    rows: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Load threshold for this row
    thr = tl.load(THRESH_ptr + row_id)
    # Process columns
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X_ptr + row_id * N + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - thr
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT_ptr + row_id * N + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # Ensure CUDA and contiguous
        assert inputs.is_cuda, "inputs must be on CUDA for Triton kernels"
        # Compute in fp32 for numeric stability; we don't use torch ops in forward
        # Flatten leading dims to rows, last dim N
        B, S, N = inputs.shape
        rows = B * S
        x = inputs.contiguous()
        # Linearize X for kernels
        X_lin = x.view(-1).to(torch.float32)

        # Allocate outputs for mean and std
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)

        # 1) Compute per-row mean and std
        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            X_lin, mean, std,
            rows=rows, N=N,
            BLOCK=512,
            num_warps=8,
        )

        # 2) Compute multiplier = ndtri(target_sparsity) on device (scalar)
        p = torch.empty(1, device=x.device, dtype=torch.float32)
        p.fill_(float(target_sparsity))
        q = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_kernel[(1,)](p, q)

        # 3) Compute threshold per row (fp32)
        threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

        # 4) Apply ReLU(x - threshold[row]) elementwise, writing fp32 to OUT
        OUT = torch.empty(rows * N, device=x.device, dtype=torch.float32)
        relu_threshold_kernel[grid_stats](
            X_lin, threshold, OUT,
            rows=rows, N=N,
            BLOCK=1024,
            num_warps=4,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)