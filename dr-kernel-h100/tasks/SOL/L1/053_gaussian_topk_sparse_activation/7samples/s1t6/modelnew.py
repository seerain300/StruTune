import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and population std over last dim N.
# X: [rows, N] where rows = batch_size * seq_len, N = intermediate_size.
# MEAN: fp32 [rows], STD: fp32 [rows]
@triton.jit
def row_stats_kernel(
    X_ptr, MEAN_ptr, STD_ptr,
    rows, N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    # Accumulate sum and sum of squares in fp32
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over columns in chunks of BLOCK
    for offs in range(0, N, BLOCK):
        cols = offs + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0)
        # x is fp32; ensure computation is fp32
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    n_float = tl.full((), N, tl.float32)
    mean = sum_val / n_float
    var = sum_sq / n_float - mean * mean  # population variance
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row, mean)
    tl.store(STD_ptr + row, std)


# Kernel 2: scalar inverse-normal CDF using Abramowitz & Stegun 5.2.23 approximation.
# Input p: device scalar tensor (1-element) with probability in (0,1), output q: 1-element tensor.
@triton.jit
def ndtri_kernel(p_ptr, q_ptr):
    p = tl.load(p_ptr)  # scalar float32
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
        q = tl.sqrt(-2.0 * tl.log(p))
        r = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    # Central region
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        r = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    # Upper region
    else:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        r = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        q = -q
    tl.store(q_ptr, q)


# Kernel 3: compute threshold per row: mean + std * multiplier
@triton.jit
def threshold_vec_kernel(mean_ptr, std_ptr, multiplier, threshold_ptr, rows):
    row = tl.program_id(0)
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    q = tl.load(multiplier)  # scalar
    thr = mean + std * q
    tl.store(threshold_ptr + row, thr)


# Kernel 4: elementwise ReLU(x - threshold[row]), write directly to [B, S, N] output
@triton.jit
def relu_threshold_kernel(
    X_ptr,         # *f32, linearized [rows*N]
    threshold_ptr, # *f32, [rows]
    OUT_ptr,       # *f32, linearized [rows*N]
    rows, N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    # Each program handles one row
    for offs in range(0, N, BLOCK):
        cols = offs + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0)
        thr = tl.load(threshold_ptr + row)  # scalar
        y = tl.maximum(x - thr, 0.0)
        tl.store(OUT_ptr + row * N + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target_sparsity: float):
        # Triton-only forward: no torch ops
        # Ensure CUDA and contiguous
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous()

        # Flatten leading dims into rows, last dim N
        B, S, N = x.shape
        rows = B * S

        # Create device fp32 buffers for compute
        X_lin = x.view(rows * N).to(torch.float32)  # linearized input in fp32
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)

        # Launch reduction kernel to compute mean and std per row
        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            X_lin, mean, std,
            rows, N,
            BLOCK=512,  # tuned chunk size for larger N
            num_warps=8,
        )

        # Compute multiplier = ndtri(target_sparsity) in Triton (scalar)
        p = torch.empty(1, device=x.device, dtype=torch.float32)
        p.fill_(float(target_sparsity))
        q = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_kernel[(1,)](p, q)  # single program for scalar compute

        # Compute threshold per row in Triton
        threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

        # Prepare output buffer (fp32) and apply ReLU per row
        OUT = torch.empty(rows * N, device=x.device, dtype=torch.float32)
        relu_threshold_kernel[grid_stats](
            X_lin, threshold, OUT,
            rows, N,
            BLOCK=1024,  # larger block for better bandwidth
            num_warps=4,
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)