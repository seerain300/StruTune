import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row sum and sum of squares over last dim N.
# X: *f32, contiguous [rows, N], ROWS: int, N: int, BLOCK: constexpr
@triton.jit
def row_reduce_kernel(X_ptr, SUM_ptr, SUMSQ_ptr, ROWS: tl.int32, N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Each program handles one row
    sum_val = 0.0
    sumsq_val = 0.0
    # Iterate over N in chunks of BLOCK
    for start in range(0, N, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N
        # Linearized indexing for row pid
        ptr = X_ptr + pid * N + offsets
        vals = tl.load(ptr, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)
    # Write partial results
    tl.store(SUM_ptr + pid, sum_val)
    tl.store(SUMSQ_ptr + pid, sumsq_val)


# Kernel 2: compute per-row mean and population std from sum and sumsq.
# SUM: [rows] f32, SUMSQ: [rows] f32, MEAN: [rows] f32, STD: [rows] f32, ROWS: int, N: int
@triton.jit
def compute_mean_std_kernel(SUM_ptr, SUMSQ_ptr, MEAN_ptr, STD_ptr, ROWS: tl.int32, N: tl.int32):
    pid = tl.program_id(axis=0)
    # Guard against overlaunch (grid=(ROWS,))
    if pid >= ROWS:
        return
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)
    # Population std: std = sqrt(sumsq/N - (sum/N)^2)
    mean = sum_val / N
    var = sumsq_val / N - mean * mean
    # Avoid negative due to rounding: clamp var to >= 0
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


# Kernel 3: scalar inverse-normal CDF using A&S 5.2.23 approximation.
# p: [1] f32, q: [1] f32
@triton.jit
def ndtri_kernel(p_ptr, q_ptr):
    # Load probability
    p = tl.load(p_ptr)
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

    # Lower tail
    mask_low = p < p_low
    if mask_low:
        q_low = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
        q_low = -poly / ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
        # Select result
        # Triton doesn't support dynamic branching on scalar tensors cleanly here,
        # so we compute both and select via mask logic: store q_low when mask_low, else upper
        # For non-low/high, compute upper path result; we'll overwrite accordingly.
        q[0] = q_low  # default; will be overwritten for mid/high

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    if mask_mid:
        q_mid = p - 0.5
        r = q_mid * q_mid
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        poly2 = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        q_mid = poly / poly2
        q[0] = q_mid

    # Upper tail
    mask_high = p > p_high
    if mask_high:
        q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
        q_high = -poly / ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
        q[0] = q_high

    # Store result
    tl.store(q_ptr, q)


# Kernel 4: compute threshold per row in Triton
# mean: [rows] f32, std: [rows] f32, q: [1] f32, threshold: [rows] f32
@triton.jit
def threshold_vec_kernel(mean_ptr, std_ptr, q_ptr, threshold_ptr, ROWS: tl.int32):
    pid = tl.program_id(axis=0)
    if pid >= ROWS:
        return
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    mult = tl.load(q_ptr)  # scalar q
    thr = mean + std * mult
    tl.store(threshold_ptr + pid, thr)


# Kernel 5: elementwise ReLU(x - threshold[row]) for each row
# X: *f32 contiguous [rows, N], threshold: [rows] f32, OUT: *f32 [rows, N]
@triton.jit
def relu_threshold_kernel(X_ptr, threshold_ptr, OUT_ptr, ROWS: tl.int32, N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    if pid >= ROWS:
        return
    thr = tl.load(threshold_ptr + pid)
    for start in range(0, N, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < N
        in_ptr = X_ptr + pid * N + offsets
        vals = tl.load(in_ptr, mask=mask, other=0.0)
        # ReLU(x - thr)
        out_vals = tl.maximum(vals - thr, 0.0)
        out_ptr = OUT_ptr + pid * N + offsets
        tl.store(out_ptr, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target_sparsity: float):
        # Ensure x is contiguous and on CUDA; compute in fp32 for stability
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        B, S, N = x.shape
        rows = B * S
        x_f32 = x.to(torch.float32).contiguous()
        # 1) Per-row reduction: sum and sum of squares
        sum_buf = torch.empty(rows, device=x.device, dtype=torch.float32)
        sumsq_buf = torch.empty(rows, device=x.device, dtype=torch.float32)
        grid_stats = (rows,)
        row_reduce_kernel[grid_stats](
            x_f32, sum_buf, sumsq_buf, rows, N, BLOCK=1024, num_warps=4
        )

        # 2) Compute per-row mean and population std in Triton-friendly way (torch ops on device)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        std = torch.empty(rows, device=x.device, dtype=torch.float32)
        compute_mean_std_kernel[grid_stats](sum_buf, sumsq_buf, mean, std, rows, N)

        # 3) Compute inverse-normal multiplier (scalar) in Triton
        p = torch.full((1,), float(target_sparsity), device=x.device, dtype=torch.float32)
        q = torch.empty(1, device=x.device, dtype=torch.float32)
        ndtri_kernel[(1,)](p, q)

        # 4) Compute per-row threshold (fp32)
        threshold = torch.empty(rows, device=x.device, dtype=torch.float32)
        threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

        # 5) Apply ReLU(x - threshold[row]) elementwise in Triton, write to fp32 OUT
        OUT = torch.empty(rows * N, device=x.device, dtype=torch.float32)
        relu_threshold_kernel[grid_stats](
            x_f32, threshold, OUT, rows, N, BLOCK=1024, num_warps=4
        )

        # Return in bf16 to match original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
