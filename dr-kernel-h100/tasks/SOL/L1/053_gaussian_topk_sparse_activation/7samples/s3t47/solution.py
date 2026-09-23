import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_reduce_kernel(X_ptr, sum_out_ptr, sumsq_out_ptr,
                             ROWS: tl.int32, F: tl.int32,
                             BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes sum[row] and sumsq[row] to sum_out_ptr and sumsq_out_ptr (float32).
    """
    pid = tl.program_id(axis=0)  # row id
    sum_val = 0.0
    sum_sq = 0.0
    # Loop over feature dimension in tiles
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        offs = pid * F + cols
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    tl.store(sum_out_ptr + pid, sum_val)
    tl.store(sumsq_out_ptr + pid, sum_sq)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (A&S 7.1.26) for the scalar p stored at p_in_ptr.
    Write result to p_out_ptr (float32).
    Clamp p to [eps, 1-eps] to avoid log(0)/log(1) edge cases.
    """
    p = tl.load(p_in_ptr)
    p = tl.maximum(p, eps)
    p = tl.minimum(p, 1.0 - eps)

    # A&S constants
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
    q = tl.sqrt(-2.0 * tl.log(p))
    poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
    denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
    z_low = poly / denom

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid
    denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    z_mid = poly_mid / denom_mid

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    denom_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    z_high = -poly_high / denom_high

    z = tl.where(p < p_low, z_low, tl.where(p > p_high, z_high, z_mid))
    tl.store(p_out_ptr, z)


@triton.jit
def row_compute_kernel(sum_ptr, sumsq_ptr, threshold_out_ptr,
                        ROWS: tl.int32, F: tl.int32,
                        multiplier: tl.float32):
    """
    For each row, compute mean, var, std, and threshold = mean + std * multiplier.
    Writes threshold[row * F + cols] to threshold_out_ptr (float32).
    """
    pid = tl.program_id(axis=0)  # row id
    sum_val = tl.load(sum_ptr + pid)
    sum_sq_val = tl.load(sumsq_ptr + pid)
    mean = sum_val / F
    var = sum_sq_val / F - mean * mean
    # std via sqrt in Triton (host must pass non-negative var)
    std = tl.sqrt(var)
    thr = mean + std * multiplier
    # Write threshold for all features
    for start in range(0, F, 1):
        offs = pid * F + start
        # Store scalar threshold at each feature position
        tl.store(threshold_out_ptr + offs, thr)


@triton.jit
def row_activation_kernel(X_ptr, Threshold_ptr, Output_ptr,
                           ROWS: tl.int32, F: tl.int32,
                           BLOCK_SIZE: tl.constexpr):
    """
    For each row, apply y = max(0, x - threshold) elementwise across features.
    X_ptr: input float32
    Threshold_ptr: per-element threshold for the row
    Output_ptr: output float32
    """
    pid = tl.program_id(axis=0)
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        offs = pid * F + cols
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        thr = tl.load(Threshold_ptr + offs, mask=mask, other=0.0)
        y = tl.maximum(x - thr, 0.0)
        tl.store(Output_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float,
                 block_size_reduce: int = 1024, num_warps_reduce: int = 8, num_stages_reduce: int = 2,
                 block_size_act: int = 1024, num_warps_act: int = 8, num_stages_act: int = 2):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        self.block_size_reduce = block_size_reduce
        self.num_warps_reduce = num_warps_reduce
        self.num_stages_reduce = num_stages_reduce
        self.block_size_act = block_size_act
        self.num_warps_act = num_warps_act
        self.num_stages_act = num_stages_act

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, F], float32 or other; we convert to float32 for compute
        B, S, F = x.shape
        rows = B * S
        x32 = x.to(torch.float32).contiguous()

        # 1) Row-wise reduction: sum and sum of squares
        sum_buf = torch.empty(rows, dtype=torch.float32, device=x32.device)
        sumsq_buf = torch.empty(rows, dtype=torch.float32, device=x32.device)

        row_stats_reduce_kernel[(rows,)](
            x32, sum_buf, sumsq_buf,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size_reduce,
            num_warps=self.num_warps_reduce,
            num_stages=self.num_stages_reduce,
        )

        # 2) Compute scalar inverse normal CDF of target_sparsity in Triton
        p_in = x32.new_tensor(self.target_sparsity)  # 1-element device scalar (float32)
        p_out = torch.empty(1, dtype=torch.float32, device=x32.device)
        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, 1e-7,  # eps for clamping
            num_warps=1,
            num_stages=1,
        )
        multiplier = p_out  # 1-element tensor (float32)

        # 3) Compute per-element threshold for each row
        threshold = torch.empty(rows * F, dtype=torch.float32, device=x32.device)

        row_compute_kernel[(rows,)](
            sum_buf, sumsq_buf, threshold,
            ROWS=rows, F=F,
            multiplier=multiplier,
            num_warps=1,
            num_stages=1,
        )

        # 4) Apply activation in Triton: y = max(0, x - threshold)
        output = torch.empty(rows * F, dtype=torch.float32, device=x32.device)
        row_activation_kernel[(rows,)](
            x32.view(rows, F),
            threshold,
            output,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size_act,
            num_warps=self.num_warps_act,
            num_stages=self.num_stages_act,
        )

        # 5) Reshape and cast to bfloat16 to match original output
        y_out = output.view(B, S, F).to(torch.bfloat16)
        return y_out


def run(*args):
    return ModelNew()(*args)
