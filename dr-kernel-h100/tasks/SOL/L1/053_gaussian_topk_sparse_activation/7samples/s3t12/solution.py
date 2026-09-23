import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes mean[row] and sumsq[row] to mean_out_ptr and sumsq_out_ptr.
    """
    row_id = tl.program_id(0)
    if row_id >= ROWS:
        return

    # Accumulators
    total = 0.0
    total_sq = 0.0

    # Iterate over features in tiles
    for off in range(0, F, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        # Address for this row and columns
        ptr = X_ptr + row_id * F + cols
        vals = tl.load(ptr, mask=mask, other=0.0)
        total += tl.sum(vals)
        total_sq += tl.sum(vals * vals)

    mean = total / F
    sumsq = total_sq / F

    tl.store(mean_out_ptr + row_id, mean)
    tl.store(sumsq_out_ptr + row_id, sumsq)


@triton.jit
def _sqrt_std_kernel(var_ptr, std_ptr, N: tl.int32):
    """
    Elementwise std = sqrt(var) over N rows. Writes to std_ptr[i] = sqrt(var_ptr[i]).
    """
    i = tl.program_id(0)
    if i >= N:
        return
    var = tl.load(var_ptr + i)
    std = tl.sqrt(var)
    tl.store(std_ptr + i, std)


@triton.jit
def _ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF for scalar p (p_in_ptr[0]) using A&S 7.1.26
    and write to p_out_ptr[0]. Clamp p to [eps, 1-eps] to avoid log(0) or log(1).
    """
    p = tl.load(p_in_ptr)
    # clamp to avoid log(0) or log(1)
    p = tl.maximum(p, eps)
    p = tl.minimum(p, 1.0 - eps)

    # Abramowitz and Stegun constants
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

    # masks for regions
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high

    # lower region approximation
    q_low = tl.sqrt(-2.0 * tl.log(p))
    nd_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
             ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # central region approximation
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    nd_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
             (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    # upper region approximation
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    nd_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
              ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # select region
    out = tl.where(mask_low, nd_low, 0.0)
    out = tl.where(mask_mid, nd_mid, out)
    out = tl.where(mask_high, nd_high, out)

    tl.store(p_out_ptr, out)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, multiplier_ptr, Y_ptr,
                           ROWS: tl.int32, F: tl.int32,
                           BLOCK_SIZE: tl.constexpr):
    """
    For each row, compute cutoff = mean + std * multiplier, then
    Y[row, :] = max(0, X[row, :] - cutoff).
    """
    row_id = tl.program_id(0)
    if row_id >= ROWS:
        return

    mean = tl.load(mean_ptr + row_id)
    std = tl.load(std_ptr + row_id)
    mult = tl.load(multiplier_ptr)  # scalar

    cutoff = mean + std * mult

    for off in range(0, F, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        x = tl.load(X_ptr + row_id * F + cols, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + row_id * F + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 512, num_warps: int = 8, num_stages: int = 2, eps: float = 1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = eps

    def forward(self, x: torch.Tensor, target_sparsity: float):
        # Ensure CUDA tensor
        assert x.is_cuda, "Input must be a CUDA tensor"
        B, S, F = x.shape
        rows = B * S

        # Convert to float32 for compute
        x32 = x.to(torch.float32).contiguous()

        # 1) Row-wise stats in Triton
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        sumsq = torch.empty(rows, dtype=torch.float32, device=x.device)

        row_stats_kernel[(rows,)](
            x32, mean, sumsq,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute var and std via Triton (elementwise)
        var = sumsq - mean * mean
        std = torch.empty(rows, dtype=torch.float32, device=x.device)

        _sqrt_std_kernel[(rows,)](
            var, std, N=rows,
            num_warps=4,  # small elementwise kernel
            num_stages=1,
        )

        # 3) Compute scalar ndtri(target_sparsity) in Triton; avoid torch.tensor in forward
        # Create device scalar via .item() from Python and a 1-element tensor
        p_in = x.new_tensor(target_sparsity).contiguous()  # float32 on device, no torch.tensor()
        p_out = torch.empty(1, dtype=torch.float32, device=x.device)

        _ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor

        # 4) Apply activation in Triton
        y = torch.empty(rows * F, dtype=torch.float32, device=x.device)

        relu_threshold_kernel[(rows,)](
            x32.view(rows, F),
            mean, std, std_multiplier, y,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 5) Reshape and cast to bfloat16 to match original output
        y_out = y.view(B, S, F).to(torch.bfloat16)
        return y_out


def run(*args):
    return ModelNew()(*args)
