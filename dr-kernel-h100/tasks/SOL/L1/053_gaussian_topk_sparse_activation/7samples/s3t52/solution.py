import torch
import triton
import triton.language as tl


@triton.jit
def row_stats_kernel(X_ptr, mean_out_ptr, sumsq_out_ptr,
                      ROWS: tl.int32, F: tl.int32,
                      BLOCK_SIZE: tl.constexpr):
    """
    For each row (0..ROWS-1), compute sum and sum of squares across F features.
    Writes mean[row] = sum/F and sumsq[row] to mean_out_ptr and sumsq_out_ptr.
    """
    row = tl.program_id(0)
    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0

    for col in range(0, F, BLOCK_SIZE):
        idx = col + tl.arange(0, BLOCK_SIZE)
        mask = idx < F
        base = row * F
        offs = base + idx
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        # x is fp16; cast to fp32 for accumulation
        x32 = x.to(tl.float32)
        sum_val += tl.sum(x32, axis=0)
        sum_sq += tl.sum(x32 * x32, axis=0)

    mean = sum_val / F
    sumsq = sum_sq  # per-row sum of squares
    # Store results
    tl.store(mean_out_ptr + row, mean)
    tl.store(sumsq_out_ptr + row, sumsq)


@triton.jit
def ndtri_scalar_kernel(p_in_ptr, p_out_ptr, eps: tl.float32):
    """
    Compute inverse standard normal CDF (ndtri) for scalar p_in[0] using A&S 7.1.26 approximation.
    Writes result to p_out_ptr[0].
    """
    p = tl.load(p_in_ptr)
    # Clamp to avoid log(0) or log(1)
    p = tl.maximum(tl.minimum(p, 1.0 - eps), eps)
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

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    y_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    q2 = p - 0.5
    r = q2 * q2
    y_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q2 / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    q3 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    y_high = -(((((c1 * q3 + c2) * q3 + c3) * q3 + c4) * q3 + c5) * q3 + c6) / \
              ((((d1 * q3 + d2) * q3 + d3) * q3 + d4) * q3 + 1.0)

    # Select result based on p
    mask_low = p < p_low
    mask_mid = (p >= p_low) & (p <= p_high)
    mask_high = p > p_high
    y = tl.where(mask_low, y_low, 0.0)
    y = tl.where(mask_mid, y_mid, y)
    y = tl.where(mask_high, y_high, y)
    tl.store(p_out_ptr, y)


@triton.jit
def relu_threshold_kernel(X_ptr, mean_ptr, std_ptr, mult_ptr, Y_ptr,
                           ROWS: tl.int32, F: tl.int32,
                           BLOCK_SIZE: tl.constexpr):
    """
    Apply activation: y = max(0, x - (mean + std * mult))
    Inputs:
      X_ptr: pointer to input [ROWS*F] flattened
      mean_ptr: pointer to mean [ROWS]
      std_ptr: pointer to std [ROWS]
      mult_ptr: pointer to scalar multiplier [1]
      Y_ptr: pointer to output [ROWS*F]
    """
    row = tl.program_id(0)
    cutoff = tl.load(mean_ptr + row) + tl.load(std_ptr + row) * tl.load(mult_ptr)
    for col in range(0, F, BLOCK_SIZE):
        idx = col + tl.arange(0, BLOCK_SIZE)
        mask = idx < F
        base = row * F
        offs = base + idx
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        # x is fp32 (from stats), keep in fp32
        y = x - cutoff
        y = tl.maximum(y, 0.0)
        tl.store(Y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 2048, num_warps: int = 8, num_stages: int = 2, eps: float = 1e-7):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.eps = eps

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Convert input to float32 for stable accumulation
        x32 = x.to(torch.float32).contiguous()
        B, S, F = x32.shape
        rows = B * S

        # 1) Compute per-row mean and sum of squares using Triton
        mean = torch.empty(rows, dtype=torch.float32, device=x32.device)
        sumsq = torch.empty(rows, dtype=torch.float32, device=x32.device)

        row_stats_kernel[(rows,)](
            x32,
            mean, sumsq,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute variance and std in Triton (elementwise per row)
        var = sumsq / F - mean * mean
        # Clamp to non-negative to avoid tiny negative due to numerical error
        var = tl.maximum(var, 0.0)  # Triton requires tensors; here we rely on PyTorch var, then Triton for activation
        # Instead, compute std in Triton using another kernel (elementwise sqrt):
        std = torch.empty_like(mean)
        # Launch elementwise sqrt kernel:
        @triton.jit
        def sqrt_row_kernel(in_ptr, out_ptr, N: tl.int32):
            row = tl.program_id(0)
            v = tl.load(in_ptr + row)
            v = tl.maximum(v, 0.0)
            s = tl.sqrt(v)
            tl.store(out_ptr + row, s)

        sqrt_row_kernel[(rows,)](
            var, std, N=rows,
            num_warps=2, num_stages=1
        )

        # 3) Compute inverse standard normal CDF for scalar target_sparsity in Triton
        p_in = x32.new_tensor(target_sparsity)  # 1-element device tensor (float32)
        p_out = torch.empty(1, dtype=torch.float32, device=x32.device)
        ndtri_scalar_kernel[(1,)](
            p_in, p_out, self.eps,
            num_warps=1,
            num_stages=1,
        )
        std_multiplier = p_out  # 1-element tensor scalar

        # 4) Apply activation in Triton: y = max(0, x - (mean + std * multiplier))
        y = torch.empty(rows * F, dtype=torch.float32, device=x32.device)
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
