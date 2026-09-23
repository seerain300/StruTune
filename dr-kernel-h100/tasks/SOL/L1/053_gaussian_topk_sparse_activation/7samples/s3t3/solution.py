import torch
import triton
import triton.language as tl


# Kernel 1: compute mean and variance (unbiased=False) per row
@triton.jit
def _row_stats_kernel(X_ptr, Mean_ptr, Var_ptr,
                      ROWS: tl.constexpr, F: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    # Accumulate sum and sum of squares over the last dimension F
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over tiles of size BLOCK_SIZE
    for off in range(0, F, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < F
        # Address: row-major flattened. Each row is contiguous across F.
        # We treat X as [ROWS, F], so the offset for this row and idx is row * F + idx.
        x = tl.load(X_ptr + row * F + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = F
    mean = sum_val / n
    var = sum_sq / n - mean * mean  # unbiased=False
    tl.store(Mean_ptr + row, mean)
    tl.store(Var_ptr + row, var)


# Kernel 2: evaluate inverse normal CDF (ndtri) for a single p (piecewise A&S 7.1.26)
@triton.jit
def _ndtri_kernel(P_ptr, Out_ptr):
    # Load p
    p = tl.load(P_ptr)
    # Constants
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Coefficients for lower and upper regions
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

    # Coefficients for central region
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

    # Compute q for lower region: sqrt(-2*log(p))
    # Note: p is in (0,1), log(p) can be negative; we need care. But for p>0, log(p) <= 0.
    # Triton's log supports float. We assume p>0. For our usage, target_sparsity in (0,1).
    q_low = tl.sqrt(-2.0 * tl.log(p))
    phi_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
              ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    val_low = -phi_low

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    phi_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
              (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    # Upper region
    q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    phi_up = (((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
             ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)
    val_up = phi_up

    # Piecewise selection
    # Triton scalar control: if p < p_low: val_low; elif p > p_high: val_up; else: phi_mid
    # Implement with tl.where
    is_low = p < p_low
    is_high = p > p_high
    val = tl.where(is_low, val_low, tl.where(is_high, val_up, phi_mid))

    tl.store(Out_ptr, val)


# Kernel 3: apply ReLU(x - (mean + std * multiplier)) per element
@triton.jit
def _relu_threshold_kernel(X_ptr, Mean_ptr, Std_ptr, Multiplier_ptr, Y_ptr,
                           ROWS: tl.constexpr, F: tl.constexpr,
                           BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    # Load mean and std for this row
    mean = tl.load(Mean_ptr + row)
    std = tl.load(Std_ptr + row)
    # Load multiplier (scalar)
    m = tl.load(Multiplier_ptr)
    cutoff = mean + std * m
    # Process elements of this row
    for off in range(0, F, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < F
        x = tl.load(X_ptr + row * F + idx, mask=mask, other=0.0)
        # ReLU: max(0, x - cutoff)
        diff = x - cutoff
        # Triton provides tl.maximum; if not, we can emulate with tl.where
        y = tl.maximum(diff, 0.0)
        tl.store(Y_ptr + row * F + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 256, num_warps: int = 8, num_stages: int = 2):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward: computes adaptive sparsity via inverse-normal thresholding
        and applies ReLU(x - threshold). All heavy computation is done in Triton.
        Input: [batch_size, seq_len, intermediate_size]
        Output: bfloat16 tensor of same shape.
        """
        assert inputs.dim() == 3, "Input must be 3D [batch, seq, features]"
        B, S, F = inputs.shape
        rows = B * S

        # Work in float32 for numerical stability
        x = inputs.to(torch.float32)

        # 1) Compute row-wise mean and var in Triton
        mean = torch.empty(rows, dtype=torch.float32, device=x.device)
        var = torch.empty(rows, dtype=torch.float32, device=x.device)

        _row_stats_kernel[(rows,)](
            x, mean, var,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 2) Compute std_multiplier (inverse normal CDF) in Triton
        # Create 1-element tensor on device for p (avoid host-side torch operations)
        p_tensor = torch.tensor([target_sparsity], dtype=torch.float32, device=x.device)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x.device)

        _ndtri_kernel[(1,)](
            p_tensor, std_multiplier,
            num_warps=1,
            num_stages=1,
        )

        # 3) Compute std from var on host (tiny op, but avoids extra Triton kernel)
        std = torch.sqrt(var)

        # 4) Apply activation in Triton
        y = torch.empty_like(x, dtype=torch.float32)

        _relu_threshold_kernel[(rows,)](
            x, mean, std, std_multiplier, y,
            ROWS=rows, F=F,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )

        # 5) Cast to bfloat16 and reshape to original
        y_bf16 = y.to(torch.bfloat16)
        # y has same shape as x, already [B, S, F], so we can return
        return y_bf16


def run(*args):
    return ModelNew()(*args)
