import torch
import triton
import triton.language as tl


@triton.jit
def row_reduce_sum_sumsq_kernel(X_ptr, sum_out_ptr, sumsq_out_ptr,
                                 F: tl.int32,
                                 BLOCK_SIZE: tl.constexpr):
    """
    Each program handles one row (program_id(0) == row id).
    It iterates over the feature dimension in tiles and accumulates sum and sum of squares.
    Writes sum[row] and sumsq[row] to sum_out_ptr and sumsq_out_ptr.
    """
    row_id = tl.program_id(axis=0)
    sum_val = 0.0
    sum_sq = 0.0
    # Loop over features in tiles
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        offs = row_id * F + cols
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    tl.store(sum_out_ptr + row_id, sum_val)
    tl.store(sumsq_out_ptr + row_id, sum_sq)


@triton.jit
def apply_relu_threshold_kernel(X2d_ptr, Y_ptr, mean_ptr, std_ptr,
                                 multiplier_value: tl.float32,
                                 F: tl.int32,
                                 BLOCK_SIZE: tl.constexpr):
    """
    Each program handles one row. It iterates over F and applies:
      y = max(0, x - (mean + std * multiplier_value))
    Writes results to Y_ptr[row * F + cols].
    """
    row_id = tl.program_id(axis=0)
    m = tl.load(mean_ptr + row_id)
    s = tl.load(std_ptr + row_id)
    threshold = m + s * multiplier_value
    for start in range(0, F, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < F
        offs = row_id * F + cols
        x = tl.load(X2d_ptr + offs, mask=mask, other=0.0)
        y = x - threshold
        y = tl.where(y > 0, y, 0.0)
        tl.store(Y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size_reduce=1024, block_size_act=1024):
        super().__init__()
        # Tunable tile sizes; 1024 works well for large F like 12288 and 16384.
        self.block_size_reduce = block_size_reduce
        self.block_size_act = block_size_act

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Gaussian-based top-k sparse activation using Triton:
        1) Compute per-row mean and std across the last dim (feature).
        2) Compute multiplier = icdf(target_sparsity) using torch.
        3) Apply y = max(0, x - (mean + std * multiplier)) with Triton.
        Returns tensor in bfloat16 like the original.
        """
        # Ensure 3D input [B, S, F]
        assert inputs.dim() == 3, "inputs must be [batch_size, seq_len, intermediate_size]"
        B, S, F = inputs.shape

        # Flatten to 2D for Triton row-wise processing
        x = inputs.contiguous()  # ensure contiguous
        x32 = x.to(torch.float32)  # compute in float32
        rows = B * S
        x2d = x32.view(rows, F)

        # 1) Per-row sum and sum of squares via Triton reduction
        sum_buf = torch.empty(rows, dtype=torch.float32, device=x32.device)
        sumsq_buf = torch.empty(rows, dtype=torch.float32, device=x32.device)

        row_reduce_sum_sumsq_kernel[(rows,)](
            x2d, sum_buf, sumsq_buf,
            F=F,
            BLOCK_SIZE=self.block_size_reduce,
            num_warps=8,
            num_stages=2,
        )

        # 2) Compute mean and std (unbiased=False) on GPU using PyTorch
        mean = sum_buf / F
        var = sumsq_buf / F - mean * mean
        std = torch.sqrt(var)  # elementwise GPU op

        # 3) Compute scalar inverse normal CDF for target_sparsity using torch
        # This is a single scalar tensor on device
        # Note: torch.distributions.normal.inverse_cdf is not available; use ppf from CDF if needed.
        # Here, use torch.special.erfinv trick or rely on torch.normal.icdf (not universally available).
        # A robust approach: use torch.quantile on a standard normal sample is overkill.
        # We instead use torch.special.erfinv only if available; otherwise approximate.
        # Given the environment, use torch.normal.ppf if available:
        try:
            import torch.distributions
            multiplier = torch.tensor(torch.distributions.Normal(0.0, 1.0).icdf(target_sparsity), dtype=torch.float32, device=x32.device)
        except Exception:
            # Fallback: erf-based inverse (assuming math.erf inverse available in torch.special)
            # If unavailable, use a high-accuracy approximation or accept PyTorch’s quantile. To keep correctness,
            # we will fall back to torch.normal.ppf if available. If not, we approximate via torch.quantile on
            # a standard normal sample, but that’s costly. As a last resort, use 0 for multiplier; but that’s incorrect.
            # Therefore, we’ll raise a clear error if icdf is not available in this environment.
            raise RuntimeError("Torch environment must provide torch.distributions.Normal.icdf for inverse CDF.")

        # 4) Apply activation in Triton
        y = torch.empty(rows * F, dtype=torch.float32, device=x32.device)
        apply_relu_threshold_kernel[(rows,)](
            x2d, y, mean, std, float(multiplier.item()),  # pass scalar
            F=F,
            BLOCK_SIZE=self.block_size_act,
            num_warps=8,
            num_stages=2,
        )

        # 5) Reshape and cast to bfloat16 to match original output
        y_out = y.view(B, S, F).to(torch.bfloat16)
        return y_out