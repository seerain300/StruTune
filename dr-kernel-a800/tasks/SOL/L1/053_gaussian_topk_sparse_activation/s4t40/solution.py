import torch
import triton
import triton.language as tl


@triton.jit
def _row_reduce_mean_std_kernel(inp_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    For each (batch, seq) row, compute mean and std across the last dimension of length N.
    inp_ptr: *f32, shape [B, S, N], contiguous
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    N: int, length of last dim
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Accumulators in f32
    sum_x = 0.0
    sum_x2 = 0.0

    # Iterate over the row in chunks
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = N
    mean = sum_x / n
    var = sum_x2 / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Write results
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def _compute_std_multiplier_kernel(p_dev_ptr, out_ptr):
    """
    Compute std_multiplier = inverse standard normal CDF(p) via A&S approximation (7.1.26).
    Stores the result to out_ptr[0].
    p_dev_ptr: *f32, shape [1] (device scalar tensor; kernel loads it)
    out_ptr: *f32, shape [1]
    """
    p = tl.load(p_dev_ptr)

    # A&S 7.1.26 approximation: two branches
    # lower region for p < 0.5; upper region for p >= 0.5 (using 1 - p for upper branch)
    p_low = 0.02425

    # Lower region constants
    c1 = -7.784695709041462e-03
    c2 = 3.224671290700398e-01
    c3 = 2.445134137142996e+00
    c4 = 3.754408661907416e+00

    # Upper region constants (mirrored)
    d1 = 7.784894002430293e-03
    d2 = 3.223964580411365e-01
    d3 = 2.400758277161838e+00
    d4 = 2.549732539343734e+00

    if p < 0.5:
        # lower region path for p
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q))
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        result = -poly / denom
    else:
        # upper region path for 1 - p
        p_hi = 1.0 - p
        q = tl.sqrt(-2.0 * tl.log(p_hi))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q))
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        result = poly / denom

    tl.store(out_ptr, result)


@triton.jit
def _sparsify_relu_kernel(inp_ptr, mean_ptr, std_ptr, threshold_ptr, out_ptr, B, S, N):
    """
    Elementwise: out = relu(input - (mean + std * threshold[0]))
    inp_ptr: *f32, shape [B, S, N]
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    threshold_ptr: *f32, shape [1] (scalar multiplier)
    out_ptr: *f32, shape [B, S, N]
    """
    pid = tl.program_id(0)
    # pid indexes over all rows (B*S)
    row = pid
    b = row // S
    s = row % S
    row_start = (b * S + s) * N

    # Load mean and std for this row
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    thresh = tl.load(threshold_ptr)  # scalar

    base = row_start

    # Apply ReLU: max(0, x - (mean + std * thresh))
    for off in range(0, N, 1024):
        idx = off + tl.arange(0, 1024)
        mask = idx < N
        x = tl.load(inp_ptr + base + idx, mask=mask, other=0.0)
        t = mean + std * thresh
        y = tl.maximum(x - t, 0.0)
        tl.store(out_ptr + base + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation:
        - Reduce per row (B*S rows) to compute mean and std across the last dim (N).
        - Compute scalar std_multiplier via Triton kernel using A&S approximation.
        - Apply elementwise sparsification: out = relu(input - (mean + std * std_multiplier)).
        Returns output with the same shape as inputs.
        """
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and compute in float32
        inp = inputs.contiguous()
        inp_f32 = inp.to(torch.float32)

        B, S, N = inp_f32.shape
        rows = B * S

        # Allocate outputs and buffers
        mean = torch.empty(rows, dtype=torch.float32, device=inp_f32.device)
        std = torch.empty(rows, dtype=torch.float32, device=inp_f32.device)
        threshold_buf = torch.empty(1, dtype=torch.float32, device=inp_f32.device)  # scalar multiplier on device

        # Launch reduction kernel
        BLOCK_SIZE = 2048  # tuned for N up to 16384; mask handles smaller N
        grid = (rows,)
        _row_reduce_mean_std_kernel[grid](inp_f32, mean, std, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)

        # Compute std_multiplier via Triton scalar kernel: inverse normal CDF of target_sparsity
        p_dev = torch.empty(1, dtype=torch.float32, device=inp_f32.device)
        p_dev.fill_(float(target_sparsity))
        _compute_std_multiplier_kernel[(1,)](p_dev, threshold_buf, num_warps=1, num_stages=1)

        # Allocate output and run elementwise sparsification
        out_f32 = torch.empty_like(inp_f32)
        _sparsify_relu_kernel[grid](
            inp_f32, mean, std, threshold_buf, out_f32, B, S, N,
            num_warps=8, num_stages=4
        )

        # Cast back to original dtype
        return out_f32.to(inputs.dtype)


def run(*args):
    return ModelNew()(*args)
