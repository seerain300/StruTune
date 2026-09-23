import torch
import triton
import triton.language as tl


@triton.jit
def _row_reduce_mean_std_kernel(inp_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    For each (batch, seq) row, compute mean and std across the last dimension of length N.
    inp_ptr: *f32, shape [B, S, N], contiguous (row-major, last dim contiguous)
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
    # Population variance (unbiased=False), matches torch.std(..., unbiased=False)
    var = sum_x2 / n - mean * mean
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

    # A&S 7.1.26 constants
    # Lower region constants (q = sqrt(-2 log(p)))
    c1 = -7.784695709041462e-03
    c2 = 3.224671290700398e-01
    c3 = 2.445134137142996e+00
    c4 = 3.754408661907416e+00

    # Upper region constants (q = sqrt(-2 log(1 - p)))
    d1 = 7.784894002430293e-03
    d2 = 3.223964580411365e-01
    d3 = 2.400758277161838e+00
    d4 = 2.549732539343734e+00

    p_low = 0.02425

    if p < 0.5:
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q))
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        result = -poly / denom
    else:
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
    inp_ptr: *f32, shape [B, S, N], contiguous (row-major)
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    threshold_ptr: *f32, shape [1] (scalar multiplier)
    out_ptr: *f32, shape [B, S, N], contiguous
    B, S, N are ints, used for grid sizing (program_id(0) selects row pid).
    """
    pid = tl.program_id(0)
    row_start = pid * N

    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    multiplier = tl.load(threshold_ptr)  # scalar

    cutoff = mean + std * multiplier

    # Process the entire row in chunks
    for off in range(0, N, 1024):
        idx = off + tl.arange(0, 1024)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # relu
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized forward that:
          - computes per-(batch,seq) mean and std across last dim (population, unbiased=False),
          - computes inverse standard normal CDF for target_sparsity in Triton,
          - applies out = relu(input - (mean + std * multiplier)) in Triton,
          - returns output cast to original dtype.
        """
        if target_sparsity == 0.0:
            # No sparsity requested: return original tensor unchanged
            return inputs

        # Ensure contiguous and compute in float32
        inp = inputs.contiguous()
        inp_f32 = inp.to(torch.float32)
        B, S, N = inp_f32.shape
        flat_in = inp_f32.view(B * S, N)

        # Allocate outputs for mean and std
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=inp_f32.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=inp_f32.device)

        # Launch reduction kernel
        # Choose BLOCK_SIZE tuned for N; use 4096 for N up to 16384 to reduce loop iterations
        BLOCK_SIZE = 4096 if N >= 4096 else (2048 if N >= 2048 else 1024)
        grid = (B * S,)
        _row_reduce_mean_std_kernel[grid](flat_in, mean_buf, std_buf, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=4)

        # Compute std_multiplier via Triton kernel (inverse-normal CDF)
        # Create a 1-element device tensor for p
        p_dev = torch.tensor(target_sparsity, dtype=torch.float32, device=inp_f32.device).unsqueeze(0)  # shape [1]
        multiplier = torch.empty(1, dtype=torch.float32, device=inp_f32.device)
        _compute_std_multiplier_kernel[(1,)](p_dev, multiplier, num_warps=1, num_stages=1)

        # Allocate output and launch sparsify+ReLU kernel
        out_flat = torch.empty_like(flat_in, dtype=torch.float32, device=inp_f32.device)
        _sparsify_relu_kernel[grid](flat_in, mean_buf, std_buf, multiplier, out_flat, B, S, N, num_warps=4, num_stages=2)

        # Reshape and cast back to original dtype
        out = out_flat.view(B, S, N)
        return out.to(inp.dtype)


def run(*args):
    return ModelNew()(*args)
