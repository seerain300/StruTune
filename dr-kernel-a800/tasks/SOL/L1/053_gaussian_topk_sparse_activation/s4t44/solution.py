import torch
import triton
import triton.language as tl


@triton.jit
def _row_reduce_mean_std_kernel(inp_ptr, mean_ptr, std_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    For each (batch, seq) row, compute mean and std across the last dimension of length N.
    inp_ptr: *f32, shape [B, S, N], contiguous (row-major: last dim contiguous)
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    N: int, length of last dim
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # Accumulators in f32
    sum_x = 0.0
    sum_x2 = 0.0

    # First pass: sum of values
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)

    # Second pass: sum of squares
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        sum_x2 += tl.sum(x * x, axis=0)

    n = N
    mean = sum_x / n
    var = sum_x2 / n - mean * mean  # population variance (unbiased=False)
    std = tl.sqrt(var)

    # Write results
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def _ndtri_scalar_kernel(p, out_ptr):
    """
    Compute inverse standard normal CDF for scalar p via A&S 26.2.23 approximation.
    Stores the result to out_ptr[0].
    p: host-passed float, target sparsity
    out_ptr: *f32, shape [1]
    """
    p = tl.float32(p)  # ensure scalar float

    # Constants (A&S 26.2.23)
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

    if p < 0.5:
        # lower region
        q = tl.sqrt(-2.0 * tl.log(p))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        result = -poly / denom
    else:
        # upper region
        p_hi = 1.0 - p
        q = tl.sqrt(-2.0 * tl.log(p_hi))
        poly = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denom = (((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0))
        result = poly / denom

    tl.store(out_ptr, result)


@triton.jit
def _sparsify_relu_kernel(inp_ptr, mean_ptr, std_ptr, threshold_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise: out = relu(input - (mean + std * threshold[0]))
    One program processes one row of length N.
    inp_ptr: *f32, shape [B, S, N]
    mean_ptr: *f32, shape [B*S]
    std_ptr: *f32, shape [B*S]
    threshold_ptr: *f32, shape [1] (scalar multiplier)
    out_ptr: *f32, shape [B, S, N]
    """
    pid = tl.program_id(0)
    row_start = pid * N

    # load mean and std for this row
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    std_multiplier = tl.load(threshold_ptr)  # scalar
    cutoff = mean + std * std_multiplier

    # iterate over row and apply relu(input - cutoff)
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(inp_ptr + row_start + idx, mask=mask, other=0.0)
        y = x - cutoff
        y = tl.maximum(y, 0.0)  # relu
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Gaussian-based top-k sparse activation.
        Computes adaptive sparsity threshold based on input statistics:
          threshold = mean + std * norm.icdf(target_sparsity)
        Then applies ReLU(input - threshold).
        """
        if target_sparsity == 0.0:
            return inputs

        # Ensure contiguous and compute in float32
        orig_dtype = inputs.dtype
        inp = inputs.contiguous()
        inp_f32 = inp.to(torch.float32)

        # Expect shape [B, S, N]; get B,S,N
        assert inp_f32.dim() == 3, "inputs must be 3D: [batch_size, seq_len, intermediate_size]"
        B = inp_f32.shape[0]
        S = inp_f32.shape[1]
        N = inp_f32.shape[2]
        total_rows = B * S

        # Allocate outputs for mean, std
        mean = torch.empty((total_rows,), dtype=torch.float32, device=inp_f32.device)
        std = torch.empty((total_rows,), dtype=torch.float32, device=inp_f32.device)

        # Launch reduction kernel: one program per row
        grid = (total_rows,)
        _row_reduce_mean_std_kernel[grid](inp_f32, mean, std, N, BLOCK_SIZE=1024, num_warps=4, num_stages=2)

        # Compute std_multiplier in Triton (scalar approximation): pass float target_sparsity
        threshold_buf = torch.empty((1,), dtype=torch.float32, device=inp_f32.device)
        _ndtri_scalar_kernel[target_sparsity](threshold_buf)  # pass scalar float as 'p'

        # Output buffer
        out_f32 = torch.empty_like(inp_f32)

        # Launch sparsification kernel: one program per row
        _sparsify_relu_kernel[grid](inp_f32, mean, std, threshold_buf, out_f32, N, BLOCK_SIZE=1024, num_warps=4, num_stages=2)

        # Cast back to original dtype
        return out_f32.to(orig_dtype)


def run(*args):
    return ModelNew()(*args)
