import torch
import triton
import triton.language as tl


# Kernel 1: compute per-row mean and std along the last dimension
# input_ptr: pointer to input tensor of shape [M, K], where M = B*S, K = last_dim
# mean_ptr: pointer to output mean vector of shape [M] (float32)
# std_ptr: pointer to output std vector of shape [M] (float32)
@triton.jit
def compute_row_stats_kernel(
    input_ptr, mean_ptr, std_ptr,
    M, K,
    stride_m, stride_k,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    # Initialize accumulators
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over columns in chunks of BLOCK_SIZE
    for col_start in range(0, K, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        # Address for this row slice: input[row, offs]
        ptrs = input_ptr + row * stride_m + offs * stride_k
        x = tl.load(ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    n = K
    # Compute mean and variance (population, unbiased=False)
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    # Clamp variance to non-negative to avoid tiny negative due to fp errors
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    tl.store(mean_ptr + row, mean)
    tl.store(std_ptr + row, std)


# Kernel 2: compute z = inverse standard normal CDF at p using Abramowitz & Stegun 5.2.23 approximation.
# output_ptr: pointer to a 1-element float32 tensor to store z.
@triton.jit
def ndtri_kernel(output_ptr, p: tl.float32):
    # p is a scalar in (0,1). We implement piecewise approximation.
    # Constants for lower region
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
        z = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
            ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
        tl.store(output_ptr, -z)  # since p < 0.5 => output is negative
        return

    # Upper region
    if p > p_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
            ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
        tl.store(output_ptr, z)
        return

    # Central region
    q = p - 0.5
    r = q * q
    z = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q / \
        (((((b1*r + b2)*r + b3)*r + b4)*r + b5) * r + 1.0)
    tl.store(output_ptr, z)


# Kernel 3: apply elementwise gating: output = relu(input - threshold[row])
# input_ptr: flattened input of length M*K
# mean_ptr, std_ptr: per-row mean and std (length M)
# z: scalar in float32
# output_ptr: flattened output of length M*K
@triton.jit
def apply_gating_kernel(
    input_ptr, mean_ptr, std_ptr, output_ptr,
    M, K, z,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    total = M * K
    for offs in range(pid * BLOCK_SIZE, total, BLOCK_SIZE):
        mask = offs < total
        x = tl.load(input_ptr + offs, mask=mask, other=0.0)
        # Compute row index for threshold
        row = offs // K
        mean_val = tl.load(mean_ptr + row)
        std_val = tl.load(std_ptr + row)
        threshold = mean_val + std_val * z
        gated = x - threshold
        gated = tl.maximum(gated, 0.0)  # relu
        tl.store(output_ptr + offs, gated, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_size: int = 1024, num_warps: int = 4, num_stages: int = 2):
        super().__init__()
        self.block_size = block_size
        self.num_warps = num_warps
        self.num_stages = num_stages

    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # If no sparsity, return input directly
        if target_sparsity == 0.0:
            return inputs

        # Ensure 3D input: [batch_size, seq_len, intermediate_size]
        assert inputs.ndim == 3, "inputs must be 3D [batch_size, seq_len, intermediate_size]"
        B, S, K = inputs.shape
        M = B * S

        # Make input contiguous
        x = inputs.contiguous()
        # Compute statistics in float32
        x32 = x.to(torch.float32)

        # Allocate per-row mean and std
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        std = torch.empty(M, dtype=torch.float32, device=x.device)

        # Launch row stats kernel: one program per row
        grid_stats = (M,)
        compute_row_stats_kernel[grid_stats](
            x32, mean, std,
            M, K,
            x32.stride(0), x32.stride(2),
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages
        )

        # Compute z via Triton ndtri kernel (store into a 1-element tensor)
        z_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        ndtri_kernel[(1,)](z_buf, target_sparsity)
        z = z_buf[0]  # scalar float32

        # Prepare flattened input and output
        x_flat = x32.view(-1)
        out_flat = torch.empty_like(x_flat)

        # Launch gating kernel over total elements
        total = M * K
        grid_gate = (triton.cdiv(total, self.block_size),)
        apply_gating_kernel[grid_gate](
            x_flat, mean, std, out_flat,
            M, K, z,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
            num_stages=self.num_stages
        )

        # Reshape back to [B, S, K] and return
        return out_flat.view(B, S, K)