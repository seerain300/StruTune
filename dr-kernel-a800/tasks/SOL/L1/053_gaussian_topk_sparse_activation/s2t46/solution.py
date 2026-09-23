import triton
import triton.language as tl


@triton.jit
def compute_invphi_kernel(p_scalar: tl.constexpr, out_ptr):
    """
    Compute inverse standard normal CDF (quantile) for probability p_scalar
    using bisection. Writes a single float32 value to out_ptr[0].
    """
    low = -6.0
    high = 6.0
    # Bisection: 50 iterations suffice for high accuracy
    for _ in range(50):
        mid = 0.5 * (low + high)
        # CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
        cdf = 0.5 * (1.0 + tl.erf(0.7071067811865476 * mid))  # 1/sqrt(2)
        if cdf > p_scalar:
            high = mid
        else:
            low = mid
    tl.store(out_ptr, mid)


@triton.jit
def row_sparsity_kernel(
    x_ptr,            # *float32, base pointer to input
    out_ptr,          # *float32, base pointer to output
    B, S, N,          # int32 dimensions
    std_multiplier,   # float32 scalar: inv-Phi(target_sparsity)
    BLOCK_SIZE: tl.constexpr,
):
    """
    One Triton program per row (b, s). Compute mean and std across N,
    then apply gating: out = max(0, x - (mean + std * std_multiplier)).
    Iterates over N in chunks of BLOCK_SIZE using while loops to handle any N.
    """
    row_id = tl.program_id(axis=0)
    b = row_id // S
    s = row_id % S

    # Base linear index for the row (assuming contiguous layout: offset = b*S*N + s*N + n)
    base = b * S * N + s * N

    # First pass: accumulate sum and sum of squares in float32
    sum_val = 0.0
    sum_sq = 0.0
    n_start = 0
    while n_start < N:
        offs = n_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x_chunk = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x_chunk, axis=0)
        sum_sq += tl.sum(x_chunk * x_chunk, axis=0)
        n_start += BLOCK_SIZE

    # Compute mean and std (population std: unbiased=False)
    mean = sum_val / N
    var = sum_sq / N - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Compute threshold
    threshold = mean + std * std_multiplier

    # Second pass: apply gating and store
    n_start = 0
    while n_start < N:
        offs = n_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x_chunk = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        diff = x_chunk - threshold
        out_chunk = tl.maximum(diff, 0.0)  # ReLU
        tl.store(out_ptr + base + offs, out_chunk, mask=mask)
        n_start += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Assume input is a single tensor x with shape [B, S, N]
        x = args[0]
        target_sparsity = float(args[1])

        # If no sparsity requested, return x unchanged.
        if target_sparsity == 0.0:
            return x

        # Ensure float32 and contiguous for Triton
        x_f32 = x.contiguous().to(torch.float32)
        B, S, N = x_f32.shape

        # Output buffer in float32
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # 1-element device buffer for std_multiplier
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)

        # Launch inv-Phi kernel: one program
        compute_invphi_kernel[(1,)](target_sparsity, std_multiplier)

        # Launch row sparsity kernel: one program per row
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32, B, S, N, float(std_multiplier[0]), BLOCK_SIZE=1024, num_warps=8
        )

        # Return in bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
