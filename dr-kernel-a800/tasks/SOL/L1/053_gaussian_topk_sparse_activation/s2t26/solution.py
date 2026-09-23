import triton
import triton.language as tl


@triton.jit
def row_sparsity_kernel(
    x_ptr: tl.pointer_type(tl.float32),
    out_ptr: tl.pointer_type(tl.float32),
    B: tl.int32,
    S: tl.int32,
    N: tl.int32,
    std_multiplier: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row: row_id in [0, B*S)
    row_id = tl.program_id(0)
    b = row_id // S
    s = row_id % S
    base = (b * S + s) * N

    # Pass 1: compute sum and sum of squares across N
    sum_val = 0.0
    sum_sq = 0.0

    for n in range(0, N, BLOCK_SIZE):
        offs = n + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / N
    var = sum_sq / N - mean * mean  # population variance (unbiased=False)
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Compute threshold = mean + std * inv-Phi(target_sparsity)
    threshold = mean + std * std_multiplier

    # Pass 2: apply ReLU gating: y = max(0, x - threshold)
    for n in range(0, N, BLOCK_SIZE):
        offs = n + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float):
        # If no sparsity, return input unchanged (still keep it on-device)
        if target_sparsity == 0.0:
            return x

        # Ensure inputs are contiguous and compute in float32 for numeric stability
        x_f32 = x.contiguous().to(torch.float32)

        B = x_f32.shape[0]
        S = x_f32.shape[1]
        N = x_f32.shape[2]

        # Allocate output (float32 for computation)
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # Compute inv-Phi(target_sparsity) using Abramowitz & Stegun 5.2.8 in pure Python
        # inv-Phi(p): solve Phi(z) = p via bisection over z in [-6, 6]
        def norm_inv(p: float) -> float:
            if p <= 0.0:
                return -6.0
            if p >= 1.0:
                return 6.0
            z_min = -6.0
            z_max = 6.0
            a1 = 0.2316419
            a2 = 0.31938153
            a3 = -0.356563782
            a4 = 1.781477937
            a5 = -1.821255978
            z = 0.5 * (z_min + z_max)
            # 30 iterations for good accuracy
            for _ in range(30):
                t = 1.0 / (1.0 + a1 * abs(z))
                poly = a5
                poly = poly * z + a4
                poly = poly * z + a3
                poly = poly * z + a2
                poly = poly * z + a1
                poly = poly * z
                cdf = 0.5 * (1.0 + poly * tl.exp(-0.5 * z * z) * t)
                if cdf > p:
                    z_max = z
                else:
                    z_min = z
                z = 0.5 * (z_min + z_max)
            return z

        std_multiplier = float(norm_inv(target_sparsity))

        # Launch row sparsity kernel: one program per row (B*S)
        grid = (B * S,)
        row_sparsity_kernel[grid](
            x_f32, out_f32, B, S, N, std_multiplier, BLOCK_SIZE=1024, num_warps=8
        )

        # Match original behavior: return in bfloat16
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
