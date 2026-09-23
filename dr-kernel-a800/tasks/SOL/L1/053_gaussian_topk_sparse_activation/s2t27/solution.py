import triton
import triton.language as tl


@triton.jit
def invPhi_bisect_kernel(p: tl.float32, out_ptr: tl.pointer(dtype=tl.float32), tol: tl.float32, iters: tl.int32):
    # Compute z = Phi^{-1}(p) via bisection on z in [-6, 6].
    # Use erf approximation (Abramowitz & Stegun 7.1.26): erf(x) ≈ sign(x) * (1 - exp(-x^2) * poly(t)), t = 1/(1 + A|x|)
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    low = tl.full((), -6.0, tl.float32)
    high = tl.full((), 6.0, tl.float32)
    for _ in range(iters):
        mid = 0.5 * (low + high)
        x = mid * 0.7071067811865476  # 1/sqrt(2)
        sign = tl.where(x >= 0.0, 1.0, -1.0)
        ax = tl.abs(x)
        A = 0.254829592
        p1 = 0.3480242
        p2 = 1.1388291
        p3 = 0.3275911
        t = 1.0 / (1.0 + A * ax)
        poly = (((p3 * t - p2) * t + p1) * t)
        erf_approx = sign * (1.0 - poly * tl.exp(-ax * ax))
        phi_mid = 0.5 * (1.0 + erf_approx)
        cond = phi_mid < p  # if phi(mid) < p, we need larger z (move low up)
        low = tl.where(cond, mid, low)
        high = tl.where(cond, high, mid)
    z_approx = 0.5 * (low + high)
    tl.store(out_ptr, z_approx)


@triton.jit
def row_stats_gate_kernel(
    x_ptr, out_ptr,
    B, S, N,
    std_multiplier_ptr,  # points to a 1-element buffer
    BLOCK_SIZE: tl.constexpr
):
    # One program per row (b, s)
    row_id = tl.program_id(0)
    b = row_id // S
    s = row_id % S
    base = b * S * N + s * N

    # First pass: compute sum and sum of squares across N
    total_sum = tl.full((), 0.0, tl.float32)
    total_sumsq = tl.full((), 0.0, tl.float32)

    offset = 0
    while offset < N:
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x_chunk = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        x_chunk = x_chunk.to(tl.float32)
        total_sum += tl.sum(x_chunk, axis=0)
        total_sumsq += tl.sum(x_chunk * x_chunk, axis=0)
        offset += BLOCK_SIZE

    mean = total_sum / N
    var = total_sumsq / N - mean * mean
    var = tl.maximum(var, 0.0)  # clamp small negatives
    std = tl.sqrt(var)

    # Read inv-Phi multiplier
    std_multiplier = tl.load(std_multiplier_ptr)
    threshold = mean + std * std_multiplier

    # Second pass: apply gating and store output
    offset = 0
    while offset < N:
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x_chunk = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        x_chunk = x_chunk.to(tl.float32)
        y_chunk = x_chunk - threshold
        y_chunk = tl.maximum(y_chunk, 0.0)  # ReLU gating
        tl.store(out_ptr + base + idx, y_chunk, mask=mask)
        offset += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # No gating requested
        if target_sparsity == 0.0:
            return x.to(torch.bfloat16)

        # Ensure contiguous and cast to float32 for compute (no torch ops)
        x_f32 = x.contiguous().to(torch.float32)
        B, S, N = x_f32.shape

        # Output buffer (float32 for compute)
        out_f32 = torch.empty((B, S, N), dtype=torch.float32, device=x_f32.device)

        # 1-element buffer for inv-Phi(std_multiplier)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=x_f32.device)

        # Compute inv-Phi(target_sparsity) in Triton
        invPhi_bisect_kernel[(1,)](
            float(target_sparsity),
            std_multiplier,
            tol=1e-6,
            iters=20,  # 20 bisection iterations for good accuracy
            num_warps=4
        )

        # Run row-wise stats + gating
        BLOCK_SIZE = 1024
        row_stats_gate_kernel[(B * S,)](
            x_f32, out_f32,
            B, S, N,
            std_multiplier,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8
        )

        # Return as bfloat16 (original code example casts to bfloat16)
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
