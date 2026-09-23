import math
import triton
import triton.language as tl


@triton.jit
def sparsity_row_kernel(x_ptr, out_ptr, B, S, N, p, BLOCK_SIZE: tl.constexpr):
    """
    One Triton program per row (b, s):
      - Computes mean and std across N (float32)
      - Computes inv-Phi(p) via bisection (p in (0,1))
      - Computes threshold = mean + std * invPhi(p)
      - Applies gating: out = max(0, x - threshold)
      - Writes float32 output for the entire row
    """
    pid = tl.program_id(0)  # 0..B*S-1
    # Compute row base offset
    row_start = pid * N
    # First pass: compute sum and sum of squares across N
    total_sum = 0.0
    total_sumsq = 0.0
    # Iterate over chunks of BLOCK_SIZE
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0).to(tl.float32)
        total_sum += tl.sum(x, axis=0)
        total_sumsq += tl.sum(x * x, axis=0)

    mean = total_sum / N
    var = total_sumsq / N - mean * mean
    # Ensure numerical stability
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Compute inv-Phi(p) using bisection on z in [-6, 6]
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    invphi = 0.0
    low = -6.0
    high = 6.0
    # Bisection iterations: 24 steps provide good precision for p in (0,1)
    for _ in range(24):
        mid = 0.5 * (low + high)
        t = 0.5 * (1.0 + tl.math.erf(mid * 0.7071067811865476))  # 1/sqrt(2) = 0.70710678...
        if t > p:
            high = mid
        else:
            low = mid
    invphi = 0.5 * (low + high)

    # threshold per row
    threshold = mean + std * invphi

    # Second pass: apply gating and write output
    for off in range(0, N, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0).to(tl.float32)
        gated = x - threshold
        # ReLU: max(0, gated)
        zero = 0.0
        out = tl.where(gated > zero, gated, zero)
        tl.store(out_ptr + row_start + idx, out, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x, target_sparsity: float):
        """
        x: input tensor of shape [B, S, N], dtype float16/float32/bfloat16
        target_sparsity: float in (0,1), e.g., 0.5
        Returns: output tensor of shape [B, S, N], dtype float32 (the harness may cast to bf16).
        """
        # Ensure we don't use torch ops in forward (strict constraint).
        # Make input contiguous and cast to float32 for stable computation in kernel.
        x_contig = x.contiguous()
        # We will pass a 1-element float32 buffer for p; but here we pass p as a Python float to Triton.
        # Allocate output as float32
        B, S, N = x_contig.shape
        out = torch.empty((B, S, N), dtype=torch.float32, device=x_contig.device)

        # Launch Triton kernel: one program per row
        grid = (B * S,)
        sparsity_row_kernel[grid](
            x_contig,
            out,
            B, S, N,
            float(target_sparsity),
            BLOCK_SIZE=1024,
            num_warps=8,
        )
        return out


def run(*args):
    return ModelNew()(*args)
