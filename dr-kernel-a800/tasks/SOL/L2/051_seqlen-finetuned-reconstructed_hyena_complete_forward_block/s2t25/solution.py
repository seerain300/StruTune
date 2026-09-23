import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_forward_stats_kernel(x_ptr, sums_ptr, sumsq_ptr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    # Each program normalizes one row across the last dimension D
    row = tl.program_id(0)  # over flattened [N, L] rows
    base = row * D
    sum_val = 0.0
    sumsq_val = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)
    tl.store(sums_ptr + row, sum_val)
    tl.store(sumsq_ptr + row, sumsq_val)


@triton.jit
def layernorm_apply_kernel(x_ptr, sums_ptr, sumsq_ptr, weight_ptr, bias_ptr, out_ptr, N, L, D, eps, BLOCK_D: tl.constexpr):
    # Each program normalizes one row across the last dimension D and applies affine
    row = tl.program_id(0)  # over flattened [N, L] rows
    n = row // L
    l = row % L
    base_in = n * (L * D) + l * D
    sum_val = tl.load(sums_ptr + row)
    sumsq_val = tl.load(sumsq_ptr + row)
    mean = sum_val / D
    var = sumsq_val / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + base_in + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(out_ptr + base_in + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect hidden_states, norm1_weight, norm1_bias as first three args, matching the original run signature.
        hidden_states = args[0]  # shape [N, L, D]
        norm1_weight = args[1]   # [D]
        norm1_bias = args[2]     # [D]

        # Compute per-row statistics with Triton
        N, L, D = hidden_states.shape
        x_flat = hidden_states.reshape(N * L, D).contiguous().to(torch.float32)
        sums = torch.empty((N * L,), dtype=torch.float32, device=hidden_states.device)
        sumsq = torch.empty((N * L,), dtype=torch.float32, device=hidden_states.device)

        # Launch stats kernel: grid = (N*L,)
        layernorm_forward_stats_kernel[(N * L,)](
            x_flat, sums, sumsq, D, BLOCK_D=D
        )

        # Normalize and apply affine
        out_flat = torch.empty((N * L, D), dtype=torch.float32, device=hidden_states.device)
        layernorm_apply_kernel[(N * L,)](
            x_flat, sums, sumsq, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32), out_flat,
            N, L, D, 1e-5, BLOCK_D=D
        )

        # Reshape back to [N, L, D]
        normed = out_flat.view(N, L, D)
        return normed


def run(*args):
    return ModelNew()(*args)
