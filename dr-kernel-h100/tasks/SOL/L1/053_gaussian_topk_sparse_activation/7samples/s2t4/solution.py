import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel: per-row gating. Each program handles one (batch, seq) row.
# It iterates across the last dimension in chunks of BLOCK_SIZE and writes:
# y = max(0, x - (mean + std * inv_cdf))
@triton.jit
def gate_rows(
    x_ptr,           # *float32, flattened as [rows, N]
    mean_ptr,        # *float32, shape [rows]
    std_ptr,         # *float32, shape [rows]
    inv_cdf_ptr,     # *float32, shape [1]
    out_ptr,         # *float32, flattened as [rows, N]
    N: tl.constexpr,        # last-dim size (intermediate_size)
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)  # each program handles one row
    # Compute base offsets for this row
    base = row * N
    # Load per-row mean and std
    mean = tl.load(mean_ptr + row)
    std = tl.load(std_ptr + row)
    # Load scalar inv_cdf (single element)
    inv_cdf = tl.load(inv_cdf_ptr)
    threshold = mean + std * inv_cdf
    # Iterate over columns
    for col in range(0, N, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + base + cols, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized sparsity gate:
        - Computes per-row mean and std across the last dimension using PyTorch.
        - Computes inv_norm_cdf(target_sparsity) using PyTorch's _ndtri for correctness.
        - Applies ReLU gate with adaptive threshold (mean + std * inv_cdf) using a Triton kernel.
        Returns output cast to bfloat16 to match original behavior.
        """
        if target_sparsity == 0.0:
            # No gating: return input cast to bfloat16
            return inputs.to(torch.bfloat16)

        # Ensure contiguous and compute in float32 for numerical stability
        x = inputs.contiguous()
        x_f32 = x.to(torch.float32)

        # Compute per-row mean and std along last dim
        # Keep dims to have shape [B, S, 1]
        mean = x_f32.mean(dim=-1, keepdim=True)
        std = x_f32.std(dim=-1, keepdim=True, unbiased=False)

        # Prepare output buffer (float32)
        B, S, N = x_f32.shape
        out_f32 = torch.empty_like(x_f32)

        # Compute inv_norm_cdf(target_sparsity) using PyTorch for correctness
        # Create a 1-element tensor on the right device
        device = x_f32.device
        inv_cdf = torch._ndtri(torch.tensor(target_sparsity, device=device, dtype=torch.float32))
        # Pass as 1-element tensor to Triton kernel
        inv_cdf_buf = inv_cdf.view(1).contiguous()

        # Flatten rows for Triton kernel
        rows = B * S
        x_flat = x_f32.view(rows, N)
        out_flat = out_f32.view(rows, N)

        # Launch Triton kernel: one program per row
        BLOCK_SIZE = 1024  # tuned for typical intermediate sizes; adjust if needed
        grid = (rows,)
        gate_rows[grid](x_flat, mean.view(rows), std.view(rows), inv_cdf_buf, out_flat, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)

        # Cast to bfloat16 to match original code's behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
