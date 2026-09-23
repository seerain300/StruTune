import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_forward_kernel(x_ptr, out_ptr, weight_ptr, bias_ptr, N, D, eps, BLOCK_D: tl.constexpr):
    # Each program handles one row (n) and normalizes across the last dimension D
    n = tl.program_id(0)
    # Base pointer for this row
    base = n * D

    # Compute sum and sum of squares over D
    sum_val = 0.0
    sumsq_val = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    mean = sum_val / D
    var = sumsq_val / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        y = y * w + b
        tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor):
        """
        Perform LayerNorm over the last dimension D for each row (N rows) using Triton.
        Inputs:
          - hidden_states: [N, L, D], float32 on CUDA device
          - norm1_weight: [D], float32
          - norm1_bias: [D], float32
        Returns:
          - normalized tensor: [N, L, D]
        """
        # Ensure contiguous and device
        x = hidden_states.contiguous()
        N, L, D = x.shape
        device = x.device

        # Prepare output
        out = torch.empty_like(x, dtype=torch.float32, device=device)

        # Triton launch: one program per row across D
        grid = (N,)
        layernorm_forward_kernel[grid](x, out, norm1_weight, norm1_bias, N, D, 1e-5, BLOCK_D=256)

        # Return Triton-computed normalized tensor
        return out


def run(*args):
    return ModelNew()(*args)
