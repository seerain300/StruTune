import math
import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    hidden_ptr,        # *bf16, [N, C]
    out_ptr,           # *bf16, [N, C]
    ln_weight_ptr,     # *bf16, [C]
    ln_bias_ptr,       # *bf16, [C]
    N,                 # int32
    C,                 # int32 (hidden size, e.g., 1536)
    eps,               # float32
    BLOCK_SIZE: tl.constexpr,
):
    """
    Per-row Layer Normalization:
    For each row i in [0, N):
      mean = sum(x_i) / C
      var  = sum(x_i^2) / C - mean^2
      inv_std = 1 / sqrt(var + eps)
      out_i = ((x_i - mean) * inv_std) * ln_weight + ln_bias
    All math in fp32, output cast back to bf16.
    """
    row = tl.program_id(0)
    if row >= N:
        return

    sum_x = 0.0
    sum_x2 = 0.0

    col = 0
    while col < C:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(hidden_ptr + row * C + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sum_x += tl.sum(x32, axis=0)
        sum_x2 += tl.sum(x32 * x32, axis=0)
        col += BLOCK_SIZE

    mean = sum_x / C
    var = sum_x2 / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    col = 0
    while col < C:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(hidden_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(out_ptr + row * C + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        """
        Triton-optimized version:
        - Layer normalization with Triton (fp32 math, bf16 I/O).
        - Spatial shuffle (pure shape transforms).
        - Two-layer MLP (fc1, GELU, fc2) using PyTorch/cuBLAS.
        """
        # Ensure tensors are contiguous
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()

        # Use Triton LN when on CUDA; otherwise fallback to PyTorch
        use_triton = hidden.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda
        N, C = hidden.shape
        out_hidden = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)

        if use_triton:
            grid = (N,)
            layer_norm_kernel[grid](
                hidden, out_hidden, ln_weight, ln_bias,
                N, C, eps,
                BLOCK_SIZE=1024,
                num_warps=4,
            )
            hidden_norm = out_hidden
        else:
            # Fallback: PyTorch LN (fp32 math), then cast back
            x = hidden
            x32 = x.to(torch.float32)
            mean = x32.mean(dim=-1, keepdim=True)
            var = x32.var(dim=-1, keepdim=True, unbiased=False)
            x_norm = (x32 - mean) / torch.sqrt(var + eps)
            ln_weight32 = ln_weight.to(torch.float32)
            ln_bias32 = ln_bias.to(torch.float32)
            hidden_norm = (x_norm * ln_weight32 + ln_bias32).to(torch.bfloat16)

        # Step 2: Spatial shuffle to merge patches (reshapes/permutes; exact as original)
        offset = 0
        shuffled_patches = []
        for i in range(grid_thw.shape[0]):
            t = grid_thw[i, 0].item()
            h = grid_thw[i, 1].item()
            w = grid_thw[i, 2].item()
            num_patches_this = t


def run(*args):
    return ModelNew()(*args)
