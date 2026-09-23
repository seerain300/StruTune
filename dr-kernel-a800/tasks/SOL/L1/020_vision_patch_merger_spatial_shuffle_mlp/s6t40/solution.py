import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_affine_kernel(
    x_ptr,        # *bfloat16, input (M, K)
    ln_w_ptr,     # *bfloat16, ln weight (K,)
    ln_b_ptr,     # *bfloat16, ln bias (K,)
    y_ptr,        # *bfloat16, output (M, K)
    M: tl.constexpr,      # number of rows
    K: tl.constexpr,      # hidden size
    eps: tl.constexpr,    # epsilon
    BLOCK: tl.constexpr,  # tile size along K
):
    row = tl.program_id(0)  # one program per row
    # First pass: compute mean and variance in FP32
    sum_ = 0.0
    sumsq_ = 0.0
    for off in range(0, K, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < K
        x = tl.load(x_ptr + row * K + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_ += tl.sum(x, axis=0)
        sumsq_ += tl.sum(x * x, axis=0)
    mean = sum_ / K
    var = sumsq_ / K - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, store as BF16
    for off in range(0, K, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < K
        x = tl.load(x_ptr + row * K + idx, mask=mask, other=0.0).to(tl.float32)
        ln_w = tl.load(ln_w_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        ln_b = tl.load(ln_b_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        y = y * ln_w + ln_b
        tl.store(y_ptr + row * K + idx, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor, ln_weight: torch.Tensor,
                ln_bias: torch.Tensor, fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor, eps: float):
        """
        Triton-optimized forward:
        - LayerNorm + affine in Triton (one program per row, FP32 compute, BF16 output).
        - Packing is done via view (num_patches % 4 == 0 in provided inputs).
        - MLP layers and GELU done with torch for robustness.
        """
        device = hidden.device
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]

        # 1) LayerNorm + affine in Triton
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()
        ln_out = torch.empty((num_patches, hidden_size), dtype=torch.bfloat16, device=device)
        BLOCK_ln = 256  # tile along K; masked loads ensure safety for K=1536
        grid_ln = (num_patches,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M=num_patches, K=hidden_size, eps=eps,
            BLOCK=BLOCK_ln, num_warps=4, num_stages=2
        )

        # 2) Packing via view: (num_patches//4, 4*hidden_size)
        # Invariant from get_inputs: num_patches % 4 == 0
        M_out = num_patches // 4
        K_packed = 4 * hidden_size  # 4 * 1536 = 6144
        packed = ln_out.view(M_out, K_packed)

        # 3) fc1: (M_out, 6144) @ (6144, 6144) + fc1_bias
        # Use torch for robustness
        K1 = K_packed  # 6144
        N1 = fc1_weight.shape[0]  # 6144
        fc1_out = torch.addmm(fc1_bias, packed, fc1_weight.transpose(0, 1))
        fc1_out = fc1_out.to(torch.bfloat16)

        # 4) GELU using torch (fast and reliable)
        # If you prefer Triton GELU, we can add a separate elementwise kernel.
        fc1_after_gelu = torch.nn.functional.gelu(fc1_out)

        # 5) fc2: (M_out, N1) @ (3584, N1) + fc2_bias
        N2 = fc2_weight.shape[0]  # 3584
        fc2_out = torch.addmm(fc2_bias, fc1_after_gelu, fc2_weight.transpose(0, 1))
        fc2_out = fc2_out.to(torch.bfloat16)

        return fc2_out


def run(*args):
    return ModelNew()(*args)
