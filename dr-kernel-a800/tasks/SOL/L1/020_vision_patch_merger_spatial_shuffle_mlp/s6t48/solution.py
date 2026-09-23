import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_affine_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, K, eps,
    BLOCK: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)
    if row >= M:
        return

    # First pass: compute mean and variance over K (in FP32)
    sum_val = 0.0
    sum_sq = 0.0
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / K
    var = sum_sq / K - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        norm = (x - mean) * inv_std
        w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = norm * w + b
        tl.store(Y_ptr + row * K + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _gelu_kernel(
    X_ptr, Y_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    # 2D grid over rows and columns
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if (pid_m < 0) or (pid_n < 0):
        return

    row = pid_m
    col0 = pid_n * BLOCK_N
    cols = col0 + tl.arange(0, BLOCK_N)

    mask = (row < M) & (cols < N)

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # GELU tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    t = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.tanh(t))
    tl.store(Y_ptr + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,
        grid_thw: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
        eps: float,
    ):
        """
        Triton-only implementation:
        1) LayerNorm (pre-shuffle) over last dim with affine (Triton)
        2) Spatial packing: view from (num_patches, 1536) to (num_patches//4, 4*1536)
        3) fc1: linear (cuBLAS-backed) (6144->6144) + bias
        4) GELU (Triton)
        5) fc2: linear (cuBLAS-backed) (6144->3584) + bias
        """
        device = hidden.device
        M = hidden.shape[0]  # num_patches
        K = hidden.shape[1]  # hidden_size = 1536

        # 1) LayerNorm with affine (Triton)
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        BLOCK_ln = 256
        grid_ln = (M,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M=M, K=K, eps=eps,
            BLOCK=BLOCK_ln, num_warps=4, num_stages=2
        )

        # 2) Spatial packing: reshape since T=1 and num_patches % 4 == 0
        # From (M, K) -> (M//4, 4*K)
        K_expanded = 4 * K  # 6144
        hidden_pack = ln_out.view(M // 4, K_expanded)

        # 3) fc1: linear (M//4, 6144) @ (6144, 6144) + bias
        fc1_out = torch.nn.functional.linear(hidden_pack, fc1_weight, fc1_bias)

        # 4) GELU activation (Triton)
        K_after = fc1_out.shape[1]  # 6144
        fc1_after_gelu = torch.empty_like(fc1_out, dtype=torch.bfloat16, device=device)
        BLOCK_N_gelu = 256
        grid_gelu = (fc1_out.shape[0], triton.cdiv(K_after, BLOCK_N_gelu))
        _gelu_kernel[grid_gelu](
            fc1_out, fc1_after_gelu,
            M=fc1_out.shape[0], N=K_after,
            BLOCK_N=BLOCK_N_gelu, num_warps=4, num_stages=2
        )

        # 5) fc2: linear (M//4, 6144) @ (3584, 6144) + bias
        out = torch.nn.functional.linear(fc1_after_gelu, fc2_weight, fc2_bias)

        return out


def run(*args):
    return ModelNew()(*args)
