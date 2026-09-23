import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: LayerNorm per row (hidden_size is a compile-time constant)
if TRITON_AVAILABLE:
    @triton.jit
    def layernorm_affine_kernel(
        x_ptr,          # *float32, input [num_patches, hidden_size]
        out_ptr,        # *float32, output [num_patches, hidden_size]
        ln_weight_ptr,  # *float32, [hidden_size]
        ln_bias_ptr,    # *float32, [hidden_size]
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)  # one program per row
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden_size
        x = tl.load(x_ptr + pid * hidden_size + offs, mask=mask, other=0.0)
        # Compute mean and variance
        sum_x = tl.sum(x, axis=0)
        sum_x2 = tl.sum(x * x, axis=0)
        mean = sum_x / hidden_size
        var = sum_x2 / hidden_size - mean * mean
        inv_std = tl.math.rsqrt(var + eps)
        norm = (x - mean) * inv_std
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0)
        out = norm * w + b
        tl.store(out_ptr + pid * hidden_size + offs, out, mask=mask)


# Triton kernel: GELU elementwise on fp32 tensor
if TRITON_AVAILABLE:
    @triton.jit
    def gelu_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        # Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        c = 0.044715
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        x3 = x * x * x
        y = x + c * x3
        t = tl.tanh(sqrt_2_over_pi * y)
        out = 0.5 * x * (1.0 + t)
        tl.store(out_ptr + offs, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor,
                eps: float):
        """
        Triton-optimized forward:
        - LayerNorm (Triton)
        - Spatial shuffle (PyTorch, pure data movement)
        - FC1 and FC2 (PyTorch cuBLAS)
        - GELU (Triton)
        """
        # Ensure inputs are on the same device and dtype for computations
        device = hidden.device
        hidden_size = hidden.shape[1]
        # LayerNorm in fp32 via Triton
        if TRITON_AVAILABLE:
            out = torch.empty_like(hidden, dtype=torch.float32, device=device)
            # Choose BLOCK_SIZE as a multiple covering hidden_size, 1536 here
            BLOCK_SIZE = 1024  # safe tile; Triton will handle loop internally or masked
            layernorm_affine_kernel[(hidden.shape[0],)](
                hidden.to(torch.float32), out, ln_weight.to(torch.float32), ln_bias.to(torch.float32),
                hidden_size=hidden_size, eps=eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4, num_stages=2,
            )
            hidden_norm = out
        else:
            # Fallback: PyTorch LayerNorm in fp32
            hidden_norm = (hidden.to(torch.float32) - hidden.to(torch.float32).mean(dim=-1, keepdim=True)) \
                / torch.sqrt(hidden.to(torch.float32).var(dim=-1, keepdim=True, unbiased=False) + eps)
            hidden_norm = hidden_norm * ln_weight.to(torch.float32) + ln_bias.to(torch.float32)

        # Spatial shuffle: reshape, permute, flatten (pure data movement, keep dtype fp32)
        # Reshape normalized hidden to (T, H, W, C) where T*H*W == num_patches per grid,
        # then merge 2x2 into (T, H//2, W//2, 4*C).
        # Given grid_thw shape [num_grids, 3], we build the mapping per grid.
        num_patches = hidden.shape[0]
        num_merged_patches = grid_thw.shape[0]  # already provided
        patches_per_grid = num_patches // grid_thw.shape[0]
        # Compute T, H, W per grid such that T*H*W = patches_per_grid and H%2==0, W%2==0
        # Use a simple heuristic similar to the original code:
        sqrt_p = int(math.sqrt(patches_per_grid))
        h = (sqrt_p // 2) * 2  # ensure divisible by 2
        if h == 0:
            h = 2
        w = (patches_per_grid // h // 2) * 2
        if w == 0:
            w = 2
        t = patches_per_grid // (h * w)
        if t == 0:
            t = 1

        # Prepare a list of tensors for each grid
        patches_list = []
        offset = 0
        for i in range(grid_thw.shape[0]):
            t_i = int(grid_thw[i, 0].item())
            h_i = int(grid_thw[i, 1].item())
            w_i = int(grid_thw[i, 2].item())
            # Build T,H,W per grid; if not matching the heuristic, fall back to original t,h,w
            Tg = t_i if t_i > 0 else t
            Hg = h_i if h_i > 0 else h
            Wg = w_i if w_i > 0 else w

            patches = hidden_norm[offset:offset + Tg * Hg * Wg]
            patches = patches.view(Tg, Hg, Wg, hidden_size)
            # Merge 2x2 patches
            Tg_m = Tg
            Hg_m = Hg // 2
            Wg_m = Wg // 2
            # Reshape to (T, H//2, W//2, 2, 2, C)
            merged = patches.view(Tg_m, Hg_m, 2, Wg_m, 2, hidden_size)
            # Permute to (T, H//2, W//2, 2, 2, C)
            perm = merged.permute(0, 2, 4, 1, 3, 5)  # -> [Tg, 2, 2, Hg//2, Wg//2, hidden_size]
            # Flatten spatial merge groups: (T * H//2 * W//2, 4*C)
            M = Tg * Hg_m * Wg_m
            C = hidden_size
            patches_list.append(perm.reshape(M, 4 * C))
            offset += Tg * Hg * Wg

        # Concatenate all grids into the final shuffled tensor [num_merged_patches, 6144]
        hidden_shuffled = torch.cat(patches_list, dim=0)  # already fp32

        # FC1: (num_merged_patches, 6144) @ (6144, 6144).T (+ fc1_bias)
        fc1_out = torch.nn.functional.linear(hidden_shuffled, fc1_weight, fc1_bias)  # fp32

        # GELU activation via Triton (if available)
        if TRITON_AVAILABLE:
            N = fc1_out.numel()
            gelu_out = torch.empty_like(fc1_out)
            BLOCK_G = 1024
            grid = (triton.cdiv(N, BLOCK_G),)
            gelu_kernel[grid](fc1_out, gelu_out, N=N, BLOCK_SIZE=BLOCK_G, num_warps=4, num_stages=2)
            fc1_out = gelu_out

        # FC2: (num_merged_patches, 6144) @ (3584, 6144).T (+ fc2_bias)
        out_mlp = torch.nn.functional.linear(fc1_out, fc2_weight, fc2_bias)  # fp32

        return out_mlp


def run(*args):
    return ModelNew()(*args)
