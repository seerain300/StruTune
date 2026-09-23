import torch
import math
import triton
import triton.language as tl


# -------------------------
# 1) Triton LayerNorm kernel
# -------------------------
@triton.jit
def layer_norm_kernel(
    hidden_ptr,        # *bf16, [N, C]
    out_ptr,           # *bf16, [N, C]
    ln_weight_ptr,     # *bf16, [C]
    ln_bias_ptr,       # *bf16, [C]
    N, C,              # int32
    eps,               # float32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # row id
    if pid >= N:
        return
    # Accumulate mean/var in fp32
    mean = 0.0
    var = 0.0
    for c0 in range(0, C, BLOCK_SIZE):
        offs = c0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(hidden_ptr + pid * C + offs, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / C

    # Compute variance
    for c0 in range(0, C, BLOCK_SIZE):
        offs = c0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(hidden_ptr + pid * C + offs, mask=mask, other=0.0).to(tl.float32)
        var += tl.sum((x - mean) * (x - mean), axis=0)
    var = var / C
    inv_std = tl.rsqrt(var + eps)

    # Normalize and affine
    for c0 in range(0, C, BLOCK_SIZE):
        offs = c0 + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(hidden_ptr + pid * C + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(out_ptr + pid * C + offs, y.to(tl.bfloat16), mask=mask)


# -------------------------
# 2) Triton GEMM with bias + epilogue
# -------------------------
@triton.jit
def matmul_bias_kernel(
    A_ptr,             # *bf16, [M, K]
    W_ptr,             # *bf16, [K, Nout]
    Bias_ptr,          # *bf16, [Nout] or None
    C_ptr,             # *bf16, [M, Nout] output
    M, K, Nout,        # int32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(A_ptr + (offs_m[:, None] * K) + k[None, :], mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(W_ptr + (k[:, None] * Nout) + offs_n[None, :], mask=(k[:, None] < K) & (offs_n[None, :] < Nout), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    # Add bias if provided
    if tl.pointer_is_contiguous(Bias_ptr):
        bias = tl.load(Bias_ptr + offs_n, mask=offs_n < Nout, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    # Store result as bf16
    tl.store(C_ptr + (offs_m[:, None] * Nout) + offs_n[None, :],
             acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < Nout))


# -------------------------
# 3) Triton GELU elementwise kernel
# -------------------------
@triton.jit
def gelu_kernel(
    x_ptr,             # *bf16, [M, K]
    y_ptr,             # *bf16, [M, K]
    M, K,              # int32
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    x = tl.load(x_ptr + (offs_m[:, None] * K) + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    c = 0.044715
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + c * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))

    tl.store(y_ptr + (offs_m[:, None] * K) + offs_k[None, :], y.to(tl.bfloat16), mask=mask)


# -------------------------
# Helper functions for grid sizing (used in forward)
# -------------------------
def _ceil_div(a, b):
    return (a + b - 1) // b


def _launch_layer_norm(hidden: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float) -> torch.Tensor:
    assert hidden.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda
    N, C = hidden.shape
    out = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)
    BLOCK = 1024  # C=1536 => 2 blocks of 1024
    grid = (N,)
    layer_norm_kernel[grid](
        hidden, out, ln_weight, ln_bias,
        N, C, float(eps),
        BLOCK_SIZE=BLOCK,
        num_warps=4,
    )
    return out


def _launch_matmul_bias(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor, BLOCK_M: int = 64, BLOCK_N: int = 128, BLOCK_K: int = 64) -> torch.Tensor:
    assert A.is_cuda and W.is_cuda and (bias is None or bias.is_cuda)
    M, K = A.shape
    K_w, Nout = W.shape
    assert K == K_w, f"Inner dim mismatch: A({M},{K}) @ W({K_w},{Nout})"

    C = torch.empty((M, Nout), dtype=torch.bfloat16, device=A.device)
    grid = ( _ceil_div(M, BLOCK_M), _ceil_div(Nout, BLOCK_N) )
    matmul_bias_kernel[grid](
        A, W, bias if bias is not None else A,  # pass a valid pointer; kernel guards with pointer_is_contiguous
        C,
        M, K, Nout,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return C


def _launch_gelu(x: torch.Tensor, BLOCK_M: int = 128, BLOCK_K: int = 128) -> torch.Tensor:
    assert x.is_cuda
    M, K = x.shape
    y = torch.empty_like(x, dtype=torch.bfloat16, device=x.device)
    grid = ( _ceil_div(M, BLOCK_M), _ceil_div(K, BLOCK_K) )
    gelu_kernel[grid](
        x, y,
        M, K,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return y


# -------------------------
# ModelNew: Triton-optimized forward
# -------------------------
class ModelNew(torch.nn.Module):
    def __init__(self, merge_size: int = 2, hidden_size: int = 1536, hidden_size_expanded: int = 6144, out_hidden_size: int = 3584, eps: float = 1e-6):
        super().__init__()
        self.merge_size = merge_size
        self.hidden_size = hidden_size
        self.hidden_size_expanded = hidden_size_expanded
        self.out_hidden_size = out_hidden_size
        self.eps = eps

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
    ) -> torch.Tensor:
        """
        Triton-optimized forward:
        1) Layer normalization (Triton) on hidden [num_patches, 1536], learnable weight/bias.
        2) Spatial shuffle (PyTorch view/permute) using provided grid_thw per workload.
        3) fc1 GEMM (Triton), then GELU (Triton).
        4) fc2 GEMM (Triton) with bias.
        Returns: [num_merged_patches, 3584] bfloat16.
        """
        # Ensure CUDA and contiguous tensors
        device = hidden.device
        assert device.type == "cuda", "Triton requires CUDA device. Move inputs to cuda."

        # 1) Triton LayerNorm
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()
        hidden_norm = _launch_layer_norm(hidden, ln_weight, ln_bias, self.eps)  # [num_patches, 1536] bf16

        # 2) Spatial shuffle: exact as original using provided grid_thw
        # We need to reproduce the original mapping:
        # For each grid i: num_patches_this = t * h * w
        # Reshape hidden_norm[offset:offset+num_patches_this] -> (t, h, w, C)
        # Merge spatial 2x2 -> (t, h//2, w//2, 4, C) -> permute to (t, h//2, w//2, 4, C) -> reshape to (t*(h//2)*(w//2), 4*C)
        # We will do this in PyTorch (view/permute) for correctness and robustness.

        N = hidden_norm.shape[0]
        C = hidden_norm.shape[1]
        T, H, W = grid_thw.shape  # grid_thw: [num_grids, 3] => per-grid (t, h, w)
        # We need to iterate over grids; however, the original helper used per-grid T/H/W inside run. Here, grid_thw gives per-grid values.
        # We will compute offsets and perform reshapes per grid.
        num_merged_patches = 0
        offset = 0
        # Compute num_merged_patches from grid_thw
        for i in range(grid_thw.shape[0]):
            t = grid_thw[i, 0].item()
            h = grid_thw[i, 1].item()
            w = grid_thw[i, 2].item()
            num_patches_this = t * h * w
            num_merged_patches += (t * (h // self.merge_size) * (w // self.merge_size))
            # Reshape and shuffle this grid
            patches = hidden_norm[offset:offset + num_patches_this]
            patches = patches.view(t, h, w, C)  # (t, h, w, C)
            h2 = h // self.merge_size
            w2 = w // self.merge_size
            patches = patches.view(t, h2, self.merge_size, w2, self.merge_size, C)  # (t, h2, 2, w2, 2, C)
            patches = patches.permute(0, 1, 3, 2, 4, 5)  # (t, h2, w2, 2, 2, C)
            patches = patches.reshape(t * h2 * w2, 4 * C)  # (num_patches_this, 6144)
            # Place into a single output buffer
            # We'll collect all grids in a list and cat at the end. But since we don't have an output buffer, we do per grid in a loop and append.
            # However, to preserve original order, we should compute offsets per grid contribution.
            # We will store into a preallocated output buffer by computing the destination index. But since we don't have per-grid base index, we will instead collect all reshaped tensors and concatenate after loop.
            # Better: allocate output directly with known num_merged_patches and compute base offset per grid based on where it starts in the output.
            # We don't have the output ordering dictated by original helper; therefore, we cannot directly write to output here. Instead, we will keep a list of reshaped tensors and concatenate after loop.

        # Since we cannot compute final output ordering without original helper, we will reconstruct output by noting that original outputs after shuffle are simply concatenation in the original patch order. The original helper computes actual grid_thw with specific rules (sqrt-based), but forward uses those grid_thw exactly. We can still compute the final output using the same logical ordering: for each grid, perform the reshape/permute, and then concatenate all grids in the original order.

        # To maintain correctness, we compute each grid's reshaped tensor and concatenate them in the same loop order. We don't have a preallocated buffer, but we can allocate at the end since num_merged_patches is known after loop.
        # Compute total num_merged_patches and allocate output
        total_num_merged = num_merged_patches
        shuffled_list = []
        offset = 0
        for i in range(grid_thw.shape[0]):
            t = grid_thw[i, 0].item()
            h = grid_thw[i, 1].item()
            w = grid_thw[i, 2].item()
            num_patches_this = t * h * w
            patches = hidden_norm[offset:offset + num_patches_this]
            patches = patches.view(t, h, w, C)
            h2 = h // self.merge_size
            w2 = w // self.merge_size
            patches = patches.view(t, h2, self.merge_size, w2, self.merge_size, C)
            patches = patches.permute(0, 1, 3, 2, 4, 5)
            patches = patches.reshape(t * h2 * w2, self.hidden_size_expanded)
            shuffled_list.append(patches)
            offset += num_patches_this
        # Concatenate all grids' shuffled patches in original order
        hidden_shuffled = torch.cat(shuffled_list, dim=0)  # [num_merged_patches, 6144], bf16

        # 3) Triton fc1: linear (no bias), output [num_merged_patches, 6144]
        # We need to ensure fc1_weight is contiguous
        fc1_weight = fc1_weight.contiguous()
        fc1_out = _launch_matmul_bias(hidden_shuffled, fc1_weight, None)  # [num_merged_patches, 6144], bf16

        # 4) Triton GELU
        fc1_out = fc1_out.contiguous()
        fc1_gelu = _launch_gelu(fc1_out)  # [num_merged_patches, 6144], bf16

        # 5) Triton fc2: linear with bias, output [num_merged_patches, 3584]
        fc2_weight = fc2_weight.contiguous()
        fc2_bias = fc2_bias.contiguous()
        output = _launch_matmul_bias(fc1_gelu, fc2_weight, fc2_bias)  # [num_merged_patches, 3584], bf16

        return output


def run(*args):
    return ModelNew()(*args)
