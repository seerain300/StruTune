import math
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    x_ptr,           # *ptr to input patches (N, H), bfloat16
    y_ptr,           # *ptr to output patches (N, H), bfloat16
    ln_weight_ptr,   # *ptr to ln_weight (H), bfloat16
    ln_bias_ptr,     # *ptr to ln_bias (H), bfloat16
    N,               # number of rows (num_patches)
    H: tl.constexpr, # hidden_size (1536)
    eps,             # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    # One Triton program per row
    row_id = tl.program_id(0)
    if row_id >= N:
        return
    row_offset = row_id * H

    # Compute sum in float32
    sum_ = 0.0
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_ += tl.sum(x, axis=0)

    mean = sum_ / H

    # Compute variance in float32
    var_sum = 0.0
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        var_sum += tl.sum((x - mean) * (x - mean), axis=0)

    var = var_sum / H
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine, store as bfloat16
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        y = y * gamma + beta
        tl.store(y_ptr + row_offset + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def linear_gemv_kernel(
    A_ptr,           # *ptr to A (M, K), bfloat16
    Wt_ptr,          # *ptr to W^T (K, N), bfloat16
    Bias_ptr,        # *ptr to bias (N), bfloat16
    Out_ptr,         # *ptr to output (M, N), float32
    M,               # number of rows in A (num_merged_patches)
    K,               # number of columns in A (hidden_size_expanded)
    N,               # output dimension (6144 for first linear, 3584 for second)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch: pid_m over rows of A, pid_n over columns of output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m < M
    mask_n = n < N
    mask_mn = mask_m[:, None] & mask_n[None, :]

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K

        # A[m, k]: shape (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + m[:, None] * K + k[None, :]
        a = tl.load(a_ptrs, mask=mask_mn, other=0.0)
        a = a.to(tl.float32)

        # Wt[k, n]: shape (BLOCK_K, BLOCK_N)
        wt_ptrs = Wt_ptr + k[:, None] * N + n[None, :]
        wt = tl.load(wt_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)
        wt = wt.to(tl.float32)

        # Fused multiply-add
        acc += tl.dot(a, wt)

    # Add bias
    b = tl.load(Bias_ptr + n, mask=mask_n, other=0.0).to(tl.float32)
    acc += b[None, :]

    # Store output (float32); Out_ptr points to float32 buffer
    out_ptrs = Out_ptr + m[:, None] * N + n[None, :]
    tl.store(out_ptrs, acc, mask=mask_mn)


@triton.jit
def gelu_kernel(
    X_ptr,           # *ptr to input (M, N), float32
    Y_ptr,           # *ptr to output (M, N), float32
    M,               # number of rows
    N,               # number of cols
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m < M
    mask_n = n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = X_ptr + m[:, None] * N + n[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)  # float32

    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    gelu = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    y_ptrs = Y_ptr + m[:, None] * N + n[None, :]
    tl.store(y_ptrs, gelu, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, axes_and_scalars: dict, device: torch.device):
        super().__init__()
        self.device = device

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
        Triton-only implementation of the original run:
        1) LayerNorm over the last dim (1536) on hidden
        2) Spatial shuffle to form (num_merged_patches, 6144) using PyTorch (data movement)
        3) Linear1: (num_merged_patches, 6144) @ fc1_weight.T + fc1_bias
        4) GELU
        5) Linear2: (num_merged_patches, 6144) @ fc2_weight.T + fc2_bias
        Returns: (num_merged_patches, 3584) in bfloat16.
        """
        assert hidden.is_cuda and grid_thw.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda and \
               fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, \
               "All inputs must be CUDA tensors"

        num_patches = hidden.shape[0]
        hidden_size = 1536
        hidden_expanded = 6144
        out_hidden_size = 3584

        # 1) Triton LayerNorm: y_norm (num_patches, 1536), bfloat16
        y_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)
        N = num_patches
        H = hidden_size
        grid_ln = (N,)
        layernorm_kernel[grid_ln](
            hidden, y_norm, ln_weight, ln_bias,
            N, H, eps,
            BLOCK_SIZE=256,
        )

        # 2) Spatial shuffle to (num_merged_patches, 6144) using PyTorch (data movement)
        # Reconstruct grids and patches as in original code
        num_grids = grid_thw.shape[0]
        patches_per_grid = num_patches // num_grids if num_patches % num_grids == 0 else 1
        offset = 0
        shuffled_patches = []
        for i in range(num_grids):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            num_patches_this = t * h * w
            patches = y_norm[offset: offset + num_patches_this]  # (num_patches_this, 1536)
            h_merged = h // 2
            w_merged = w // 2
            patches = patches.view(t, h_merged, 2, w_merged, 2, hidden_size)
            patches = patches.permute(0, 1, 3, 2, 4, 5)  # (T, H/2, W/2, 2, 2, C)
            patches = patches.reshape(t * h_merged * w_merged, hidden_expanded)
            shuffled_patches.append(patches)
            offset += num_patches_this
        hidden_shuffled = torch.cat(shuffled_patches, dim=0)  # (num_merged_patches, 6144), bfloat16

        # 3) Triton Linear1: (M, K) @ (K, N) + bias -> (M, N), float32
        M = hidden_shuffled.shape[0]  # num_merged_patches
        K = hidden_expanded  # 6144
        N1 = 6144
        W1_T = fc1_weight.transpose(0, 1).contiguous()  # (6144, 6144), bfloat16
        out1 = torch.empty((M, N1), dtype=torch.float32, device=hidden.device)
        grid1 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        linear_gemv_kernel[grid1](
            hidden_shuffled, W1_T, fc1_bias,
            out1,
            M, K, N1,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        # 4) Triton GELU
        gelu_out = torch.empty_like(out1, dtype=torch.float32, device=hidden.device)
        grid_gelu = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        gelu_kernel[grid_gelu](
            out1, gelu_out,
            M, N1,
            BLOCK_M=64, BLOCK_N=64,
        )

        # 5) Triton Linear2: (M, K) @ (K, OUT) + bias -> (M, OUT), float32
        out_hidden = 3584
        W2_T = fc2_weight.transpose(0, 1).contiguous()  # (6144, 3584), bfloat16
        out2 = torch.empty((M, out_hidden), dtype=torch.float32, device=hidden.device)
        grid2 = (triton.cdiv(M, 64), triton.cdiv(out_hidden, 64))
        linear_gemv_kernel[grid2](
            gelu_out, W2_T, fc2_bias,
            out2,
            M, K, out_hidden,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        # Return in bfloat16 to match original behavior
        return out2.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
