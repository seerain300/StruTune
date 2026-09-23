import math
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_affine_kernel(
    hidden_ptr,        # *bf16, [num_patches, hidden_size]
    ln_weight_ptr,     # *bf16, [hidden_size]
    ln_bias_ptr,       # *bf16, [hidden_size]
    out_ptr,           # *bf16, [num_patches, hidden_size]
    num_patches,       # int32
    hidden_size,       # int32
    eps,               # float32
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles one row (patch)
    # If pid >= num_patches, exit (masking protects loads)
    # Compute mean and variance in fp32
    sum_val = 0.0
    sum_sq = 0.0

    # First pass: compute sum and sum of squares
    c0 = 0
    while c0 < hidden_size:
        offs = c0 + tl.arange(0, BLOCK_C)
        mask = offs < hidden_size
        ptrs = hidden_ptr + pid * hidden_size + offs
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        c0 += BLOCK_C

    mean = sum_val / hidden_size
    var = sum_sq / hidden_size - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    c0 = 0
    while c0 < hidden_size:
        offs = c0 + tl.arange(0, BLOCK_C)
        mask = offs < hidden_size
        ptrs_in = hidden_ptr + pid * hidden_size + offs
        x = tl.load(ptrs_in, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        ptrs_out = out_ptr + pid * hidden_size + offs
        tl.store(ptrs_out, y.to(tl.bfloat16), mask=mask)
        c0 += BLOCK_C


@triton.jit
def spatial_shuffle_to_fc1_kernel(
    ln_out_ptr,         # *bf16, [num_patches, hidden_size]
    grid_thw_ptr,       # *int64, [num_grids, 3]
    out_fc1_ptr,        # *bf16, [num_merged_patches, hidden_size_expanded] (preallocated)
    num_patches,        # int32
    hidden_size,        # int32
    hidden_size_expanded,  # int32, 4 * hidden_size
    num_grids,          # int32
):
    pid = tl.program_id(0)  # grid id
    if pid >= num_grids:
        return

    # Load T, H, W for this grid
    t_i = tl.load(grid_thw_ptr + pid * 3 + 0)
    h_i = tl.load(grid_thw_ptr + pid * 3 + 1)
    w_i = tl.load(grid_thw_ptr + pid * 3 + 2)

    # Total patches for this grid
    patches_this = t_i * h_i * w_i

    # Iterate over original patches in this grid
    p = 0
    while p < patches_this:
        # Original coordinates
        HW = h_i * w_i
        t = p // HW
        r = p % HW
        h = r // w_i
        w = r % w_i
        i0 = t
        j0 = h * w_i + w

        # Merged coordinates
        i1 = i0 // 2
        j1 = j0 // 2

        # Row in shuffled output
        Wm = w_i // 2
        row = i1 * Wm + j1

        # Iterate over expanded features c = 0..hidden_size_expanded-1
        # hidden_size_expanded = 4 * hidden_size
        c = 0
        while c < hidden_size_expanded:
            c1 = c // 4
            c2 = c % 4
            original_feature = c1 * 4 + c2
            # Load from LayerNorm output: [num_patches, hidden_size]
            ln_ptrs = ln_out_ptr + p * hidden_size + original_feature
            val = tl.load(ln_ptrs).to(tl.float32)
            # Store to out_fc1 at [row, c]
            out_ptrs = out_fc1_ptr + row * hidden_size_expanded + c
            tl.store(out_ptrs, val.to(tl.bfloat16))
            c += 1
        p += 1


@triton.jit
def matmul_bias_kernel(
    A_ptr,             # *bf16, [M, K]
    B_ptr,             # *bf16, [K, N]
    bias_ptr,          # *bf16, [N]
    C_ptr,             # *bf16, [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + m0 * K + k_offsets[None, :] * M  # shape (BM, BK)
        b_ptrs = B_ptr + k_offsets[:, None] * N + n0     # shape (BK, BN)

        a_mask = (m0 + tl.arange(0, BLOCK_M))[:, None] < M
        b_mask = (n0 + tl.arange(0, BLOCK_N))[None, :] < N
        k_mask = k_offsets[None, :] < K

        a = tl.load(a_ptrs, mask=a_mask & k_mask[None, :], other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & b_mask, other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
        k0 += BLOCK_K

    # Add bias
    bias = tl.load(bias_ptr + n0 + tl.arange(0, BLOCK_N), mask=(n0 + tl.arange(0, BLOCK_N)) < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store to C
    c_ptrs = C_ptr + (m0 + tl.arange(0, BLOCK_M))[:, None] * N + (n0 + tl.arange(0, BLOCK_N))[None, :]
    c_mask = (m0 + tl.arange(0, BLOCK_M))[:, None] < M & (n0 + tl.arange(0, BLOCK_N))[None, :] < N
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def gelu_tanh_kernel(
    x_ptr,            # *bf16, [M, N]
    out_ptr,          # *bf16, [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * 1  # 1x1 grid not used; we tile with larger blocks
    n0 = pid_n * N  # not used; we use broadcasting

    # For simplicity, we process whole matrix: M,N
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptrs).to(tl.float32)
    # GELU tanh approximation
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    t = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.tanh(t))
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, y.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        """
        hidden: [num_patches, hidden_size] (bf16, contiguous)
        grid_thw: [num_grids, 3] int64 (T, H, W per grid)
        ln_weight, ln_bias: [hidden_size] bf16
        fc1_weight: [hidden_size_expanded, hidden_size_expanded] bf16
        fc1_bias: [hidden_size_expanded] bf16
        fc2_weight: [out_hidden_size, hidden_size_expanded] bf16
        fc2_bias: [out_hidden_size] bf16
        eps: float
        Returns: [num_merged_patches, out_hidden_size] bf16
        """
        assert hidden.is_cuda and grid_thw.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda
        assert fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda

        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        hidden_size_expanded = hidden_size * 4
        # Allocate layernorm output
        ln_out = torch.empty_like(hidden)  # [num_patches, hidden_size], bf16
        # Launch layernorm_affine kernel: one program per row
        BLOCK_C = 128
        grid0 = (num_patches,)
        layernorm_affine_kernel[grid0](
            hidden, ln_weight, ln_bias, ln_out,
            num_patches, hidden_size, eps,
            BLOCK_C,
            num_warps=4, num_stages=2,
        )

        # Allocate first linear input (shuffled): [num_merged_patches, hidden_size_expanded]
        # We need to compute num_merged_patches. Since original code derives it from num_patches and num_grids,
        # we can compute using the same formula as get_inputs: patches_per_grid = num_patches // num_grids,
        # actual_patches_per_grid = T*H*W as derived by get_inputs. Here, we use the provided grid_thw to derive it:
        # Each grid has its own T,H,W. Sum T*H*W across grids equals num_patches. To get num_merged_patches:
        # The original code sets num_merged_patches = sum over grids of T * (H//2) * (W//2). We approximate with
        # sum over grids of (grid_thw[:,0]) * (grid_thw[:,1]//2) * (grid_thw[:,2]//2).
        # Compute num_merged_patches (host side):
        num_merged_patches = 0
        for i in range(grid_thw.shape[0]):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            num_merged_patches += t * (h // 2) * (w // 2)

        out_fc1 = torch.empty((num_merged_patches, hidden_size_expanded), dtype=torch.bfloat16, device=hidden.device)

        # Launch spatial_shuffle_to_fc1_kernel: one program per grid
        spatial_shuffle_to_fc1_kernel[(grid_thw.shape[0],)](
            ln_out, grid_thw, out_fc1,
            num_patches, hidden_size, hidden_size_expanded, grid_thw.shape[0],
            num_warps=4, num_stages=2,
        )

        # First Linear: A = out_fc1 [M, K], B = fc1_weight.T [K, N], bias = fc1_bias [N]
        # We will run matmul_bias_kernel. Ensure fc1_weight is [K, N] where K=hidden_size_expanded, N=hidden_size_expanded.
        # Weight for PyTorch nn.Linear is [N, K], so transpose here.
        B_fc1 = fc1_weight.t().contiguous()  # [hidden_size_expanded, hidden_size_expanded] bf16
        M = out_fc1.shape[0]
        K = out_fc1.shape[1]
        N = B_fc1.shape[1]

        # Allocate output for first linear
        fc1_out = torch.empty((M, N), dtype=torch.bfloat16, device=hidden.device)

        # Launch matmul_bias_kernel for first linear
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_fc1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_bias_kernel[grid_fc1](
            out_fc1, B_fc1, fc1_bias, fc1_out,
            M, N, K,
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # GELU activation: elementwise on fc1_out
        gelu_out = torch.empty_like(fc1_out)
        # For simplicity, use whole matrix grid (1x1). Gelu_tanh_kernel processes full tensor.
        grid_gelu = (1, 1)
        gelu_tanh_kernel[grid_gelu](
            fc1_out, gelu_out,
            M, N,
            num_warps=4, num_stages=2,
        )

        # Second Linear: A = gelu_out [M, K2], B = fc2_weight.T [K2, N2], bias = fc2_bias [N2]
        # Transpose fc2_weight (PyTorch nn.Linear weight is [N2, K2])
        B_fc2 = fc2_weight.t().contiguous()  # [hidden_size_expanded, out_hidden_size] bf16
        M2 = M
        K2 = N  # N from first linear = hidden_size_expanded
        N2 = B_fc2.shape[1]  # out_hidden_size

        # Allocate output for second linear
        out = torch.empty((M2, N2), dtype=torch.bfloat16, device=hidden.device)

        # Launch matmul_bias_kernel for second linear
        grid_fc2 = (triton.cdiv(M2, BLOCK_M), triton.cdiv(N2, BLOCK_N))
        matmul_bias_kernel[grid_fc2](
            gelu_out, B_fc2, fc2_bias, out,
            M2, N2, K2,
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
