import math
import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr,           # *ptr to input patches (N, H), bfloat16
    y_ptr,           # *ptr to output patches (N, H), bfloat16
    ln_weight_ptr,   # *ptr to ln_weight (H), bfloat16
    ln_bias_ptr,     # *ptr to ln_bias (H), bfloat16
    N,               # number of rows (num_patches)
    H: tl.constexpr, # hidden_size (1536)
    eps,             # epsilon (float)
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    row_id = tl.program_id(0)
    if row_id >= N:
        return
    row_offset = row_id * H

    # Compute mean in float32
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
        diff = x - mean
        var_sum += tl.sum(diff * diff, axis=0)
    var = var_sum / H
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine, store as bfloat16
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        gamma = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        y = y * gamma + beta
        tl.store(y_ptr + row_offset + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def linear_kernel(
    A_ptr,            # *ptr to A (M, K), float32 (we'll pass float32 tensors)
    Wt_ptr,           # *ptr to W^T (N, K), float32 (transposed weight)
    Bias_ptr,         # *ptr to bias (N), float32
    C_ptr,            # *ptr to output (M, N), float32
    M,                # rows of A
    K,                # cols of A / rows of W^T
    N,                # cols of W^T / rows of output
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: (ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # A_tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Wt_tile: [BLOCK_K, BLOCK_N] (from W^T of shape (N, K))
        wt_ptrs = Wt_ptr + offs_n[None, :] * K + offs_k[:, None]
        wt_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        Wt_tile = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, Wt_tile)

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Store
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def gelu_tanh_kernel(
    X_ptr,            # *ptr to input (M, K), float32
    Y_ptr,            # *ptr to output (M, K), float32
    M,                # number of rows
    K,                # number of cols
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    k_start = pid_k * BLOCK_K

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_k = k_start + tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    x = tl.load(x_ptrs, mask=mask, other=0.0)  # float32

    # GELU tanh approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))

    y_ptrs = Y_ptr + offs_m[:, None] * K + offs_k[None, :]
    tl.store(y_ptrs, y, mask=mask)


def triton_layer_norm(hidden: torch.Tensor,
                      ln_weight: torch.Tensor,
                      ln_bias: torch.Tensor,
                      eps: float):
    """
    Triton LayerNorm over last dimension. Returns bfloat16 tensor of shape like hidden.
    """
    N = hidden.shape[0]
    H = hidden.shape[1]
    assert hidden.dtype == torch.bfloat16, "hidden must be bfloat16"
    assert ln_weight.shape == (H,), "ln_weight must have shape (hidden_size,)"
    assert ln_bias.shape == (H,), "ln_bias must have shape (hidden_size,)"

    # Output tensor
    out = torch.empty_like(hidden)

    # Choose BLOCK_SIZE (power of two <= H) for good reduction; 256 or 128 are fine for H=1536.
    # We'll use 256 for speed, masks handle tails.
    BLOCK_SIZE = 256

    # Launch Triton kernel: one program per row
    grid = (N,)
    layer_norm_kernel[grid](
        hidden, out, ln_weight, ln_bias, N, H, eps, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4, num_stages=2,
    )
    return out


def triton_linear(A: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor):
    """
    Triton GEMM: A (M, K) float32 @ weight.T (N, K) float32 + bias (N) -> (M, N) float32.
    weight is original (K, N), we pass as W^T here.
    """
    M, K = A.shape
    N = weight.shape[1]  # K dimension in weight corresponds to input features, output rows = N in weight
    # Ensure dtypes and device
    assert A.dtype == torch.float32, "A must be float32 for Triton linear kernel"
    # We will treat weight.T as (N, K) via indexing. For performance, we create a transposed view for addressing:
    # Triton kernel expects Wt_ptr pointing to (N, K). We can pass weight as is and compute row-major addressing.
    # But simpler: create Wt by swapping strides or materialize. To avoid materialization, pass weight and compute
    # as Wt[i, k] = weight[k, i]. Triton will read weight in transposed order by passing appropriate pointer arithmetic.
    # For simplicity and correctness, materialize Wt as a contiguous (N, K) tensor:
    Wt = weight.transpose(0, 1).contiguous()  # (N, K)
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    # Tile sizes. For K=6144, N up to 6144/3584, M up to num_merged_patches, use moderate tiles.
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    linear_kernel[grid](
        A, Wt, bias, C,
        M, K, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return C


def triton_gelu(x: torch.Tensor):
    """
    Triton GELU using tanh approximation. Input x is float32 (M, K), output is float32.
    """
    M, K = x.shape
    y = torch.empty_like(x)

    BLOCK_M = 64
    BLOCK_K = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    gelu_tanh_kernel[grid](
        x, y, M, K,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return y


class ModelNew(torch.nn.Module):
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
          1) LayerNorm (bfloat16 -> bfloat16)
          2) Spatial "shuffle" done as in original (PyTorch reshape/permute)
          3) First linear layer (Triton GEMM) -> float32
          4) GELU (Triton)
          5) Second linear layer (Triton GEMM) -> float32
          6) Return bfloat16 (cast at the end)
        """
        device = hidden.device
        # Step 1: Triton LayerNorm
        hidden_norm = triton_layer_norm(hidden, ln_weight, ln_bias, eps)

        # Step 2: Spatial "shuffle" to produce (num_merged_patches, hidden_size_expanded=6144).
        # We implement the exact logic as in the original to ensure correctness:
        # Construct per-grid THW from grid_thw, then reconstruct patches and reshape/permute.
        # Note: grid_thw shape is (num_grids, 3) with entries (T, H, W).
        num_grids = grid_thw.shape[0]
        num_patches = hidden_norm.shape[0]
        hidden_size = hidden_norm.shape[1]  # 1536
        hidden_expanded = 6144
        patches_per_grid = (num_patches // num_grids) if num_grids > 0 else num_patches

        shuffled_patches = []
        offset = 0
        for i in range(num_grids):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            num_patches_this = t * h * w
            if num_patches_this > 0:
                patches = hidden_norm[offset: offset + num_patches_this]
                h_merged = h // 2
                w_merged = w // 2
                patches = patches.view(t, h_merged, 2, w_merged, 2, hidden_size)
                patches = patches.permute(0, 1, 3, 2, 4, 5)  # (t, h_merged, w_merged, 2, 2, hidden_size)
                patches = patches.reshape(t * h_merged * w_merged, hidden_expanded)
                shuffled_patches.append(patches)
                offset += num_patches_this

        hidden_shuffled = torch.cat(shuffled_patches, dim=0)  # (num_merged_patches, hidden_expanded)

        # Step 3: First linear (Triton GEMM)
        # hidden_shuffled: (M, K) where M=num_merged_patches, K=hidden_expanded
        M = hidden_shuffled.shape[0]
        K = hidden_shuffled.shape[1]
        # fc1_weight is (K, N) where N=K=6144 for first linear
        A = hidden_shuffled.to(torch.float32)  # compute in float32
        Wt = fc1_weight.transpose(0, 1).contiguous()  # (N, K)
        out1 = triton_linear(A, Wt, fc1_bias)  # (M, N) float32

        # Step 4: GELU (Triton)
        out1_gelu = triton_gelu(out1)

        # Step 5: Second linear (Triton GEMM)
        # fc2_weight is (OUT_N, K) = (3584, 6144)
        Wt2 = fc2_weight.transpose(0, 1).contiguous()  # (K, OUT_N)
        out2 = triton_linear(out1_gelu, Wt2, fc2_bias)  # (M, OUT_N) float32

        # Step 6: Return bfloat16 to match original
        return out2.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
