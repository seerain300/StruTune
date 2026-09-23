import math
import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr,           # *ptr input (N, H) bfloat16
    y_ptr,           # *ptr output (N, H) bfloat16
    ln_weight_ptr,   # *ptr (H) bfloat16
    ln_bias_ptr,     # *ptr (H) bfloat16
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

    # Pass 1: sum
    sum_ = 0.0
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_ += tl.sum(x, axis=0)
    mean = sum_ / H

    # Pass 2: variance
    var_sum = 0.0
    for off in range(0, H, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        var_sum += tl.sum((x - mean) * (x - mean), axis=0)
    var = var_sum / H
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 3: normalize and apply affine, store bfloat16
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
def linear_kernel(
    A_ptr,           # *ptr A (M, K) float32
    B_ptr,           # *ptr B (N, K) float32 (second dim is K, i.e., rows=6144, cols=6144 or 3584)
    Bias_ptr,        # *ptr Bias (N) float32
    C_ptr,           # *ptr output (M, N) float32
    M,               # rows of A
    K,               # K dimension of A/B
    N,               # cols of output
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for off_k in range(0, K, BLOCK_K):
        k_idx = off_k + tl.arange(0, BLOCK_K)
        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + off_m[:, None] * K + k_idx[None, :]
        a_mask = (off_m[:, None] < M) & (k_idx[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N] where B is (N, K)
        b_ptrs = B_ptr + off_n[None, :] * K + k_idx[:, None]
        b_mask = (k_idx[:, None] < K) & (off_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(Bias_ptr + off_n, mask=off_n < N, other=0.0)  # (BLOCK_N,)
    acc = acc + bias[None, :]

    # Write back
    c_ptrs = C_ptr + off_m[:, None] * N + off_n[None, :]
    c_mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def gelu_kernel(
    x_ptr,           # *ptr input (M, N) float32
    y_ptr,           # *ptr output (M, N) float32
    M,               # rows
    N,               # cols
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (off_m[:, None] < M) & (off_n[None, :] < N)

    x = tl.load(x_ptr + off_m[:, None] * N + off_n[None, :], mask=mask, other=0.0)
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    tl.store(y_ptr + off_m[:, None] * N + off_n[None, :], y, mask=mask)


def triton_layer_norm(hidden: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float) -> torch.Tensor:
    # hidden: (num_patches, hidden_size) bfloat16
    # ln_weight, ln_bias: (hidden_size) bfloat16
    N, H = hidden.shape
    y = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)
    # Choose a block size; 256 works well for H=1536
    BLOCK_SIZE = 256
    grid = (N,)
    layer_norm_kernel[grid](
        hidden, y, ln_weight, ln_bias, N, H, eps, BLOCK_SIZE
    )
    return y


def triton_linear(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor, BLOCK_M: int = 128, BLOCK_N: int = 128, BLOCK_K: int = 64) -> torch.Tensor:
    """
    Compute C = A @ W^T + bias, where:
      - A: (M, K) float32
      - W: (N, K) float32  (i.e., original weight shape, we will use it as B = (N, K))
      - bias: (N) float32
      Output: (M, N) float32
    """
    M, K = A.shape
    N_w, K_w = W.shape
    assert K == K_w, "A's K must match W's K"
    # W^T would be (K, N), but we can treat W as (N, K) and in the kernel load it as (BLOCK_K, BLOCK_N)
    C = torch.empty((M, N_w), dtype=torch.float32, device=A.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_w, BLOCK_N))
    linear_kernel[grid](A, W, bias, C, M, K, N_w, BLOCK_M, BLOCK_N, BLOCK_K)
    return C


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply GELU to x (M, N) float32 in Triton.
    Returns y (M, N) float32.
    """
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.float32, device=x.device)
    BLOCK_M = 128
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gelu_kernel[grid](x, y, M, N, BLOCK_M, BLOCK_N)
    return y


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect the same signature as the original run: (hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps)
        hidden = args[0]
        grid_thw = args[1]
        ln_weight = args[2]
        ln_bias = args[3]
        fc1_weight = args[4]  # (6144, 6144)
        fc1_bias = args[5]    # (6144,)
        fc2_weight = args[6]  # (3584, 6144)
        fc2_bias = args[7]    # (3584,)
        eps = args[8]

        device = hidden.device
        assert hidden.is_cuda, "Inputs must be on CUDA for Triton kernels"
        hidden = hidden.contiguous()

        # Step 1: Triton LayerNorm over each patch vector (num_patches x 1536)
        hidden_ln = triton_layer_norm(hidden, ln_weight.to(torch.bfloat16).contiguous(), ln_bias.to(torch.bfloat16).contiguous(), eps)

        # Step 2: Spatial shuffle (data movement using PyTorch, no torch.cat)
        # Build per-grid tensors and concatenate them in PyTorch without cat
        # We need to reconstruct the grid shape for each grid from grid_thw (shape: num_grids x 3)
        patches_per_grid = grid_thw.shape[0]  # Not needed; grid_thw already provides per-grid t,h,w

        per_grid_tensors = []
        offset = 0
        for i in range(grid_thw.shape[0]):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            num_patches_this = t * h * w
            patch_chunk = hidden_ln[offset: offset + num_patches_this]
            # Reshape to (T, H/2, 2, W/2, 2, C), then permute to (T, H/2, W/2, 2, 2, C)
            h_merged = h // 2
            w_merged = w // 2
            patch_chunk = patch_chunk.view(t, h_merged, 2, w_merged, 2, 1536)
            patch_chunk = patch_chunk.permute(0, 1, 3, 2, 4, 5)  # (t, h_merged, w_merged, 2, 2, 1536)
            # Flatten to (t * h_merged * w_merged, 4 * 1536) = (t * h_merged * w_merged, 6144)
            patch_chunk = patch_chunk.reshape(t * h_merged * w_merged, 6144)
            per_grid_tensors.append(patch_chunk)
            offset += num_patches_this

        # Now we have a list of tensors, each of shape (num_patches_i, 6144).
        # The original code concatenates them to get hidden_shuffled. Here we keep them in a list because
        # we will compute the first linear on each independently. The final output is independent of this
        # concatenation, since we will compute two linear layers on each per-grid tensor and return a tensor
        # of shape (num_merged_patches, 3584). We will infer num_merged_patches from the sum of all
        # per-grid row counts, but since we need to return exactly the shape the original would produce,
        # we assume that the caller provides grid_thw such that total rows equals num_merged_patches.
        # To avoid torch.cat, we compute outputs per grid and then combine via PyTorch ops using indexing.
        # For simplicity in this environment, we assume the caller provides correct grid_thw and num_merged_patches.
        # Here, we proceed by computing the first linear for each tensor and then GELU and second linear.
        # We'll return a tensor of shape (num_merged_patches, 3584).

        # However, we need to know num_merged_patches from inputs. The original function signature doesn't
        # pass num_merged_patches explicitly, but get_inputs builds grid_thw and returns tensors. Since we
        # cannot query it here, we will infer it from per_grid_tensors:
        # total_num_merged = sum(tensor.shape[0] for tensor in per_grid_tensors)
        total_num_merged = sum(t.shape[0] for t in per_grid_tensors)
        output = torch.empty((total_num_merged, fc2_weight.shape[0]), dtype=torch.float32, device=device)

        idx_start = 0
        # For each per-grid tensor, compute linear1 -> GELU -> linear2 and write into output sequentially
        for pt in per_grid_tensors:
            # First linear: pt (M, 6144) @ fc1_weight.T (6144, 6144) + fc1_bias (6144)
            # We need fc1_weight.T (6144, 6144)
            W1_T = fc1_weight.T.contiguous()  # (6144, 6144), float32
            bias1 = fc1_bias.to(torch.float32).contiguous()
            # Convert pt to float32 for compute
            pt_f32 = pt.to(torch.float32).contiguous()
            y1 = triton_linear(pt_f32, W1_T, bias1)  # (M, 6144), float32
            # GELU in Triton
            y1_gelu = triton_gelu(y1)  # (M, 6144), float32
            # Second linear: y1_gelu (M, 6144) @ fc2_weight (3584, 6144)^T + fc2_bias (3584)
            W2_T = fc2_weight.T.contiguous()  # (6144, 3584), float32
            bias2 = fc2_bias.to(torch.float32).contiguous()
            y2 = triton_linear(y1_gelu, W2_T, bias2)  # (M, 3584), float32
            # Write into output
            output[idx_start: idx_start + y2.shape[0], :] = y2
            idx_start += y2.shape[0]

        # Return output in bfloat16 to match original signature expectations
        return output.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
