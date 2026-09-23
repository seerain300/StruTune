import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: batched matrix-vector multiplication.
# Computes Y[n, t, k] = sum_j X[n, t, j] * W[k, j] for all n, t, k.
# X: [N, T, M], W: [K, M], Y: [N, T, K]
# We launch one program per (n, t) row and loop over M and K in tiles.
@triton.jit
def linear_bmm_kernel(
    X_ptr,         # *const T (bf16) pointer to input matrix X [N, T, M]
    W_ptr,         # *const T (bf16) pointer to weight matrix W [K, M]
    Y_ptr,         # *T (bf16) pointer to output Y [N, T, K]
    N, T, M, K,    # int32 sizes
    x_stride0, x_stride1, x_stride2,  # strides for X
    w_stride0, w_stride1,             # strides for W (rows, cols)
    y_stride0, y_stride1, y_stride2,  # strides for Y
    BLOCK_M: tl.constexpr,            # tile size over M
    BLOCK_K: tl.constexpr,            # tile size over K
):
    # program id: one instance per (n, t) row
    pid = tl.program_id(axis=0)
    n = pid // T
    t = pid % T

    # If pid >= N*T, we should mask out, but here we assume grid matches N*T
    # Base offsets for this row
    # X offset for (n, t, 0): n*x_stride0 + t*x_stride1
    # We'll compute row base
    # Note: Triton indexing uses element strides, not bytes, which is fine.

    # Accumulator for this (n, t) row, across K
    # We'll keep in float32 for numeric stability.
    # We use a vector acc[BLOCK_K] and loop over K in steps of BLOCK_K
    # Initialize accumulator to zeros
    # We'll iterate over K tiles and accumulate into a vector 'acc'

    # Outer loop over K in tiles of BLOCK_K
    for ko in range(0, K, BLOCK_K):
        offs_k = ko + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        # Initialize acc for this tile
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)

        # Loop over M in tiles of BLOCK_M
        for m0 in range(0, M, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < M

            # Load X[n, t, offs_m] as bf16, convert to float32
            x_ptrs = X_ptr + n * x_stride0 + t * x_stride1 + offs_m[None, :] * x_stride2
            x_vals = tl.load(x_ptrs, mask=mask_m[None, :], other=0.0)
            x_vals = x_vals.to(tl.float32)  # [1, BLOCK_M]

            # Load W[offs_k, offs_m] as bf16, convert to float32
            w_ptrs = W_ptr + offs_k[:, None] * w_stride0 + offs_m[None, :] * w_stride1
            w_mask = mask_k[:, None] & mask_m[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
            w_vals = w_vals.to(tl.float32)  # [BLOCK_K, BLOCK_M]

            # Accumulate: acc += sum over m of (x_vals * w_vals), i.e., acc += w_vals @ x_vals.T
            # x_vals is [1, BM]; w_vals is [BK, BM]; we want [BK, 1]
            # We can multiply and reduce over axis=1 (BM dimension)
            # acc_k is [BK]
            acc += tl.sum(w_vals * x_vals, axis=1)

        # Store results to Y[n, t, offs_k]
        y_ptrs = Y_ptr + n * y_stride0 + t * y_stride1 + offs_k * y_stride2
        # Cast back to original dtype (bf16) for output
        y_vals = acc.to(tl.bfloat16)
        tl.store(y_ptrs, y_vals, mask=mask_k)


def triton_linear_bmm(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    X: [N, T, M], W: [K, M], returns Y: [N, T, K]
    Uses Triton kernel to compute Y = X @ W^T, elementwise across (N, T).
    Accumulates in float32, outputs in bf16 to match original code's dtype.
    """
    assert X.is_cuda and W.is_cuda, "Triton kernel requires CUDA tensors"
    assert X.dtype == torch.bfloat16 and W.dtype == torch.bfloat16, "Inputs must be bfloat16"

    N, T, M = X.shape
    K, Mw = W.shape
    assert M == Mw, "Mismatch in M dimension for X and W"

    # Allocate output
    Y = torch.empty((N, T, K), device=X.device, dtype=torch.bfloat16)

    # Choose tile sizes. Since M=15360, K=1024, we can set BLOCK_M=256 and BLOCK_K=128
    # Grid: one program per (n, t) row
    grid = (N * T,)

    # Strides (in elements)
    x_stride0, x_stride1, x_stride2 = X.stride()
    w_stride0, w_stride1 = W.stride()
    y_stride0, y_stride1, y_stride2 = Y.stride()

    # Heuristic: num_warps based on K (small), maybe 4 or 8
    num_warps = 4

    linear_bmm_kernel[grid](
        X, W, Y,
        N, T, M, K,
        x_stride0, x_stride1, x_stride2,
        w_stride0, w_stride1,
        y_stride0, y_stride1, y_stride2,
        BLOCK_M=256, BLOCK_K=128,
        num_warps=num_warps,
    )
    return Y


# Triton kernel for elementwise addition of positional embeddings:
# Y: [N, T, K], add position t slice: pos_emb[t, :] for each row
@triton.jit
def add_pos_embed_kernel(
    Y_ptr,          # *T (bf16) pointer to output [N, T, K]
    pos_ptr,        # *T (bf16) pointer to positional embedding [T, K]
    N, T, K,        # int32 sizes
    y_stride0, y_stride1, y_stride2,  # strides for Y
    p_stride0, p_stride1,             # strides for pos (rows, cols)
    BLOCK_T: tl.constexpr,            # tile size over T
    BLOCK_K: tl.constexpr,            # tile size over K
):
    pid = tl.program_id(axis=0)  # one program per row (n, t)
    n = pid // T
    t = pid % T

    # Tile over K dimension
    for ko in range(0, K, BLOCK_K):
        offs_k = ko + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

    # We can write a 2D grid with axis0 over N*T and axis1 over T tiles,
    # but for simplicity, we loop over T in the kernel. Since T varies, we keep
    # one program per (n, t) and compute that row. The grid ensures we cover all (n, t).
    # For safety, we compute only for the current (n, t) row.

    # Load pos[t, :] as bf16 and add to Y[n, t, :]
    # Iterate K in tiles
    for ko in range(0, K, BLOCK_K):
        offs_k = ko + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        y_ptrs = Y_ptr + n * y_stride0 + t * y_stride1 + offs_k * y_stride2
        y_vals = tl.load(y_ptrs, mask=mask_k, other=0.0)  # bf16
        # pos[t, offs_k]
        pos_ptrs = pos_ptr + t * p_stride0 + offs_k * p_stride1
        pos_vals = tl.load(pos_ptrs, mask=mask_k, other=0.0)  # bf16
        # Add in float32 for stability
        y_vals = y_vals.to(tl.float32) + pos_vals.to(tl.float32)
        y_vals = y_vals.to(tl.bfloat16)
        tl.store(y_ptrs, y_vals, mask=mask_k)


def triton_add_pos_embed(Y: torch.Tensor, pos_emb: torch.Tensor):
    """
    Y: [N, T, K] (bf16), pos_emb: [T, K] (bf16)
    Add pos_emb[t, :] to each row Y[n, t, :].
    """
    assert Y.is_cuda and pos_emb.is_cuda, "Triton kernel requires CUDA tensors"
    assert Y.dtype == torch.bfloat16 and pos_emb.dtype == torch.bfloat16, "Inputs must be bfloat16"

    N, T, K = Y.shape
    # Grid: one program per (n, t) row
    grid = (N * T,)

    y_stride0, y_stride1, y_stride2 = Y.stride()
    p_stride0, p_stride1 = pos_emb.stride()

    # Choose BLOCK sizes
    BLOCK_T = 1  # one t per program
    BLOCK_K = 128  # tile over K

    add_pos_embed_kernel[grid](
        Y, pos_emb,
        N, T, K,
        y_stride0, y_stride1, y_stride2,
        p_stride0, p_stride1,
        BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K,
        num_warps=2,
    )
    return Y


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Expect the same 10 args as original: input_features, conv weights and biases,
        # conv_out_weight, positional_embedding, embed_scale (float).
        # Note: We will not use Triton for convs and permutation in this concise version.
        # We will use Triton for the linear projection and final addition.
        # Extract arguments (match original signature)
        # args[0] is input_features
        input_features = args[0]
        conv2d1_weight = args[1]
        conv2d1_bias = args[2]
        conv2d2_weight = args[3]
        conv2d2_bias = args[4]
        conv2d3_weight = args[5]
        conv2d3_bias = args[6]
        conv_out_weight = args[7]
        positional_embedding = args[8]
        embed_scale = args[9]

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Ensure conv_out_weight is on same device and dtype
        # The original code uses xavier to create conv_out_weight of [d_model, conv_out_dim]
        # But in this setup, conv_out_weight is provided and is [1024, 3840].
        # We will use it as-is. Triton will expect bf16; the original uses bfloat16 in get_inputs().
        # We'll cast to bf16 if necessary (but provided tensors are bf16 already).
        # Compute linear projection with Triton
        # If Triton is not available, fall back to torch linear.
        if TRITON_AVAILABLE and x.is_cuda and conv_out_weight.is_cuda:
            # Ensure dtypes: bfloat16
            x_in = x
            # Linear via Triton: Y = x_in @ conv_out_weight.T
            # Shapes: x_in [N, T, 15360], conv_out_weight [1024, 3840], we need W^T [3840, 1024].
            # But conv_out_weight is [K=1024, M=3840]. So we can pass it as-is.
            # The kernel expects X[M], W[K, M], so conv_out_weight is already in the right shape.
            # Important: x_in should be contiguous along last dim for best performance.
            x_in = x_in.contiguous()
            Y = triton_linear_bmm(x_in, conv_out_weight)
        else:
            # Fallback: PyTorch linear
            Y = F.linear(x, conv_out_weight)

        # Scale embeddings
        Y = Y * embed_scale

        # Add positional embeddings (slice to T and add)
        # positional_embedding is [max_source_positions, d_model] = [1500, 1024]
        # Our T varies. We need pos_emb[:T, :], i.e., [:Y.shape[1], :]
        # Since positional_embedding is contiguous, slicing is fine.
        pos_slice = positional_embedding[:Y.shape[1], :]
        if TRITON_AVAILABLE and Y.is_cuda and pos_slice.is_cuda:
            # Add in Triton
            Y = triton_add_pos_embed(Y, pos_slice)
        else:
            # Fallback: PyTorch add
            Y = Y + pos_slice.to(Y.dtype)

        return Y


def run(*args):
    return ModelNew()(*args)
