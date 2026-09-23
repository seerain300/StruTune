import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def linear_gelu_bf16_kernel(
    X_ptr,       # *ptr to input x, shape [M, K], contiguous (after reshape)
    W_ptr,       # *ptr to weight, shape [N, K], contiguous (conv_out_weight)
    Y_ptr,       # *ptr to output y, shape [M, N], contiguous
    M,           # int: number of rows (B*T)
    N,           # int: number of output features (d_model)
    K,           # int: input features (conv_out_dim)
    stride_xm,   # int: stride of X along M (normally K)
    stride_xk,   # int: stride of X along K (normally 1)
    stride_wk,   # int: stride of W along K (normally 1)
    stride_wn,   # int: stride of W along N (normally K)
    stride_ym,   # int: stride of Y along M
    stride_yn,   # int: stride of Y along N
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Tile indices
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in bfloat16
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)

    # Constants for GELU tanh approximation
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715

    # Loop over K dimension in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load x block [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        x_block = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.bfloat16)

        # Apply GELU tanh approximation in-register
        # gelu(x) = 0.5 * x * (1 + tanh(c0 * (x + c1 * x^3)))
        x_cubed = x_block * x_block * x_block
        inner = c0 * (x_block + c1 * x_cubed)
        gelu_x = 0.5 * x_block * (1.0 + tl.tanh(inner))

        # Load W block [BLOCK_K, BLOCK_N] as W[d, k]
        w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        w_block = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.bfloat16)

        # Accumulate: acc += gelu(x_block) @ w_block^T
        acc += tl.dot(gelu_x, w_block)

    # Store results
    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask)


def triton_linear_gelu_bf16(x_2d: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Compute y = GELU(x) @ W^T for x of shape [M, K] and W of shape [N, K], returning y of shape [M, N].
    x_2d: [B*T, K], contiguous, dtype bfloat16
    w: [N, K], contiguous, dtype bfloat16
    """
    assert x_2d.is_cuda and w.is_cuda, "Triton kernel requires CUDA tensors"
    assert x_2d.dtype == torch.bfloat16 and w.dtype == torch.bfloat16, "This kernel expects bfloat16 inputs"
    M = x_2d.shape[0]
    N = w.shape[0]
    K = w.shape[1]

    y = torch.empty((M, N), device=x_2d.device, dtype=x_2d.dtype)

    stride_xm = x_2d.stride(0)
    stride_xk = x_2d.stride(1)
    stride_wn = w.stride(0)
    stride_wk = w.stride(1)
    stride_ym = y.stride(0)
    stride_yn = y.stride(1)

    # Tile sizes; you can tune these based on your GPU and shapes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    linear_gelu_bf16_kernel[grid](
        x_2d, w, y,
        M, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=2,
    )
    return y


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Unpack inputs
        input_features = args[0]
        conv2d1_weight = args[1]
        conv2d1_bias = args[2]
        conv2d2_weight = args[3]
        conv2d2_bias = args[4]
        conv2d3_weight = args[5]
        conv3_bias = args[6]
        conv_out_weight = args[7]  # [d_model, conv_out_dim]
        positional_embedding = args[8]  # [max_source_positions, d_model]
        embed_scale = args[9]  # float

        # Stage 1: Conv2d (1 -> 384 channels) + GELU (PyTorch for correctness)
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        x = F.conv2d(x, conv2d3_weight, conv3_bias, stride=2, padding=1)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Ensure dtype is bfloat16 for Triton kernel
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        if conv_out_weight.dtype != torch.bfloat16:
            conv_out_weight = conv_out_weight.to(torch.bfloat16)

        # Flatten x to [M, K] with M=B*T
        B, T, K = x.shape
        D, K_w = conv_out_weight.shape
        assert K == K_w, "Mismatch between input features and weight K dimension"

        x_2d = x.reshape(-1, K).contiguous()

        # Triton fused GELU + linear: compute y[b, t, d] = sum_k GELU(x[b, t, k]) * conv_out_weight[d, k]
        y = triton_linear_gelu_bf16(x_2d, conv_out_weight)  # y shape: [B*T, D]

        # Reshape back to [B, T, D]
        y = y.view(B, T, D)

        # Scale embeddings
        y = y * embed_scale

        # Add positional embeddings
        seq_len = y.shape[1]  # time_after_conv
        pos_embed = positional_embedding[:seq_len, :].to(y.dtype).to(y.device).unsqueeze(0)  # [1, time_after_conv, d_model]
        y = y + pos_embed

        return y


def run(*args):
    return ModelNew()(*args)
