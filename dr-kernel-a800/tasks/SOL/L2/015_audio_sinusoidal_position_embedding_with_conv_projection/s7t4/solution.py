import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def matmul_kernel_2d(
    x_ptr,           # *bf16 [M, K], row-major: stride_xm = K, stride_xk = 1
    wT_ptr,          # *bf16 [K, N], row-major: stride_wtk = N, stride_wtn = 1
    y_ptr,           # *bf16 [M, N], row-major: stride_ym = N, stride_yn = 1
    M, N, K,         # int32 sizes
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile id along M dimension
    pid_n = tl.program_id(1)  # tile id along N dimension

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)

    # Loop over K dimension in BLOCK_K chunks
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k  # [BLOCK_K]
        # Load A tile: [BLOCK_M, BLOCK_K] from x [M, K]
        a_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)
        # Load B tile: [BLOCK_K, BLOCK_N] from wT [K, N]
        b_ptrs = wT_ptr + k_idx[:, None] * stride_wtk + offs_n[None, :] * stride_wtn
        b = tl.load(b_ptrs, mask=(k_idx[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        # Accumulate
        acc += tl.dot(a, b)

    # Store results
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_linear_proj(x_rowwise: torch.Tensor, w_T: torch.Tensor) -> torch.Tensor:
    """
    x_rowwise: [B, S, K], bfloat16, CUDA
    w_T: [K, N], bfloat16, CUDA (conv_out_weight transposed: [3840, 1024])
    Returns y: [B, S, N], bfloat16
    """
    assert x_rowwise.is_cuda and w_T.is_cuda, "Triton kernel requires CUDA tensors"
    B, S, K = x_rowwise.shape
    K_w, N = w_T.shape
    assert K_w == K, "Transposed weight K must match x K dimension"

    # Ensure bfloat16 and contiguous
    x = x_rowwise.contiguous().to(torch.bfloat16)       # [B, S, K]
    wT = w_T.contiguous().to(torch.bfloat16)            # [K, N]
    M = B * S
    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)  # [B*S, N]

    # Strides
    stride_xm = K
    stride_xk = 1
    stride_wtk = N
    stride_wtn = 1
    stride_ym = N
    stride_yn = 1

    # Tile sizes: tuned for this problem size; can be adjusted later
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel_2d[grid](
        x.view(M, K), wT, y,
        M, N, K,
        stride_xm, stride_xk,
        stride_wtk, stride_wtn,
        stride_ym, stride_yn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )
    return y.view(B, S, N)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Args order matches the helper:
        # 0=input_features, 1-3 conv weights/bias, 4-6 conv2d2 weights/bias, 7-9 conv2d3 weights/bias,
        # 10=conv_out_weight, 11=positional_embedding, 12=embed_scale
        (
            input_features,
            conv2d1_weight, conv2d1_bias,
            conv2d2_weight, conv2d2_bias,
            conv2d3_weight, conv2d3_bias,
            conv_out_weight,
            positional_embedding,
            embed_scale
        ) = args

        device = input_features.device
        assert device.type == 'cuda', "ModelNew expects CUDA device"

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
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # [B, S, 3840]

        # Linear projection: y = x @ conv_out_weight.T -> [B, S, 1024]
        # conv_out_weight is [1024, 3840]; we pass its transpose [3840, 1024].
        conv_out_weight_T = conv_out_weight.t().contiguous().to(torch.bfloat16)  # [3840, 1024]
        y_row = triton_linear_proj(x.to(torch.bfloat16), conv_out_weight_T)     # [B, S, 1024]

        # Scale by embed_scale
        y_row = y_row * float(embed_scale)

        # Add positional embedding [S, 1024], broadcast over batch
        pos_emb = positional_embedding[:t, :].contiguous().to(torch.bfloat16)
        y_row = y_row + pos_emb  # broadcasting along batch dimension

        return y_row


def run(*args):
    return ModelNew()(*args)
