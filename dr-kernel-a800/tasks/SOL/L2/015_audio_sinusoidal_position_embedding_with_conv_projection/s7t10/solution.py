import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton GEMM: X_rowwise [M, K] dot WT [K, N] -> Y [M, N]
if TRITON_AVAILABLE:
    @triton.jit
    def matmul_linear_kernel(X_ptr, WT_ptr, Y_ptr,
                              M, K, N,
                              stride_xm, stride_xk,
                              stride_wtk, stride_wtn,
                              stride_ym, stride_yn,
                              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_N
        offs_n = pid_n * BLOCK_K

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.bfloat16)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            # X tile: [BLOCK_N, BLOCK_K]
            x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            # WT tile: [BLOCK_K, BLOCK_N], WT[k, n] = WT_ptr[k, n]
            wt_ptrs = WT_ptr + offs_k[:, None] * stride_wtk + offs_n[None, :] * stride_wtn
            wt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)
            acc += tl.dot(x, wt)

        y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(y_ptrs, acc, mask=y_mask)


    @triton.jit
    def scale_elementwise_kernel(Y_flat_ptr, scale, N_elems: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * 1024 + tl.arange(0, 1024)
        mask = offs < N_elems
        y = tl.load(Y_flat_ptr + offs, mask=mask, other=0.0)
        y = y * scale
        tl.store(Y_flat_ptr + offs, y, mask=mask)


    @triton.jit
    def add_pos_emb_kernel(Y_flat_ptr, pos_flat_ptr, N_elems: tl.constexpr, S: tl.constexpr, N: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * 1024 + tl.arange(0, 1024)
        mask = offs < N_elems
        s = offs // N
        n = offs % N
        pos_ptrs = pos_flat_ptr + s * N + n
        pos = tl.load(pos_ptrs, mask=mask, other=0.0)
        y = tl.load(Y_flat_ptr + offs, mask=mask, other=0.0)
        y = y + pos
        tl.store(Y_flat_ptr + offs, y, mask=mask)


def triton_linear_gemv(x_rowwise: torch.Tensor, w: torch.Tensor, embed_scale: float) -> torch.Tensor:
    """
    x_rowwise: [B, S, K], bfloat16, CUDA
    w: [N, K], bfloat16, CUDA (conv_out_weight)
    Returns y: [B, S, N], bfloat16
    """
    B, S, K = x_rowwise.shape
    N = w.shape[0]
    assert w.shape[1] == K, "Weight K must match x K dimension"
    x = x_rowwise.contiguous()       # [B*S, K]
    w_t = w.contiguous()             # [N, K]; WT[k, n] = w_t[n, k]

    y_flat = torch.empty((B * S * N,), device=x.device, dtype=torch.bfloat16)

    # Strides for X: row-major, last dim contiguous
    stride_xm = x.stride(0)  # typically K
    stride_xk = x.stride(1)  # typically 1

    # WT [N, K]; access WT[k, n] via strides
    stride_wtk = w_t.stride(1)  # typically 1
    stride_wtn = w_t.stride(0)  # typically K

    # Y is [B*S, N] flattened: row-major, stride_ym = N, stride_yn = 1
    stride_ym = N
    stride_yn = 1

    M = B * S
    BLOCK_N = 128
    BLOCK_K = 128
    grid = (triton.cdiv(M, BLOCK_N), triton.cdiv(N, BLOCK_K))
    matmul_linear_kernel[grid](
        x, w_t, y_flat,
        M, K, N,
        stride_xm, stride_xk,
        stride_wtk, stride_wtn,
        stride_ym, stride_yn,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )

    y = y_flat.view(B, S, N)
    return y


def triton_scale_inplace(y_flat: torch.Tensor, scale: float, N_elems: int):
    grid = (triton.cdiv(N_elems, 1024),)
    scale_elementwise_kernel[grid](y_flat, scale, N_elems=N_elems, num_warps=4, num_stages=2)


def triton_add_pos_emb_inplace(y_flat: torch.Tensor, pos_flat: torch.Tensor, S: int, N: int, N_elems: int):
    grid = (triton.cdiv(N_elems, 1024),)
    add_pos_emb_kernel[grid](y_flat, pos_flat, N_elems=N_elems, S=S, N=N, num_warps=4, num_stages=2)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
        # conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        input_features = args[0]
        conv2d1_weight = args[1]
        conv2d1_bias = args[2]
        conv2d2_weight = args[3]
        conv2d2_bias = args[4]
        conv2d3_weight = args[5]
        conv2d3_bias = args[6]
        conv_out_weight = args[7]  # [N=1024, K=3840]
        positional_embedding = args[8]  # [time_after_conv, 1024], bfloat16
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

        # Flatten to [B, S, K] for linear GEMM
        B, S, K = x.shape  # S = time_after_conv, K = 3840

        # Triton GEMM: x_rowwise [B*S, K] @ conv_out_weight^T [K, N] -> [B*S, N]
        y = triton_linear_gemv(x, conv_out_weight, embed_scale)  # y: [B, S, N]

        # Flatten for elementwise ops
        y_flat = y.view(-1)  # [B*S*N]
        N_elems = y_flat.numel()

        # Triton elementwise scale in-place
        triton_scale_inplace(y_flat, float(embed_scale), N_elems)

        # Triton add positional embedding (broadcast over batch). pos_emb is [S, N].
        pos_flat = positional_embedding[:S, :].contiguous().view(-1)  # [S*N]
        triton_add_pos_emb_inplace(y_flat, pos_flat, S, 1024, N_elems)

        # Reshape back to [B, S, 1024]
        y = y_flat.view(B, S, 1024)

        return y


def run(*args):
    return ModelNew()(*args)
