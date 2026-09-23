import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def conv2d_depthwise_kernel(
    residual_ptr,        # *f32, [B, C, H, W]
    weight_ptr,          # *f32, [C, 1, 7, 7]
    out_ptr,             # *f32, [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program ids
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_wblk = tl.program_id(2)

    # decode b, c
    b = pid_bc // C
    c = pid_bc % C
    h_out = pid_h

    # output spatial vector
    w_start = pid_wblk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W_out

    # accumulate over 7x7 kernel
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    # weight vector for channel c (kernel is per-channel)
    # weight is laid out as [C, 1, 7, 7] contiguous -> index = c * 49 + k
    for kh in range(7):
        for kw in range(7):
            weight_idx = c * 49 + kh * 7 + kw
            w_val = tl.load(weight_ptr + weight_idx)
            # input coordinates
            h_in = h_out + kh - PAD_H
            w_in = w_offsets - PAD_W
            in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask_w
            # compute linear index into residual[b, c, h_in, w_in]
            base = b * C * H * W + c * H * W + h_in * W + w_in
            val = tl.load(residual_ptr + base, mask=in_bounds, other=0.0)
            acc += val * w_val

    # store results
    out_base = b * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_offsets
    tl.store(out_ptr + out_base, acc, mask=mask_w)


@triton.jit
def linear_matmul_kernel(
    A_ptr,      # *f32, [M, K], contiguous
    B_ptr,      # *f32, [N, K], contiguous (we use B as is; result is A @ B^T)
    C_ptr,      # *f32, [M, N], contiguous
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # load A tiles: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * K + offs_k[None, :])
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # load B tiles as B^T: [BLOCK_K, BLOCK_N]
        # B is [N, K]; we want B^T[k, n] = B[n, k]
        b_ptrs = B_ptr + (offs_n[None, :] * K + offs_k[:, None])
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        B_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(A_tile, B_tile)

    # store C tile
    c_ptrs = C_ptr + (offs_m[:, None] * N + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def gelu_tanh_kernel(
    in_ptr,     # *f32, input flattened, length M
    out_ptr,    # *f32, output flattened, length M
    M: tl.constexpr,
):
    pid = tl.program_id(0)
    BLOCK = 1024
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    tanh_inner = tl.tanh(inner)
    y = 0.5 * x * (1.0 + tanh_inner)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def layernorm_mean_var_kernel(
    x_ptr,           # *f32, input [B, C, H, W] (we reduce over C for each (b,h,w))
    mean_ptr,        # *f32, [B, H, W]
    var_ptr,         # *f32, [B, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    # Launch over (B, H, W) and reduce over C
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # loop over channels C
    for c in range(C):
        base = pid_b * C * H * W + c * H * W + pid_h * W + pid_w
        val = tl.load(x_ptr + base)
        sum_val += val
        sum_sq += val * val

    mean = sum_val / C
    var = sum_sq / C - mean * mean

    mean_store = pid_b * H * W + pid_h * W + pid_w
    var_store = pid_b * H * W + pid_h * W + pid_w
    tl.store(mean_ptr + mean_store, mean)
    tl.store(var_ptr + var_store, var)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, residual, dwconv_weight, layernorm_weight, pwconv1_weight, grn_weight, pwconv2_weight, drop_mask, drop_path_prob, eps):
        # Dimensions
        B, C, H, W = residual.shape
        C4 = pwconv1_weight.shape[0]

        # 1) Depthwise conv via Triton
        x_dwconv = torch.empty((B, C, H, W), device=residual.device, dtype=residual.dtype)
        BLOCK_W = 128
        grid = (B * C, H, triton.cdiv(W, BLOCK_W))
        conv2d_depthwise_kernel[grid](
            residual, dwconv_weight, x_dwconv,
            B=B, C=C, H=H, W=W,
            H_out=H, W_out=W,
            PAD_H=3, PAD_W=3,
            BLOCK_W=BLOCK_W
        )

        # 2) Per-channel mean/var over spatial dims (we need NHWC permutation to compute across W).
        mean = torch.empty((B, H, W), device=residual.device, dtype=residual.dtype)
        var = torch.empty((B, H, W), device=residual.device, dtype=residual.dtype)
        grid_mean_var = (B, H, W)
        layernorm_mean_var_kernel[grid_mean_var](
            x_dwconv, mean, var,
            B=B, C=C, H=H, W=W
        )

        # 3) Normalize and apply per-channel layernorm_weight
        x_nhwc = x_dwconv.permute(0, 2, 3, 1)


def run(*args):
    return ModelNew()(*args)
