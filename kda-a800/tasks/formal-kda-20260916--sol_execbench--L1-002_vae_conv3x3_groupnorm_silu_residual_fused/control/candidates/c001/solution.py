"""
KDA candidate c001 — L1/002 VAE fused residual block.

Pipeline (matches PyTorch reference exactly):
    residual = x
    t1  = conv2d(x,  conv1_weight, stride=1, pad=1)        # 3x3, C=256
    t2  = silu(group_norm(t1, groups=32, w=norm1_w, b=norm1_b, eps))
    t3  = conv2d(t2, conv2_weight, stride=1, pad=1)
    out = silu(group_norm(t3, groups=32, w=norm2_w, b=norm2_b, eps)) + residual

All compute is done in Triton. PyTorch is used only for tensor metadata,
allocation, and kernel launch (no Torch/CPU/NumPy compute fallback).

c001 = correctness anchor:
  - implicit-GEMM 3x3 conv, NCHW, FP32 accumulate, input_precision="ieee"
  - two-pass biased-variance GroupNorm stats, one program per (batch, group)
  - fused GN-apply + SiLU (+ residual add on the second GroupNorm)
  - conservative fixed block sizes (no autotune yet)
"""

import torch
import triton
import triton.language as tl

_NUM_GROUPS = 32
_KS = 3

# ---------------------------------------------------------------------------
# Convolution: implicit GEMM over the 9 taps of a 3x3 kernel, stride=1, pad=1.
# Output tile [BLOCK_M output pixels, BLOCK_N output channels]; K = C_in.
# ---------------------------------------------------------------------------
@triton.jit
def _conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    B, H, W, M,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # flat output-pixel index
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # output channel index
    m_mask = offs_m < M
    n_mask = offs_n < C

    HW = H * W
    b = offs_m // HW
    rem = offs_m % HW
    oh = rem // W
    ow = rem % W

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for kh in tl.static_range(0, _KS):
        ih = oh + kh - 1
        ih_valid = (ih >= 0) & (ih < H)
        for kw in tl.static_range(0, _KS):
            iw = ow + kw - 1
            iw_valid = (iw >= 0) & (iw < W)
            hw_valid = ih_valid & iw_valid & m_mask
            base_x = b * (C * HW) + ih * W + iw          # [BLOCK_M], add ci*HW below
            tap = kh * _KS + kw
            for k0 in tl.static_range(0, C, BLOCK_K):
                cin = k0 + tl.arange(0, BLOCK_K)          # always < C (BLOCK_K | C)
                # X tile [BLOCK_M, BLOCK_K]
                x_ptrs = x_ptr + base_x[:, None] + cin[None, :] * HW
                xt = tl.load(x_ptrs, mask=hw_valid[:, None], other=0.0)
                # W tile [BLOCK_K, BLOCK_N]: weight[co, ci, kh, kw]
                w_ptrs = w_ptr + offs_n[None, :] * (C * _KS * _KS) + cin[:, None] * (_KS * _KS) + tap
                wt = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)
                acc += tl.dot(xt, wt, input_precision=IP)

    out_ptrs = out_ptr + b[:, None] * (C * HW) + offs_n[None, :] * HW + (oh * W + ow)[:, None]
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# GroupNorm statistics: one program per (batch, group).
# Groups of C/G=8 consecutive channels are contiguous in NCHW memory, so a
# group is a contiguous block of length 8*H*W starting at pid * group_size.
# Two-pass (mean, then Sum (x-mean)^2) for numerical safety; biased variance.
# ---------------------------------------------------------------------------
@triton.jit
def _gn_stats_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    group_size, eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * group_size
    gs_f = group_size.to(tl.float32)

    acc_sum = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + start + idx, mask=mask, other=0.0)
        acc_sum += v
    mean = tl.sum(acc_sum) / gs_f

    acc_var = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + start + idx, mask=mask, other=0.0)
        d = tl.where(mask, v - mean, 0.0)
        acc_var += d * d
    var = tl.sum(acc_var) / gs_f
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


# ---------------------------------------------------------------------------
# GroupNorm apply + SiLU (+ optional residual add), fully elementwise.
#   y = (v - mean_g) * rstd_g * weight[c] + bias[c]
#   y = y * sigmoid(y)            (SiLU)
#   out = y (+ residual)          (residual = original x on the 2nd GroupNorm)
# ---------------------------------------------------------------------------
@triton.jit
def _gn_apply_silu_kernel(
    t_ptr, mean_ptr, rstd_ptr, w_ptr, b_ptr, out_ptr, res_ptr,
    numel, group_size, HW,
    C: tl.constexpr, ADD_RES: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < numel

    gid = idx // group_size            # (batch, group) statistic index
    c = (idx // HW) % C                # channel for affine params

    v = tl.load(t_ptr + idx, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + gid, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + gid, mask=mask, other=0.0)
    wc = tl.load(w_ptr + c, mask=mask, other=0.0)
    bc = tl.load(b_ptr + c, mask=mask, other=0.0)

    y = (v - mean) * rstd * wc + bc
    y = y * tl.sigmoid(y)
    if ADD_RES:
        r = tl.load(res_ptr + idx, mask=mask, other=0.0)
        y = y + r
    tl.store(out_ptr + idx, y, mask=mask)


# ---------------------------------------------------------------------------
# Host-side orchestration (metadata / allocation / launch only).
# ---------------------------------------------------------------------------
_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 64
_STATS_BLOCK = 1024
_APPLY_BLOCK = 1024
_IP = "ieee"          # c001: full FP32 accumulation, no TF32 rounding


def _conv3x3(x, w, C, B, H, W):
    M = B * H * W
    out = torch.empty_like(x)
    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(C, _BLOCK_N))
    _conv3x3_kernel[grid](
        x, w, out,
        B, H, W, M,
        C=C,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        IP=_IP,
        num_warps=4, num_stages=2,
    )
    return out


def _gn_stats(t, B, C, H, W):
    group_size = (C // _NUM_GROUPS) * H * W
    n_groups = B * _NUM_GROUPS
    mean = torch.empty((n_groups,), dtype=torch.float32, device=t.device)
    rstd = torch.empty((n_groups,), dtype=torch.float32, device=t.device)
    return mean, rstd, group_size, n_groups


def run(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    conv1_weight = conv1_weight.contiguous()
    conv2_weight = conv2_weight.contiguous()
    norm1_weight = norm1_weight.contiguous()
    norm1_bias = norm1_bias.contiguous()
    norm2_weight = norm2_weight.contiguous()
    norm2_bias = norm2_bias.contiguous()

    B, C, H, W = x.shape
    HW = H * W
    numel = B * C * H * W
    eps_f = float(eps)

    # ---- Path 1: conv1 -> GN1 -> SiLU ----
    t1 = _conv3x3(x, conv1_weight, C, B, H, W)
    mean1, rstd1, group_size, _ = _gn_stats(t1, B, C, H, W)
    _gn_stats_kernel[(B * _NUM_GROUPS,)](
        t1, mean1, rstd1, group_size, eps_f, BLOCK=_STATS_BLOCK, num_warps=4,
    )
    t2 = torch.empty_like(x)
    grid_apply = (triton.cdiv(numel, _APPLY_BLOCK),)
    _gn_apply_silu_kernel[grid_apply](
        t1, mean1, rstd1, norm1_weight, norm1_bias, t2, t1,
        numel, group_size, HW,
        C=C, ADD_RES=False, BLOCK=_APPLY_BLOCK, num_warps=4,
    )

    # ---- Path 2: conv2 -> GN2 -> SiLU -> +residual ----
    t3 = _conv3x3(t2, conv2_weight, C, B, H, W)
    mean2, rstd2, group_size2, _ = _gn_stats(t3, B, C, H, W)
    _gn_stats_kernel[(B * _NUM_GROUPS,)](
        t3, mean2, rstd2, group_size2, eps_f, BLOCK=_STATS_BLOCK, num_warps=4,
    )
    out = torch.empty_like(x)
    _gn_apply_silu_kernel[grid_apply](
        t3, mean2, rstd2, norm2_weight, norm2_bias, out, x,
        numel, group_size2, HW,
        C=C, ADD_RES=True, BLOCK=_APPLY_BLOCK, num_warps=4,
    )

    return out
