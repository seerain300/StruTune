"""
KDA candidate c004 — L1/002 VAE fused residual block.

Pipeline (matches PyTorch reference exactly):
    residual = x
    t1  = conv2d(x,  conv1_weight, stride=1, pad=1)        # 3x3, C=256
    t2  = silu(group_norm(t1, groups=32, w=norm1_w, b=norm1_b, eps))
    t3  = conv2d(t2, conv2_weight, stride=1, pad=1)
    out = silu(group_norm(t3, groups=32, w=norm2_w, b=norm2_b, eps)) + residual

All compute is done in Triton. PyTorch is used only for tensor metadata,
allocation, and kernel launch (no Torch/CPU/NumPy compute fallback).

--------------------------------------------------------------------------
c004 = uniform-compile-fault recovery (single lever) vs c003.
--------------------------------------------------------------------------
c001 (ieee), c002 (tf32) and c003 (no input_precision kwarg + host-side
inv_group_size + plain range loops) ALL failed identically: every one of the
five workloads returned RUNTIME_ERROR with max_abs == max_rel == 0.0, i.e.
no kernel ever produced output. That is a trace/compile-time fault, and its
uniformity (shape- and precision-independent) means it lives in a construct
shared, unchanged, by all three candidates — and it must be in the FIRST
kernel launched (the conv), since a later kernel's fault would still let the
conv write nonzero output (max_abs > 0) which we never observed.

The one such construct present, unchanged, in every candidate and unique to
the conv kernel is the **capture of the module-global Python int `_KS`
inside the `@triton.jit` body** (`range(0, _KS)`, `kh*_KS+kw`, `_KS*_KS`).
Free-variable capture of module globals inside a jitted kernel is fragile
across Triton builds (some resolve it, some raise at trace time). The
GroupNorm stats/apply kernels reference no globals — consistent with the
conv being the uniform faulting kernel.

c004's SINGLE lever: make the kernel-size a proper `tl.constexpr` kernel
parameter `KS` and remove ALL module-global capture from the conv jit body
(everything now flows through kernel arguments / constexprs). fp32 `tl.dot`
(no precision kwarg → native TF32 tensor-core path on sm_80) is unchanged,
as are the GroupNorm stats/apply kernels and all block sizes. If this still
fails identically, the fault is not global-capture and the next candidate
bisects `tl.dot` itself.
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
    C: tl.constexpr, KS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # flat output-pixel index
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # output channel index
    m_mask = offs_m < M
    n_mask = offs_n < C

    HW = H * W
    KK = KS * KS
    b = offs_m // HW
    rem = offs_m % HW
    oh = rem // W
    ow = rem % W

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for kh in range(0, KS):
        ih = oh + kh - 1
        ih_valid = (ih >= 0) & (ih < H)
        for kw in range(0, KS):
            iw = ow + kw - 1
            iw_valid = (iw >= 0) & (iw < W)
            hw_valid = ih_valid & iw_valid & m_mask
            base_x = b * (C * HW) + ih * W + iw          # [BLOCK_M], add ci*HW below
            tap = kh * KS + kw
            for k0 in range(0, C, BLOCK_K):
                cin = k0 + tl.arange(0, BLOCK_K)          # always < C (BLOCK_K | C)
                # X tile [BLOCK_M, BLOCK_K]
                x_ptrs = x_ptr + base_x[:, None] + cin[None, :] * HW
                xt = tl.load(x_ptrs, mask=hw_valid[:, None], other=0.0)
                # W tile [BLOCK_K, BLOCK_N]: weight[co, ci, kh, kw]
                w_ptrs = w_ptr + offs_n[None, :] * (C * KK) + cin[:, None] * KK + tap
                wt = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)
                acc += tl.dot(xt, wt)

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
    group_size, inv_group_size, eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * group_size

    acc_sum = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + start + idx, mask=mask, other=0.0)
        acc_sum += v
    mean = tl.sum(acc_sum) * inv_group_size

    acc_var = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + start + idx, mask=mask, other=0.0)
        d = tl.where(mask, v - mean, 0.0)
        acc_var += d * d
    var = tl.sum(acc_var) * inv_group_size
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


def _conv3x3(x, w, C, B, H, W):
    M = B * H * W
    out = torch.empty_like(x)
    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(C, _BLOCK_N))
    _conv3x3_kernel[grid](
        x, w, out,
        B, H, W, M,
        C=C, KS=_KS,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return out


def _gn_stats(t, B, C, H, W, eps):
    group_size = (C // _NUM_GROUPS) * H * W
    n_groups = B * _NUM_GROUPS
    mean = torch.empty((n_groups,), dtype=torch.float32, device=t.device)
    rstd = torch.empty((n_groups,), dtype=torch.float32, device=t.device)
    _gn_stats_kernel[(n_groups,)](
        t, mean, rstd,
        group_size, 1.0 / float(group_size), float(eps),
        BLOCK=_STATS_BLOCK, num_warps=4,
    )
    return mean, rstd, group_size


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

    grid_apply = (triton.cdiv(numel, _APPLY_BLOCK),)

    # ---- Path 1: conv1 -> GN1 -> SiLU ----
    t1 = _conv3x3(x, conv1_weight, C, B, H, W)
    mean1, rstd1, group_size = _gn_stats(t1, B, C, H, W, eps_f)
    t2 = torch.empty_like(x)
    _gn_apply_silu_kernel[grid_apply](
        t1, mean1, rstd1, norm1_weight, norm1_bias, t2, t1,
        numel, group_size, HW,
        C=C, ADD_RES=False, BLOCK=_APPLY_BLOCK, num_warps=4,
    )

    # ---- Path 2: conv2 -> GN2 -> SiLU -> +residual ----
    t3 = _conv3x3(t2, conv2_weight, C, B, H, W)
    mean2, rstd2, group_size2 = _gn_stats(t3, B, C, H, W, eps_f)
    out = torch.empty_like(x)
    _gn_apply_silu_kernel[grid_apply](
        t3, mean2, rstd2, norm2_weight, norm2_bias, out, x,
        numel, group_size2, HW,
        C=C, ADD_RES=True, BLOCK=_APPLY_BLOCK, num_warps=4,
    )

    return out
