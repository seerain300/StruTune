"""
KDA candidate c006 — L1/002 VAE fused residual block.

Pipeline (matches PyTorch reference exactly):
    residual = x
    t1  = conv2d(x,  conv1_weight, stride=1, pad=1)        # 3x3, C=256
    t2  = silu(group_norm(t1, groups=32, w=norm1_w, b=norm1_b, eps))
    t3  = conv2d(t2, conv2_weight, stride=1, pad=1)
    out = silu(group_norm(t3, groups=32, w=norm2_w, b=norm2_b, eps)) + residual

All compute is done in Triton. PyTorch is used only for tensor metadata,
allocation, and kernel launch (no Torch/CPU/NumPy compute fallback).

--------------------------------------------------------------------------
c006 = weight prepermute (single lever) vs the c005 best-parent.
--------------------------------------------------------------------------
c005 (conv autotune) reached geomean 0.4183x (5/5 pass). The conv is still the
bottleneck. In c005 the weight tile for tap (kh,kw) is loaded from the native
`(Cout, Cin, 3, 3)` layout with pointer math
    w_ptr + cout*(C*KK) + cin*KK + tap ,
so within a W tile [cin, cout] the `cout` (BLOCK_N) columns are strided by
C*KK = 2304 and the `cin` (BLOCK_K) rows are strided by KK = 9 — i.e. the tile
is fully uncoalesced. Every output M-tile re-reads the entire 2.4 MB weight
(from L2), and the strided pattern wastes L2 transactions. On the big
workloads there are thousands of M-tiles, so this L2 traffic is non-trivial.

c006's SINGLE lever: a one-time Triton permute of each conv weight from
`(Cout, Cin, 3, 3)` to tap-contiguous `(KH, KW, Cin, Cout)` (flat layout
[KK, Cin, Cout]). The conv then loads a W tile with pointer math
    w_perm + tap*(Cin*Cout) + cin*Cout + cout ,
so the `cout` (BLOCK_N) columns become stride-1 contiguous → coalesced loads
and a natural row-major `tl.dot` B-operand. Weights are tiny (256*256*9 =
590K elems = 2.4 MB) so the permute is a cheap, memory-bound pass done once
per conv and reused across all M-tiles.

Everything else is byte-identical to c005: the conv autotune config set, the
implicit-GEMM math / masking, fp32 `tl.dot` (native TF32 path on sm_80), the
GroupNorm stats/apply kernels, and all dtypes.

Correctness: the permute is a pure index remap (verified below: dst
(tap,cin,cout) <- src (cout,cin,tap)); the conv's numerical result is
unchanged because it still contracts the same weight[cout,cin,kh,kw] against
the same input pixels — only the storage order of the operand changed.
BLOCK_K ∈ {64,128,256} still divides C=256 so the cin dim needs no mask;
BLOCK_N is still masked by n_mask.
"""

import torch
import triton
import triton.language as tl

_NUM_GROUPS = 32
_KS = 3


# ---------------------------------------------------------------------------
# One-time weight permute: (Cout, Cin, KH, KW) -> [KK, Cin, Cout] (tap-major,
# cout contiguous). Pure index remap; reused across all conv output tiles.
# ---------------------------------------------------------------------------
@triton.jit
def _permute_weight_kernel(
    src_ptr, dst_ptr,
    Cout, Cin, KK, numel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)          # flat index into dst [KK, Cin, Cout]
    mask = idx < numel

    cout = idx % Cout
    tmp = idx // Cout
    cin = tmp % Cin
    tap = tmp // Cin                                  # kh*KW + kw

    # src [Cout, Cin, KK]: flat = cout*(Cin*KK) + cin*KK + tap
    src_idx = cout * (Cin * KK) + cin * KK + tap
    v = tl.load(src_ptr + src_idx, mask=mask, other=0.0)
    tl.store(dst_ptr + idx, v, mask=mask)


# ---------------------------------------------------------------------------
# Convolution: implicit GEMM over the 9 taps of a 3x3 kernel, stride=1, pad=1.
# Output tile [BLOCK_M output pixels, BLOCK_N output channels]; K = C_in.
# Weights are pre-permuted to [KK, Cin, Cout] (cout contiguous).
# ---------------------------------------------------------------------------
def _conv_configs():
    cfgs = []
    for bm, bn, bk, w, s in [
        (64, 64, 64, 4, 2),      # c004 baseline
        (64, 128, 64, 4, 3),
        (128, 64, 64, 4, 3),
        (128, 128, 64, 8, 3),
        (128, 128, 64, 8, 2),
        (64, 256, 64, 8, 2),
        (256, 64, 64, 8, 2),
        (128, 256, 64, 8, 2),
        (256, 128, 64, 8, 2),
        (64, 64, 128, 4, 2),
        (128, 128, 128, 8, 2),
    ]:
        cfgs.append(triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
            num_warps=w, num_stages=s,
        ))
    return cfgs


@triton.autotune(configs=_conv_configs(), key=["M", "C", "H", "W"])
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
    CC = C * C                                            # per-tap stride in permuted weight
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
                # W tile [BLOCK_K, BLOCK_N] from permuted [KK, Cin, Cout]:
                #   w_perm[tap, cin, cout] ; cout (=offs_n) is stride-1 contiguous
                w_ptrs = w_ptr + tap * CC + cin[:, None] * C + offs_n[None, :]
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
_STATS_BLOCK = 1024
_APPLY_BLOCK = 1024
_PERMUTE_BLOCK = 1024


def _permute_weight(w, C):
    # w: (Cout, Cin, KH, KW) contiguous -> dst: [KK, Cin, Cout] contiguous
    KK = _KS * _KS
    numel = C * C * KK
    dst = torch.empty((KK, C, C), dtype=w.dtype, device=w.device)
    grid = (triton.cdiv(numel, _PERMUTE_BLOCK),)
    _permute_weight_kernel[grid](
        w, dst,
        C, C, KK, numel,
        BLOCK=_PERMUTE_BLOCK, num_warps=4,
    )
    return dst


def _conv3x3(x, w_perm, C, B, H, W):
    M = B * H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(C, meta["BLOCK_N"]))
    _conv3x3_kernel[grid](
        x, w_perm, out,
        B, H, W, M,
        C=C, KS=_KS,
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

    # One-time weight permute to tap-contiguous [KK, Cin, Cout].
    w1p = _permute_weight(conv1_weight, C)
    w2p = _permute_weight(conv2_weight, C)

    # ---- Path 1: conv1 -> GN1 -> SiLU ----
    t1 = _conv3x3(x, w1p, C, B, H, W)
    mean1, rstd1, group_size = _gn_stats(t1, B, C, H, W, eps_f)
    t2 = torch.empty_like(x)
    _gn_apply_silu_kernel[grid_apply](
        t1, mean1, rstd1, norm1_weight, norm1_bias, t2, t1,
        numel, group_size, HW,
        C=C, ADD_RES=False, BLOCK=_APPLY_BLOCK, num_warps=4,
    )

    # ---- Path 2: conv2 -> GN2 -> SiLU -> +residual ----
    t3 = _conv3x3(t2, w2p, C, B, H, W)
    mean2, rstd2, group_size2 = _gn_stats(t3, B, C, H, W, eps_f)
    out = torch.empty_like(x)
    _gn_apply_silu_kernel[grid_apply](
        t3, mean2, rstd2, norm2_weight, norm2_bias, out, x,
        numel, group_size2, HW,
        C=C, ADD_RES=True, BLOCK=_APPLY_BLOCK, num_warps=4,
    )

    return out
