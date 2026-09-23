"""
c001 — Correctness-first fully-Triton baseline for
L1/002_vae_conv3x3_groupnorm_silu_residual_fused  (H100 / sm_90, FP32).

Pipeline (all compute in Triton; PyTorch only for allocation / launch plumbing):
    x -> Conv3x3(pad1) -> GroupNorm(32) -> SiLU
      -> Conv3x3(pad1) -> GroupNorm(32) -> SiLU -> + x

Design (see docs/plan.md, candidate c001):
  * Conv as an implicit-GEMM: for each of the 9 (kh,kw) taps, accumulate
    [BLOCK_M, C_in] x [C_in, BLOCK_N] TF32 tensor-core dots into an FP32 tile.
    M = B*H*W (output pixels), N = C_out = 256, K = C_in = 256.
  * GroupNorm reduction is a split (partial -> finalize) reduction so it scales
    both the many-small-group and few-huge-group regimes. Partials are summed
    with a tl.sum tree reduction in FP32 (more accurate than atomics/naive).
  * Normalize + affine + SiLU (+ residual on stage 2) are fused into one kernel
    that reads the conv buffer once.

NCHW layout throughout. Group g of batch b occupies the contiguous block
[bg*group_size, (bg+1)*group_size) with bg = b*32 + g and group_size = 8*H*W,
because C = 256 = 32*8 (verified in docs/draft.md).
"""

import torch
import triton
import triton.language as tl

GROUPS = 32
CH_PER_G = 8  # 256 / 32

# Conv tile sizes (fixed for c001; tuning is a later candidate).
CB_M = 64
CB_N = 64
CB_K = 64

BLOCK_STATS = 2048   # elements reduced per partial-stats program
BLOCK_APPLY = 1024   # elements per normalize/apply program


@triton.jit
def _conv3x3_kernel(
    X, Wt, Out,
    C, H, W, M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    hw = H * W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # output pixel indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)    # output channels

    m_mask = offs_m < M
    n_mask = offs_n < N

    # decompose flat output-pixel index -> (b, oh, ow)
    b = offs_m // hw
    rem = offs_m % hw
    oh = rem // W
    ow = rem % W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # base offset into X for the (b, ih, iw) part of each row (channel added later)
    for kh in tl.static_range(3):
        ih = oh + kh - 1
        h_ok = (ih >= 0) & (ih < H)
        for kw in tl.static_range(3):
            iw = ow + kw - 1
            w_ok = (iw >= 0) & (iw < W)
            valid = m_mask & h_ok & w_ok           # [BLOCK_M]
            row_base = b * (C * hw) + ih * W + iw   # [BLOCK_M]
            tap = kh * 3 + kw
            for kc in range(0, C, BLOCK_K):
                offs_k = kc + tl.arange(0, BLOCK_K)
                k_mask = offs_k < C
                # X tile [BLOCK_M, BLOCK_K]: channel stride = hw
                x_off = row_base[:, None] + (offs_k * hw)[None, :]
                x_mask = valid[:, None] & k_mask[None, :]
                x_tile = tl.load(X + x_off, mask=x_mask, other=0.0)
                # W tile [BLOCK_K, BLOCK_N]: w[oc, ic, kh, kw]
                #   flat = oc*(C*9) + ic*9 + tap
                w_off = (offs_k * 9 + tap)[:, None] + (offs_n * (C * 9))[None, :]
                w_mask = k_mask[:, None] & n_mask[None, :]
                w_tile = tl.load(Wt + w_off, mask=w_mask, other=0.0)
                acc += tl.dot(x_tile, w_tile, input_precision="tf32")

    # store to NCHW buffer: out[b, oc, oh, ow] = b*C*hw + oc*hw + oh*W + ow
    out_off = (b * (C * hw) + oh * W + ow)[:, None] + (offs_n * hw)[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(Out + out_off, acc, mask=out_mask)


@triton.jit
def _gn_partial_kernel(
    Buf, PSum, PSq,
    group_size, num_tiles,
    BLOCK: tl.constexpr,
):
    bg = tl.program_id(0)
    t = tl.program_id(1)
    offs = t * BLOCK + tl.arange(0, BLOCK)
    mask = offs < group_size
    gidx = bg * group_size + offs
    x = tl.load(Buf + gidx, mask=mask, other=0.0)
    s = tl.sum(x)
    sq = tl.sum(x * x)
    tl.store(PSum + bg * num_tiles + t, s)
    tl.store(PSq + bg * num_tiles + t, sq)


@triton.jit
def _gn_finalize_kernel(
    PSum, PSq, Mean, Rstd,
    group_size, num_tiles, eps,
    BLOCK_TILES: tl.constexpr,
):
    bg = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TILES)
    mask = offs < num_tiles
    s = tl.load(PSum + bg * num_tiles + offs, mask=mask, other=0.0)
    sq = tl.load(PSq + bg * num_tiles + offs, mask=mask, other=0.0)
    total = tl.sum(s)
    total_sq = tl.sum(sq)
    n = group_size.to(tl.float32)
    mean = total / n
    var = total_sq / n - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Mean + bg, mean)
    tl.store(Rstd + bg, rstd)


@triton.jit
def _gn_apply_kernel(
    Buf, Mean, Rstd, Weight, Bias, Xres, Out,
    C, hw, group_size,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bg = tl.program_id(0)
    t = tl.program_id(1)
    offs = t * BLOCK + tl.arange(0, BLOCK)
    mask = offs < group_size
    gidx = bg * group_size + offs

    g = bg % GROUPS
    channel = g * CH_PER_G + (offs // hw)   # [BLOCK], values in [g*8, g*8+8)

    x = tl.load(Buf + gidx, mask=mask, other=0.0)
    mean = tl.load(Mean + bg)
    rstd = tl.load(Rstd + bg)
    w = tl.load(Weight + channel, mask=mask, other=0.0)
    bb = tl.load(Bias + channel, mask=mask, other=0.0)

    y = (x - mean) * rstd * w + bb
    y = y * tl.sigmoid(y)  # SiLU

    if ADD_RESIDUAL:
        r = tl.load(Xres + gidx, mask=mask, other=0.0)
        y = y + r

    tl.store(Out + gidx, y, mask=mask)


def _conv3x3(x, weight, out):
    B, C, H, W = x.shape
    M = B * H * W
    N = C
    grid = (triton.cdiv(M, CB_M), triton.cdiv(N, CB_N))
    _conv3x3_kernel[grid](
        x, weight, out,
        C, H, W, M, N,
        BLOCK_M=CB_M, BLOCK_N=CB_N, BLOCK_K=CB_K,
        num_warps=4, num_stages=2,
    )


def _group_norm_stats(buf, B, C, H, W, eps):
    hw = H * W
    group_size = CH_PER_G * hw
    num_bg = B * GROUPS
    num_tiles = triton.cdiv(group_size, BLOCK_STATS)

    psum = torch.empty((num_bg, num_tiles), device=buf.device, dtype=torch.float32)
    psq = torch.empty((num_bg, num_tiles), device=buf.device, dtype=torch.float32)

    _gn_partial_kernel[(num_bg, num_tiles)](
        buf, psum, psq,
        group_size, num_tiles,
        BLOCK=BLOCK_STATS, num_warps=4,
    )

    mean = torch.empty((num_bg,), device=buf.device, dtype=torch.float32)
    rstd = torch.empty((num_bg,), device=buf.device, dtype=torch.float32)
    block_tiles = triton.next_power_of_2(num_tiles)
    _gn_finalize_kernel[(num_bg,)](
        psum, psq, mean, rstd,
        group_size, num_tiles, float(eps),
        BLOCK_TILES=block_tiles, num_warps=4,
    )
    return mean, rstd


def _group_norm_apply(buf, mean, rstd, weight, bias, out, x_res,
                      B, C, H, W, add_residual):
    hw = H * W
    group_size = CH_PER_G * hw
    num_bg = B * GROUPS
    num_tiles = triton.cdiv(group_size, BLOCK_APPLY)
    _gn_apply_kernel[(num_bg, num_tiles)](
        buf, mean, rstd, weight, bias,
        x_res if x_res is not None else buf, out,
        C, hw, group_size,
        ADD_RESIDUAL=add_residual,
        BLOCK=BLOCK_APPLY, num_warps=4,
    )


@torch.no_grad()
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
    x = x.contiguous()
    B, C, H, W = x.shape

    w1 = conv1_weight.contiguous()
    w2 = conv2_weight.contiguous()
    n1w = norm1_weight.contiguous()
    n1b = norm1_bias.contiguous()
    n2w = norm2_weight.contiguous()
    n2b = norm2_bias.contiguous()

    # Stage 1: conv1 -> gn1 -> silu
    buf1 = torch.empty_like(x)
    _conv3x3(x, w1, buf1)
    mean1, rstd1 = _group_norm_stats(buf1, B, C, H, W, eps)
    act1 = torch.empty_like(x)
    _group_norm_apply(buf1, mean1, rstd1, n1w, n1b, act1, None,
                      B, C, H, W, add_residual=False)

    # Stage 2: conv2 -> gn2 -> silu -> + x
    buf2 = torch.empty_like(x)
    _conv3x3(act1, w2, buf2)
    mean2, rstd2 = _group_norm_stats(buf2, B, C, H, W, eps)
    out = torch.empty_like(x)
    _group_norm_apply(buf2, mean2, rstd2, n2w, n2b, out, x,
                      B, C, H, W, add_residual=True)

    return out
