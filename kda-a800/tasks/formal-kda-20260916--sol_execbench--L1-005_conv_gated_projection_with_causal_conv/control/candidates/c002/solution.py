"""
Solution for L1/005 conv_gated_projection_with_causal_conv  (candidate c002)

Target: NVIDIA A800 (sm_80, Ampere). Compute is 100% Triton; PyTorch is used only for
tensor metadata / launch plumbing (shapes, dtypes, empty allocations, free reshape views).
NO Torch/CPU/NumPy/CUDA-ext computational fallback.

Fused pipeline (all in row-major (M, .) space, M = B*S), eliminating the reference's two
transposes and the .contiguous() copy:

  K1  BCx = Xv @ in_proj_weight^T + in_proj_bias           -> (M, 3H)   [tl.dot, fp32 accum]
  K2  y   = gating -> causal depthwise conv(k=4) -> gating  -> (M, H)    [fp32 middle]
  K3  out = y  @ out_proj_weight^T + out_proj_bias          -> (M, H)    [tl.dot, fp32 accum]

Index map (see docs/draft.md 1.1):
  chunks of BCx: Bgate=[0,H), Cgate=[H,2H), Xproj=[2H,3H)
  Bx[m,h] = BCx[m,h] * BCx[m,2H+h]
  conv_out[m,h] = conv_bias[h] + sum_{k=0..3} conv_weight[h,0,k] * Bx[m-(3-k), h]
                  (tap valid iff (m mod S) >= (3-k)  -> causal pad == batch boundary)
  y[m,h] = BCx[m,H+h] * conv_out[m,h]

c002 = c001 + @triton.autotune on the shared GEMM kernel (K1 & K3), keyed on (M,N,K).
       Single variable vs c001: GEMM tile-selection policy. K2 unchanged.
       Motivation: c001's fixed 64x64x32 tile underperforms cuBLAS at large M (WL3 0.889x).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune config menu for the GEMM (curated Ampere bf16 tiles, draft 5).
# All feedback shapes are mask-free: M in {1024,2048,8192} (mult. 256),
# N in {2048,6144} (mult. 256), K=2048 (mult. 128) -> every tile divides evenly.
# ---------------------------------------------------------------------------
def _gemm_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    ]


# ---------------------------------------------------------------------------
# K1 / K3 : tiled GEMM with bias.   C = A @ Wt^T + bias
#   A  : (M, K)  row-major
#   Wt : (N, K)  row-major   (F.linear weight, so C = A @ Wt^T)
#   C  : (M, N)  row-major bf16
# ---------------------------------------------------------------------------
@triton.autotune(configs=_gemm_configs(), key=['M', 'N', 'K'])
@triton.jit
def _gemm_bias_kernel(
    A, Wt, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = Wt + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_rem = K - k0
        a = tl.load(a_ptrs, mask=m_mask & (offs_k[None, :] < k_rem), other=0.0)
        w = tl.load(w_ptrs, mask=n_mask & (offs_k[:, None] < k_rem), other=0.0)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=m_mask & n_mask)


# ---------------------------------------------------------------------------
# K2 : fused gating + causal depthwise conv(k=4) + gating.
#   BCx : (M, 3H) bf16      Cw : (H, KC) bf16 (view of conv_weight (H,1,KC))
#   Cb  : (H,)   bf16       Y  : (M, H)  bf16 out
# Row-parallel simple variant: re-load neighbor Bx per tap.  (unchanged from c001)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_conv_kernel(
    BCx, Cw, Cb, Y,
    M, S, H,
    stride_bm, stride_bn,
    stride_ym, stride_yh,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
    KC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    m_mask = offs_m < M
    h_mask = offs_h < H
    s = offs_m % S  # position within batch, (BLOCK_M,)

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    for k in range(0, KC):
        d = (KC - 1) - k              # tap offset: k=KC-1 -> current sample (d=0)
        src = offs_m - d
        valid = m_mask & (s >= d)     # causal pad + batch-boundary guard
        ld_mask = valid[:, None] & h_mask[None, :]
        bg = tl.load(BCx + src[:, None] * stride_bm + offs_h[None, :] * stride_bn,
                     mask=ld_mask, other=0.0).to(tl.float32)
        xp = tl.load(BCx + src[:, None] * stride_bm + (2 * H + offs_h[None, :]) * stride_bn,
                     mask=ld_mask, other=0.0).to(tl.float32)
        w = tl.load(Cw + offs_h * KC + k, mask=h_mask, other=0.0).to(tl.float32)  # (BLOCK_H,)
        acc += bg * xp * w[None, :]   # zero where !valid (bg/xp loaded as 0)

    cb = tl.load(Cb + offs_h, mask=h_mask, other=0.0).to(tl.float32)
    conv_out = acc + cb[None, :]

    cg = tl.load(BCx + offs_m[:, None] * stride_bm + (H + offs_h[None, :]) * stride_bn,
                 mask=m_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
    y = cg * conv_out
    tl.store(Y + offs_m[:, None] * stride_ym + offs_h[None, :] * stride_yh,
             y.to(tl.bfloat16), mask=m_mask[:, None] & h_mask[None, :])


# --- K2 fixed launch configuration (unchanged from c001) ---
_BM2, _BH2 = 64, 64


@torch.no_grad()
def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    B, S, H = x.shape
    M = B * S
    TH = in_proj_weight.shape[0]   # 3H
    KC = conv_weight.shape[2]

    # --- metadata / shape guards (allowed: no compute) ---
    assert H == 2048, f"unexpected hidden_size {H}"
    assert TH == 3 * H, f"unexpected triple_hidden {TH}"
    assert KC == 4, f"unexpected conv_kernel_size {KC}"
    assert x.dtype == torch.bfloat16

    xc = x if x.is_contiguous() else x.contiguous()
    Xv = xc.reshape(M, H)                     # free view of (B,S,H) contiguous
    Cw = conv_weight.reshape(H, KC)           # free view of (H,1,KC) contiguous

    BCx = torch.empty((M, TH), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((M, H), device=x.device, dtype=torch.bfloat16)
    out = torch.empty((M, H), device=x.device, dtype=torch.bfloat16)

    # ---- K1: GEMM1 (M x H) @ (H x 3H) -> (M, 3H) ----
    grid1 = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(TH, META['BLOCK_N']),)
    _gemm_bias_kernel[grid1](
        Xv, in_proj_weight, in_proj_bias, BCx,
        M, TH, H,
        Xv.stride(0), Xv.stride(1),
        in_proj_weight.stride(0), in_proj_weight.stride(1),
        BCx.stride(0), BCx.stride(1),
    )

    # ---- K2: fused gating + causal depthwise conv + gating -> (M, H) ----
    grid2 = (triton.cdiv(M, _BM2), triton.cdiv(H, _BH2))
    _fused_conv_kernel[grid2](
        BCx, Cw, conv_bias, y,
        M, S, H,
        BCx.stride(0), BCx.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=_BM2, BLOCK_H=_BH2, KC=KC,
    )

    # ---- K3: GEMM2 (M x H) @ (H x H) -> (M, H) ----
    grid3 = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(H, META['BLOCK_N']),)
    _gemm_bias_kernel[grid3](
        y, out_proj_weight, out_proj_bias, out,
        M, H, H,
        y.stride(0), y.stride(1),
        out_proj_weight.stride(0), out_proj_weight.stride(1),
        out.stride(0), out.stride(1),
    )

    return out.reshape(B, S, H)
