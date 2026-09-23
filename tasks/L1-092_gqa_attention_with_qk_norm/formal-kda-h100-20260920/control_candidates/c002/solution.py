"""
Solution c002 — L1/092 GQA Attention with QK-Norm (GLM-4.5-Air block), H100 / sm_90.

Parent c001 (geomean 2.1858x). SINGLE CHANGE vs c001: GEMM tile config now
selected by M. Large-M (>=1024) GEMMs use a high-throughput Hopper tile
(BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=3) to lift
tensor-core utilization on the compute-bound projections that dominate the weak
large-M/high-batch shapes ((16,128), (16,256), (32,256)). Small-M (<1024) keeps
c001's proven tile (64,128,64, warps=4, stages=3) so those shapes cannot regress.
All compute stays Triton (tl.dot, fp32 accumulate); PyTorch only for metadata /
allocation / launch plumbing (NO computational fallback).

Pipeline (matches task/definition.json reference exactly):
  1. QKV projections (Triton GEMM + bias, fp32 accum).
  2. QK RMSNorm(d=128, fp32) fused with NeoX rotate-half RoPE (Triton).
  3. GQA causal FlashAttention (Triton, online softmax, fp32 accum).
  4. Output projection (Triton GEMM, no bias, fp32 accum).
"""

import torch
import triton
import triton.language as tl

# ---- Fixed model constants (from task/definition.json) ---------------------
H = 96            # num_attention_heads
KV = 8            # num_key_value_heads
G = H // KV       # 12  groups (num_key_value_groups)
D = 128           # head_dim
HALF = D // 2     # 64  rotate-half split
HID = 4096        # hidden_size
QO = H * D        # 12288  q_out_features
KVO = KV * D      # 1024   kv_out_features
SCALING = D ** -0.5          # 1/sqrt(128)
LOG2E = 1.4426950408889634   # log2(e), for exp2-based softmax


# ===========================================================================
# GEMM:  C[M,N] = A[M,K] @ W[N,K]^T (+ bias[N])      (== F.linear semantics)
# ===========================================================================
@triton.jit
def _gemm_kernel(
    A, W, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
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
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k0 * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, w)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    c = acc.to(C.dtype.element_ty)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _gemm(a, w, bias, out):
    M, K = a.shape
    N = w.shape[0]
    # Tile config selected by M. Large-M projections are compute-bound (they
    # dominate the weak high-batch shapes); use a wide Hopper tile to raise
    # tensor-core utilization. Small-M keeps c001's proven tile (no regression).
    if M >= 1024:
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 256, 64, 8
        num_warps, num_stages = 8, 3
    else:
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 128, 64, 8
        num_warps, num_stages = 4, 3
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _gemm_kernel[grid](
        a, w, bias if bias is not None else a, out,
        M, N, K,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=num_warps, num_stages=num_stages,
    )


# ===========================================================================
# Fused QK RMSNorm(fp32) + NeoX rotate-half RoPE
#   X: [M, NHEAD, D]  ->  Out: [M, NHEAD, D]
# ===========================================================================
@triton.jit
def _norm_rope_kernel(
    X, Wn, Cos, Sin, Out,
    M,
    stride_xm, stride_xh, stride_xd,
    stride_cm, stride_cd,
    stride_om, stride_oh, stride_od,
    eps,
    D: tl.constexpr, HALF: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_h = tl.arange(0, HALF)

    xbase = offs_m[:, None] * stride_xm + h * stride_xh
    x1 = tl.load(X + xbase + offs_h[None, :] * stride_xd, mask=mask_m[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(X + xbase + (offs_h[None, :] + HALF) * stride_xd, mask=mask_m[:, None], other=0.0).to(tl.float32)

    var = (tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / D
    rstd = 1.0 / tl.sqrt(var + eps)  # [BLOCK_M]

    w1 = tl.load(Wn + offs_h).to(tl.float32)
    w2 = tl.load(Wn + offs_h + HALF).to(tl.float32)
    xn1 = x1 * rstd[:, None] * w1[None, :]
    xn2 = x2 * rstd[:, None] * w2[None, :]

    cbase = offs_m[:, None] * stride_cm
    cos1 = tl.load(Cos + cbase + offs_h[None, :] * stride_cd, mask=mask_m[:, None], other=0.0).to(tl.float32)
    cos2 = tl.load(Cos + cbase + (offs_h[None, :] + HALF) * stride_cd, mask=mask_m[:, None], other=0.0).to(tl.float32)
    sin1 = tl.load(Sin + cbase + offs_h[None, :] * stride_cd, mask=mask_m[:, None], other=0.0).to(tl.float32)
    sin2 = tl.load(Sin + cbase + (offs_h[None, :] + HALF) * stride_cd, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # rotate_half: rot[:HALF] = -x[HALF:], rot[HALF:] = x[:HALF]
    out1 = xn1 * cos1 - xn2 * sin1
    out2 = xn2 * cos2 + xn1 * sin2

    obase = offs_m[:, None] * stride_om + h * stride_oh
    tl.store(Out + obase + offs_h[None, :] * stride_od, out1.to(Out.dtype.element_ty), mask=mask_m[:, None])
    tl.store(Out + obase + (offs_h[None, :] + HALF) * stride_od, out2.to(Out.dtype.element_ty), mask=mask_m[:, None])


def _norm_rope(x, wn, cos, sin, nhead, eps):
    M = x.shape[0]
    out = torch.empty_like(x)
    xv = x.view(M, nhead, D)
    ov = out.view(M, nhead, D)
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M), nhead)
    _norm_rope_kernel[grid](
        xv, wn, cos, sin, ov,
        M,
        xv.stride(0), xv.stride(1), xv.stride(2),
        cos.stride(0), cos.stride(1),
        ov.stride(0), ov.stride(1), ov.stride(2),
        float(eps),
        D=D, HALF=HALF, BLOCK_M=BLOCK_M,
        num_warps=4, num_stages=2,
    )
    return out


# ===========================================================================
# GQA causal FlashAttention (forward, online softmax).
#   Q: [B,S,H,D]   K,V: [B,S,KV,D]   ->  O: [B,S,H,D]
# ===========================================================================
@triton.jit
def _attn_kernel(
    Q, K, V, O,
    B, S,
    sqb, sqs, sqh, sqd,
    skb, sks, skh, skd,
    svb, svs, svh, svd,
    sob, sos, soh, sod,
    qk_scale,
    H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    b = off_bh // H
    h = off_bh % H
    kv_h = h // G

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + b * sqb + offs_m[:, None] * sqs + h * sqh + offs_d[None, :] * sqd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    hi = tl.minimum((start_m + 1) * BLOCK_M, S)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + b * skb + offs_n[:, None] * sks + kv_h * skh + offs_d[None, :] * skd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < S, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * qk_scale  # [BLOCK_M, BLOCK_N] fp32
        causal = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < S)
        qk = tl.where(causal, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + b * svb + offs_n[:, None] * svs + kv_h * svh + offs_d[None, :] * svd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < S, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    o_ptrs = O + b * sob + offs_m[:, None] * sos + h * soh + offs_d[None, :] * sod
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < S)


def _attention(q, k, v, B, S):
    # q: [B,S,H,D], k/v: [B,S,KV,D]  (contiguous flat views)
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    _attn_kernel[grid](
        q, k, v, o,
        B, S,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        SCALING * LOG2E,
        H=H, G=G, D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return o


# ===========================================================================
# Entry point
# ===========================================================================
@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_norm_eps: float,
):
    B, S, _ = hidden_states.shape
    M = B * S

    hs = hidden_states.reshape(M, HID).contiguous()
    cos2d = cos.reshape(M, D).contiguous()
    sin2d = sin.reshape(M, D).contiguous()

    q_proj_weight = q_proj_weight.contiguous()
    k_proj_weight = k_proj_weight.contiguous()
    v_proj_weight = v_proj_weight.contiguous()
    o_proj_weight = o_proj_weight.contiguous()

    # --- 1. QKV projections ---
    q = torch.empty((M, QO), device=hs.device, dtype=hs.dtype)
    k = torch.empty((M, KVO), device=hs.device, dtype=hs.dtype)
    v = torch.empty((M, KVO), device=hs.device, dtype=hs.dtype)
    _gemm(hs, q_proj_weight, q_proj_bias, q)
    _gemm(hs, k_proj_weight, k_proj_bias, k)
    _gemm(hs, v_proj_weight, v_proj_bias, v)

    # --- 2. QK RMSNorm + RoPE ---
    q = _norm_rope(q, q_norm_weight, cos2d, sin2d, H, rms_norm_eps)
    k = _norm_rope(k, k_norm_weight, cos2d, sin2d, KV, rms_norm_eps)

    # --- 3. GQA causal FlashAttention ---
    qv = q.view(B, S, H, D)
    kv = k.view(B, S, KV, D)
    vv = v.view(B, S, KV, D)
    attn = _attention(qv, kv, vv, B, S)      # [B,S,H,D]
    attn = attn.view(M, QO)

    # --- 4. Output projection (no bias) ---
    out = torch.empty((M, HID), device=hs.device, dtype=hs.dtype)
    _gemm(attn, o_proj_weight, None, out)

    return out.view(B, S, HID)
