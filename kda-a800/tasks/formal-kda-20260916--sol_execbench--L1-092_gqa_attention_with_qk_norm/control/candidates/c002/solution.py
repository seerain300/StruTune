"""
L1/092 GQA Attention with QK Norm (GLM-4.5-Air) — Triton implementation.

Candidate c002: c001 baseline with a performance-tuned projection/output GEMM.
  Single design delta vs c001: the generic fixed-tile GEMM (BLOCK 64x64x32,
  per-iteration M/K masking, no L2 swizzle) is replaced by an autotuned,
  L2-cache-swizzled (GROUP_M) matmul with fp32 accumulation via tl.dot(a,b,acc)
  and an EVEN_K fast path. The RMSNorm+RoPE kernel and the flash-attention
  kernel are byte-for-byte identical to c001.

Pipeline (all compute in Triton; PyTorch only for allocation / metadata / launch):
  1. Q/K/V projections   : tuned tiled GEMM  x[M,4096] @ W[N,4096]^T + bias
  2. RMSNorm(fp32)+RoPE   : fused elementwise kernel over head_dim=128 (Q and K)
  3. Flash attention      : GQA (kv_head = q_head // 12), causal, online softmax fp32
  4. Output projection    : tuned tiled GEMM  attn[M,12288] @ Wo[4096,12288]^T (no bias)

Fixed constants for this task:
  HIDDEN=4096  NH=96  NKV=8  HEAD_DIM=128  GROUPS=12  Q_OUT=12288  KV_OUT=1024
  SCALE = 128 ** -0.5
"""

import torch
import triton
import triton.language as tl

# ---- fixed problem constants -------------------------------------------------
HIDDEN = 4096
NH = 96
NKV = 8
HEAD_DIM = 128
GROUPS = NH // NKV          # 12
Q_OUT = NH * HEAD_DIM       # 12288
KV_OUT = NKV * HEAD_DIM     # 1024
SCALE = HEAD_DIM ** -0.5    # 0.08838834764831845


# ============================================================================
# 1) Tuned tiled GEMM:  C[M,N] = A[M,K] @ Bw[N,K]^T (+ bias[N])
#    A row-major [M,K], Bw row-major [N,K] (i.e. F.linear weight layout).
#    - L2 swizzle over program ids (GROUP_M) for weight/activation reuse.
#    - fp32 accumulation via tl.dot(a, b, acc).
#    - EVEN_K fast path (K divisible by BLOCK_K -> no inner-loop K masking).
#    - M/N handled with modulo-wrapped loads + masked store (no per-iter M/N mask).
# ============================================================================
_GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
]


@triton.autotune(configs=_GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _gemm_kernel(
    A, Bw, C, BIAS,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr, EVEN_K: tl.constexpr,
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

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = Bw + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_rem = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if HAS_BIAS:
        bias = tl.load(BIAS + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16),
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def _gemm(a, w, out, bias):
    """out[M,N] = a[M,K] @ w[N,K]^T (+ bias)."""
    M, K = a.shape
    N, Kw = w.shape
    assert Kw == K
    even_k = (K % 32 == 0) and (K % 64 == 0)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    _gemm_kernel[grid](
        a, w, out, bias if bias is not None else a,
        M, N, K,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None, EVEN_K=even_k,
    )


# ============================================================================
# 2) Fused RMSNorm(fp32) + RoPE over head_dim (one program per (row m, head h))
#    x layout [M, NHEADS*D]; cos/sin layout [M, D]; out same as x.
#    RoPE: out = xn*cos + rotate_half(xn)*sin
#          rotate_half(xn)[d<64] = -xn[d+64];  [d>=64] = xn[d-64]
# ============================================================================
@triton.jit
def _norm_rope_kernel(
    X, OUT, W, COS, SIN,
    eps,
    stride_xm, stride_cm,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_d = tl.arange(0, D)
    base = pid_m * stride_xm + pid_h * D
    x = tl.load(X + base + offs_d).to(tl.float32)

    var = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + offs_d).to(tl.float32)
    xn = x * inv * w

    half = D // 2
    src = tl.where(offs_d < half, offs_d + half, offs_d - half)
    x_src = tl.load(X + base + src).to(tl.float32)
    w_src = tl.load(W + src).to(tl.float32)
    xn_src = x_src * inv * w_src
    sign = tl.where(offs_d < half, -1.0, 1.0)

    cos = tl.load(COS + pid_m * stride_cm + offs_d).to(tl.float32)
    sin = tl.load(SIN + pid_m * stride_cm + offs_d).to(tl.float32)

    out = xn * cos + sign * xn_src * sin
    tl.store(OUT + base + offs_d, out.to(tl.bfloat16))


# ============================================================================
# 3) Flash attention (GQA, causal, online softmax fp32)
#    q_r [M, Q_OUT] with head h at cols [h*D : h*D+D],  m = b*S + s
#    k_r [M, KV_OUT], v [M, KV_OUT] with kv head at cols [kv*D : kv*D+D]
#    attn out [M, Q_OUT] (head-merged, ready for output GEMM)
# ============================================================================
@triton.jit
def _flash_kernel(
    Q, K, V, O,
    S,
    scale,
    NH: tl.constexpr, NKV: tl.constexpr, GROUPS: tl.constexpr,
    QO: tl.constexpr, KVO: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // NH
    h = bh % NH
    kv = h // GROUPS

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_row = (b * S + offs_m)                       # [BLOCK_M]
    q_ptrs = Q + q_row[:, None] * QO + h * D + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)

    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_row = (b * S + offs_n)
        k_ptrs = K + k_row[:, None] * KVO + kv * D + offs_d[None, :]
        v_ptrs = V + k_row[:, None] * KVO + kv * D + offs_d[None, :]
        n_mask = offs_n < S
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale               # [BLOCK_M, BLOCK_N] fp32
        causal = (offs_n[None, :] <= offs_m[:, None]) & n_mask[None, :]
        qk = tl.where(causal, qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = O + q_row[:, None] * QO + h * D + offs_d[None, :]
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=offs_m[:, None] < S)


# ============================================================================
# Host entry point
# ============================================================================
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
    dev = hidden_states.device
    eps = float(rms_norm_eps)

    x = hidden_states.reshape(M, HIDDEN)
    cos2 = cos.reshape(M, HEAD_DIM)
    sin2 = sin.reshape(M, HEAD_DIM)

    # --- 1) projections -------------------------------------------------------
    q = torch.empty((M, Q_OUT), dtype=torch.bfloat16, device=dev)
    k = torch.empty((M, KV_OUT), dtype=torch.bfloat16, device=dev)
    v = torch.empty((M, KV_OUT), dtype=torch.bfloat16, device=dev)
    _gemm(x, q_proj_weight, q, q_proj_bias)
    _gemm(x, k_proj_weight, k, k_proj_bias)
    _gemm(x, v_proj_weight, v, v_proj_bias)

    # --- 2) RMSNorm + RoPE on Q and K ----------------------------------------
    q_r = torch.empty_like(q)
    k_r = torch.empty_like(k)
    _norm_rope_kernel[(M, NH)](
        q, q_r, q_norm_weight, cos2, sin2,
        eps, Q_OUT, HEAD_DIM, D=HEAD_DIM, num_warps=4,
    )
    _norm_rope_kernel[(M, NKV)](
        k, k_r, k_norm_weight, cos2, sin2,
        eps, KV_OUT, HEAD_DIM, D=HEAD_DIM, num_warps=4,
    )

    # --- 3) flash attention (GQA, causal) ------------------------------------
    attn = torch.empty((M, Q_OUT), dtype=torch.bfloat16, device=dev)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(S, BLOCK_M), B * NH)
    _flash_kernel[grid](
        q_r, k_r, v, attn,
        S, SCALE,
        NH, NKV, GROUPS,
        Q_OUT, KV_OUT, HEAD_DIM,
        BLOCK_M, BLOCK_N,
        num_warps=4, num_stages=2,
    )

    # --- 4) output projection (no bias) --------------------------------------
    out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=dev)
    _gemm(attn, o_proj_weight, out, None)

    return out.reshape(B, S, HIDDEN)
