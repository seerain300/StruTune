"""
KDA candidate c002 — L2/015 Audio Sinusoidal PE + Conv Projection.

Parent: c001 (0/5 RUNTIME_ERROR — no output produced). Change vs c001:
  * The uniform, shape-independent RUNTIME_ERROR points at a compile/import
    failure rather than a numerical/masking bug. The one library-dependent site
    is the GELU erf. c001 used `tl.math.erf`; a naive fix (`libdevice.erf` via
    `from triton.language.extra import libdevice`) is equally version-fragile —
    if that import path is wrong on this stack it fails identically.
  * FIX: replace the library erf with a fully self-contained polynomial erf
    (Abramowitz-Stegun 7.1.26; max abs err ~1.5e-7, far below bf16 precision)
    built only from core Triton builtins (`tl.exp`, `tl.where`, arithmetic).
    This removes every `libdevice`/`tl.math` dependency — the sole plausible
    compile-failure surface — so the pipeline should compile and establish the
    correctness baseline. All kernel bodies are otherwise identical to c001,
    keeping the change attributable.

Fully-Triton 4-kernel pipeline (no Torch/CPU/NumPy compute fallback):

  K-A conv1 (Cin=1) + GELU   -> intermediate1 NHWC  [B, 40, T1, 384]
  K-B conv2 (implicit GEMM)  + GELU -> intermediate2 NHWC [B, 20, T2, 384]
  K-C conv3 (implicit GEMM)  + GELU, written directly into the linear
      A-matrix layout [B, T3, 3840] with column k = c*10 + f (fused reshape)
  K-D linear (GEMM) + scale + positional-embedding add -> out bf16 [B, T3, 1024]

All conv/linear accumulation is fp32; intermediates are rounded to bf16 at the
same sites as the reference (round conv output to bf16 -> GELU in fp32 ->
round to bf16). GELU is exact-erf (matches F.gelu default).

PyTorch is used only for tensor metadata / buffer allocation / launch plumbing.
"""

import torch
import triton
import triton.language as tl

# ---- constants (from task/definition.json) ----
D_MODEL = 1024
NUM_MEL = 80
HIDDEN = 384          # downsample_hidden_size (conv channels)
FREQ0 = 80
FREQ1 = 40
FREQ2 = 20
FREQ3 = 10
CONV_OUT_DIM = 3840   # 384 * 10
KSZ = 3
INV_SQRT2 = 0.7071067811865476


@triton.jit
def _erf(x):
    # Self-contained erf (Abramowitz-Stegun 7.1.26), fp32, max abs err ~1.5e-7.
    # Uses only the most primitive Triton builtins (tl.where, tl.exp, arithmetic)
    # so it compiles on any Triton/sm_80 stack with no libdevice/tl.math surface.
    sign = tl.where(x < 0.0, -1.0, 1.0)
    ax = tl.where(x < 0.0, -x, x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * tl.exp(-ax * ax)
    return sign * y


@triton.jit
def _gelu(x):
    # exact-erf GELU, computed in fp32 (matches torch F.gelu default)
    return 0.5 * x * (1.0 + _erf(x * INV_SQRT2))


# ============================ K-A: conv1 (Cin=1) + GELU ======================
@triton.jit
def conv1_kernel(
    inp_ptr,        # input_features NCHW [B,1,80,Tin] bf16
    w_ptr,          # conv2d1_weight [384,1,3,3] bf16
    b_ptr,          # conv2d1_bias [384] bf16
    out_ptr,        # intermediate1 NHWC [B,40,T1,384] bf16
    B, TIN, T1, M,  # M = B*40*T1
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # output channel (cout)

    HW = FREQ1 * T1
    b = offs_m // HW
    rem = offs_m % HW
    ho = rem // T1          # freq out [0,40)
    wo = rem % T1           # time out [0,T1)
    m_valid = offs_m < M
    n_valid = offs_n < HIDDEN

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kh in tl.static_range(0, KSZ):
        hi = ho * 2 - 1 + kh
        h_ok = (hi >= 0) & (hi < FREQ0)
        for kw in tl.static_range(0, KSZ):
            wi = wo * 2 - 1 + kw
            w_ok = (wi >= 0) & (wi < TIN)
            valid = m_valid & h_ok & w_ok
            a_off = (b * FREQ0 + hi) * TIN + wi
            a = tl.load(inp_ptr + a_off, mask=valid, other=0.0).to(tl.float32)  # [BLOCK_M]
            wv = tl.load(w_ptr + offs_n * (KSZ * KSZ) + (kh * KSZ + kw),
                         mask=n_valid, other=0.0).to(tl.float32)                # [BLOCK_N]
            acc += a[:, None] * wv[None, :]

    bias = tl.load(b_ptr + offs_n, mask=n_valid, other=0.0).to(tl.float32)
    acc += bias[None, :]
    y = acc.to(tl.bfloat16).to(tl.float32)     # round conv output to bf16
    g = _gelu(y).to(tl.bfloat16)

    row = (b * FREQ1 + ho) * T1 + wo           # [BLOCK_M]
    o_off = row[:, None] * HIDDEN + offs_n[None, :]
    tl.store(out_ptr + o_off, g, mask=m_valid[:, None] & n_valid[None, :])


# =================== K-B: conv2 (implicit GEMM) + GELU -> NHWC ===============
@triton.jit
def conv_nhwc_kernel(
    inp_ptr,        # NHWC [B,HIN,WIN,CIN] bf16
    w_ptr,          # weight [COUT,CIN,3,3] bf16
    b_ptr,          # bias [COUT] bf16
    out_ptr,        # NHWC [B,HOUT,WOUT,COUT] bf16
    B, HIN, WIN, HOUT, WOUT, M,   # M = B*HOUT*WOUT
    CIN: tl.constexpr, COUT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # cout
    offs_k = tl.arange(0, BLOCK_K)

    HW = HOUT * WOUT
    b = offs_m // HW
    rem = offs_m % HW
    ho = rem // WOUT
    wo = rem % WOUT
    m_valid = offs_m < M
    n_valid = offs_n < COUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kh in tl.static_range(0, KSZ):
        hi = ho * 2 - 1 + kh
        h_ok = (hi >= 0) & (hi < HIN)
        for kw in tl.static_range(0, KSZ):
            wi = wo * 2 - 1 + kw
            w_ok = (wi >= 0) & (wi < WIN)
            valid = m_valid & h_ok & w_ok
            base = ((b * HIN + hi) * WIN + wi) * CIN     # [BLOCK_M] NHWC base
            tap = kh * KSZ + kw
            for k0 in tl.range(0, CIN, BLOCK_K):
                kk = k0 + offs_k
                k_ok = kk < CIN
                a = tl.load(inp_ptr + base[:, None] + kk[None, :],
                            mask=valid[:, None] & k_ok[None, :], other=0.0)     # [BM,BK] bf16
                w = tl.load(w_ptr + offs_n[None, :] * (CIN * KSZ * KSZ)
                            + kk[:, None] * (KSZ * KSZ) + tap,
                            mask=k_ok[:, None] & n_valid[None, :], other=0.0)    # [BK,BN] bf16
                acc += tl.dot(a, w, out_dtype=tl.float32)

    bias = tl.load(b_ptr + offs_n, mask=n_valid, other=0.0).to(tl.float32)
    acc += bias[None, :]
    y = acc.to(tl.bfloat16).to(tl.float32)
    g = _gelu(y).to(tl.bfloat16)

    row = (b * HOUT + ho) * WOUT + wo
    o_off = row[:, None] * COUT + offs_n[None, :]
    tl.store(out_ptr + o_off, g, mask=m_valid[:, None] & n_valid[None, :])


# ===== K-C: conv3 (implicit GEMM) + GELU -> linear A-matrix [B,T3,3840] ======
@triton.jit
def conv3_to_A_kernel(
    inp_ptr,        # NHWC [B,20,T2,384] bf16
    w_ptr,          # weight [384,384,3,3] bf16
    b_ptr,          # bias [384] bf16
    A_ptr,          # [B, T3, 3840] bf16 ; col = cout*10 + f
    B, HIN, WIN, T3, M,           # HOUT=10, WOUT=T3 ; M = B*10*T3
    CIN: tl.constexpr, COUT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # cout
    offs_k = tl.arange(0, BLOCK_K)

    HW = FREQ3 * T3
    b = offs_m // HW
    rem = offs_m % HW
    ho = rem // T3          # freq out f in [0,10)
    wo = rem % T3           # time out t in [0,T3)
    m_valid = offs_m < M
    n_valid = offs_n < COUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kh in tl.static_range(0, KSZ):
        hi = ho * 2 - 1 + kh
        h_ok = (hi >= 0) & (hi < HIN)
        for kw in tl.static_range(0, KSZ):
            wi = wo * 2 - 1 + kw
            w_ok = (wi >= 0) & (wi < WIN)
            valid = m_valid & h_ok & w_ok
            base = ((b * HIN + hi) * WIN + wi) * CIN
            tap = kh * KSZ + kw
            for k0 in tl.range(0, CIN, BLOCK_K):
                kk = k0 + offs_k
                k_ok = kk < CIN
                a = tl.load(inp_ptr + base[:, None] + kk[None, :],
                            mask=valid[:, None] & k_ok[None, :], other=0.0)
                w = tl.load(w_ptr + offs_n[None, :] * (CIN * KSZ * KSZ)
                            + kk[:, None] * (KSZ * KSZ) + tap,
                            mask=k_ok[:, None] & n_valid[None, :], other=0.0)
                acc += tl.dot(a, w, out_dtype=tl.float32)

    bias = tl.load(b_ptr + offs_n, mask=n_valid, other=0.0).to(tl.float32)
    acc += bias[None, :]
    y = acc.to(tl.bfloat16).to(tl.float32)
    g = _gelu(y).to(tl.bfloat16)

    # write to A[b, t=wo, c*10 + f] ; row = b*T3 + wo ; col = offs_n*FREQ3 + ho
    row = b * T3 + wo
    col = offs_n * FREQ3 + ho[:, None]           # [BM,BN]
    a_off = row[:, None] * CONV_OUT_DIM + col
    tl.store(A_ptr + a_off, g, mask=m_valid[:, None] & n_valid[None, :])


# ============= K-D: linear + scale + positional-embedding add ================
@triton.jit
def linear_kernel(
    A_ptr,          # [M, 3840] bf16
    w_ptr,          # conv_out_weight [1024, 3840] bf16
    pe_ptr,         # positional_embedding [1500, 1024] bf16
    out_ptr,        # [M, 1024] bf16   (M = B*T3)
    M, T3, embed_scale,
    K: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_valid = offs_m < M
    n_valid = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        kk = k0 + offs_k
        k_ok = kk < K
        a = tl.load(A_ptr + offs_m[:, None] * K + kk[None, :],
                    mask=m_valid[:, None] & k_ok[None, :], other=0.0)
        w = tl.load(w_ptr + offs_n[None, :] * K + kk[:, None],
                    mask=k_ok[:, None] & n_valid[None, :], other=0.0)
        acc += tl.dot(a, w, out_dtype=tl.float32)

    # round linear output to bf16 (as reference), scale (x*32 exact in bf16), add PE
    x = acc.to(tl.bfloat16).to(tl.float32) * embed_scale
    t = offs_m % T3
    pe = tl.load(pe_ptr + t[:, None] * N + offs_n[None, :],
                 mask=m_valid[:, None] & n_valid[None, :], other=0.0).to(tl.float32)
    out = (x + pe).to(tl.bfloat16)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             out, mask=m_valid[:, None] & n_valid[None, :])


def _hout(n):
    return (n - 1) // 2 + 1


@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv2d1_weight: torch.Tensor,
    conv2d1_bias: torch.Tensor,
    conv2d2_weight: torch.Tensor,
    conv2d2_bias: torch.Tensor,
    conv2d3_weight: torch.Tensor,
    conv2d3_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    positional_embedding: torch.Tensor,
    embed_scale: float,
):
    # ---- structural guards ----
    assert input_features.dtype == torch.bfloat16
    assert input_features.dim() == 4 and input_features.shape[1] == 1
    B, _, mel, TIN = input_features.shape
    assert mel == NUM_MEL
    assert conv_out_weight.shape == (D_MODEL, CONV_OUT_DIM)
    dev = input_features.device

    input_features = input_features.contiguous()
    conv2d1_weight = conv2d1_weight.contiguous()
    conv2d2_weight = conv2d2_weight.contiguous()
    conv2d3_weight = conv2d3_weight.contiguous()
    conv_out_weight = conv_out_weight.contiguous()
    positional_embedding = positional_embedding.contiguous()

    T1 = _hout(TIN)
    T2 = _hout(T1)
    T3 = _hout(T2)
    assert _hout(FREQ0) == FREQ1 and _hout(FREQ1) == FREQ2 and _hout(FREQ2) == FREQ3
    assert T3 <= positional_embedding.shape[0]

    inter1 = torch.empty((B, FREQ1, T1, HIDDEN), dtype=torch.bfloat16, device=dev)
    inter2 = torch.empty((B, FREQ2, T2, HIDDEN), dtype=torch.bfloat16, device=dev)
    Amat = torch.empty((B, T3, CONV_OUT_DIM), dtype=torch.bfloat16, device=dev)
    out = torch.empty((B, T3, D_MODEL), dtype=torch.bfloat16, device=dev)

    es = float(embed_scale)

    # ---- K-A: conv1 ----
    BM1, BN1 = 64, 128
    M1 = B * FREQ1 * T1
    grid1 = (triton.cdiv(M1, BM1), triton.cdiv(HIDDEN, BN1))
    conv1_kernel[grid1](
        input_features, conv2d1_weight, conv2d1_bias, inter1,
        B, TIN, T1, M1,
        BLOCK_M=BM1, BLOCK_N=BN1, num_warps=4, num_stages=2,
    )

    # ---- K-B: conv2 (NHWC out) ----
    BM, BN, BK = 64, 64, 64
    M2 = B * FREQ2 * T2
    grid2 = (triton.cdiv(M2, BM), triton.cdiv(HIDDEN, BN))
    conv_nhwc_kernel[grid2](
        inter1, conv2d2_weight, conv2d2_bias, inter2,
        B, FREQ1, T1, FREQ2, T2, M2,
        CIN=HIDDEN, COUT=HIDDEN,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=4, num_stages=3,
    )

    # ---- K-C: conv3 -> linear A-matrix ----
    M3 = B * FREQ3 * T3
    grid3 = (triton.cdiv(M3, BM), triton.cdiv(HIDDEN, BN))
    conv3_to_A_kernel[grid3](
        inter2, conv2d3_weight, conv2d3_bias, Amat,
        B, FREQ2, T2, T3, M3,
        CIN=HIDDEN, COUT=HIDDEN,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=4, num_stages=3,
    )

    # ---- K-D: linear + scale + PE ----
    Mlin = B * T3
    BLM, BLN, BLK = 64, 64, 64
    gridL = (triton.cdiv(Mlin, BLM), triton.cdiv(D_MODEL, BLN))
    linear_kernel[gridL](
        Amat.view(Mlin, CONV_OUT_DIM), conv_out_weight, positional_embedding,
        out.view(Mlin, D_MODEL),
        Mlin, T3, es,
        K=CONV_OUT_DIM, N=D_MODEL,
        BLOCK_M=BLM, BLOCK_N=BLN, BLOCK_K=BLK, num_warps=4, num_stages=3,
    )

    return out
