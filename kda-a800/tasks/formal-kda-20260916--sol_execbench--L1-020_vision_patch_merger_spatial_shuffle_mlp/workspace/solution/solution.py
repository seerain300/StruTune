import torch
import triton
import triton.language as tl

# =============================================================================
# L1/020 Vision Patch Merger: LayerNorm (pre-shuffle) + 2x2 spatial shuffle +
# 2-layer GELU MLP. Target: NVIDIA A800 (sm_80). Compute in Triton only;
# PyTorch used only for tensor metadata / launch plumbing / index construction.
#
# Pipeline:
#   1. K1  : per-input-patch LayerNorm (fp32, over C=1536) fused with the 2x2
#            spatial shuffle -> A_shuffled[M, E=6144] bf16.
#   2. K2  : FC1 = A @ fc1_weight^T + fc1_bias, fp32 accum, then exact-erf GELU
#            (with reference-matching bf16-before-GELU rounding) -> bf16.
#   3. K3  : FC2 = gelu @ fc2_weight^T + fc2_bias, fp32 accum -> bf16 output.
#
# c003 changes vs c002 (both c001 & c002 failed all 5 with RUNTIME_ERROR and no
# traceback; erf/trans were already ruled out by c002 => defect is in shared
# code). Prime suspect for a *uniform* failure: the @triton.autotune sweep — one
# oversized config (BLOCK_N=256/num_stages=3/num_warps=8) can fault at launch and
# poison the CUDA context, cascading to RUNTIME_ERROR for every workload.
# Isolation change:
#   * REMOVE @triton.autotune entirely; use ONE conservative fixed GEMM config
#     (BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=3) that is
#     low on shared memory and safe on sm_80.
#   * Remove the needless early `return` in K1.
# Semantics, fp32 LN, poly-erf GELU, and direct weight-tile load are unchanged,
# so this cleanly tests the autotune hypothesis.
# =============================================================================

C = 1536            # hidden_size
E = 6144            # hidden_size_expanded = 4 * C
O = 3584            # out_hidden_size
MERGE = 2
BLOCK_C = 2048      # power-of-two cover of C=1536 (masked)
INV_SQRT2 = 0.7071067811865476

# Fixed, conservative GEMM tile (no autotune).
GEMM_BM = 64
GEMM_BN = 128
GEMM_BK = 32
GEMM_GROUP_M = 8
GEMM_WARPS = 4
GEMM_STAGES = 3


# -----------------------------------------------------------------------------
# Version-portable exact-erf (Abramowitz & Stegun 7.1.26). Max abs error 1.5e-7,
# far inside the tightest tolerance (atol 0.0014). Uses only tl.exp / arithmetic.
# -----------------------------------------------------------------------------
@triton.jit
def _erf_poly(x):
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    ax = tl.where(x < 0.0, -x, x)   # |x|
    t = 1.0 / (1.0 + p * ax)
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    y = 1.0 - poly * tl.exp(-ax * ax)
    return tl.where(x < 0.0, -y, y)


# -----------------------------------------------------------------------------
# K1: fused LayerNorm + spatial shuffle
# One program per output (merged) row; loops beta = 0..3 over the 2x2 block.
# For each beta it gathers source patch row src_idx[m, beta], LayerNorms it in
# fp32 over C, applies affine, and stores 1536 bf16 values contiguously at
# columns [beta*C : (beta+1)*C] of A_shuffled row m.
# -----------------------------------------------------------------------------
@triton.jit
def ln_shuffle_kernel(
    hidden_ptr,      # [P, C] bf16
    src_idx_ptr,     # [M, 4] int32 : source global patch row for each (m, beta)
    lnw_ptr,         # [C] bf16
    lnb_ptr,         # [C] bf16
    out_ptr,         # [M, E] bf16
    eps,
    stride_hp, stride_hc,
    stride_om, stride_oe,
    BLOCK_C: tl.constexpr,
    C_REAL: tl.constexpr,
):
    m = tl.program_id(0)

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C_REAL

    lnw = tl.load(lnw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    lnb = tl.load(lnb_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    inv_n = 1.0 / C_REAL

    for beta in range(0, 4):
        src = tl.load(src_idx_ptr + m * 4 + beta).to(tl.int64)
        x = tl.load(hidden_ptr + src * stride_hp + offs * stride_hc,
                    mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) * inv_n
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) * inv_n
        rstd = 1.0 / tl.sqrt(var + eps)
        y = xc * rstd * lnw + lnb
        o = out_ptr + m * stride_om + (beta * C_REAL + offs) * stride_oe
        tl.store(o, y.to(tl.bfloat16), mask=mask)


# -----------------------------------------------------------------------------
# K2/K3: tiled bf16 GEMM  Out[M,N] = A[M,K] @ W[N,K]^T + bias[N]
# fp32 accumulator; optional exact-erf GELU epilogue (K2 only) with
# reference-matching bf16-before-GELU rounding. Weight tile is loaded directly
# as [BLOCK_K, BLOCK_N] (no tl.trans). Single fixed config (no autotune).
# -----------------------------------------------------------------------------
@triton.jit
def gemm_kernel(
    A_ptr, W_ptr, bias_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_om, stride_on,
    APPLY_GELU: tl.constexpr,
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

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # Weight W is [N, K] row-major; load tile as [BLOCK_K, BLOCK_N] (== W^T tile)
    # so tl.dot(a[M,K], w[K,N]) needs no transpose primitive.
    w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem),
                    other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, w)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    if APPLY_GELU:
        # Match reference: FC1 output is rounded to bf16 (cuBLAS) before GELU.
        xb = acc.to(tl.bfloat16).to(tl.float32)
        acc = 0.5 * xb * (1.0 + _erf_poly(xb * INV_SQRT2))

    out = acc.to(tl.bfloat16)
    o_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# -----------------------------------------------------------------------------
# Host-side gather-index construction (metadata / plumbing only).
# Builds src_idx[M, 4] int32 giving, for output row m and block beta, the global
# input patch row to gather. One small grid_thw -> CPU transfer (G<=4).
# -----------------------------------------------------------------------------
def _build_src_idx(grid_thw, M, device):
    grid_list = grid_thw.tolist()  # single small sync; G<=4 rows of [t,h,w]
    src_idx = torch.empty((M, 4), dtype=torch.int32)  # CPU
    row_off = 0
    merged_off = 0
    for t, h, w in grid_list:
        hm = h // MERGE
        wm = w // MERGE
        n = t * hm * wm
        if n == 0:
            continue
        m_local = torch.arange(n, dtype=torch.int64)
        ww_m = m_local % wm
        tmp = m_local // wm
        hh_m = tmp % hm
        tt = tmp // hm
        for beta in range(4):
            a = beta // 2
            b = beta % 2
            h_idx = hh_m * MERGE + a
            w_idx = ww_m * MERGE + b
            src_local = (tt * h + h_idx) * w + w_idx
            src_idx[merged_off:merged_off + n, beta] = (row_off + src_local).to(torch.int32)
        row_off += t * h * w
        merged_off += n
    return src_idx.to(device, non_blocking=True)


@torch.no_grad()
def run(
    hidden: torch.Tensor,
    grid_thw: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    eps: float,
):
    device = hidden.device
    P = hidden.shape[0]
    M = P // 4

    hidden = hidden.contiguous()
    ln_weight = ln_weight.contiguous()
    ln_bias = ln_bias.contiguous()
    fc1_weight = fc1_weight.contiguous()
    fc1_bias = fc1_bias.contiguous()
    fc2_weight = fc2_weight.contiguous()
    fc2_bias = fc2_bias.contiguous()

    # ---- gather index (plumbing) ----
    src_idx = _build_src_idx(grid_thw, M, device)

    # ---- K1: LayerNorm + spatial shuffle -> A_shuffled[M, E] ----
    a_shuffled = torch.empty((M, E), dtype=torch.bfloat16, device=device)
    ln_shuffle_kernel[(M,)](
        hidden, src_idx, ln_weight, ln_bias, a_shuffled,
        float(eps),
        hidden.stride(0), hidden.stride(1),
        a_shuffled.stride(0), a_shuffled.stride(1),
        BLOCK_C=BLOCK_C, C_REAL=C,
    )

    # ---- K2: FC1 (+bias +GELU) -> [M, E] ----
    fc1_out = torch.empty((M, E), dtype=torch.bfloat16, device=device)
    grid1 = (triton.cdiv(M, GEMM_BM) * triton.cdiv(E, GEMM_BN),)
    gemm_kernel[grid1](
        a_shuffled, fc1_weight, fc1_bias, fc1_out,
        M, E, E,
        a_shuffled.stride(0), a_shuffled.stride(1),
        fc1_weight.stride(0), fc1_weight.stride(1),
        fc1_out.stride(0), fc1_out.stride(1),
        APPLY_GELU=True,
        BLOCK_M=GEMM_BM, BLOCK_N=GEMM_BN, BLOCK_K=GEMM_BK, GROUP_M=GEMM_GROUP_M,
        num_warps=GEMM_WARPS, num_stages=GEMM_STAGES,
    )

    # ---- K3: FC2 (+bias) -> [M, O] ----
    output = torch.empty((M, O), dtype=torch.bfloat16, device=device)
    grid2 = (triton.cdiv(M, GEMM_BM) * triton.cdiv(O, GEMM_BN),)
    gemm_kernel[grid2](
        fc1_out, fc2_weight, fc2_bias, output,
        M, O, E,
        fc1_out.stride(0), fc1_out.stride(1),
        fc2_weight.stride(0), fc2_weight.stride(1),
        output.stride(0), output.stride(1),
        APPLY_GELU=False,
        BLOCK_M=GEMM_BM, BLOCK_N=GEMM_BN, BLOCK_K=GEMM_BK, GROUP_M=GEMM_GROUP_M,
        num_warps=GEMM_WARPS, num_stages=GEMM_STAGES,
    )

    return output
