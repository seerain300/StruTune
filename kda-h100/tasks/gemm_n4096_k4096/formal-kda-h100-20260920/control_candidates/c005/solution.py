"""KDA candidate c005 — gemm_n4096_k4096 (FlashInfer, H100 / sm_90).

Operation: C = A @ B.T   (A [M,K] f16, B [N,K] f16, C [M,N] f16; N=K=4096).

Lineage: parent = c004 (geomean ~0.58x; deterministic M-bucket table, BLOCK_N=32,
BLOCK_K=128 on small M, K-mask removed). Prior findings:
  * c003 (reject): 128->256 CTAs via BLOCK_N=16 did nothing => small-M is NOT
    occupancy/bandwidth bound beyond one wave.
  * c004 (accept): halving the serial K-loop (BLOCK_K 64->128, 64->32 iters) shaved
    ~5-7% off the small-M sol floor => the per-CTA serial-K dependency chain IS a
    real component of the ~0.05ms floor. Remaining gap to cuBLAS (~0.028ms) ~1.8x.

c005 hypothesis (H7): even at BLOCK_K=128 each small-M CTA still walks all K=4096
serially (32 dependent MMA steps) while only ~128 CTAs (one wave) are resident, so
the machine is under-occupied ALONG THE K AXIS. Split-K partitions K across SPLIT_K
CTAs (each does 4096/SPLIT_K of the contraction -> 8 K-iters at SPLIT_K=4), raising
resident CTAs to n_ntiles*SPLIT_K = 128*4 = 512 with INDEPENDENT work that overlaps
latency — the axis c003 could not exercise. This directly attacks the serial-K floor.

Design (deterministic, FP32 throughout, no atomics -> deterministic reduction):
  * Stage 1 `_gemm_splitk_kernel`: grid (num_mtiles*num_ntiles, SPLIT_K). Each CTA
    computes its (m,n) tile over its K-slice [pid_k*Kc, (pid_k+1)*Kc) in FP32 and
    STORES (plain, no atomics) into a separate partial buffer Cp[SPLIT_K, M, N] fp32.
  * Stage 2 `_reduce_cast_kernel`: sums Cp over the SPLIT_K axis in FP32 and casts to
    FP16 into C. Fixed reduction order => bit-deterministic run to run.
  * K-slice Kc = K // SPLIT_K = 1024 is an exact multiple of BLOCK_K=128 (8 iters),
    and SPLIT_K | 4096, so K coverage is exact with no tail and no overlap.
  * Gate: split-K only for the band 16 <= M <= 256 (enough work to amortize the 2nd
    launch + partial traffic). Tiny M (<16) and large M (>256) keep the proven c004
    single-pass path (extra launches would dominate tiny; large is compute-bound).

Numerics: FP32 partial accumulate + FP32 reduction, FP16 only at the final cast —
matches cuBLAS FP32-accumulate. M-masked A-loads / partial-stores / C-stores.
Triton is the only compute path (torch.empty is allocation plumbing only).
"""

import torch
import triton
import triton.language as tl


# ------------------------- single-pass kernel (c004) -------------------------
@triton.jit
def _gemm_nt_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
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
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_am = offs_m % M
    offs_bn = offs_n % N

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ------------------------- split-K stage 1 (partials) ------------------------
@triton.jit
def _gemm_splitk_kernel(
    A, B, Cp,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cps, stride_cpm, stride_cpn,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
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

    offs_am = offs_m % M
    offs_bn = offs_n % N

    # K-slice for this split. Kc = K // SPLIT_K is an exact multiple of BLOCK_K.
    Kc = K // SPLIT_K
    k_start = pid_k * Kc

    a_ptrs = A + (offs_am[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak)
    b_ptrs = B + ((k_start + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, Kc, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cp_ptrs = Cp + (pid_k * stride_cps
                    + offs_cm[:, None] * stride_cpm
                    + offs_cn[None, :] * stride_cpn)
    cp_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(cp_ptrs, acc, mask=cp_mask)


# ------------------------- split-K stage 2 (reduce+cast) ---------------------
@triton.jit
def _reduce_cast_kernel(
    Cp, C,
    M, N,
    stride_cps, stride_cpm, stride_cpn,
    stride_cm, stride_cn,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        p_ptrs = Cp + (s * stride_cps
                       + offs_m[:, None] * stride_cpm
                       + offs_n[None, :] * stride_cpn)
        acc += tl.load(p_ptrs, mask=mask, other=0.0)

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def _single_pass_config(M):
    """c004 configs for the non-split path (tiny M<16 and large M>256)."""
    if M <= 64:
        return dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=128, GROUP_M=1,
                    num_warps=4, num_stages=4)
    elif M <= 128:
        return dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=128, GROUP_M=1,
                    num_warps=4, num_stages=4)
    elif M <= 256:
        return dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=128, GROUP_M=8,
                    num_warps=4, num_stages=4)
    else:
        return dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, GROUP_M=8,
                    num_warps=8, num_stages=3)


def run(A, B):
    assert A.dim() == 2 and B.dim() == 2, "A, B must be 2-D"
    M, K = A.shape
    N, Kb = B.shape
    assert K == Kb, "K mismatch between A and B"

    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Split-K band: enough work to amortize the second launch + partial traffic,
    # and the per-CTA serial-K chain is the dominant floor here.
    if 16 <= M <= 256:
        SPLIT_K = 4
        BLOCK_M = 64 if M <= 64 else 128
        BLOCK_N = 32
        BLOCK_K = 128
        GROUP_M = 8

        num_pid_m = triton.cdiv(M, BLOCK_M)
        num_pid_n = triton.cdiv(N, BLOCK_N)

        Cp = torch.empty((SPLIT_K, M, N), dtype=torch.float32, device=A.device)

        grid1 = (num_pid_m * num_pid_n, SPLIT_K)
        _gemm_splitk_kernel[grid1](
            A, B, Cp,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            Cp.stride(0), Cp.stride(1), Cp.stride(2),
            SPLIT_K=SPLIT_K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
            num_warps=4, num_stages=3,
        )

        RBM, RBN = 64, 128
        grid2 = (triton.cdiv(M, RBM), triton.cdiv(N, RBN))
        _reduce_cast_kernel[grid2](
            Cp, C,
            M, N,
            Cp.stride(0), Cp.stride(1), Cp.stride(2),
            C.stride(0), C.stride(1),
            SPLIT_K=SPLIT_K,
            BLOCK_M=RBM, BLOCK_N=RBN,
            num_warps=4, num_stages=2,
        )
        return C

    # Non-split single-pass path (c004).
    cfg = _single_pass_config(M)
    grid = (triton.cdiv(M, cfg['BLOCK_M']) * triton.cdiv(N, cfg['BLOCK_N']),)
    _gemm_nt_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=cfg['BLOCK_M'], BLOCK_N=cfg['BLOCK_N'], BLOCK_K=cfg['BLOCK_K'],
        GROUP_M=cfg['GROUP_M'],
        num_warps=cfg['num_warps'], num_stages=cfg['num_stages'],
    )
    return C
