"""KDA candidate c006 — NT GEMM with deterministic two-pass Split-K (modest, M-adaptive).

Task: C = A @ B.T, fp16, A[M,K], B[N,K], C[M,N]; N=K=4096, M variable/small.
Target: NVIDIA A800 (sm_80, Ampere). Triton-only compute.

Lineage / evidence:
  * c001/c004/c005 plain no-split-K NT GEMM all plateau at ~0.56-0.57x geomean.
    Analysis: for skinny M there is a single M-tile, so the block count is just
    num_pid_n = ceil(N/BLOCK_N) (32 at BLOCK_N=128, 64 at BLOCK_N=64) -> most of the
    108 SMs sit idle and HBM (reading the 33.5MB B once) is under-fed. We hit ~440 GB/s
    vs cuBLAS ~735 GB/s -> the plain path is OCCUPANCY-limited on small M.
  * c002 atomic split-K: INVALID (autotune re-launch w/o reset_to_zero overflowed C32).
  * c003 atomic split-K (fixed reset): VALID but 0.38x — atomic contention + full torch.zeros
    memset of C32 + separate cast pass + over-split (SPLIT_K 8-16 => 4-8 iter K loops killed
    cp.async steady state) cost more than the occupancy bought. That confounded H3; it did NOT
    cleanly test whether occupancy is the win.

Single coherent hypothesis this candidate tests (H3, cleanly):
  Raising the block count with a MODEST split of K (SPLIT_K 2-4, so K-loops stay >=16 iters and
  cp.async pipelining survives) and combining partials with a DETERMINISTIC two-pass reduction
  (no atomics, no contention, no full-buffer memset) restores occupancy on small M and beats the
  plain plateau — while reading B exactly once (each K-slice read once across the SPLIT_K blocks).

Design:
  * SPLIT_K is chosen in Python from M (closed-form heuristic, generalizes beyond the 5 feedback
    M's): target at least one full 108-SM wave of output tiles, capped at 4, powers of two.
      - M<=64  -> 1 m-tile  -> SPLIT_K=4  (32 n-tiles * 4 = 128 blocks)
      - M~128  -> 2 m-tiles -> SPLIT_K=2  (64 * 2 = 128 blocks)
      - M>=240 -> >=4 m-tiles (>=128 blocks) -> SPLIT_K=1
  * SPLIT_K==1 fast path: the proven plain deterministic kernel writes fp16 straight to C — no
    partials, no reduce pass — so large M (M=240) is byte-for-byte the c005 path (no regression).
  * SPLIT_K>1 path (kernel 1): each (pid_mn, pid_k) block reduces its K-slice [k_start,k_start+K/SK)
    with fp32 accum and writes the partial to Cpart[pid_k, m, n] (fp32 [SPLIT_K, M, N]). Every
    (split, valid-m, n) is written exactly once -> no zero-init needed, fully deterministic.
  * Reduction (kernel 2): sums the SPLIT_K fp32 partials per output tile, casts to fp16, stores C.

Correctness (static checklist):
  * a tile: offs_m*stride_am + (k_start+offs_k)*stride_ak (stride_ak=1 -> coalesced along k).
  * b tile: offs_n*stride_bn + (k_start+offs_k)*stride_bk (stride_bk=1 -> coalesced), shape
    [BLOCK_N, BLOCK_K]; acc += tl.dot(a, tl.trans(b)) = sum_k a[m,k]*b[n,k] = (A@B.T)[m,n]. Correct.
  * K=4096; K/SPLIT_K in {4096,2048,1024} is a multiple of every BLOCK_K in {32,64,128} -> no K
    masking inside the loop. N=4096 divisible by every BLOCK_N -> no N masking. M may be < BLOCK_M
    -> mask offs_m<M on A load (other=0.0), on the partial store, and on the C store.
  * fp32 accumulate everywhere (matches torch fp16->fp32 accumulate); a single final cast to fp16.
    Two-pass fp32 reduction stays at fp32-ULP error, trivially inside atol=rtol=0.01 (ratio 0.99).
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------------------------
# Plain no-split-K kernel (SPLIT_K == 1 fast path). Identical structure to the c005 baseline:
# coalesced [BLOCK_N, BLOCK_K] B load + in-register transpose, fp32 accum, fp16 store to C.
# --------------------------------------------------------------------------------------------
def _plain_configs():
    specs = [
        (16, 64, 128, 8, 4, 3),
        (16, 128, 128, 8, 4, 3),
        (16, 128, 256, 8, 8, 3),
        (16, 256, 128, 8, 8, 3),
        (32, 128, 128, 8, 8, 3),
        (32, 128, 256, 8, 8, 3),
        (64, 128, 128, 8, 8, 3),
        (64, 256, 128, 8, 8, 3),
        (128, 128, 128, 8, 8, 3),
        (128, 256, 64, 8, 8, 3),
        (128, 256, 128, 8, 8, 3),
        (256, 128, 128, 8, 8, 3),
    ]
    return [
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "GROUP_SIZE_M": GM},
            num_warps=nw,
            num_stages=ns,
        )
        for BM, BN, BK, GM, nw, ns in specs
    ]


@triton.autotune(configs=_plain_configs(), key=["M"])
@triton.jit
def _gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    m_mask = offs_m[:, None] < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=m_mask)


# --------------------------------------------------------------------------------------------
# Split-K partial kernel (SPLIT_K > 1). Writes fp32 partials to Cpart[SPLIT_K, M, N].
# --------------------------------------------------------------------------------------------
def _splitk_configs():
    # Keep BLOCK_M=16 (min MMA tile -> least padding waste on skinny M) and BLOCK_K<=64 so a
    # K-slice of K/SPLIT_K has >=16 loop iters (SPLIT_K<=4, K=4096) -> cp.async steady state.
    specs = [
        (16, 64, 64, 8, 4, 3),
        (16, 64, 64, 8, 4, 4),
        (16, 64, 32, 8, 4, 4),
        (16, 128, 64, 8, 4, 3),
        (16, 128, 64, 8, 8, 3),
        (16, 128, 32, 8, 8, 4),
        (32, 64, 64, 8, 4, 3),
        (32, 128, 64, 8, 8, 3),
    ]
    return [
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "GROUP_SIZE_M": GM},
            num_warps=nw,
            num_stages=ns,
        )
        for BM, BN, BK, GM, nw, ns in specs
    ]


@triton.autotune(configs=_splitk_configs(), key=["M", "SPLIT_K"])
@triton.jit
def _splitk_kernel(
    A, B, Cpart,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_ps, stride_pm, stride_pn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_mn = pid // SPLIT_K
    pid_k = pid % SPLIT_K

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    k_per_split = K // SPLIT_K
    k_start = pid_k * k_per_split

    a_ptrs = A + (offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak)
    b_ptrs = B + (offs_n[:, None] * stride_bn + (k_start + offs_k[None, :]) * stride_bk)
    m_mask = offs_m[:, None] < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k_per_split, BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    p_ptrs = Cpart + (pid_k * stride_ps + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn)
    tl.store(p_ptrs, acc, mask=m_mask)


@triton.jit
def _reduce_kernel(
    Cpart, C,
    M, N,
    stride_ps, stride_pm, stride_pn,
    stride_cm, stride_cn,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m[:, None] < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    p_ptrs = Cpart + (offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn)
    for _ in range(0, SPLIT_K):
        acc += tl.load(p_ptrs, mask=m_mask, other=0.0)
        p_ptrs += stride_ps

    c = acc.to(tl.float16)
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=m_mask)


def _pick_split_k(M):
    # Estimate output-tile count with reference tiling BLOCK_M~64, BLOCK_N~128 (32 n-tiles).
    # Target >=1 full 108-SM wave; modest cap of 4 (keep K-loops long, partial traffic small).
    m_tiles = (M + 63) // 64
    base = m_tiles * 32
    sk = 1
    while sk < 4 and base * sk < 108:
        sk *= 2
    return sk


def run(A, B):
    assert A.dim() == 2 and B.dim() == 2, "A,B must be 2D"
    M, K = A.shape
    N, Kb = B.shape
    assert K == Kb, "inner dims must match (A[M,K], B[N,K])"

    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    split_k = _pick_split_k(M)

    if split_k == 1:
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )
        _gemm_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )
        return C

    # Split-K: deterministic two-pass. Partials in fp32 [SPLIT_K, M, N].
    Cpart = torch.empty((split_k, M, N), device=A.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]) * split_k,
    )
    _splitk_kernel[grid](
        A, B, Cpart,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        Cpart.stride(0), Cpart.stride(1), Cpart.stride(2),
        SPLIT_K=split_k,
    )

    RBM, RBN = 32, 128
    reduce_grid = (triton.cdiv(M, RBM) * triton.cdiv(N, RBN),)
    _reduce_kernel[reduce_grid](
        Cpart, C,
        M, N,
        Cpart.stride(0), Cpart.stride(1), Cpart.stride(2),
        C.stride(0), C.stride(1),
        SPLIT_K=split_k,
        BLOCK_M=RBM,
        BLOCK_N=RBN,
    )
    return C
