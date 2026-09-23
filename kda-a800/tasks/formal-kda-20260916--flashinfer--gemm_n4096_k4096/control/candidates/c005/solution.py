"""KDA candidate c005 — NT GEMM, B-read-efficiency focused (coalesced [N,K] load + wide BLOCK_K).

Task: C = A @ B.T, fp16, A[M,K], B[N,K], C[M,N]; N=K=4096, M variable/small.
Target: NVIDIA A800 (sm_80, Ampere). Triton-only compute.

Lineage / evidence:
  * c001 plain NT GEMM: VALID 0.56x. c004 wider plain autotune: VALID 0.56x (same plateau).
  * c003 atomic split-K: VALID but 0.38x (contention + extra passes). c002 invalid.
  * All plain variants plateau at ~0.56x => ~440 GB/s effective on the 33.5MB B read,
    whereas cuBLAS (the reference) runs ~44-70us => ~735 GB/s. The op is dominated by
    reading B once; the gap to cuBLAS is B-read EFFICIENCY, not occupancy alone
    (small-M configs already launch >=32-128 blocks).

Single coherent hypothesis this candidate tests (H: B-read efficiency is the bottleneck):
  Restructure the loads to the canonical Ampere NT form and widen the K burst:
    * Load B tile as [BLOCK_N, BLOCK_K] with the stride-1 K axis as the INNER (contiguous)
      tile dimension -> fully coalesced 128-bit HBM loads; transpose in-register
      (tl.trans, Ampere ldmatrix.trans) before the MMA. (c004 loaded B as [BK,BN] with
      the stride-1 axis on the outer tile dim and relied on the layout optimizer.)
    * Widen BLOCK_K to 128/256 so each contiguous B burst is longer -> larger, fewer HBM
      transactions and better DRAM burst utilization. K=4096 divisible by all BLOCK_K.
  Everything else stays the proven deterministic no-split-K path (B read exactly once,
  fp32 accumulate, fp16 store, autotune key=['M']). No atomics, no scratch, no cast pass.

Correctness (static checklist):
  * a tile: offs_m*stride_am + offs_k*stride_ak (stride_ak=1 -> coalesced along k).
  * b tile: offs_n*stride_bn + offs_k*stride_bk (stride_bk=1 -> coalesced along k), shape
    [BLOCK_N, BLOCK_K]. acc = tl.dot(a, tl.trans(b)) computes
    acc[m,n] = sum_k a[m,k]*b[n,k] = (A @ B.T)[m,n]. Correct.
  * N,K = 4096 divisible by every BLOCK_N/BLOCK_K -> no N/K masking. M may be < BLOCK_M ->
    mask offs_m < M on A load (other=0.0) and on C store.
  * fp32 accumulator (matches torch fp16->fp32 accumulate); cast to fp16 at store.
"""

import torch
import triton
import triton.language as tl


def _configs():
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages)
    # Emphasis: wide BLOCK_K (128/256) for long coalesced B bursts; single-M-tile options
    # for every feedback M so B is read exactly once.
    specs = [
        # small M single-tile, wide K bursts, high occupancy via smaller BLOCK_N
        (16, 64, 128, 8, 4, 3),
        (16, 64, 256, 8, 4, 3),
        (16, 128, 128, 8, 4, 3),
        (16, 128, 256, 8, 8, 3),
        (16, 256, 128, 8, 8, 3),
        (32, 64, 128, 8, 4, 3),
        (32, 128, 128, 8, 8, 3),
        (32, 128, 256, 8, 8, 3),
        # mid M
        (64, 128, 128, 8, 8, 3),
        (64, 128, 256, 8, 8, 3),
        (64, 256, 128, 8, 8, 3),
        # larger M
        (128, 128, 128, 8, 8, 3),
        (128, 256, 64, 8, 8, 3),
        (128, 256, 128, 8, 8, 3),
        (256, 128, 64, 8, 8, 3),
        (256, 128, 128, 8, 8, 3),
    ]
    cfgs = []
    for BM, BN, BK, GM, nw, ns in specs:
        cfgs.append(
            triton.Config(
                {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "GROUP_SIZE_M": GM},
                num_warps=nw,
                num_stages=ns,
            )
        )
    return cfgs


@triton.autotune(configs=_configs(), key=["M"])
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

    # A tile [BLOCK_M, BLOCK_K]: k contiguous (stride_ak==1) -> coalesced.
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B tile [BLOCK_N, BLOCK_K]: k contiguous (stride_bk==1) -> coalesced; transpose in-reg.
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
