"""KDA candidate c004 — Plain NT GEMM, occupancy-tuned (no split-K, no atomics).

Task: C = A @ B.T, fp16, A[M,K], B[N,K], C[M,N]; N=K=4096, M variable/small.
Target: NVIDIA A800 (sm_80, Ampere). Triton-only compute.

Lineage / evidence:
  * c001 plain NT GEMM (autotune BLOCK_N in {64,128,256}, BLOCK_M in {16,32,64},
    num_stages 3|4, no split-K): VALID, geomean 0.56x. Best valid baseline so far.
  * c002 atomic split-K: INVALID (autotune re-launch overflowed the un-reset atomic
    buffer -> non-finite).
  * c003 atomic split-K with reset_to_zero fix: VALID but SLOWER (0.38x) — the
    tl.atomic_add contention + full fp32 memset + separate cast pass cost more than the
    occupancy split-K bought. => split-K's reduction overhead is net-negative here.

Single structural change vs the c001 baseline: keep the plain (no-split-K, no-atomic,
deterministic fp16 store) GEMM, but WIDEN the autotune space to raise concurrency on the
memory-bound small-M cases the c001 space could not reach:
  * add small BLOCK_N (32) so a single M-tile yields ~128 N-tiles (>= 108 SMs) with no
    reduction pass — pure occupancy for M=4/48/64 where c001 launched too few blocks;
  * add deeper num_stages (5) for more cp.async pipeline depth to hide HBM latency
    (the dominant cost for the mem-bound skinny shapes);
  * add L2 GROUP_SIZE_M reordering so B N-blocks are reused across M-tiles for larger M
    (M=128/240, where num_pid_m > 1);
  * keep large BLOCK_N/BLOCK_M options so the key=['M'] tuner still picks MMA-efficient
    tiles for the near-ridge M=240 case.
No atomics, no scratch, no cast pass, B read exactly once, deterministic output.

Correctness (static checklist):
  * A[m,k]=m*stride_am+k*stride_ak ; B[n,k]=n*stride_bn+k*stride_bk ;
    C[m,n]=m*stride_cm+n*stride_cn. Strides taken from tensors (contiguous enforced).
  * b tile is [BLOCK_K, BLOCK_N] = B.T view: offs_k*stride_bk + offs_n*stride_bn, i.e.
    logical (k,n) at B[n,k]. tl.dot(a,b) with a=[BM,BK] -> acc[BM,BN]. Correct A@B.T.
  * N=4096 divisible by every BLOCK_N, K=4096 divisible by every BLOCK_K -> no N/K mask.
    M may be < BLOCK_M -> mask offs_m < M on A load (other=0.0) and on C store.
  * fp32 accumulator (matches torch fp16->fp32 accumulate); cast to fp16 at store.
"""

import torch
import triton
import triton.language as tl


def _configs():
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages)
    specs = [
        # --- occupancy-first (small M, memory bound): small BLOCK_N -> many blocks ---
        (16, 32, 64, 8, 4, 4),
        (16, 32, 64, 8, 4, 5),
        (16, 32, 32, 8, 4, 5),
        (16, 64, 64, 8, 4, 4),
        (16, 64, 64, 8, 4, 5),
        (32, 32, 64, 8, 4, 4),
        (32, 64, 64, 8, 4, 4),
        (32, 64, 64, 8, 8, 4),
        # --- balanced ---
        (16, 128, 64, 8, 4, 4),
        (64, 64, 64, 8, 8, 4),
        (64, 128, 64, 8, 8, 4),
        (64, 128, 64, 8, 8, 5),
        # --- MMA-efficient (larger M, near ridge) ---
        (64, 256, 64, 8, 8, 3),
        (128, 128, 64, 8, 8, 4),
        (128, 256, 64, 8, 8, 3),
        (128, 128, 32, 8, 8, 4),
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
    # L2-friendly grouped ordering along M.
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
    # B.T view: logical (k, n) lives at B[n, k] -> offset k*stride_bk + n*stride_bn
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    m_mask = offs_m[:, None] < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
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
