# KDA A800 best solution: gemm_n4096_k4096
# candidate: c007  |  feedback: 0.58x  |  final (authoritative): 0.59x
# campaign formal-kda-20260916 (A800, g0056)  |  evaluations: 8
# source: tasks/formal-kda-20260916--flashinfer--gemm_n4096_k4096/control/candidates/c007/solution.py (sha256-frozen snapshot)

"""KDA candidate c007 — plain NT GEMM, occupancy x burst-width corner (BLOCK_N=32 + wide BLOCK_K).

Task: C = A @ B.T, fp16, A[M,K], B[N,K], C[M,N]; N=K=4096, M variable/small.
Target: NVIDIA A800 (sm_80, Ampere). Triton-only compute.

Lineage / evidence:
  * c001/c004/c005 plain no-split-K NT GEMM all plateau at ~0.56-0.57x geomean (~440 GB/s on the
    33.5MB B read; cuBLAS ref ~735 GB/s). c005 (coalesced [N,K] B load + tl.trans, wide BLOCK_K)
    is the best valid baseline (0.57x).
  * c003 atomic split-K 0.38x; c006 deterministic two-pass split-K 0.49x. BOTH split-K reduction
    strategies lose -> the extra pass / atomic contention costs more than the occupancy it buys at
    these ~50-100us latencies. Split-K is exhausted for this skinny NT GEMM.

Root-cause the plateau targets (single coherent hypothesis for c007):
  For skinny M (single M-tile) with B read exactly once, the block count equals ceil(N/BLOCK_N).
  To fill A800's 108 SMs we need BLOCK_N=32 -> 128 blocks. But the two prior plain sweeps never
  combined that with wide K bursts:
    * c004 used BLOCK_N=32 only with narrow BLOCK_K<=64 -> full occupancy but short/narrow HBM
      bursts and short K-loops.
    * c005 used wide BLOCK_K=128/256 only with BLOCK_N>=64 -> <=64 blocks (~60% occupancy).
  c007 tests the UNTESTED intersection: BLOCK_N=32 (128 blocks = full occupancy) TOGETHER WITH
  wide BLOCK_K=128/256 (long coalesced B bursts) and deep cp.async pipelining (num_stages 4/5),
  with NO split-K and NO reduction pass. Hypothesis: maximizing occupancy AND per-block burst
  width simultaneously in the plain path lifts small-M above the 0.57x plateau.

  The kernel body is byte-for-byte the proven c005 kernel (coalesced [BLOCK_N,BLOCK_K] B load +
  in-register tl.trans, fp32 accumulate, fp16 store). The ONLY change vs c005 is the autotune
  config list: add the BLOCK_N=32 x wide-BLOCK_K x deep-stage configs for small M, while keeping
  c005's proven larger-M configs so autotune key=['M'] preserves M=128/240 (which pick BLOCK_M>=64
  to avoid re-reading B; BLOCK_N=32 with many m-tiles would re-read B and is only offered to the
  single-M-tile regime by having BLOCK_M small there).

Correctness (static checklist, unchanged from c005):
  * a tile [BLOCK_M,BLOCK_K]: offs_m*stride_am + offs_k*stride_ak (stride_ak=1 -> coalesced along k).
  * b tile [BLOCK_N,BLOCK_K]: offs_n*stride_bn + offs_k*stride_bk (stride_bk=1 -> coalesced);
    acc = tl.dot(a, tl.trans(b)) = sum_k a[m,k]*b[n,k] = (A @ B.T)[m,n]. Correct.
  * N=K=4096 divisible by every BLOCK_N in {32,64,128,256} and BLOCK_K in {32,64,128,256}
    -> no N/K masking. M may be < BLOCK_M -> mask offs_m<M on A load (other=0.0) and C store.
  * fp32 accumulator (matches torch fp16->fp32 accumulate); cast to fp16 at store. Deterministic.
"""

import torch
import triton
import triton.language as tl


def _configs():
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages)
    specs = [
        # --- NEW: full-occupancy (BLOCK_N=32 -> 128 blocks for a single M-tile) x wide K burst ---
        (16, 32, 128, 8, 4, 4),
        (16, 32, 128, 8, 4, 5),
        (16, 32, 256, 8, 4, 4),
        (16, 32, 256, 8, 8, 4),
        (32, 32, 128, 8, 4, 4),
        (32, 32, 256, 8, 4, 4),
        # --- 64-block occupancy variants with wide K (bridge between c004 and c005) ---
        (16, 64, 128, 8, 4, 4),
        (16, 64, 256, 8, 8, 3),
        (16, 64, 128, 8, 4, 5),
        (32, 64, 128, 8, 4, 4),
        # --- c005 proven small-M configs (kept so autotune never regresses below baseline) ---
        (16, 128, 128, 8, 4, 3),
        (16, 128, 256, 8, 8, 3),
        (16, 256, 128, 8, 8, 3),
        (32, 128, 128, 8, 8, 3),
        (32, 128, 256, 8, 8, 3),
        # --- c005 proven mid/large-M configs (BLOCK_M>=64 -> few m-tiles, B read ~once) ---
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
