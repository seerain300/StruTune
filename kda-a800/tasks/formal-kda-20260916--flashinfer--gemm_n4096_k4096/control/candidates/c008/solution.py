"""KDA candidate c008 — M<=8 non-MMA GEMV-streaming path + c007 tl.dot path for M>8.

Task: C = A @ B.T, fp16, A[M,K], B[N,K], C[M,N]; N=K=4096, M variable/small.
Target: NVIDIA A800 (sm_80, Ampere). Triton-only compute.

Lineage / evidence:
  * Plain tl.dot NT GEMM plateau: c001 0.56, c004 0.56, c005 0.57, c007 0.58 (best valid).
  * Split-K exhausted: c003 atomic 0.38, c006 deterministic two-pass 0.49 (both slower — the
    reduction pass/atomic contention costs more than the occupancy at ~50-100us latencies).
  * For the tiniest M the op is NOT at HBM peak: M=4 runs ~74us for a 33.5MB B read => ~450 GB/s,
    far below cuBLAS ~746 GB/s and the ~2 TB/s peak (~17us). So at tiny M the kernel is dominated
    by launch/latency/per-block setup, not raw bandwidth. The tl.dot path also pads M=4 -> 16 MMA
    rows (75% wasted MMA rows) and pays MMA operand-staging (ldmatrix/smem) overhead per block.

Single coherent hypothesis this candidate tests (H8):
  For very small M (M<=8) a LEAN non-MMA GEMV-streaming kernel — each block owns a strip of
  BLOCK_N output columns, streams K with coalesced [BLOCK_N,BLOCK_K] B loads and a small
  [BLOCK_M,BLOCK_K] A load, and accumulates with a plain fp32 broadcast-multiply + reduce
  (tl.sum over K), NO tensor cores, NO 16-row padding — reduces per-block setup/latency and lifts
  tiny-M throughput above the tl.dot plateau (0.58x). Compute is vector-ALU but the op is
  memory/latency bound at tiny M so FMA throughput is not the ceiling; B is still read exactly once
  (same HBM traffic as the MMA path).

Dispatch: M<=8 -> GEMV kernel; M>8 -> the proven c007 tl.dot kernel (byte-for-byte), so M=48/64/
128/240 are unchanged (they already pick BLOCK_M>=16..256 and use MMA efficiently).

Correctness (static checklist):
  GEMV kernel:
   * a tile [BLOCK_M,BLOCK_K]: offs_m*stride_am + offs_k*stride_ak (stride_ak=1 -> coalesced),
     mask offs_m<M (other=0.0). BLOCK_M=8 constexpr covers M<=8.
   * b tile [BLOCK_N,BLOCK_K]: offs_n*stride_bn + offs_k*stride_bk (stride_bk=1 -> coalesced).
   * acc[m,n] += sum_k a[m,k]*b[n,k] via tl.sum(a[:,None,:]*b[None,:,:], axis=2) over each K-chunk
     -> total = sum_k a[m,k]*b[n,k] = (A@B.T)[m,n]. Correct. fp32 multiply+accumulate; fp16 store.
   * K=4096 divisible by every BLOCK_K in {16,32,64}; N=4096 divisible by every BLOCK_N in
     {32,64,128} -> no N/K masking. Only the M rows are masked.
  tl.dot kernel: identical to c007 (see that record) — coalesced [BLOCK_N,BLOCK_K] B load +
  tl.trans, fp32 accum, fp16 store, mask offs_m<M; N=K=4096 divisible by all blocks.
"""

import torch
import triton
import triton.language as tl


# ============================================================================================
# Plain tl.dot NT GEMM (c007, unchanged) — used for M > 8.
# ============================================================================================
def _dot_configs():
    specs = [
        (16, 32, 128, 8, 4, 4),
        (16, 32, 128, 8, 4, 5),
        (16, 32, 256, 8, 4, 4),
        (16, 32, 256, 8, 8, 4),
        (32, 32, 128, 8, 4, 4),
        (32, 32, 256, 8, 4, 4),
        (16, 64, 128, 8, 4, 4),
        (16, 64, 256, 8, 8, 3),
        (16, 64, 128, 8, 4, 5),
        (32, 64, 128, 8, 4, 4),
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


@triton.autotune(configs=_dot_configs(), key=["M"])
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


# ============================================================================================
# GEMV-streaming NT kernel (no MMA) — used for M <= 8. BLOCK_M fixed at 8, rows masked to M.
# ============================================================================================
def _gemv_configs():
    # (BLOCK_N, BLOCK_K, num_warps, num_stages). BLOCK_M is fixed 8. Keep BLOCK_N*BLOCK_K modest
    # so the [8,BLOCK_N,BLOCK_K] broadcast product's per-thread register footprint is bounded.
    specs = [
        (32, 32, 8, 3),
        (32, 16, 4, 4),
        (32, 64, 8, 3),
        (64, 32, 8, 3),
        (64, 16, 8, 4),
        (128, 16, 8, 3),
        (64, 32, 8, 4),
        (32, 32, 4, 4),
    ]
    return [
        triton.Config(
            {"BLOCK_N": BN, "BLOCK_K": BK},
            num_warps=nw,
            num_stages=ns,
        )
        for BN, BK, nw, ns in specs
    ]


@triton.autotune(configs=_gemv_configs(), key=["M"])
@triton.jit
def _gemv_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    m_mask = offs_m[:, None] < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]
        b = tl.load(b_ptrs).to(tl.float32)                          # [BLOCK_N, BLOCK_K]
        # acc[m,n] += sum_k a[m,k]*b[n,k]
        acc += tl.sum(a[:, None, :] * b[None, :, :], axis=2)
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

    if M <= 8:
        grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
        _gemv_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=8,
        )
        return C

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
