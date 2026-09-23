"""KDA candidate c003 — Split-K NT GEMM (fp32 atomic reduction + Triton cast).

Task: C = A @ B.T, fp16, A[M,K], B[N,K], C[M,N]; N=K=4096, M variable/small.
Target: NVIDIA A800 (sm_80, Ampere). Triton-only compute.

Lineage:
  * c001 (plain NT GEMM, no split-K): correct, 0.56x geomean — too few output tiles
    (<=32) to occupy 108 SMs / hide HBM latency on skinny M (confirmed H2).
  * c002 (this file's structure, atomic split-K) evaluated INVALID: 0/5, all
    "C: non-finite output mismatch". Root cause identified by reading the installed
    triton 3.5.0 autotuner: @triton.autotune benchmarks each candidate config by
    launching the kernel many times via do_bench. Because the kernel uses
    tl.atomic_add into C32 and NO reset_to_zero was declared, every benchmark launch
    accumulated into the SAME un-zeroed C32 buffer -> the fp32 sum grew unbounded ->
    overflowed fp16 on the cast -> inf/nan. The GEMM math itself is correct.

Single change vs c002:  add reset_to_zero=["C32"] to the autotuner.
  * The autotuner's pre_hook now zeroes C32 before each benchmarked launch (so every
    timed config sees a single clean accumulation) and once more (reset_only=True)
    before the real launch on the first/autotuning call. On later calls the config is
    cached, the benchmark is skipped, and run() already allocates a fresh torch.zeros
    C32 each invocation, so correctness holds. This directly enables testing H3 (does
    split-K occupancy beat cuBLAS on skinny M) which c002 could not reach.

Design (unchanged, restated for the static checklist):
  * READ B EXACTLY ONCE. Each program owns one (pid_n, pid_k): it reads B[offs_n,
    k_slice]. Across pid_k the K axis is partitioned (disjoint), across pid_n the N
    axis is partitioned (disjoint) -> total B HBM traffic = 1x. Split-K adds occupancy
    without re-reading B. A is re-read per pid_n but A is tiny (<=2MB) and lives in the
    40MB L2, so its HBM traffic stays ~1x. => still HBM-bound on B (~33.5MB).
  * fp32 accumulation (matches torch.matmul fp16->fp32 accumulate); cast to fp16 last.
  * K=4096 divisible by SPLIT_K (power of two) and each K-chunk (K/SPLIT_K in
    {256,512,1024,2048}) divisible by BLOCK_K=64 -> no K masking in the inner loop.
    N=4096 divisible by every BLOCK_N -> no N masking. M may be < BLOCK_M -> mask rows
    offs_m < M on the A load and the atomic store.
  * Autotune key=['M']: per M the tuner picks a single-M-tile BLOCK_M plus a SPLIT_K
    that pushes the block count past ~2x108. Warmup (3 iters) absorbs the one-time
    autotune so it is excluded from the evaluator's timing.
"""

import torch
import triton
import triton.language as tl


def _configs():
    # (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages)
    # For every feedback M there is a "single M-tile (num_pid_m==1) + SPLIT_K for
    # occupancy" option (B read once), plus safe fallbacks.
    # BLOCK_K fixed at 64: divides every K-chunk (K/SPLIT_K in {256,512,1024,2048}).
    specs = [
        # M ~ 4  -> BLOCK_M=16 single tile, high split
        (16, 128, 64, 8, 4, 3),
        (16, 128, 64, 16, 4, 3),
        (16, 64, 64, 16, 4, 3),
        # M ~ 48/64 -> BLOCK_M=64 single tile
        (64, 128, 64, 8, 8, 3),
        (64, 64, 64, 8, 8, 3),
        (64, 128, 64, 4, 8, 3),
        # M ~ 128 -> BLOCK_M=128 single tile
        (128, 128, 64, 4, 8, 3),
        (128, 64, 64, 4, 8, 3),
        (128, 64, 64, 8, 8, 3),
        # M ~ 240 -> BLOCK_M=256 single tile, split for occupancy (B read once)
        (256, 64, 64, 4, 8, 3),
        (256, 64, 64, 2, 8, 3),
        (256, 32, 64, 4, 8, 3),
    ]
    cfgs = []
    for BM, BN, BK, SK, nw, ns in specs:
        cfgs.append(
            triton.Config(
                {
                    "BLOCK_M": BM,
                    "BLOCK_N": BN,
                    "BLOCK_K": BK,
                    "SPLIT_K": SK,
                },
                num_warps=nw,
                num_stages=ns,
            )
        )
    return cfgs


@triton.autotune(configs=_configs(), key=["M"], reset_to_zero=["C32"])
@triton.jit
def _gemm_splitk_kernel(
    A, B, C32,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # K-slice owned by this split.  K divisible by SPLIT_K, and (K//SPLIT_K) divisible
    # by BLOCK_K -> no K masking needed.
    K_per = K // SPLIT_K
    k_start = pid_k * K_per

    a_ptrs = A + (offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak)
    # B.T view: logical (k, n) lives at B[n, k] -> offset n*stride_bn + k*stride_bk
    b_ptrs = B + ((k_start + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    m_mask = offs_m[:, None] < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K_per, BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C32 + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.atomic_add(c_ptrs, acc, mask=m_mask)


@triton.jit
def _cast_f32_to_f16_kernel(C32, C16, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    v = tl.load(C32 + offs, mask=mask)
    tl.store(C16 + offs, v.to(tl.float16), mask=mask)


def run(A, B):
    assert A.dim() == 2 and B.dim() == 2, "A,B must be 2D"
    M, K = A.shape
    N, Kb = B.shape
    assert K == Kb, "inner dims must match (A[M,K], B[N,K])"

    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    # fp32 accumulation buffer for split-K atomic reduction (must start zeroed).
    C32 = torch.zeros((M, N), device=A.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    _gemm_splitk_kernel[grid](
        A, B, C32,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C32.stride(0), C32.stride(1),
    )

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    n_elem = M * N
    BLOCK = 1024
    cast_grid = (triton.cdiv(n_elem, BLOCK),)
    _cast_f32_to_f16_kernel[cast_grid](C32, C, n_elem, BLOCK=BLOCK)
    return C
