"""KDA candidate c002 — Split-K NT GEMM (fp32 atomic reduction + Triton cast).

Task: C = A @ B.T, fp16, A[M,K], B[N,K], C[M,N]; N=K=4096, M variable/small.
Target: NVIDIA A800 (sm_80, Ampere). Triton-only compute.

Motivation (from c001 evidence, docs/plan.md decision log):
  * c001 (plain NT GEMM, no split-K) was correct but only 0.56x geomean: for skinny M
    there are too few output tiles (<=32) to occupy 108 SMs / hide HBM latency (H2).
  * Reference (cuBLAS) itself runs ~44-70us on these shapes, well above the ~17us HBM
    floor -> real headroom exists.

Single structural change vs c001: add a Split-K dimension to the reduction so the block
count scales up and saturates HBM bandwidth.

Key design points:
  * READ B EXACTLY ONCE.  B[N,K] does not depend on M, so if BLOCK_M >= M (a single
    M-tile, num_pid_m == 1) then every K-slice of B is read by exactly one block ->
    total B traffic = 1x.  Split-K then adds parallelism (occupancy) WITHOUT re-reading B.
    We therefore let the autotuner (key=['M']) choose, per M, a BLOCK_M that covers M in
    one tile plus a SPLIT_K that pushes the block count past ~2x108.
  * fp32 accumulation (matches torch.matmul fp16->fp32 accumulate).
  * Split-K partials combined with tl.atomic_add into an fp32 buffer, then a Triton cast
    kernel writes fp16 C.  c001 showed the evaluator is tolerance-based (atol=rtol=0.01,
    matched_ratio=0.99), not bit-exact, so atomic (fp32-ULP, value-safe) is admissible.
    The extra buffer is only C-sized (<=3.9MB) regardless of SPLIT_K, unlike a two-pass
    [SPLIT_K,M,N] scratch whose traffic grows with SPLIT_K.
  * K=4096 divisible by SPLIT_K (power of two) and each K-chunk divisible by BLOCK_K=64 ->
    no K masking in the inner loop.  N=4096 divisible by every BLOCK_N -> no N masking.
    M may be < BLOCK_M -> mask rows offs_m < M on the A load and the atomic store.
"""

import torch
import triton
import triton.language as tl


def _configs():
    # (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages)
    # Curated so that for every feedback M there exists a "single M-tile (num_pid_m==1)
    # + SPLIT_K for occupancy" option (B read once), plus safe multi-tile fallbacks.
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


@triton.autotune(configs=_configs(), key=["M"])
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
