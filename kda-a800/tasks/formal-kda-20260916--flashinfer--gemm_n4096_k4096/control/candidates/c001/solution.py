"""KDA candidate c001 — baseline autotuned NT GEMM (no split-K).

Task: C = A @ B.T, fp16, A[M,K], B[N,K], C[M,N]; N=K=4096, M variable/small.
Target: NVIDIA A800 (sm_80, Ampere). Triton-only compute.

Design (see docs/draft.md, docs/plan.md):
  * NT layout: for both A and B the contraction axis K is the contiguous/fast axis.
    B is stored [N,K]; B.T is logically [K,N] with strides (stride_bk=1, stride_bn=K).
  * Standard grouped-along-M output-tile mapping for L2 locality.
  * fp32 accumulator (matches torch.matmul fp16->fp32 accumulate), fp16 store.
  * K=4096 is divisible by every BLOCK_K used -> no K-mask in the inner loop.
    N=4096 is divisible by every BLOCK_N used -> no N-mask.
    M may be < BLOCK_M -> mask rows with offs_m < M (other=0.0) on A load and C store.

This candidate establishes correctness + a real per-M speed baseline. It intentionally
does NOT use split-K (that is c002+).
"""

import torch
import triton
import triton.language as tl


def _configs():
    # Explicit, curated config list (keeps autotune cheap in case the evaluator
    # times a cold call). Covers BLOCK_M in {16,32,64} for the small-M regime,
    # BLOCK_N in {64,128,256} for load width vs block-count trade-off, and a
    # couple of pipeline depths. Triton autotune prunes configs that exceed smem.
    specs = [
        (16, 64, 64, 4, 4),
        (16, 128, 64, 4, 4),
        (16, 256, 64, 8, 4),
        (16, 128, 32, 4, 3),
        (32, 64, 64, 4, 4),
        (32, 128, 64, 4, 4),
        (32, 256, 64, 8, 4),
        (32, 128, 32, 4, 3),
        (64, 64, 64, 4, 4),
        (64, 128, 64, 4, 4),
        (64, 256, 64, 8, 4),
        (64, 128, 64, 8, 4),
    ]
    cfgs = []
    for BM, BN, BK, nw, ns in specs:
        cfgs.append(
            triton.Config(
                {
                    "BLOCK_M": BM,
                    "BLOCK_N": BN,
                    "BLOCK_K": BK,
                    "GROUP_SIZE_M": 8,
                },
                num_warps=nw,
                num_stages=ns,
            )
        )
    return cfgs


@triton.autotune(configs=_configs(), key=["M"])
@triton.jit
def _gemm_nt_kernel(
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
    # B.T view: element (k, n) lives at B[n, k] -> offset offs_n*stride_bn + offs_k*stride_bk
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    m_mask = offs_m[:, None] < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        # K divisible by BLOCK_K -> no k mask; only mask rows beyond M.
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
    _gemm_nt_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C
