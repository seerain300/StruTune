"""KDA candidate c008 — gemm_n4096_k4096 (FlashInfer, H100 / sm_90).

Operation: C = A @ B.T   (A [M,K] f16, B [N,K] f16, C [M,N] f16; N=K=4096).

Lineage: parent = c006 (deterministic M-bucket table; M<=64 BLOCK_K=256, all other
buckets from c004). Findings:
  * c003 (reject): >1 wave of CTAs doesn't help -> not occupancy/BW bound.
  * c004 (accept): shorter serial K-loop (BLOCK_K 64->128) lowers the small-M floor.
  * c005 (reject, 0.38x): split-K adds partial-buffer traffic ~= B itself -> retired.
  * c006 (accept): BLOCK_K=256 on M<=64 ~8% faster/shape (within-run A/B).
  * c007 (reject, 0.50x): BLOCK_K=256 on the 129-256 bucket needed stages 4->2, which
    killed the pipeline (BLOCK_M=128) and collapsed that bucket to ~0.37x. So the
    K-loop lever is EXHAUSTED for BLOCK_M=128 buckets (stages>=4 required, capping
    BLOCK_K=128 at that BLOCK_M).

c008 hypothesis (H10): the 129<=M<=256 bucket is the weakest (~0.46-0.54x) and is the
ONLY small bucket using 2 M-tiles (BLOCK_M=128, GROUP_M=8). Its deficit may be the
2-M-tile scheduling, NOT the K-loop. Re-tile it with BLOCK_M=64 -> 4 M-tiles * 128
N-tiles = 512 CTAs, KEEPING the proven-good BLOCK_K=128 / num_stages=4 pipeline
(smem A[64,128]+B[128,32] f16 *4 = 48KB, ample). This isolates the M-tiling variable:
same per-CTA K-loop as the healthy M<=64 bucket, just more/smaller M-tiles.

Change vs c006 (one bucket, one variable): 129<=M<=256 -> BLOCK_M 128->64. BLOCK_N=32,
BLOCK_K=128, GROUP_M=8, num_warps=4, num_stages=4 unchanged. Buckets M<=64, 65<M<=128,
and M>256 are byte-for-byte c006.

Numerics unchanged: FP32 accumulate, FP16 store, M-masked A-loads / C-stores.
No split-K. Triton is the only compute path.
"""

import torch
import triton
import triton.language as tl


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

    # group-M raster for L2 locality (consecutive CTAs share a B n-tile)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Wrap row/col offsets into range so all loads are in-bounds; wrong rows
    # produced by wrapping are discarded by the store mask below.
    offs_am = offs_m % M
    offs_bn = offs_n % N

    # A tile: [BLOCK_M, BLOCK_K]
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B tile gathered as [BLOCK_K, BLOCK_N]: b[kk, nn] = B[offs_n[nn], offs_k[kk]]
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # K=4096 is an exact multiple of every BLOCK_K used (64/128/256) => no tail.
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


def _select_config(M):
    """Deterministic M-bucket -> tile/launch config.

    vs c006 the ONLY change is BLOCK_M 128->64 on the 129<=M<=256 bucket (2->4
    M-tiles), keeping its BLOCK_K=128 / num_stages=4 pipeline. All other buckets are
    byte-for-byte c006.
    """
    if M <= 64:
        # c006: 1 M-tile * 128 N-tiles = 128 CTAs; BLOCK_K=256 => 16 K-iters.
        return dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=256, GROUP_M=1,
                    num_warps=4, num_stages=4)
    elif M <= 128:
        # c006: BLOCK_K=128, 1 M-tile.
        return dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=128, GROUP_M=1,
                    num_warps=4, num_stages=4)
    elif M <= 256:
        # c008: BLOCK_M 128->64 => up to 4 M-tiles * 128 N-tiles = 512 CTAs,
        # same proven BLOCK_K=128 / num_stages=4 pipeline.
        return dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=128, GROUP_M=8,
                    num_warps=4, num_stages=4)
    else:
        # c004/c006 large/compute-bound config (holds M=8192 near parity).
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

    cfg = _select_config(M)
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
