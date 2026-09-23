"""KDA candidate c009 — gemm_n4096_k4096 (FlashInfer, H100 / sm_90).

Operation: C = A @ B.T   (A [M,K] f16, B [N,K] f16, C [M,N] f16; N=K=4096).

Lineage: parent = c006 (best; deterministic M-bucket table; M<=64 BLOCK_K=256, other
small buckets BLOCK_K=128, large bucket BLOCK_M=128/BLOCK_N=256/BLOCK_K=64/stages=3).
The small-M lever ladder is exhausted (c003 CTA-count, c005 split-K, c004/c006 K-loop,
c007 K-loop@BM128, c008 M-tiling all falsified/exhausted). One direction was never
touched: the LARGE/compute-bound bucket (M>256) config, inherited unchanged since c002.

Evidence: the 3 medium shapes are well below parity while M=8192 is fine:
  M=972 ~0.79x, M=2053 ~0.69-0.72x, M=2379 ~0.74-0.77x, M=8192 ~0.93x.
The large bucket runs BLOCK_K=64 => 64 serial K-iters at num_stages=3. On Hopper wgmma
GEMM, deeper software pipelining hides global-load latency (KernelWiki pipeline-stages:
tutorial GEMM 695->940 TFLOPS from more stages). M=8192 has ~7.8 waves so its tail is
amortized and it already sits near parity; the medium shapes (M=972/2053/2379 => ~1-2.3
waves) are more pipeline/tail sensitive and should benefit most from more stages.

c009 hypothesis (H11): the large bucket is under-pipelined at num_stages=3; raising it
to 4 lifts the medium compute-bound shapes toward M=8192's level without regressing
M=8192. smem for the large tile A[128,64]+B[64,256] f16 = 49152 B/stage * 4 = 192 KB
< 228 KB -> fits.

Change vs c006 (single variable): large bucket (M>256) num_stages 3 -> 4. ALL small
buckets (M<=64, 65-128, 129-256) are byte-for-byte c006. Same single-pass NT kernel.

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

    vs c006 the ONLY change is num_stages 3->4 on the large bucket (M>256). All small
    buckets are byte-for-byte c006.
    """
    if M <= 64:
        # c006: 1 M-tile * 128 N-tiles = 128 CTAs; BLOCK_K=256 => 16 K-iters.
        return dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=256, GROUP_M=1,
                    num_warps=4, num_stages=4)
    elif M <= 128:
        # c006: BLOCK_K=128.
        return dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=128, GROUP_M=1,
                    num_warps=4, num_stages=4)
    elif M <= 256:
        # c006: BLOCK_K=128, 2 M-tiles.
        return dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=128, GROUP_M=8,
                    num_warps=4, num_stages=4)
    else:
        # c009: large/compute-bound config, num_stages 3->4 (deeper pipeline).
        # smem A[128,64]+B[64,256] f16 = 49152 B/stage * 4 = 192 KB < 228 KB.
        return dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, GROUP_M=8,
                    num_warps=8, num_stages=4)


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
