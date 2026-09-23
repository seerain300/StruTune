"""KDA candidate c002 — gemm_n4096_k4096 (FlashInfer, H100 / sm_90).

Operation: C = A @ B.T
  A : [M, K] float16   (row-major, K contiguous)
  B : [N, K] float16   (row-major, K contiguous)   -- B is logically transposed
  C : [M, N] float16
  N = K = 4096 (constants); M varies.

c002 role: raise small-M occupancy with a DETERMINISTIC M-bucketed dispatch.

Diagnosis from c001 (geomean 0.47x): small-M speedup was flat ~0.42-0.49x across
the entire M<=256 sweep and did NOT scale with M — the occupancy-starved signature.
c001's autotune preferred BLOCK_N in {128,256} => only 16-32 CTAs launch against
132 SMs (12-24% occupancy), and the kernel sat at a flat ~0.065 ms floor far above
the ~5-10 us L2/HBM bandwidth floor for streaming B once (33.5 MB). Autotune also
produced an M=128 cliff (0.33x), evidence of selection noise.

Fix (single coherent change vs c001):
  * Replace autotune with a deterministic M -> (BLOCK_M, BLOCK_N, ...) table.
  * Use BLOCK_N=32 for all small M (<=256) so there are always 4096/32 = 128
    N-tiles => ~128 CTAs (~one full wave over 132 SMs) even for a single M-tile.
  * Size BLOCK_M >= M for M<=128 so exactly one M-tile is launched (no redundant
    B re-reads); allow 2 M-tiles up to M=256.
  * Keep the proven large-M config (BLOCK_N=256) for M>256 so M=8192 stays ~parity.

Everything else is identical to c001: same tiled NT kernel, FP32 accumulation,
FP16 output, group-M raster, M-masked loads/stores, no split-K. Triton is the only
compute path.
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
    for k0 in range(0, K, BLOCK_K):
        k_rem = K - k0
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem), other=0.0)
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

    Targets ~128 concurrently-resident CTAs (one wave over 132 SMs) for small M
    by using BLOCK_N=32 (=> 128 N-tiles), while keeping one M-tile (BLOCK_M >= M)
    wherever possible to avoid redundant B re-reads.
    """
    if M <= 64:
        # 1 M-tile * 128 N-tiles = 128 CTAs, no redundant B read.
        return dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=64, GROUP_M=1,
                    num_warps=4, num_stages=4)
    elif M <= 128:
        # 1 M-tile * 128 N-tiles = 128 CTAs, no redundant B read.
        return dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=64, GROUP_M=1,
                    num_warps=4, num_stages=4)
    elif M <= 256:
        # 2 M-tiles * 128 N-tiles = 256 CTAs (~2 waves).
        return dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=64, GROUP_M=8,
                    num_warps=4, num_stages=4)
    else:
        # Large/compute-bound M (972, 2053, 2379, 8192): keep the proven
        # big-tile config that held M=8192 near parity in c001.
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
