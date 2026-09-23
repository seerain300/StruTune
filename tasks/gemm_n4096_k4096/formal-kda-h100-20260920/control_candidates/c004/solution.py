"""KDA candidate c004 — gemm_n4096_k4096 (FlashInfer, H100 / sm_90).

Operation: C = A @ B.T   (A [M,K] f16, B [N,K] f16, C [M,N] f16; N=K=4096).

Lineage: parent = c002 (geomean 0.53x, deterministic M-bucket table, BLOCK_N=32
for small M => ~128 CTAs / one wave). c003 (reject) halved BLOCK_N 32->16 to push
128->256 CTAs and got NO speedup (~0.02 lower in-band) => small-M is NOT occupancy/
bandwidth bound beyond one wave; it is latency / fixed-overhead bound. The small-M
sol floor (~0.058 ms) is invariant to CTA count and sits ~2x above cuBLAS (~0.028 ms).

c004 hypothesis (H6): that floor is dominated by the per-CTA serial K-loop — with
BLOCK_K=64 the accumulator chain has 4096/64 = 64 dependent load->mma->acc iterations,
each also paying an always-false K-mask compare (K % BLOCK_K == 0). Cutting the loop
in half and removing the mask should shave loop/issue latency where compute is trivial.

c004 change (single coherent 'reduce K-loop cost' step vs c002):
  * Raise BLOCK_K 64 -> 128 on the small-M buckets (M<=256): 64 -> 32 iterations.
  * Remove the always-false K-mask from the kernel: loads are unconditional. Safe
    for EVERY bucket because K=4096 is a multiple of every BLOCK_K used
    (64 for large, 128 for small) => the last iteration is fully in-bounds.
  * Large/compute-bound bucket (M>256) keeps BLOCK_K=64 / BLOCK_N=256 (only the mask
    removal touches it, a pure simplification) so M=8192 stays ~parity.

Everything else identical to c002: same tiled NT kernel, FP32 accumulation, FP16
output, group-M raster, M-masked A-loads/C-stores. No split-K. Triton is the only
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
    # K=4096 is an exact multiple of every BLOCK_K used (64/128) => no K tail,
    # so loads are unconditional (the always-false mask is removed).
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

    vs c002 the ONLY change is BLOCK_K 64->128 on the small-M buckets (M<=256),
    halving the serial K-loop from 64 to 32 iterations. The large bucket (M>256)
    keeps c002's BLOCK_K=64 / BLOCK_N=256. (The kernel-level K-mask removal is a
    separate, globally-safe simplification.)
    """
    if M <= 64:
        # 1 M-tile * 128 N-tiles = 128 CTAs (~one wave), no redundant B read.
        return dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=128, GROUP_M=1,
                    num_warps=4, num_stages=4)
    elif M <= 128:
        # 1 M-tile * 128 N-tiles = 128 CTAs, no redundant B read.
        return dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=128, GROUP_M=1,
                    num_warps=4, num_stages=4)
    elif M <= 256:
        # 2 M-tiles * 128 N-tiles = 256 CTAs (~2 waves).
        return dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=128, GROUP_M=8,
                    num_warps=4, num_stages=4)
    else:
        # Large/compute-bound M (972, 2053, 2379, 8192): keep the proven c002
        # big-tile config that held M=8192 near parity.
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
