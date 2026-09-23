"""KDA candidate c003 — gemm_n4096_k4096 (FlashInfer, H100 / sm_90).

Operation: C = A @ B.T   (A [M,K] f16, B [N,K] f16, C [M,N] f16; N=K=4096).

Lineage: parent = c002 (geomean 0.53x). c002 replaced c001's autotune with a
deterministic M-bucket table and used BLOCK_N=32 for small M -> 128 N-tiles ->
~128 CTAs (one wave), lifting small-M ~0.44x -> ~0.52x.

Diagnosis carried into c003: the small-M sol-time floor is still ~0.054 ms even at
M=16, ~8x above the ~7 us floor for one L2 pass of B (33.5 MB), while only ~128 CTAs
run (single M-tile, BLOCK_N=32 => ~1 CTA/SM, 4 SMs idle). That is a memory-level-
parallelism / latency limit at 1 CTA per SM, not a compute limit.

c003 change (single variable): for the single-M-tile small range 8 < M <= 128, set
BLOCK_N = 16 (was 32). 4096/16 = 256 N-tiles => 256 CTAs => ~2 CTAs/SM across all 132
SMs, doubling in-flight B loads (MLP) and eliminating idle SMs. BLOCK_N=16 is a valid
wgmma N; MMA is half as wide but we are memory-bound so that is free.

Everything else is IDENTICAL to c002:
  * tiny M<=8 stays (BM=64, BN=32, GROUP_M=1)  -> no tiny-M regression.
  * 128 < M <= 256 stays (BM=128, BN=32, GROUP_M=8) -> 256 CTAs already, no over-split.
  * M > 256 stays on the proven large-tile config (BN=256) that held M=8192 ~parity.
Same tiled NT kernel, FP32 accumulation, FP16 output, group-M raster, M-masking,
no split-K. Triton is the only compute path.
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

    vs c002 the ONLY change is BLOCK_N 32->16 for the single-M-tile range
    8 < M <= 128 (=> 256 N-tiles => 256 CTAs => ~2 CTAs/SM). All other buckets
    are byte-for-byte the c002 configs.
    """
    if M <= 8:
        # tiny: keep c002 config exactly (avoid extra-CTA overhead regression).
        return dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=64, GROUP_M=1,
                    num_warps=4, num_stages=4)
    elif M <= 64:
        # 1 M-tile * 256 N-tiles = 256 CTAs (~2/SM). BLOCK_N=16 (was 32).
        return dict(BLOCK_M=64, BLOCK_N=16, BLOCK_K=64, GROUP_M=1,
                    num_warps=4, num_stages=4)
    elif M <= 128:
        # 1 M-tile * 256 N-tiles = 256 CTAs (~2/SM). BLOCK_N=16 (was 32).
        return dict(BLOCK_M=128, BLOCK_N=16, BLOCK_K=64, GROUP_M=1,
                    num_warps=4, num_stages=4)
    elif M <= 256:
        # unchanged from c002: 2 M-tiles * 128 N-tiles = 256 CTAs (~2 waves).
        return dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=64, GROUP_M=8,
                    num_warps=4, num_stages=4)
    else:
        # unchanged from c002: proven large/compute-bound config.
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
