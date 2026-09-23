import torch
import triton
import triton.language as tl


def _configs():
    # Cover the wide M range (M_tot spans 384 -> 24576 across workloads).
    # tf32x3 issues 3 MMAs per K-step. SMEM per stage = (BM*BK + BK*BN)*4B; the
    # product BK*stages is capped so fp32 SMEM stays within the H100 ~228KB/SM
    # budget. c005 adds BK64 large-tile variants (fewer K-loop iters -> less
    # loop/mask overhead on compute-bound cases) and num_warps sweeps.
    # tf32x3 correctness is config-independent (see docs/draft.md).
    cfgs = [
        # small-M friendly (many N-tiles -> more parallelism / occupancy)
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
        # medium
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        # large-M friendly (bigger tiles -> higher tensor-core efficiency)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=16, num_stages=3),
        # GROUP_M sweep for L2 reuse of W on the large-M cases
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 16}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=3),
    ]
    return cfgs


@triton.autotune(configs=_configs(), key=["M_text", "M_img"])
@triton.jit
def _fused_matmul_wt_kernel(
    A_enc, A_hid, W, C_enc, C_hid,
    M_text, M_img, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Single fused launch computing BOTH weight-shared GEMMs:
        C_enc[M_text, N] = A_enc[M_text, K] @ W.T
        C_hid[M_img,  N] = A_hid[M_img,  K] @ W.T
    over one grid whose M-tiles span both regions (encoder tiles first, then
    image tiles). Region resolved per-program from its global m-tile index; W
    (37.7 MB) stays hot in L2 across both regions. tf32x3 compute (accuracy-safe
    vs the true-fp32 reference; see docs/draft.md). Tile config is autotuned per
    (M_text, M_img) shape (docs/plan.md Phase C). num_m_text is computed here
    from the autotuned BLOCK_M so the host grid and kernel stay consistent.
    """
    pid = tl.program_id(0)
    num_m_text = tl.cdiv(M_text, BLOCK_M)
    num_m_img = tl.cdiv(M_img, BLOCK_M)
    num_pid_m = num_m_text + num_m_img
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped-M swizzle for L2 locality.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Region selection: encoder tiles occupy [0, num_m_text); image tiles follow.
    # No tile straddles the boundary (each region's M padded to whole BLOCK_M tiles).
    if pid_m < num_m_text:
        A = A_enc
        C = C_enc
        M = M_text
        row_tile = pid_m
    else:
        A = A_hid
        C = C_hid
        M = M_img
        row_tile = pid_m - num_m_text

    offs_m = (row_tile * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # b_tile[k, n] = W[n, k]
    w_ptrs = W + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        acc = tl.dot(a, w, acc, input_precision="tf32x3")
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    offs_cm = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@torch.no_grad()
def run(hidden_states, encoder_hidden_states, process_weight):
    """Flux concatenated sequence processing.

    Mathematically identical to: concat([encoder, hidden], dim=1) @ W.T, then split.
    The projection is row-wise independent, so concat/split are eliminated and the
    two weight-shared GEMMs are computed in a single fused Triton launch spanning
    both sequence regions, with the tile config autotuned per shape.
    """
    B, I, H = hidden_states.shape
    T = encoder_hidden_states.shape[1]
    N = H
    K = H

    # Zero-copy views for contiguous inputs; plumbing only (no compute).
    enc2d = encoder_hidden_states.reshape(-1, H)   # [M_text, H]
    hid2d = hidden_states.reshape(-1, H)           # [M_img,  H]
    M_text = enc2d.shape[0]
    M_img = hid2d.shape[0]

    proc_enc = torch.empty((M_text, N), device=enc2d.device, dtype=torch.float32)
    proc_hid = torch.empty((M_img, N), device=hid2d.device, dtype=torch.float32)

    grid = lambda META: (
        (triton.cdiv(M_text, META["BLOCK_M"]) + triton.cdiv(M_img, META["BLOCK_M"]))
        * triton.cdiv(N, META["BLOCK_N"]),
    )

    # A_enc/A_hid share strides (row-major [M, H]); C likewise.
    _fused_matmul_wt_kernel[grid](
        enc2d, hid2d, process_weight, proc_enc, proc_hid,
        M_text, M_img, N, K,
        enc2d.stride(0), enc2d.stride(1),
        process_weight.stride(0), process_weight.stride(1),
        proc_enc.stride(0), proc_enc.stride(1),
    )

    return proc_enc.reshape(B, T, H), proc_hid.reshape(B, I, H)
