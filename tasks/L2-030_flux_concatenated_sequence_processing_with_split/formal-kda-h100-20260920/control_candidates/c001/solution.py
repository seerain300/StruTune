import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_wt_kernel(
    A, W, C,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Computes C[M, N] = A[M, K] @ W[N, K].T  with fp32 accumulation.

    W is stored row-major as [N, K]; b_tile[k, n] = W[n, k] is read transposed
    via a stride swap (no materialized transpose). Compute path is tf32x3, which
    is accuracy-safe against the true-fp32 reference (see docs/draft.md).
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Wrap row/col offsets so out-of-range loads stay in-bounds; masked at store.
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # b_tile[k, n] = W[n, k]  ->  W + n*stride_wn + k*stride_wk
    w_ptrs = W + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        acc = tl.dot(a, w, acc, input_precision="tf32x3")
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _gemm_wt(a, w):
    """a: [M, K] fp32 row-major; w: [N, K] fp32 row-major. Returns [M, N] = a @ w.T."""
    M, K = a.shape
    N, Kw = w.shape
    assert K == Kw, f"K mismatch: {K} vs {Kw}"
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    _matmul_wt_kernel[grid](
        a, w, c,
        M, N, K,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=4, num_stages=3,
    )
    return c


@torch.no_grad()
def run(hidden_states, encoder_hidden_states, process_weight):
    """Flux concatenated sequence processing.

    Mathematically identical to: concat([encoder, hidden], dim=1) @ W.T, then split.
    The projection is row-wise independent, so concat/split are eliminated and we
    compute two weight-shared GEMMs directly (see docs/draft.md, docs/plan.md).
    """
    B, I, H = hidden_states.shape
    T = encoder_hidden_states.shape[1]

    # Zero-copy views for contiguous inputs; plumbing only (no compute).
    enc2d = encoder_hidden_states.reshape(-1, H)   # [B*T, H]
    hid2d = hidden_states.reshape(-1, H)           # [B*I, H]

    proc_enc = _gemm_wt(enc2d, process_weight)     # [B*T, H]
    proc_hid = _gemm_wt(hid2d, process_weight)     # [B*I, H]

    return proc_enc.reshape(B, T, H), proc_hid.reshape(B, I, H)
