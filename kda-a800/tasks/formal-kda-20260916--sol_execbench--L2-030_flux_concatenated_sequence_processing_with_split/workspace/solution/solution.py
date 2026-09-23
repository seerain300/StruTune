"""L2/030 Flux concatenated sequence processing with split — Triton solution.

Algebraic simplification (see docs/draft.md §2):
    reference: concat([enc, img], dim=1) @ W.T, then split back
    ==  (enc @ W.T,  img @ W.T)      # row-for-row identical, no cat/split

Each stream is a GEMM  Y = X @ W.T  with X:[M,H], W:[H,H] (nn.Linear weight,
Y[m,n] = sum_k X[m,k] * W[n,k]), K = N = H = 3072.

Compute path is Triton only. PyTorch is used solely for tensor metadata,
output allocation, and kernel launch. No Torch/CPU/NumPy/CUDA-ext fallback.
"""

import torch
import triton
import triton.language as tl


# ---- c002 fixed configuration (Phase C tiling sweep; parent c001) -----------
# Single-axis change from c001: num_warps 4 -> 8. Rationale: the 128x128 fp32
# accumulator (16384 floats) spread over only 4 warps (128 threads) is ~128
# acc-registers/thread -> heavy register spill and poor MMA parallelism (c001
# hit ~6 TFLOPS vs ~52 tf32x3 ceiling). 8 warps (256 threads) -> 64 acc-regs/
# thread and 2x MMA parallelism. Smem footprint is unchanged from c001 (which
# compiled/ran fine), so there is no compile risk from this change.
_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_K = 32
_GROUP_M = 8
_NUM_WARPS = 8
_NUM_STAGES = 3
_PRECISION = "tf32x3"   # near-fp32 accuracy at tensor-core speed (Ampere)


@triton.jit
def _gemm_wt_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    PREC: tl.constexpr,
):
    """Compute Y = X @ W.T where W is stored as [N, K] (Y[m,n]=sum_k X[m,k]W[n,k])."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # X tile [BLOCK_M, BLOCK_K]; W tile as [BLOCK_K, BLOCK_N] via W[n,k] transpose.
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < k_remaining)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(x, w, acc, input_precision=PREC)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=y_mask)


def _project(x3d: torch.Tensor, w: torch.Tensor, y3d: torch.Tensor) -> None:
    """Launch Y = X @ W.T over a flattened [B*S, H] stream, writing into y3d."""
    B, S, H = x3d.shape
    N, K = w.shape          # N = out = H, K = in = H
    M = B * S
    if M == 0:
        return
    x2d = x3d.reshape(M, H)         # no copy when contiguous
    y2d = y3d.reshape(M, N)
    grid = (triton.cdiv(M, _BLOCK_M) * triton.cdiv(N, _BLOCK_N),)
    _gemm_wt_kernel[grid](
        x2d, w, y2d,
        M, N, K,
        x2d.stride(0), x2d.stride(1),
        w.stride(0), w.stride(1),
        y2d.stride(0), y2d.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        GROUP_M=_GROUP_M, PREC=_PRECISION,
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
    )


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
):
    """Flux concat->linear->split, fused as two independent per-stream GEMMs.

    Args:
        hidden_states:         [B, I, H] fp32 image latents
        encoder_hidden_states: [B, T, H] fp32 text conditioning
        process_weight:        [H, H]    fp32 linear weight (Y = X @ W.T)

    Returns:
        (processed_encoder [B, T, H], processed_hidden [B, I, H]) fp32
    """
    B, I, H = hidden_states.shape
    T = encoder_hidden_states.shape[1]

    processed_encoder = torch.empty(
        (B, T, H), device=encoder_hidden_states.device, dtype=torch.float32
    )
    processed_hidden = torch.empty(
        (B, I, H), device=hidden_states.device, dtype=torch.float32
    )

    # Same weight W stays hot across both launches.
    _project(encoder_hidden_states, process_weight, processed_encoder)
    _project(hidden_states, process_weight, processed_hidden)

    return processed_encoder, processed_hidden
