import torch
import math
import triton
import triton.language as tl


@triton.jit
def qk_matmul_kernel(
    Q_ptr, K_ptr, L_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, D: tl.constexpr,
    sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute logits L = Q @ K^T expanded on heads:
      - Q: [Nq, Hq, D]
      - K: [Nk, Hkv, D], where Hkv can be less than Hq; we expand K to [Nk, Hq, D] via repeat_interleave on host.
      - L: [Nq, Hq, Nk]
    """
    pid0 = tl.program_id(0)  # over queries
    pid1 = tl.program_id(1)  # over heads
    pid2 = tl.program_id(2)  # over kv tokens

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = pid2 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over head dimension (D)
    for d in range(0, D, 1):
        # Load Q tile [BLOCK_M, 1]
        Q_tile = tl.load(
            Q_ptr + q_offsets[:, None] * (Hq * D) + pid1 * D + d,
            mask=mask_q[:, None],
            other=0.0
        )  # shape (BLOCK_M, 1)

        # Load K tile [1, BLOCK_N]
        K_tile = tl.load(
            K_ptr + k_offsets[None, :] * (Hq * D) + pid1 * D + d,
            mask=mask_k[None, :],
            other=0.0
        )  # shape (1, BLOCK_N)

        acc += Q_tile * K_tile  # broadcast to (BLOCK_M, BLOCK_N)

    # Scale by sm_scale and store
    acc = acc * sm_scale
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def apply_causal_mask_kernel(
    L_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr,
    delta: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Apply forward-looking causal mask to L: for each query q, allow only j < q + 1 + delta.
    L: [Nq, Hq, Nk], float32.
    """
    pid0 = tl.program_id(0)  # over queries
    pid1 = tl.program_id(1)  # over heads
    pid2 = tl.program_id(2)  # over kv tokens

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    j_offsets = pid2 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_q = q_offsets < Nq
    mask_j = j_offsets < Nk

    # Load current logits tile
    tile = tl.load(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + j_offsets[None, :],
        mask=mask_q[:, None] & mask_j[None, :],
        other=0.0
    )

    # Compute allowed mask: j < (q + 1 + delta)
    allowed = j_offsets[None, :] < (q_offsets[:, None] + 1 + delta)

    # Apply mask: set disallowed positions to -inf
    tile = tl.where(allowed, tile, -float('inf'))
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + j_offsets[None, :],
        tile,
        mask=mask_q[:, None] & mask_j[None, :]
    )


@triton.jit
def masked_softmax_dimN_kernel(
    L_ptr, S_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Softmax over Nk (KV tokens) for each (q, head): S = softmax(L, dim=Nk).
    L: [Nq, Hq, Nk], float32
    S: [Nq, Hq, Nk], float32
    """
    pid0 = tl.program_id(0)  # over queries
    pid1 = tl.program_id(1)  # over heads
    pid2 = tl.program_id(2)  # over blocks of Nk

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    j_offsets = pid2 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_j = j_offsets < Nk

    # Load logits tile
    logits = tl.load(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + j_offsets[None, :],
        mask=mask_q[:, None] & mask_j[None, :],
        other=-float('inf')
    )

    # Row-wise max for numerical stability
    row_max = tl.max(logits, axis=1)  # [BLOCK_M]
    logits = logits - row_max[:, None]

    # exp and sum
    exp_logits = tl.exp(logits)
    row_sum = tl.sum(exp_logits, axis=1)  # [BLOCK_M]
    softmax = exp_logits / row_sum[:, None]

    tl.store(
        S_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + j_offsets[None, :],
        softmax,
        mask=mask_q[:, None] & mask_j[None, :]
    )


@triton.jit
def attn_dot_v_kernel(
    O_ptr, V_ptr, Y_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute Y = O @ V^T where:
      - O: [Nq, Hq, Nk]
      - V: [Nk, Hq, D] (we pass expanded Hq=32 here, D=128)
      - Y: [Nq, Hq, D]
    """
    pid0 = tl.program_id(0)  # over queries
    pid1 = tl.program_id(1)  # over heads
    pid2 = tl.program_id(2)  # over D blocks

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        mask_d = d_offsets < D

        # Load O tile: [BLOCK_M, BLOCK_N]
        O_tile = tl.load(
            O_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0
        )

        # Load V tile as [BLOCK_N, BLOCK_D]
        V_tile = tl.load(
            V_ptr + k_offsets[:, None] * (Hq * D) + pid1 * D + d_offsets[None, :],
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0
        )

        acc += tl.dot(O_tile, tl.trans(V_tile))

    # Store Y: [Nq, Hq, D] => index = nq*(Hq*D) + h*D + d
    tl.store(
        Y_ptr + q_offsets[:, None] * (Hq * D) + pid1 * D + d_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_d[None, :]
    )


@triton.jit
def lse_segment_kernel(
    L_ptr, LSE_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr,
    sm_scale: tl.float32,
    log2: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute logsumexp over Nk for each (q, head) in L and divide by ln(2).
    L: [Nq, Hq, Nk], float32
    LSE_ptr: [Nq, Hq], float32
    """
    pid0 = tl.program_id(0)  # over queries
    pid1 = tl.program_id(1)  # over heads

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_q = q_offsets < Nq

    # Initialize running max and sum per q
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over Nk in tiles
    for j_start in range(0, Nk, BLOCK_N):
        j_offsets = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_j = j_offsets < Nk

        # Load logits tile [BLOCK_M, BLOCK_N]
        tile = tl.load(
            L_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + j_offsets[None, :],
            mask=mask_q[:, None] & mask_j[None, :],
            other=-float('inf')
        )

        # Multiply by sm_scale before max
        tile = tile * sm_scale

        # Compute tile max per row
        tile_max = tl.max(tile, axis=1)  # [BLOCK_M]
        new_m = tl.maximum(m, tile_max)

        # Update sum: s = s * exp(m - new_m) + sum(exp(tile - new_m))
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(tile - new_m[:, None]), axis=1)

        m = new_m

    # Final logsumexp = log(s) + m
    lse = tl.log(s) + m  # [BLOCK_M]
    # Divide by ln(2)
    lse = lse / log2

    # Store to LSE_ptr
    tl.store(
        LSE_ptr + q_offsets,
        lse,
        mask=mask_q
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        q: [total_q, 32, 128], bfloat16
        k: [total_kv, 8, 128], bfloat16
        v: [total_kv, 8, 128], bfloat16
        qo_indptr: [len_indptr], int32
        kv_indptr: [len_indptr], int32
        sm_scale: float32 scalar
        Returns (output: [total_q, 32, 128], lse: [total_q, 32], both float32)
        """
        # Ensure device is CUDA and tensors are contiguous
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernels require CUDA tensors"
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        device = q.device

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        GQA_ratio = num_qo_heads // num_kv_heads
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)  # [Nq, 32, 128]
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)  # [Nq, 32]

        # Loop over segments
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            Nq = q_end - q_start
            Nk = kv_end - kv_start

            # Slice original Q, K, V
            q_batch = q[q_start:q_end]  # [Nq, 32, 128], bfloat16
            k_batch = k[kv_start:kv_end]  # [Nk, 8, 128], bfloat16
            v_batch = v[kv_start:kv_end]  # [Nk, 8, 128], bfloat16

            # Expand K and V by GQA ratio on head dim
            k_expanded = k_batch.repeat_interleave(GQA_ratio, dim=1)  # [Nk, 32, 128]
            v_expanded = v_batch.repeat_interleave(GQA_ratio, dim=1)  # [Nk, 32, 128]

            # Allocate logits and softmax buffers (float32)
            L = torch.empty((Nq, num_qo_heads, Nk), dtype=torch.float32, device=device)
            S = torch.empty((Nq, num_qo_heads, Nk), dtype=torch.float32, device=device)
            Y = torch.empty((Nq, num_qo_heads, head_dim), dtype=torch.float32, device=device)

            # Launch qk_matmul_kernel: compute Q @ K_expanded^T (scaled by sm_scale)
            BLOCK_M = 64
            BLOCK_N = 64
            grid = (triton.cdiv(Nq, BLOCK_M), num_qo_heads, triton.cdiv(Nk, BLOCK_N))
            qk_matmul_kernel[grid](
                q_batch, k_expanded, L,
                Nq, Nk,
                32, 128,
                sm_scale,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
            )

            # Apply causal mask
            BLOCK_M2 = 64
            BLOCK_N2 = 64
            grid_mask = (triton.cdiv(Nq, BLOCK_M2), num_qo_heads, triton.cdiv(Nk, BLOCK_N2))
            apply_causal_mask_kernel[grid_mask](
                L,
                Nq, Nk,
                32,
                Nk - Nq,
                BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2
            )

            # Softmax over Nk
            grid_softmax = (triton.cdiv(Nq, BLOCK_M2), num_qo_heads, triton.cdiv(Nk, BLOCK_N2))
            masked_softmax_dimN_kernel[grid_softmax](
                L, S,
                Nq, Nk,
                32,
                BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2
            )

            # Compute output = S @ V_expanded^T
            BLOCK_M3 = 64
            BLOCK_N3 = 64
            BLOCK_D = 128
            grid_attn = (triton.cdiv(Nq, BLOCK_M3), num_qo_heads, triton.cdiv(head_dim, BLOCK_D))
            attn_dot_v_kernel[grid_attn](
                S, v_expanded, Y,
                Nq, Nk,
                32, 128,
                BLOCK_M=BLOCK_M3, BLOCK_N=BLOCK_N3, BLOCK_D=BLOCK_D
            )

            # Store output for this segment
            output[q_start:q_end] = Y

            # Compute LSE for this segment
            log2 = 1.4426950408889634  # ln(2)
            grid_lse = (triton.cdiv(Nq, BLOCK_M2), num_qo_heads)
            lse_segment_kernel[grid_lse](
                L, lse[q_start:q_end],
                Nq, Nk,
                32,
                sm_scale, log2,
                BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2
            )

        # Return output as bfloat16 and lse as float32 (to match original behavior)
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
