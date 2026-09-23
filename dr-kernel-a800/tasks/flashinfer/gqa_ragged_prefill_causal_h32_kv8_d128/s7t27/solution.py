import math
import torch
import triton
import triton.language as tl


@triton.jit
def qk_matmul_kernel(
    Q_ptr, K_ptr, L_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, D: tl.constexpr,
    sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute L = Q @ K^T where:
      - Q: [Nq, Hq, D], row-major
      - K: [Nk, Hq, D], row-major (note: we process expanded Hq in caller; here Hq is the original kv heads)
      - L: [Nq, Hq, Nk], row-major float32
    Grid: (pid0 over Nq tiles, pid1 over Nk tiles, pid2 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)  # head index in [0, Hq)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        mask_d = d_offsets < D

        # Load Q tile: [BLOCK_M, BLOCK_D], Q layout: [Nq, Hq, D] => index = nq*(Hq*D) + h*D + d
        Q_tile = tl.load(
            Q_ptr + q_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_q[:, None] & mask_d[None, :],
            other=0.0
        )

        # Load K tile: [BLOCK_N, BLOCK_D], K layout: [Nk, Hq, D] => index = nk*(Hq*D) + h*D + d
        K_tile = tl.load(
            K_ptr + k_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0
        )

        # Outer product accumulation: (BLOCK_M x BLOCK_D) @ (BLOCK_D x BLOCK_N) -> (BLOCK_M x BLOCK_N)
        acc += tl.dot(Q_tile, tl.trans(K_tile))

    acc = acc * sm_scale

    # Store to L: layout [Nq, Hq, Nk] => index = nq*(Hq*Nk) + h*Nk + j
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def apply_causal_mask_kernel(
    L_ptr, Nq: tl.int32, Nk: tl.int32, delta: tl.int32, sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr
):
    """
    Apply forward-looking causal mask to L: for each q (rows) and head (h), allow only j < (q + 1 + delta).
    """
    pid0 = tl.program_id(0)  # over Nq tiles
    pid1 = tl.program_id(1)  # over heads
    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_q = q_offsets < Nq

    # We process one head per program; BLOCK_H set to 1
    for h in range(0, BLOCK_H):
        if h >= 1:  # BLOCK_H=1, but keep for future extensibility
            break

        for n in range(0, Nk, BLOCK_N):
            j_offsets = n + tl.arange(0, BLOCK_N)
            mask_j = j_offsets < Nk

            # Load tile L[q, h, j]
            L_tile = tl.load(
                L_ptr + q_offsets[:, None] * (1 * Nk) + h * Nk + j_offsets[None, :],
                mask=mask_q[:, None] & mask_j[None, :],
                other=0.0
            )

            # Compute q vector for this tile: q + (n // BLOCK_N) would be the tile base query row, but here each row is independent.
            # Instead, compute q per element via q_offsets.
            # Causal condition: j < (q + 1 + delta) for each q
            cond = j_offsets[None, :] < (q_offsets[:, None] + 1 + delta)
            L_tile = tl.where(cond, L_tile, -float('inf'))

            tl.store(
                L_ptr + q_offsets[:, None] * (1 * Nk) + h * Nk + j_offsets[None, :],
                L_tile,
                mask=mask_q[:, None] & mask_j[None, :]
            )


@triton.jit
def softmax_dimN_kernel(
    L_ptr, O_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr, sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr
):
    """
    Softmax over the Nk dimension for each (q, head): O[q, h, j] = softmax(L[q, h, j] * sm_scale)
    """
    pid0 = tl.program_id(0)  # over Nq tiles
    pid1 = tl.program_id(1)  # over heads

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_q = q_offsets < Nq

    for h in range(0, BLOCK_H):
        if h >= 1:  # BLOCK_H=1
            break

        # First pass: compute max over j
        max_val = -float('inf')
        for n in range(0, Nk, BLOCK_N):
            j_offsets = n + tl.arange(0, BLOCK_N)
            mask_j = j_offsets < Nk
            L_tile = tl.load(
                L_ptr + q_offsets[:, None] * (Hq * Nk) + h * Nk + j_offsets[None, :],
                mask=mask_q[:, None] & mask_j[None, :],
                other=-float('inf')
            )
            max_val = tl.maximum(max_val, tl.max(L_tile, axis=1))

        # Compute sum of exp(L * sm_scale - max) over j
        sum_val = 0.0
        for n in range(0, Nk, BLOCK_N):
            j_offsets = n + tl.arange(0, BLOCK_N)
            mask_j = j_offsets < Nk
            L_tile = tl.load(
                L_ptr + q_offsets[:, None] * (Hq * Nk) + h * Nk + j_offsets[None, :],
                mask=mask_q[:, None] & mask_j[None, :],
                other=-float('inf')
            )
            exp_tile = tl.exp(L_tile * sm_scale - max_val[:, None])
            sum_val += tl.sum(exp_tile, axis=1)

        # Second pass: write normalized outputs
        for n in range(0, Nk, BLOCK_N):
            j_offsets = n + tl.arange(0, BLOCK_N)
            mask_j = j_offsets < Nk
            L_tile = tl.load(
                L_ptr + q_offsets[:, None] * (Hq * Nk) + h * Nk + j_offsets[None, :],
                mask=mask_q[:, None] & mask_j[None, :],
                other=-float('inf')
            )
            O_tile = tl.exp(L_tile * sm_scale - max_val[:, None]) / sum_val[:, None]
            tl.store(
                O_ptr + q_offsets[:, None] * (Hq * Nk) + h * Nk + j_offsets[None, :],
                O_tile,
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
      - V: [Nk, Hq, D] (note: original V is kv heads; we use Nk rows here)
      - Y: [Nq, Hq, D]
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)  # head index

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
            O_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0
        )

        # Load V tile as [BLOCK_N, BLOCK_D]
        V_tile = tl.load(
            V_ptr + k_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0
        )

        # Accumulate
        acc += tl.dot(O_tile, tl.trans(V_tile))

    # Store Y: layout [Nq, Hq, D] => index = nq*(Hq*D) + h*D + d
    tl.store(
        Y_ptr + q_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_d[None, :]
    )


@triton.jit
def lse_segment_kernel(
    L_ptr, LSE_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr, delta: tl.int32,
    sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute LSE[q, h] = logsumexp_j( L[q, h, j] * sm_scale ) / ln(2) for the segment.
    Only j < (q + 1 + delta) are considered; others contribute 0.
    We process per (q,h) and scan Nk in tiles.
    """
    pid0 = tl.program_id(0)  # over Nq tiles
    pid1 = tl.program_id(1)  # over heads

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_q = q_offsets < Nq

    for h in range(0, Hq):
        max_val = -float('inf')
        for n in range(0, Nk, BLOCK_N):
            j_offsets = n + tl.arange(0, BLOCK_N)
            mask_j = j_offsets < Nk
            cond = j_offsets[None, :] < (q_offsets[:, None] + 1 + delta)
            L_tile = tl.load(
                L_ptr + q_offsets[:, None] * (Hq * Nk) + h * Nk + j_offsets[None, :],
                mask=mask_q[:, None] & mask_j[None, :] & cond,
                other=-float('inf')
            )
            # LSE over allowed j
            tile_lse = tl.log(tl.sum(tl.exp(L_tile * sm_scale), axis=1))
            # Update global max
            max_val = tl.maximum(max_val, tile_lse)

        # sum of exp over allowed j
        sum_val = 0.0
        for n in range(0, Nk, BLOCK_N):
            j_offsets = n + tl.arange(0, BLOCK_N)
            mask_j = j_offsets < Nk
            cond = j_offsets[None, :] < (q_offsets[:, None] + 1 + delta)
            L_tile = tl.load(
                L_ptr + q_offsets[:, None] * (Hq * Nk) + h * Nk + j_offsets[None, :],
                mask=mask_q[:, None] & mask_j[None, :] & cond,
                other=-float('inf')
            )
            exp_tile = tl.exp(L_tile * sm_scale)
            sum_val += tl.sum(exp_tile, axis=1)

        lse_vec = max_val + tl.log(sum_val)  # logsumexp = max + log(sum(exp(... - max)))
        # Store per q element
        tl.store(
            LSE_ptr + q_offsets * Hq + h,
            lse_vec,
            mask=mask_q
        )


class ModelNew(torch.nn.Module):
    def __init__(self, num_qo_heads=32, num_kv_heads=8, head_dim=128):
        super().__init__()
        assert num_qo_heads == 32, "This implementation expects num_qo_heads=32."
        assert num_kv_heads == 8, "This implementation expects num_kv_heads=8."
        assert head_dim == 128, "This implementation expects head_dim=128."
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_ratio = num_qo_heads // num_kv_heads
        self.sm_scale = 1.0 / math.sqrt(head_dim)
        self.inv_ln2 = 1.0 / math.log(2.0)

        # Triton tuning parameters
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_D = 64

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale=None):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda, "All tensors must be on CUDA for Triton."

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        assert len_indptr >= 1, "len_indptr must be at least 1"

        # Allocate outputs (float32 for compute)
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=q.device)

        # Cast to float32 for compute (matches original behavior)
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        # Process each batch segment
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Extract segments
            q_batch = q_f32[q_start:q_end]                 # [Nq, 32, 128]
            k_batch = k_f32[kv_start:kv_end]              # [Nk, 8, 128]
            v_batch = v_f32[kv_start:kv_end]              # [Nk, 8, 128]

            Nq = q_batch.shape[0]
            Nk = k_batch.shape[0]
            Hq = self.num_kv_heads  # original kv heads

            # Expand K and V by GQA ratio
            k_expanded = k_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [Nk, 32, 128]
            v_expanded = v_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [Nk, 32, 128]

            # 1) Compute logits L = q @ k_expanded^T
            L = torch.empty((Nq, 32, Nk), dtype=torch.float32, device=q.device)
            grid = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), 32)
            qk_matmul_kernel[grid](
                q_batch, k_expanded, L,
                Nq, Nk, 32, 128,
                self.sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_D=self.BLOCK_D,
                num_warps=4, num_stages=2
            )

            # 2) Apply forward-looking causal mask: j < (q + 1 + delta)
            delta = Nk - Nq
            grid_mask = (triton.cdiv(Nq, self.BLOCK_M), 32)
            apply_causal_mask_kernel[grid_mask](
                L, Nq, Nk, delta, self.sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_H=1,
                num_warps=4, num_stages=1
            )

            # 3) Softmax over Nk (dim=-1)
            O = torch.empty((Nq, 32, Nk), dtype=torch.float32, device=q.device)
            grid_softmax = (triton.cdiv(Nq, self.BLOCK_M), 32)
            softmax_dimN_kernel[grid_softmax](
                L, O,
                Nq, Nk, 32, self.sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_H=1,
                num_warps=4, num_stages=2
            )

            # 4) Compute output = O @ V_expanded^T
            Y = torch.empty((Nq, 32, 128), dtype=torch.float32, device=q.device)
            grid_attn = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), 32)
            attn_dot_v_kernel[grid_attn](
                O, v_expanded, Y,
                Nq, Nk,
                32, 128,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_D=self.BLOCK_D,
                num_warps=4, num_stages=2
            )

            # 5) Write segment to output
            output[q_start:q_end] = Y

            # 6) Compute LSE for this segment: logsumexp over allowed j / ln(2)
            grid_lse = (triton.cdiv(Nq, self.BLOCK_M), 32)
            segment_lse = torch.empty((Nq, 32), dtype=torch.float32, device=q.device)
            lse_segment_kernel[grid_lse](
                L, segment_lse,
                Nq, Nk, 32, delta, self.sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N,
                num_warps=4, num_stages=1
            )
            lse[q_start:q_end] = segment_lse / math.log(2.0)

        # Return outputs: output in bfloat16, lse in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
