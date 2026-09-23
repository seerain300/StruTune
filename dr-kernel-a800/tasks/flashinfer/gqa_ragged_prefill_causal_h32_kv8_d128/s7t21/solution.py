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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute L = Q @ K^T where:
      - Q: [Nq, Hq, D]
      - K: [Nk, Hq, D] (note: K is length Nk but we use expanded heads via repeat_interleave)
      - L: [Nq, Hq, Nk] float32
    Grid: (pid0 over Nq tiles, pid1 over Nk tiles, pid2 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        mask_d = d_offsets < D

        # Load Q tile: [BLOCK_M, BLOCK_D]
        Q_tile = tl.load(
            Q_ptr + q_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_q[:, None] & mask_d[None, :],
            other=0.0
        )

        # Load K tile: [BLOCK_N, BLOCK_D]
        K_tile = tl.load(
            K_ptr + k_offsets[:, None] * (Hq * D) + pid2 * D + d_offsets[None, :],
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0
        )

        # Outer product accumulation
        acc += tl.dot(Q_tile, tl.trans(K_tile))  # [BLOCK_M, BLOCK_N]

    # Scale by sm_scale (as in original)
    acc = acc * sm_scale

    # Store to L
    L_ptrs = L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :]
    tl.store(L_ptrs, acc, mask=mask_q[:, None] & mask_k[None, :])


@triton.jit
def apply_causal_mask_kernel(
    L_ptr, MaskedL_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr,
    q_positions_ptr, kv_positions_ptr,
    delta: tl.int32,  # Nk - Nq
    sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr
):
    """
    Apply causal mask: for each (q, h, kv), if kv >= (q + 1 + delta) -> set to -inf.
    Inputs:
      - L_ptr: [Nq, Hq, Nk] float32 logits
      - MaskedL_ptr: same shape
      - q_positions_ptr: [Nq] int32
      - kv_positions_ptr: [Nk] int32
    Grid: (ceil_div(Nq, BLOCK_M), ceil_div(Nk, BLOCK_N), ceil_div(Hq, BLOCK_H))
    """
    pid0 = tl.program_id(0)  # q tiles
    pid1 = tl.program_id(1)  # kv tiles
    pid2 = tl.program_id(2)  # head tiles

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)
    kv_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)
    h_offsets = pid2 * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_q = q_offsets < Nq
    mask_kv = kv_offsets < Nk
    mask_h = h_offsets < Hq

    # Load positions
    q_positions = tl.load(q_positions_ptr + q_offsets, mask=mask_q, other=0)
    kv_positions = tl.load(kv_positions_ptr + kv_offsets, mask=mask_kv, other=0)

    # Causal condition: kv < (q + 1 + delta)
    cond = kv_positions[None, :] < (q_positions[:, None] + 1 + delta)

    # Loop over heads in this tile
    for h_idx in range(BLOCK_H):
        h = h_offsets[h_idx]
        if h >= Hq:
            break
        # Load L values for this tile
        L_ptrs = L_ptr + q_offsets[:, None] * (Hq * Nk) + h * Nk + kv_offsets[None, :]
        L_vals = tl.load(L_ptrs, mask=(mask_q[:, None] & mask_kv[None, :]), other=0.0)
        neg_inf = -float('inf')
        masked_vals = tl.where(cond, L_vals, neg_inf)
        # Store masked values
        Masked_ptrs = MaskedL_ptr + q_offsets[:, None] * (Hq * Nk) + h * Nk + kv_offsets[None, :]
        tl.store(Masked_ptrs, masked_vals, mask=(mask_q[:, None] & mask_kv[None, :]))


@triton.jit
def softmax_dimN_kernel(
    L_ptr, Out_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Softmax along Nk (last dim) for each (q, h).
    Inputs:
      - L_ptr: [Nq, Hq, Nk] logits
      - Out_ptr: [Nq, Hq, Nk] float32 output of softmax
    Grid: (Nq, Hq)
    """
    q_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    if q_idx >= Nq or h_idx >= Hq:
        return

    k_offsets = tl.arange(0, BLOCK_N)  # use all Nk if BLOCK_N == Nk
    mask_k = k_offsets < Nk

    # Load vector for this (q_idx, h_idx)
    L_vec = tl.load(L_ptr + q_idx * (Hq * Nk) + h_idx * Nk + k_offsets, mask=mask_k, other=-float('inf'))
    # Scale as original does softmax(logits / sm_scale)
    L_vec = L_vec / sm_scale
    max_val = tl.max(L_vec, axis=0)
    L_vec = L_vec - max_val
    exp_vec = tl.exp(L_vec)
    sum_val = tl.sum(exp_vec, axis=0)
    softmax_vec = exp_vec / sum_val
    Out_ptrs = Out_ptr + q_idx * (Hq * Nk) + h_idx * Nk + k_offsets
    tl.store(Out_ptrs, softmax_vec, mask=mask_k)


@triton.jit
def attn_dot_v_kernel(
    Attn_ptr, V_ptr, Out_ptr,
    Nq: tl.int32, Nk: tl.int32, D: tl.constexpr,
    Hq: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute Out = Attn @ V where:
      - Attn: [Nq, Hq, Nk]
      - V: [Nk, Hq, D] (expanded heads)
      - Out: [Nq, Hq, D] float32
    Grid: (Nq, Hq)
    """
    q_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    if q_idx >= Nq or h_idx >= Hq:
        return

    acc = tl.zeros((D,), dtype=tl.float32)

    # Loop over kv tokens in tiles
    for k_start in range(0, Nk, BLOCK_Q):
        k_offsets = k_start + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
        mask_k = k_offsets < Nk

        # Load attention vector for this (q_idx, h_idx) over tile k_offsets
        attn_vec = tl.load(Attn_ptr + q_idx * (Hq * Nk) + h_idx * Nk + k_offsets, mask=mask_k, other=0.0)  # [BLOCK_Q]

        # Load V tile: [BLOCK_Q, D]
        V_tile = tl.load(
            V_ptr + k_offsets[:, None] * (Hq * D) + h_idx * D + tl.arange(0, D)[None, :],
            mask=mask_k[:, None],
            other=0.0
        )  # [BLOCK_Q, D]

        # Accumulate: sum over BLOCK_Q
        acc += tl.sum(attn_vec[:, None] * V_tile, axis=0)  # [D]

    # Store output
    Out_ptrs = Out_ptr + q_idx * (Hq * D) + h_idx * D + tl.arange(0, D)
    tl.store(Out_ptrs, acc, mask=True)  # mask is all true for D


@triton.jit
def lse_segment_kernel(
    L_ptr, LSE_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr,
    delta: tl.int32,  # Nk - Nq
    q_start: tl.int32,
    sm_scale: tl.float32,
    inv_ln2: tl.float32,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Compute lse = logsumexp over masked L for this segment and head, divided by ln(2).
    For each (q, h), consider all kv positions. We compute:
      m = max(L), s = sum(exp(L - m)), lse = (log(1 + s)) * ln(2).
    Store to LSE_ptr[h * Nq + (q_start + q_idx)] for all q_idx in this segment.
    """
    h_idx = tl.program_id(0)  # one program per head
    if h_idx >= Hq:
        return

    # Compute global max m across the segment
    m = -float('inf')
    for q in range(0, Nq, BLOCK_Q):
        q_offsets = q + tl.arange(0, BLOCK_Q)
        mask_q = q_offsets < Nq
        for k in range(0, Nk, BLOCK_K):
            k_offsets = k + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < Nk
            # Load L tile [BLOCK_Q, BLOCK_K]
            L_tile = tl.load(
                L_ptr + q_offsets[:, None] * (Hq * Nk) + h_idx * Nk + k_offsets[None, :],
                mask=mask_q[:, None] & mask_k[None, :],
                other=-float('inf')
            )
            # Apply causal mask: kv < (q + 1 + delta)
            cond = k_offsets[None, :] < (q_offsets[:, None] + 1 + delta)
            L_tile = tl.where(cond, L_tile, -float('inf'))
            # Reduce max across K then across Q
            max_k = tl.max(L_tile, axis=1)  # [BLOCK_Q]
            m = tl.maximum(m, tl.max(max_k, axis=0))

    # Compute sum exp(L - m) across the segment
    s = 0.0
    for q in range(0, Nq, BLOCK_Q):
        q_offsets = q + tl.arange(0, BLOCK_Q)
        mask_q = q_offsets < Nq
        for k in range(0, Nk, BLOCK_K):
            k_offsets = k + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < Nk
            L_tile = tl.load(
                L_ptr + q_offsets[:, None] * (Hq * Nk) + h_idx * Nk + k_offsets[None, :],
                mask=mask_q[:, None] & mask_k[None, :],
                other=-float('inf')
            )
            cond = k_offsets[None, :] < (q_offsets[:, None] + 1 + delta)
            L_tile = tl.where(cond, L_tile, -float('inf'))
            # sum over k: add sum of exp((L - m) * sm_scale) across K for each Q
            # First, scale, subtract max, exp, and sum
            L_scaled = (L_tile - m) * sm_scale
            exp_tile = tl.exp(L_scaled)
            # Reduce over K -> [BLOCK_Q], then sum across Q
            s += tl.sum(exp_tile, axis=1)  # [BLOCK_Q]
        s = s  # keep s as scalar accumulator

    lse_val = tl.log(1.0 + s) * inv_ln2  # original divides by ln(2), here multiply by 1/ln(2)

    # Write lse for all q in this segment
    for q in range(Nq):
        tl.store(LSE_ptr + h_idx * Nq + (q_start + q), lse_val)


# Triton-optimized ModelNew
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants as per original assertions
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)  # 1/sqrt(128)
        self.inv_ln2 = 1.0 / math.log(2.0)  # 1/ln(2)

        # Triton tuning parameters
        self.BLOCK_M = 32
        self.BLOCK_N = 64
        self.BLOCK_D = 64  # D=128
        self.BLOCK_Q = 64
        self.BLOCK_K = 128

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda, "All tensors must be on CUDA for Triton."

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        assert len_indptr >= 1, "len_indptr must be at least 1"

        # Constants
        Hq = self.num_qo_heads
        D = self.head_dim
        g = self.gqa_ratio
        inv_ln2 = self.inv_ln2
        sm_scale = float(sm_scale) if sm_scale is not None else self.sm_scale

        # Output buffers (float32 for compute, cast later)
        output = torch.empty((total_q, Hq, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, Hq), dtype=torch.float32, device=q.device)

        # Convert to float32 for compute (matches original behavior)
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        # Precompute qo positions and kv positions (for causal mask)
        q_positions = torch.arange(total_q, device=q.device, dtype=torch.int32)
        kv_positions = torch.arange(total_kv, device=k.device, dtype=torch.int32)

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

            # GQA expansion
            k_expanded = k_batch.repeat_interleave(g, dim=1)  # [Nk, 32, 128]
            v_expanded = v_batch.repeat_interleave(g, dim=1)  # [Nk, 32, 128]

            # Allocate logits buffer L_tmp [Nq, 32, Nk] float32
            L_tmp = torch.empty((Nq, Hq, Nk), dtype=torch.float32, device=q.device)

            # Launch Triton matmul kernel: Q=q_batch, K=k_expanded, L=L_tmp
            grid_qk = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), Hq)
            qk_matmul_kernel[grid_qk](
                q_batch, k_expanded, L_tmp,
                Nq, Nk,
                Hq=Hq, D=D,
                sm_scale=sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_D=self.BLOCK_D,
            )

            # Apply causal mask in Triton (write MaskedL_tmp)
            MaskedL_tmp = torch.empty_like(L_tmp)
            grid_mask = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), triton.cdiv(Hq, 8))
            apply_causal_mask_kernel[grid_mask](
                L_tmp, MaskedL_tmp,
                Nq, Nk,
                Hq,
                q_positions, kv_positions,
                delta=Nk - Nq,
                sm_scale=sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_H=8,
            )

            # Softmax along Nk (last dim) for each (q, head) in Triton
            SoftmaxOut = torch.empty_like(MaskedL_tmp)  # store softmax result
            grid_softmax = (Nq, Hq)
            softmax_dimN_kernel[grid_softmax](
                MaskedL_tmp, SoftmaxOut,
                Nq, Nk,
                Hq=Hq, sm_scale=sm_scale,
                BLOCK_M=1, BLOCK_N=128,
            )

            # Final output: SoftmaxOut @ v_expanded -> [Nq, 32, 128]
            Out_tmp = torch.empty((Nq, Hq, D), dtype=torch.float32, device=q.device)
            grid_out = (Nq, Hq)
            attn_dot_v_kernel[grid_out](
                SoftmaxOut, v_expanded, Out_tmp,
                Nq, Nk, D=D,
                Hq=Hq,
                BLOCK_Q=self.BLOCK_Q, BLOCK_D=self.BLOCK_D,
            )

            # Store into output[q_start:q_end] as float32
            output[q_start:q_end] = Out_tmp

            # Compute lse per segment and head using Triton
            grid_lse = (Hq,)  # one program per head
            lse_segment_kernel[grid_lse](
                L_tmp, lse,  # write into lse[h, q_start + q_idx]
                Nq, Nk,
                Hq=Hq,
                delta=Nk - Nq,
                q_start=q_start,
                sm_scale=sm_scale,
                inv_ln2=inv_ln2,
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K,
            )

        # Cast output to bfloat16 to match original output dtype
        output = output.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
