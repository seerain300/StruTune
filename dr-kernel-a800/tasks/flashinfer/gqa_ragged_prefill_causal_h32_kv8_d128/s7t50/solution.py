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
      - Q: [Nq, Hq, D]
      - K: [Nk, Hq, D] (note: K is length Nk but we use expanded heads)
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
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_k[None, :]
    )


@triton.jit
def apply_causal_mask_kernel(
    L_ptr, Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr, delta: tl.int32
):
    """
    Apply forward-looking causal mask to L: set L[q, h, j] = -inf if j >= (q + 1 + delta).
    Grid: (pid0 over Nq tiles, pid1 over Nk tiles, pid2 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    q_idx = pid0
    kv_idx = pid1
    head = pid2

    # Load the tile and mask
    acc = tl.load(
        L_ptr + q_idx * (Hq * Nk) + head * Nk + kv_idx,
        mask=(q_idx < Nq) & (kv_idx < Nk),
        other=0.0
    )

    # Compute condition: kv_idx < (q_idx + 1 + delta)
    cond = kv_idx < (q_idx + 1 + delta)
    acc = tl.where(cond, acc, -float('inf'))

    # Store back
    tl.store(
        L_ptr + q_idx * (Hq * Nk) + head * Nk + kv_idx,
        acc,
        mask=(q_idx < Nq) & (kv_idx < Nk)
    )


@triton.jit
def softmax_dimN_kernel(
    L_ptr, Out_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Softmax along Nk (last dim) for each (q, head), in-place into Out_ptr (which shares memory with L_ptr).
    Grid: (pid0 over Nq tiles, pid2 over heads)
    """
    pid0 = tl.program_id(0)
    pid2 = tl.program_id(1)

    q_offsets = pid0 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq

    # Initialize m and s
    m = tl.full((BLOCK_N,), -float('inf'), dtype=tl.float32)
    s = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Reduction to compute max and sum
    for j_start in range(0, Nk, BLOCK_N):
        j_offsets = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_j = j_offsets < Nk

        # Load tile [BLOCK_N]
        tile = tl.load(
            L_ptr + q_offsets * (Hq * Nk) + pid2 * Nk + j_offsets,
            mask=mask_q & mask_j,
            other=-float('inf')
        )
        tile_max = tl.max(tile, axis=0)
        new_m = tl.maximum(m, tile_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(tile - new_m), axis=0)
        m = new_m

    # Normalize
    inv_s = 1.0 / s
    for j_start in range(0, Nk, BLOCK_N):
        j_offsets = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_j = j_offsets < Nk

        tile = tl.load(
            L_ptr + q_offsets * (Hq * Nk) + pid2 * Nk + j_offsets,
            mask=mask_q & mask_j,
            other=-float('inf')
        )
        out = tl.exp(tile - m) * inv_s
        tl.store(
            Out_ptr + q_offsets * (Hq * Nk) + pid2 * Nk + j_offsets,
            out,
            mask=mask_q & mask_j
        )


@triton.jit
def attn_dot_v_kernel(
    Attn_ptr, V_ptr, Y_ptr,
    Nq: tl.int32, Nk: tl.int32, D: tl.constexpr,
    Hq: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Compute Y = Attn @ V where:
      - Attn: [Nq, Hq, Nk]
      - V: [Nk, Hq, D]
      - Y: [Nq, Hq, D]
    Grid: (pid0 over Nq tiles, pid2 over heads)
    """
    pid0 = tl.program_id(0)
    pid2 = tl.program_id(1)

    q_offsets = pid0 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq

    acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    for j_start in range(0, Nk, BLOCK_N):
        j_offsets = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_j = j_offsets < Nk

        # Load Attn tile [BLOCK_N]
        attn_tile = tl.load(
            Attn_ptr + q_offsets * (Hq * Nk) + pid2 * Nk + j_offsets,
            mask=mask_q & mask_j,
            other=0.0
        )  # [BLOCK_N]

        # Load V tile [BLOCK_N, D]
        V_tile = tl.load(
            V_ptr + j_offsets[:, None] * (Hq * D) + pid2 * D + tl.arange(0, D),
            mask=mask_j[:, None],
            other=0.0
        )  # [BLOCK_N, D]

        # Outer product accumulate
        acc += attn_tile[:, None] * V_tile  # [BLOCK_N, D]

    tl.store(
        Y_ptr + q_offsets * (Hq * D) + pid2 * D + tl.arange(0, D),
        acc,
        mask=mask_q[:, None]
    )


@triton.jit
def lse_segment_kernel(
    L_ptr, LSE_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, sm_scale: tl.float32, log2: tl.float32,
    BLOCK_N: tl.constexpr
):
    """
    Compute LSE per (q, head) over Nk: logsumexp(L[q, h, :]) / log(2) for masked L.
    Grid: (pid0 over Nq tiles, pid1 over heads)
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    q_offsets = pid0 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq

    m = tl.full((BLOCK_N,), -float('inf'), dtype=tl.float32)
    s = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for j_start in range(0, Nk, BLOCK_N):
        j_offsets = j_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_j = j_offsets < Nk

        tile = tl.load(
            L_ptr + q_offsets * (Hq * Nk) + pid1 * Nk + j_offsets,
            mask=mask_q & mask_j,
            other=-float('inf')
        )  # [BLOCK_N]
        tile = tile * sm_scale
        tile_max = tl.max(tile, axis=0)
        new_m = tl.maximum(m, tile_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(tile - new_m), axis=0)
        m = new_m

    lse = tl.log(s) + m  # [BLOCK_N]
    lse = lse / log2
    tl.store(
        LSE_ptr + q_offsets * Hq + pid1,
        lse,
        mask=mask_q
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tile sizes; chosen for D=128 and reasonable Nk. Can be tuned.
        self.BLOCK_M = 32
        self.BLOCK_N = 128
        self.BLOCK_D = 128

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
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # Output and LSE tensors (float32)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Convert to float32 for compute
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        log2 = 1.0 / math.log(2.0)

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
            delta = Nk - Nq  # can be negative; condition handles it

            # GQA expansion
            k_expanded = k_batch.repeat_interleave(gqa_ratio, dim=1)  # [Nk, 32, 128]
            v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)  # [Nk, 32, 128]

            # Allocate logits buffer L_tmp [Nq, 32, Nk] float32
            L_tmp = torch.empty((Nq, 32, Nk), dtype=torch.float32, device=device)

            # Launch Triton matmul kernel: Q=q_batch, K=k_expanded, L=L_tmp
            grid_qk = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), 32)
            qk_matmul_kernel[grid_qk](
                q_batch, k_expanded, L_tmp,
                Nq, Nk,
                Hq=32, D=128,
                sm_scale=sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_D=self.BLOCK_D
            )

            # Apply causal mask
            grid_mask = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), 32)
            apply_causal_mask_kernel[grid_mask](
                L_tmp,
                Nq, Nk,
                Hq=32, delta=delta,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
            )

            # Softmax along Nk, in-place into L_tmp
            grid_softmax = (triton.cdiv(Nq, self.BLOCK_M), 32)
            softmax_dimN_kernel[grid_softmax](
                L_tmp, L_tmp,  # in-place write to L_tmp
                Nq, Nk,
                Hq=32,
                BLOCK_N=self.BLOCK_N
            )

            # Compute attention output Y = softmax @ v_expanded
            Y = torch.empty((Nq, 32, head_dim), dtype=torch.float32, device=device)
            grid_attn = (triton.cdiv(Nq, self.BLOCK_M), 32)
            attn_dot_v_kernel[grid_attn](
                L_tmp, v_expanded, Y,
                Nq, Nk, D=128,
                Hq=32,
                BLOCK_N=self.BLOCK_N
            )

            # Write output for this segment
            output[q_start:q_start + Nq] = Y

            # Compute LSE for this segment: logsumexp of masked L_tmp along Nk / log(2)
            LSE = torch.empty((Nq, 32), dtype=torch.float32, device=device)
            grid_lse = (triton.cdiv(Nq, self.BLOCK_M), 32)
            lse_segment_kernel[grid_lse](
                L_tmp, LSE,
                Nq, Nk,
                Hq=32, sm_scale=sm_scale, log2=log2,
                BLOCK_N=self.BLOCK_N
            )
            lse[q_start:q_start + Nq] = LSE

        return output, lse


def run(*args):
    return ModelNew()(*args)
