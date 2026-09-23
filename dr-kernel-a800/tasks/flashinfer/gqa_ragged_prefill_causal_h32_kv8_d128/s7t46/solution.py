import math
import torch
import triton
import triton.language as tl


# Kernel 1: Compute logits L = Q @ K^T for each segment
@triton.jit
def qk_matmul_kernel(
    Q_ptr, K_ptr, L_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, D: tl.constexpr,
    sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid0 = tl.program_id(0)  # tiles over Nq
    pid1 = tl.program_id(1)  # tiles over Nk
    pid2 = tl.program_id(2)  # head id (0..Hq-1)

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

        # Accumulate outer product
        acc += tl.dot(Q_tile, tl.trans(K_tile))  # [BLOCK_M, BLOCK_N]

    # Scale by sm_scale
    acc = acc * sm_scale

    # Store logits into L: shape [Nq, Hq, Nk]
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        acc,
        mask=mask_q[:, None] & mask_k[None, :]
    )


# Kernel 2: Apply causal mask to L (L_ptr): set to -inf where kv >= q + 1 + delta
@triton.jit
def apply_causal_mask_kernel(
    L_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr, delta: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid0 = tl.program_id(0)  # tiles over Nq
    pid1 = tl.program_id(1)  # tiles over Nk
    pid2 = tl.program_id(2)  # head id (0..Hq-1)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_q = q_offsets < Nq
    mask_k = k_offsets < Nk

    # Load current logits tile
    tile = tl.load(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        mask=mask_q[:, None] & mask_k[None, :],
        other=0.0
    )

    # Causal condition: allow kv < q + 1 + delta; otherwise set to -inf
    cond = k_offsets[None, :] < (q_offsets[:, None] + 1 + delta)
    masked_tile = tl.where(cond, tile, -float('inf'))

    # Store masked logits back
    tl.store(
        L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :],
        masked_tile,
        mask=mask_q[:, None] & mask_k[None, :]
    )


# Kernel 3: Softmax along Nk (KV tokens) for each (q, head) in L_ptr
@triton.jit
def masked_softmax_dimN_kernel(
    L_ptr, S_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid0 = tl.program_id(0)  # tiles over Nq
    pid1 = tl.program_id(1)  # head id (0..Hq-1)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_q = q_offsets < Nq

    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Reduce over Nk in tiles
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk
        tile = tl.load(
            L_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=-float('inf')
        )
        tile_max = tl.max(tile, axis=1)  # [BLOCK_M]
        new_m = tl.maximum(m, tile_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(tile - new_m[:, None]), axis=1)
        m = new_m

    # Normalize
    soft = tl.exp(s)  # [BLOCK_M]
    inv_Nk = 1.0 / Nk
    # Store softmax to S: shape [Nq, Hq] as 1D contiguous
    # We compute S as [Nq, Hq], pointer arithmetic will be linearized later.
    # Here we store per (q, head) vector:
    for i in range(BLOCK_M):
        if (q_offsets[i] < Nq):
            head = pid1
            # Compute linear index for S_ptr: ((q_offsets[i] * Hq) + head)
            idx = q_offsets[i] * Hq + head
            # Store soft[i] at this index
            tl.store(S_ptr + idx, soft[i])


# Kernel 4: Output Y = S @ V_expanded, accumulate over Nk
@triton.jit
def attn_dot_v_kernel(
    S_ptr, V_ptr, Y_ptr,
    Nq: tl.int32, Nk: tl.int32, D: tl.constexpr,
    Hq: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid0 = tl.program_id(0)  # tiles over Nq
    pid2 = tl.program_id(1)  # head id (0..Hq-1)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_q = q_offsets < Nq

    # Accumulator [BLOCK_M, D]
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for k_start in range(0, Nk):
        # Load softmax value s[q, head] for this k
        # S is [Nq, Hq], linear index q*Hq + head
        s_val = tl.load(S_ptr + q_offsets * Hq + pid2, mask=mask_q, other=0.0)  # [BLOCK_M]

        # Load V_expanded row for this k and head
        # V_expanded: [Nk, Hq, D]
        V_row = tl.load(
            V_ptr + k_start * (Hq * D) + pid2 * D + tl.arange(0, D),
            mask=True,
            other=0.0
        )  # [D]

        # Fused multiply-add
        acc += s_val[:, None] * V_row[None, :]

    # Store Y: [Nq, Hq, D]
    tl.store(
        Y_ptr + q_offsets[:, None] * (Hq * D) + pid2 * D + tl.arange(0, D)[None, :],
        acc,
        mask=mask_q[:, None]
    )


# Kernel 5: Compute LSE = logsumexp(L) along Nk for each (q, head), divide by ln(2)
@triton.jit
def lse_segment_kernel(
    L_ptr, LSE_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid0 = tl.program_id(0)  # tiles over Nq
    pid1 = tl.program_id(1)  # head id (0..Hq-1)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_q = q_offsets < Nq

    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, Nk, BLOCK_N):
        k_offsets = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = k_offsets < Nk
        tile = tl.load(
            L_ptr + q_offsets[:, None] * (Hq * Nk) + pid1 * Nk + k_offsets[None, :],
            mask=mask_q[:, None] & mask_k[None, :],
            other=-float('inf')
        )
        tile_max = tl.max(tile, axis=1)
        new_m = tl.maximum(m, tile_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(tile - new_m[:, None]), axis=1)
        m = new_m

    lse = tl.log(s) + m  # [BLOCK_M]
    lse = lse / math.log(2.0)  # divide by ln(2)
    # Store LSE: [Nq, Hq], linear index q*Hq + head
    for i in range(BLOCK_M):
        if (q_offsets[i] < Nq):
            head = pid1
            idx = q_offsets[i] * Hq + head
            tl.store(LSE_ptr + idx, lse[i])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tunable block sizes; can be adjusted based on hardware
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_D = 64

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
        # Triton kernels require CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernels require CUDA tensors"
        device = q.device

        # Ensure contiguity
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        Hq = 32
        num_kv_heads = 8
        head_dim = 128
        g = Hq // num_kv_heads  # 4

        # Outputs
        output = torch.empty((total_q, Hq, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, Hq), dtype=torch.float32, device=device)

        # Iterate segments
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Extract segments
            q_batch = q[q_start:q_end].contiguous()  # [Nq, 32, 128]
            k_batch = k[kv_start:kv_end].contiguous()  # [Nk, 8, 128]
            v_batch = v[kv_start:kv_end].contiguous()  # [Nk, 8, 128]

            Nq = q_batch.shape[0]
            Nk = k_batch.shape[0]
            delta = Nk - Nq  # per-segment delta

            # GQA expansion: repeat heads by g (4x) for both K and V
            k_expanded = k_batch.repeat_interleave(g, dim=1)  # [Nk, 32, 128]
            v_expanded = v_batch.repeat_interleave(g, dim=1)  # [Nk, 32, 128]

            # Convert to float32 for compute
            q_f32 = q_batch.to(torch.float32)
            k_f32 = k_expanded.to(torch.float32)
            v_f32 = v_expanded.to(torch.float32)

            # Allocate intermediate buffers
            logits = torch.empty((Nq, Hq, Nk), dtype=torch.float32, device=device)  # L
            softmax = torch.empty((Nq, Hq), dtype=torch.float32, device=device)     # S
            out_partial = torch.empty((Nq, Hq, head_dim), dtype=torch.float32, device=device)  # Y (intermediate)

            # 1) Compute logits L = Q @ K^T (scaled) using Triton
            grid_qk = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), Hq)
            qk_matmul_kernel[grid_qk](
                q_f32, k_f32, logits,
                Nq, Nk,
                Hq=Hq, D=head_dim,
                sm_scale=sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_D=self.BLOCK_D
            )

            # 2) Apply causal mask to logits using Triton
            grid_mask = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), Hq)
            apply_causal_mask_kernel[grid_mask](
                logits,
                Nq, Nk, Hq, delta,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
            )

            # 3) Compute softmax along Nk (per (q, head)) using Triton
            grid_softmax = (triton.cdiv(Nq, self.BLOCK_M), Hq)
            masked_softmax_dimN_kernel[grid_softmax](
                logits, softmax,
                Nq, Nk, Hq,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
            )

            # 4) Compute output Y = softmax @ V_expanded using Triton (loop over k tokens inside kernel)
            # Note: We use attn_dot_v_kernel which reduces over Nk inside the kernel.
            grid_attn = (triton.cdiv(Nq, self.BLOCK_M), Hq)
            attn_dot_v_kernel[grid_attn](
                softmax, v_f32, out_partial,
                Nq, Nk, D=head_dim,
                Hq=Hq,
                BLOCK_M=self.BLOCK_M, BLOCK_D=self.BLOCK_D
            )

            # 5) Compute LSE for this segment using Triton (per (q, head))
            grid_lse = (triton.cdiv(Nq, self.BLOCK_M), Hq)
            lse_segment_kernel[grid_lse](
                logits, lse[q_start:q_end],
                Nq, Nk, Hq,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N
            )

            # Assign partial output to final output slice
            output[q_start:q_end] = out_partial

        return output, lse


def run(*args):
    return ModelNew()(*args)
