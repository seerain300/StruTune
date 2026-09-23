import torch
import math
import triton
import triton.language as tl


# Triton kernel: L = Q @ K^T
# Q: [Nq, Hq, D]      -> we pass q_batch[q_start:q_end]
# K: [Nk, Hq, D]      -> we pass k_batch[kv_start:kv_end] expanded to [Nk, Hq, D] via repeat_interleave on heads
# L_tmp: [Nq, Hq, Nk] -> float32 output buffer
@triton.jit
def qk_matmul_kernel(
    Q_ptr, K_ptr, L_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, D: tl.constexpr,
    sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (pid0 over Nq tiles, pid1 over Nk tiles, pid2 over Hq)
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    k_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
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

    # Store L_tmp
    L_ptrs = L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + k_offsets[None, :]
    tl.store(L_ptrs, acc, mask=mask_q[:, None] & mask_k[None, :])


# Triton kernel: softmax along last dimension (Nk) for each [Nq, Hq] row
# X: [Nq, Hq, Nk] input logits
# Y: [Nq, Hq, Nk] output softmax
@triton.jit
def softmax_lastdim_kernel(
    X_ptr, Y_ptr,
    Nq: tl.int32, Nk: tl.int32, Hq: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)  # over Hq
    # Each program handles one (Nq, Hq) pair and a BLOCK_N tile over Nk
    q = pid0
    h = pid1

    k_offsets = tl.arange(0, BLOCK_N)
    mask_k = k_offsets < Nk

    # Load row slice
    x_ptrs = X_ptr + q * (Hq * Nk) + h * Nk + k_offsets
    x = tl.load(x_ptrs, mask=mask_k, other=-float('inf'))

    # Numerically stable softmax: subtract max
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    exp_sum = tl.sum(exp_x, axis=0)
    soft = exp_x / exp_sum

    # Store
    y_ptrs = Y_ptr + q * (Hq * Nk) + h * Nk + k_offsets
    tl.store(y_ptrs, soft, mask=mask_k)


# Triton kernel: Output = Softmax @ V_expanded
# Softmax: [Nq, Hq, Nk] -> we'll pass softmax (already masked) from softmax_lastdim_kernel
# V_expanded: [Nk, Hq, D] -> we pass v_expanded with heads expanded by GQA ratio
# Output: [Nq, Hq, D] -> write into output buffer
@triton.jit
def attn_matmul_kernel(
    Soft_ptr, V_ptr, Out_ptr,
    Nq: tl.int32, Nk: tl.int32, D: tl.int32, Hq: tl.constexpr,
    G: tl.constexpr,  # GQA ratio, e.g., 4
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (pid0 over Nq tiles, pid1 over Hq tiles, pid2 over D tiles)
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    h_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N], but we iterate Hq
    d_offsets = pid2 * BLOCK_D + tl.arange(0, BLOCK_D)   # [BLOCK_D]
    mask_q = q_offsets < Nq
    mask_d = d_offsets < D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kv positions (Nk dimension) in chunks
    for k_start in range(0, Nk, BLOCK_N):
        k_offsets2 = k_start + h_offsets  # [BLOCK_N]
        mask_k = k_offsets2 < Nk

        # Load softmax slice: [BLOCK_M, BLOCK_N]
        soft_ptrs = Soft_ptr + q_offsets[:, None] * (Hq * Nk) + (h_offsets[None, :]) * Nk
        soft = tl.load(soft_ptrs, mask=mask_q[:, None] & mask_k[None, :], other=0.0)

        # Load V_expanded slice per head: V_ptr has [Nk, Hq, D], we need to map head index
        # For Triton, we cannot loop Hq easily; instead, compute for each h and accumulate across expanded heads.
        # Since G is known, we can iterate h and accumulate:
        for h_idx in range(0, Hq):
            v_ptrs = V_ptr + k_offsets2[:, None] * (Hq * D) + h_idx * D + d_offsets[None, :]
            v_tile = tl.load(v_ptrs, mask=mask_k[:, None] & mask_d[None, :], other=0.0)
            acc += tl.dot(soft[:, :, None] * v_tile, tl.ones((BLOCK_N, 1), dtype=tl.float32))  # effectively multiply then sum along k

        # Note: The above loop over Hq is necessary because we need to combine contributions from all heads for each q,h.
        # Better approach: reformulate V as [Nk*G, Hq, D] but Triton pointer arithmetic needs compile-time shapes.
        # Instead, keep the simple loop which Triton can unroll for small Hq (e.g., 32). For larger, this would be inefficient.
        # To keep Triton usage, we implement a second loop over h (unrolled) and accumulate properly by computing v for each head and multiplying by softmax and summing. This is still acceptable and avoids PyTorch ops.

        # The above pattern keeps code simple and ensures we stay in Triton. If performance becomes an issue, consider using torch for the final einsum to avoid Triton's complexity, but here we are required to stay in Triton.
        # We will maintain correctness by looping over Hq and accumulating.

    # Store output
    out_ptrs = Out_ptr + q_offsets[:, None] * (Hq * D) + h_offsets[None, :] * D + d_offsets[None, :]
    tl.store(out_ptrs, acc, mask=mask_q[:, None] & mask_h[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants as in original
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)  # 1/sqrt(128)
        # Triton tuning parameters
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_D = 64
        self.BLOCK_M_S = 64
        self.BLOCK_N_S = 128  # for softmax
        self.BLOCK_M_OUT = 64
        self.BLOCK_N_OUT = 32
        self.BLOCK_D_OUT = 64

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda, "All tensors must be on CUDA for Triton."

        device = q.device
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        assert len_indptr >= 1, "len_indptr must be at least 1"

        # Output buffers (float32 compute)
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=device)

        # Convert to float32 for compute; qo_indptr/kv_indptr are int32
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        q_positions = torch.arange(total_q, device=device, dtype=torch.int32)
        kv_positions = torch.arange(total_kv, device=device, dtype=torch.int32)

        # Process each segment
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Extract batch slices
            q_batch = q_f32[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_batch = k_f32[kv_start:kv_end]  # [num_kv_tokens, 8, 128]
            v_batch = v_f32[kv_start:kv_end]  # [num_kv_tokens, 8, 128]

            num_q_tokens = q_batch.shape[0]
            num_kv_tokens = k_batch.shape[0]

            # Expand K/V by GQA ratio (4)
            k_expanded = k_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]

            # Allocate logits buffer [num_q_tokens, 32, num_kv_tokens]
            logits = torch.empty((num_q_tokens, self.num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)

            # Launch qk_matmul_kernel
            grid_qk = (triton.cdiv(num_q_tokens, self.BLOCK_M), triton.cdiv(num_kv_tokens, self.BLOCK_N), self.num_qo_heads)
            qk_matmul_kernel[grid_qk](
                q_batch, k_expanded, logits,
                num_q_tokens, num_kv_tokens,
                Hq=self.num_qo_heads, D=self.head_dim,
                sm_scale=sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_D=self.BLOCK_D
            )

            # Apply causal mask: kv < (q_idx + 1 + delta), delta = num_kv_tokens - num_q_tokens
            delta = num_kv_tokens - num_q_tokens
            # Create index tensors on device
            q_positions = torch.arange(num_q_tokens, device=device, dtype=torch.int32)
            kv_positions = torch.arange(num_kv_tokens, device=device, dtype=torch.int32)
            cond = (kv_positions[None, :] < (q_positions[:, None] + 1 + delta)).to(torch.int32)
            cond = cond.to(torch.bool)
            # Mask logits: set invalid positions to -inf for stability
            logits[~cond] = float('-inf')

            # Softmax along last dimension (Nk) via Triton kernel
            grid_soft = (num_q_tokens, self.num_qo_heads)
            soft = torch.empty((num_q_tokens, self.num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)
            softmax_lastdim_kernel[grid_soft](
                logits, soft,
                num_q_tokens, num_kv_tokens, Hq=self.num_qo_heads,
                BLOCK_M=self.BLOCK_M_S, BLOCK_N=self.BLOCK_N_S
            )

            # Compute output = soft @ v_expanded (einsum 'qhk,khd->qhd')
            out_batch = torch.empty((num_q_tokens, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
            grid_out = (triton.cdiv(num_q_tokens, self.BLOCK_M_OUT),
                        triton.cdiv(self.num_qo_heads, self.BLOCK_N_OUT),
                        triton.cdiv(self.head_dim, self.BLOCK_D_OUT))
            attn_matmul_kernel[grid_out](
                soft, v_expanded, out_batch,
                num_q_tokens, num_kv_tokens, self.head_dim, Hq=self.num_qo_heads, G=self.gqa_ratio,
                BLOCK_M=self.BLOCK_M_OUT, BLOCK_N=self.BLOCK_N_OUT, BLOCK_D=self.BLOCK_D_OUT
            )

            # Store results for this segment
            output[q_start:q_end] = out_batch
            # LSE: compute logsumexp of logits (before softmax) per (q,h), then divide by ln(2)
            lse_seg = torch.logsumexp(logits.float(), dim=-1) * self.inv_ln2  # shape [num_q_tokens, 32]
            lse[q_start:q_end] = lse_seg

        # Cast outputs to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
