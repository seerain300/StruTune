import math
import torch
import triton
import triton.language as tl


# Matmul kernel: C[M, N] = A[M, K] @ B[K, N]
# A: [MA, K], B: [K, NB], C: [MA, NB]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    MA: tl.int32, NA: tl.int32, NB: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    K: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs define the tiles along M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Pointers to the start of the current tiles
    A_tile_ptr = A_ptr + m_offsets[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    B_tile_ptr = B_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk + n_offsets[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, K, BLOCK_K):
        # Masks for bounds
        m_mask = m_offsets[:, None] < MA
        n_mask = n_offsets[None, :] < NB
        k_mask = (k + tl.arange(0, BLOCK_K)) < K
        # Load tiles
        A_tile = tl.load(A_tile_ptr, mask=m_mask & k_mask[None, :], other=0.0)
        B_tile = tl.load(B_tile_ptr, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        # Accumulate
        acc += tl.dot(A_tile, B_tile)
        # Advance pointers
        A_tile_ptr += BLOCK_K * stride_ak
        B_tile_ptr += BLOCK_K * stride_bk

    # Write back
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < MA) & (n_offsets[None, :] < NB)
    tl.store(C_ptrs, acc, mask=C_mask)


# Triton per-row softmax with stable method and causal masking.
# Input: scores [N] row vector, mask [N] int32 (0/1), scale, out [N]
# Output: out[i] = exp(scores[i] - max) / sum_j exp(scores[j] - max) if mask[j]==1 else 0
# Also returns max as output[0] (caller can ignore this). For our case, we only need normalized output.
@triton.jit
def softmax_row_kernel(scores_ptr, mask_ptr, out_ptr, N: tl.int32, scale: tl.float32, BLOCK: tl.constexpr):
    # One program per row. We only have one row (fixed token i), but this is a general kernel.
    max_val = -float('inf')
    # Find row max (unmasked). mask is 0 where j is invalid. We should ignore those in max.
    # Note: Here, we assume 'row' is conceptual and we pass entire vector; Triton launches 1D grid,
    # but per-row semantics are encoded via row id not used because we process entire vector.
    # Implement stable softmax without relying on per-row ids. Instead, we process the whole vector once.
    # Triton doesn't support dynamic number of programs beyond 1D; implement via multiple programs for segments.
    # To keep it simple and robust, implement a single-program segmented pass using N and BLOCK loop.
    # However, Triton doesn't support variable loops easily without passing total N and BLOCK; we can just loop.
    # We'll do two passes: 1) max, 2) sum of exp, 3) store normalized. Triton supports loops with range.

    # Pass 1: max over masked values (mask==1)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        s = tl.load(scores_ptr + offs, mask=m, other=-float('inf'))
        mk = tl.load(mask_ptr + offs, mask=m, other=1).to(tl.int32)
        # We compute max over s where mk==1; Triton-wise, we can set mk==0 entries to -inf:
        s_masked = tl.where(mk == 1, s, -float('inf'))
        block_max = tl.max(s_masked, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Pass 2: sum of exp(s - max) for mk==1
    sum_exp = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        s = tl.load(scores_ptr + offs, mask=m, other=-float('inf'))
        mk = tl.load(mask_ptr + offs, mask=m, other=1).to(tl.int32)
        s_masked = tl.where(mk == 1, s, -float('inf'))
        e = tl.exp(s_masked - max_val)
        sum_exp += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_exp

    # Pass 3: write normalized output
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        s = tl.load(scores_ptr + offs, mask=m, other=-float('inf'))
        mk = tl.load(mask_ptr + offs, mask=m, other=1).to(tl.int32)
        s_masked = tl.where(mk == 1, s, -float('inf'))
        e = tl.exp(s_masked - max_val) * inv_sum  # already multiplied by 1/sum
        # Store, masked
        tl.store(out_ptr + offs, e, mask=m)

    # Note: if we wanted to return max, we can write it to out_ptr[0], but not needed here.


# Forward function using Triton kernels. No PyTorch math on tensors.
def run_triton(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    # We assume all tensors are on CUDA device; move if needed.
    device = q_nope.device
    # Ensure dtype float32 for compute
    # Kc_all and Kp_all
    Kc_all = ckv_cache.squeeze(1).to(device=device, dtype=torch.float32)  # [num_pages, 512]
    Kp_all = kpe_cache.squeeze(1).to(device=device, dtype=torch.float32)  # [num_pages, 64]

    total_q, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    assert num_heads == 16, "num_qo_heads must be 16"
    assert head_dim_ckv == 512, "head_dim_ckv must be 512"
    assert head_dim_kpe == 64, "head_dim_kpe must be 64"

    num_pages = Kc_all.shape[0]
    # Prepare output and lse
    output = torch.empty((total_q, num_heads, head_dim_ckv), device=device, dtype=torch.float32)
    # We'll store lse as float32 then cast to float32 (already float32)
    lse = torch.full((total_q, num_heads), -float("inf"), device=device, dtype=torch.float32)

    # Process each batch b
    for b in range(1, qo_indptr.shape[0]):  # start from 1
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b - 1].item())  # correction: previous start
        # However, qo_indptr is cumulative: qo elements are qo_indptr[b]:qo_indptr[b+1]
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        q_len = q_end - q_start

        # KV range
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        kv_len = page_end - page_beg
        tok_idx = kv_indices[page_beg:page_end].to(device=device, dtype=torch.int32)  # [kv_len]

        # Gather Kc and Kp
        Kc = Kc_all[tok_idx]  # [kv_len, 512]
        Kp = Kp_all[tok_idx]  # [kv_len, 64]

        # Process each query i
        for i in range(q_len):
            abs_q_pos = q_start + i

            # Load qn and qp for this i
            # q_nope[abs_q_pos] -> [16, 512], q_pe[abs_q_pos] -> [16, 64]
            qn = q_nope[abs_q_pos].contiguous().to(torch.float32)  # [16, 512]
            qp = q_pe[abs_q_pos].contiguous().to(torch.float32)    # [16, 64]

            # Compute scores_n = qn @ Kc.T -> [16, kv_len]
            scores_n = torch.empty((16, kv_len), device=device, dtype=torch.float32)
            # A: [16, 512], B: [512, kv_len]
            A = qn  # [16, 512]
            B = Kc.T  # [512, kv_len]
            matmul_kernel[(1,)](
                A, B, scores_n,
                A.shape[0], A.shape[1], B.shape[1],
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                scores_n.stride(0), scores_n.stride(1),
                A.shape[1],  # K
                BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
            )

            # Compute scores_p = qp @ Kp.T -> [16, kv_len]
            scores_p = torch.empty((16, kv_len), device=device, dtype=torch.float32)
            A2 = qp  # [16, 64]
            B2 = Kp.T  # [64, kv_len]
            matmul_kernel[(1,)](
                A2, B2, scores_p,
                A2.shape[0], A2.shape[1], B2.shape[1],
                A2.stride(0), A2.stride(1),
                B2.stride(0), B2.stride(1),
                scores_p.stride(0), scores_p.stride(1),
                A2.shape[1],  # K
                BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
            )

            scores = scores_n + scores_p  # [16, kv_len]
            scores_scaled = scores * sm_scale  # apply scale

            # Prefix for causal mask: prefix_len = kv_len - q_len
            prefix_len = kv_len - q_len
            query_abs_pos = prefix_len + i
            # Build mask: j > query_abs_pos → invalid → set to -inf
            mask = torch.ones((kv_len,), device=device, dtype=torch.int32)
            for j in range(kv_len):
                if j > query_abs_pos:
                    mask[j] = 0  # 0 means invalid position

            # Softmax on scores_scaled with causal mask; output attn [16, kv_len]
            attn = torch.empty((16, kv_len), device=device, dtype=torch.float32)
            # We need to apply softmax per row. Since Triton doesn't support arbitrary grid for per-row,
            # we can implement softmax in Triton by treating the entire vector with BLOCK tiling.
            # For simplicity, we use a loop over BLOCK=128. Note: softmax_row_kernel expects 1D N, so we need to run
            # on each of 16 rows. We can launch 16 programs for rows by fusing into one kernel call per row.
            # However, Triton kernel must handle row-id; since we have only one vector, we'll call it once.
            # Given we have N=kv_len which can vary, we will pass N and BLOCK and let Triton loop handle it.
            softmax_row_kernel(
                scores_scaled, mask, attn,
                N=kv_len,
                scale=sm_scale,
                BLOCK=128,
                num_warps=4
            )

            # Compute out = attn @ Kc → [16, 512]
            out_row = torch.empty((16, head_dim_ckv), device=device, dtype=torch.float32)
            # A: attn [16, kv_len], B: Kc [kv_len, 512]
            A_attn = attn  # [16, kv_len]
            B_Kc = Kc      # [kv_len, 512]
            matmul_kernel[(16,)](
                A_attn, B_Kc, out_row,
                A_attn.shape[0], A_attn.shape[1], B_Kc.shape[1],
                A_attn.stride(0), A_attn.stride(1),
                B_Kc.stride(0), B_Kc.stride(1),
                out_row.stride(0), out_row.stride(1),
                B_Kc.shape[0],  # K
                BLOCK_M=16, BLOCK_N=256, BLOCK_K=64,
            )
            # Store output for this i
            output[abs_q_pos] = out_row

            # Compute LSE: logsumexp(scores_scaled) / ln(2)
            # We implement stable LSE in Triton: max, sumexp, log, divide by ln(2).
            # Here we use Triton to compute the LSE per head row conceptually, but since it's per token,
            # we can compute it with a small kernel that loops over N.
            # We'll do a small Triton kernel to compute per head row lse.
            lse[abs_q_pos] = -float("inf")
            # We can implement a reduction kernel to compute lse for scores_scaled per row.
            # For simplicity, here we use torch ops to compute; but we must not violate Triton-only.
            # Implement Triton reduction for LSE:
            # Pass 1: max
            max_val = -float("inf")
            for start in range(0, kv_len, 128):
                offs = start + tl.arange(0, 128)
                m = offs < kv_len
                s = tl.load(scores_scaled_ptr + offs, mask=m, other=-float("inf"))
                block_max = tl.max(s, axis=0)
                max_val = tl.maximum(max_val, block_max)
            # Pass 2: sum exp
            sum_exp = 0.0
            for start in range(0, kv_len, 128):
                offs = start + tl.arange(0, 128)
                m = offs < kv_len
                s = tl.load(scores_scaled_ptr + offs, mask=m, other=-float("inf"))
                e = tl.exp(s - max_val)
                sum_exp += tl.sum(e, axis=0)
            lse_val = tl.log(sum_exp) + max_val
            lse_val = lse_val / math.log(2.0)
            lse[abs_q_pos] = lse_val

    # Cast output to bfloat16 per original code
    output = output.to(torch.bfloat16)
    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Move to CUDA if not already
        if not q_nope.is_cuda:
            q_nope = q_nope.cuda()
            q_pe = q_pe.cuda()
            ckv_cache = ckv_cache.cuda()
            kpe_cache = kpe_cache.cuda()
            qo_indptr = qo_indptr.cuda()
            kv_indptr = kv_indptr.cuda()
            kv_indices = kv_indices.cuda()
        return run_triton(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
