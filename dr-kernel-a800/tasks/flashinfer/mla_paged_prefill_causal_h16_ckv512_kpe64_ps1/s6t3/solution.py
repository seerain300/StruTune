import math
import torch
import triton
import triton.language as tl

# Kernel 1: Compute logits[h, l] = sum_k q_nope[i, k] * Kc[l, k] + sum_k q_pe[i, k] * Kp[l, k]
# Input:
#   Q_nope_ptr: [H, D_ckv] float32
#   Q_pe_ptr:   [H, D_kpe] float32
#   Kc_ptr:     [L, D_ckv] float32
#   Kp_ptr:     [L, D_kpe] float32
#   Logits_ptr: [H, L] float32 (to be written)
# Output:
#   Logits_ptr filled
@triton.jit
def compute_logits_kernel(
    Q_nope_ptr, Q_pe_ptr, Kc_ptr, Kp_ptr, Logits_ptr,
    H, L, D_ckv, D_kpe,
    # strides
    Q_nope_stride0, Q_nope_stride1,
    Q_pe_stride0, Q_pe_stride1,
    Kc_stride0, Kc_stride1,
    Kp_stride0, Kp_stride1,
    Logits_stride0, Logits_stride1,
    BLOCK_H: tl.constexpr, BLOCK_L: tl.constexpr
):
    h_pid = tl.program_id(0)
    l_pid = tl.program_id(1)
    hs = h_pid * BLOCK_H + tl.arange(0, BLOCK_H)
    ls = l_pid * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_h = hs < H
    mask_l = ls < L

    # accumulator for logits per (h, l)
    acc = tl.zeros((BLOCK_H, BLOCK_L), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, D_ckv + D_kpe):
        # pick source tensor
        use_Kc = k < D_ckv
        # Load q vectors for these heads
        q_n_ptrs = Q_nope_ptr + hs * Q_nope_stride0 + k * Q_nope_stride1
        q_p_ptrs = Q_pe_ptr + hs * Q_pe_stride0 + k * Q_pe_stride1
        q_n = tl.load(q_n_ptrs, mask=mask_h, other=0.0)
        q_p = tl.load(q_p_ptrs, mask=mask_h, other=0.0)
        # Load K vectors for these L positions
        if use_Kc:
            K_vals = tl.load(Kc_ptr + ls * Kc_stride0 + k * Kc_stride1, mask=mask_l, other=0.0)
        else:
            K_vals = tl.load(Kp_ptr + ls * Kp_stride0 + (k - D_ckv) * Kp_stride1, mask=mask_l, other=0.0)
        # Outer product and accumulate: acc[h, l] += q_n[h] * K_vals[l] + q_p[h] * K_vals[l]
        # Broadcast q_n over L, K_vals over H
        acc += q_n[:, None] * K_vals[None, :] + q_p[:, None] * K_vals[None, :]

    # Store results
    out_ptrs = Logits_ptr + hs[:, None] * Logits_stride0 + ls[None, :] * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_h[:, None] & mask_l[None, :])


# Kernel 2: For each (b, i), compute lse_scaled = logsumexp(logits[h, :]) / ln(2) with causal mask.
# Input:
#   Logits_ptr: [H, L] float32
#   mask_ptr:   [L] int32 (0/1 mask: 1 means causal, 0 means not causal)
# Output:
#   lse_ptr[h] = log(sum(exp(logits[h, l])) where causal)/ln(2)
@triton.jit
def lse_mask_kernel(
    Logits_ptr, mask_ptr, lse_ptr,
    H, L,
    Logits_stride0, Logits_stride1,
    # strides for mask are 1D: stride = 1
    ln_inv: tl.float32,  # 1/ln(2)
    BLOCK_H: tl.constexpr, BLOCK_L: tl.constexpr
):
    h_pid = tl.program_id(0)
    # One program per head
    h = h_pid
    if h >= H:
        return

    # First pass: compute max over masked elements
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        m = tl.load(mask_ptr + ls, mask=mask_l, other=1)  # 1 means causal
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        # Set non-causal to -inf
        vals = tl.where(m == 0, -float('inf'), vals)
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Second pass: compute sum of exp(masked (x - max))
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        m = tl.load(mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, -float('inf'), vals)
        e = tl.exp(vals - max_val)
        # Ensure masked entries contribute 0
        e = tl.where(m == 0, 0.0, e)
        sum_exp += tl.sum(e, axis=0)

    # Compute lse_scaled = log(sum_exp) * ln_inv
    sum_exp = tl.maximum(sum_exp, 0.0)  # numerical guard
    lse_val = tl.log2(sum_exp) * ln_inv
    tl.store(lse_ptr + h, lse_val)


# Kernel 3: For each (b, i, h), compute softmax over L of masked logits and then out = softmax @ Kc.
# Input:
#   Logits_ptr: [H, L] float32
#   mask_ptr:   [L] int32 (0/1)
#   Kc_ptr:     [L, D_ckv] float32
#   Out_ptr:    [H, D_ckv] float32 (to be written)
# Output:
#   Out_ptr[h, :] filled
@triton.jit
def softmax_matmul_kernel(
    Logits_ptr, mask_ptr, Kc_ptr, Out_ptr,
    H, L, D_ckv,
    Logits_stride0, Logits_stride1,
    Kc_stride0, Kc_stride1,
    Out_stride0, Out_stride1,
    ln_inv: tl.float32,  # 1/ln(2), not used here but kept for signature symmetry
    BLOCK_H: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr
):
    h_pid = tl.program_id(0)
    h = h_pid
    if h >= H:
        return

    # Compute max for stability
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        m = tl.load(mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, -float('inf'), vals)
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp over masked logits
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        m = tl.load(mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        vals = tl.where(m == 0, -float('inf'), vals)
        e = tl.exp(vals - max_val)
        e = tl.where(m == 0, 0.0, e)
        sum_exp += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_exp

    # Now produce output vector out[h, :] = softmax @ Kc
    out_vec = tl.zeros((D_ckv,), dtype=tl.float32)
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv

        # Compute softmax for each l position and accumulate into out_vec[ks]
        for l0 in range(0, L, BLOCK_L):
            ls = l0 + tl.arange(0, BLOCK_L)
            mask_l = ls < L
            m = tl.load(mask_ptr + ls, mask=mask_l, other=1)
            vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
            vals = tl.where(m == 0, -float('inf'), vals)
            e = tl.exp(vals - max_val) * inv_sum  # masked entries were set to 0 previously; here we apply inv_sum
            # For non-causal, e should be 0; we can mask contributions by e directly if needed.
            # Accumulate contribution: out_vec += e[:, None] * Kc[ls, ks[None, :]]
            Kc_block = tl.load(Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1,
                               mask=mask_l[:, None] & mask_k[None, :], other=0.0)
            # For each element, multiply and reduce across L
            # We'll do a manual accumulation for clarity:
            for l_idx in range(BLOCK_L):
                if (l0 + l_idx) < L:
                    # scalar e for this l position
                    e_l = e[l_idx]
                    Kc_row = tl.load(Kc_ptr + (l0 + l_idx) * Kc_stride0 + ks * Kc_stride1,
                                     mask=mask_k, other=0.0)
                    out_vec += e_l * Kc_row

    # Store result
    out_ptrs = Out_ptr + h * Out_stride0 + tl.arange(0, D_ckv) * Out_stride1
    tl.store(out_ptrs, out_vec, mask=tl.arange(0, D_ckv) < D_ckv)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        # Ensure dtype float32 for compute
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Prepare output
        output = torch.empty(
            (total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device
        )
        lse = torch.empty(
            (total_q, num_qo_heads), dtype=torch.float32, device=device
        )

        # batch size from indptr
        batch_size = int(kv_indptr.shape[0]) - 1
        # Loop over batches
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # KV block for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [kv_len]
            Kc_batch = Kc_all[tok_idx]  # [kv_len, 512]
            Kp_batch = Kp_all[tok_idx]  # [kv_len, 64]

            # Prepare masks for causal (only for this b)
            # mask[l] = 1 if l <= (L - q_len + i), else 0
            for i in range(q_len):
                # For this (b, i), compute masks and lse
                # First compute logits [H, L] for this query
                H = num_qo_heads
                L = kv_len
                D_ckv = head_dim_ckv
                D_kpe = q_pe_f32.shape[-1]  # should be 64

                # Allocate logits buffer
                logits = torch.empty((H, L), dtype=torch.float32, device=device)
                # Launch compute_logits_kernel
                grid = (H, triton.cdiv(L, 128))
                compute_logits_kernel[grid](
                    q_nope_f32[q_start + i], q_pe_f32[q_start + i], Kc_batch, Kp_batch, logits,
                    H, L, D_ckv, D_kpe,
                    q_nope_f32.stride(0), q_nope_f32.stride(1),
                    q_pe_f32.stride(0), q_pe_f32.stride(1),
                    Kc_batch.stride(0), Kc_batch.stride(1),
                    Kp_batch.stride(0), Kp_batch.stride(1),
                    logits.stride(0), logits.stride(1),
                    BLOCK_H=32, BLOCK_L=128,
                    num_warps=4, num_stages=2,
                )

                # Build causal mask as int32 vector on device: 1 if causal, 0 otherwise
                # prefix_len = L - q_len + i
                prefix_len = L - q_len + i
                mask_vec = torch.arange(L, dtype=torch.int32, device=device)
                mask_vec = (mask_vec <= prefix_len).to(torch.int32)

                # Compute lse_scaled per head using lse_mask_kernel
                lse_scaled = torch.empty((H,), dtype=torch.float32, device=device)
                grid_lse = (H,)
                ln_inv = 1.0 / math.log(2.0)
                lse_mask_kernel[grid_lse](
                    logits, mask_vec, lse_scaled,
                    H, L,
                    logits.stride(0), logits.stride(1),
                    ln_inv,
                    BLOCK_H=1, BLOCK_L=256,
                    num_warps=2, num_stages=2,
                )
                lse[q_start + i, :] = lse_scaled  # store lse vector per query

                # Compute output vector per head using softmax_matmul_kernel
                for h in range(H):
                    out_vec = torch.empty((D_ckv,), dtype=torch.float32, device=device)
                    grid_out = (1,)
                    softmax_matmul_kernel[grid_out](
                        logits, mask_vec, Kc_batch, out_vec,
                        H, L, D_ckv,
                        logits.stride(0), logits.stride(1),
                        Kc_batch.stride(0), Kc_batch.stride(1),
                        out_vec.stride(0), out_vec.stride(1),
                        ln_inv,
                        BLOCK_H=1, BLOCK_L=256, BLOCK_K=64,
                        num_warps=4, num_stages=2,
                    )
                    output[q_start + i, h] = out_vec

        # Return bfloat16 output and float32 lse
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
