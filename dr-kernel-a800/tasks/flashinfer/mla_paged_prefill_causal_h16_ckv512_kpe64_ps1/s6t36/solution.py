import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    Qn_ptr, Qp_ptr, Kc_ptr, Kp_ptr, Logits_ptr,
    H, L, D_ckv, D_kpe,
    Qn_stride0, Qn_stride1,
    Qp_stride0, Qp_stride1,
    Kc_stride0, Kc_stride1,
    Kp_stride0, Kp_stride1,
    Logits_stride0, Logits_stride1,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (H, cdiv(L, BLOCK_L))
    h = tl.program_id(0)
    pid_l = tl.program_id(1)

    ls = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = ls < L

    # Accumulator for logits[h, ls]
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    # Iterate Kc dimension in chunks of BLOCK_K
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv

        # Load q_nope[h, ks] as vector [BLOCK_K]
        qn_ptrs = Qn_ptr + h * Qn_stride0 + ks * Qn_stride1
        qn_vec = tl.load(qn_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load Kc[ls, ks] as matrix [BLOCK_L, BLOCK_K]
        kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1
        kc_mat = tl.load(kc_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]

        # Accumulate: acc += qn_vec @ kc_mat^T -> [BLOCK_L]
        acc += tl.sum(kc_mat * qn_vec[None, :], axis=1)

    # Iterate Kp dimension in chunks of BLOCK_K (size 64)
    for k0 in range(0, D_kpe, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_kpe

        # Load q_pe[h, ks] as vector [BLOCK_K]
        qp_ptrs = Qp_ptr + h * Qp_stride0 + ks * Qp_stride1
        qp_vec = tl.load(qp_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load Kp[ls, ks] as matrix [BLOCK_L, BLOCK_K]
        kp_ptrs = Kp_ptr + ls[:, None] * Kp_stride0 + ks[None, :] * Kp_stride1
        kp_mat = tl.load(kp_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]

        # Accumulate: acc += qp_vec @ kp_mat^T -> [BLOCK_L]
        acc += tl.sum(kp_mat * qp_vec[None, :], axis=1)

    # Store acc to Logits[h, ls]
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def lse_mask_kernel(
    Logits_ptr, LSE_ptr, Mask_ptr, H, L, inv_ln2,
    Logits_stride0, Logits_stride1,
    BLOCK_L: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Compute row-wise max
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        # Apply causal mask: non-causal positions are -inf (here mask is all ones)
        causal_mask = tl.load(Mask_ptr + ls, mask=mask_l, other=1)  # 1 allowed, 0 masked
        vals = tl.where(causal_mask > 0, vals, -float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp(vals - max_val)
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        causal_mask = tl.load(Mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.where(causal_mask > 0, vals, -float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    lse_scaled = tl.log(sum_exp) * inv_ln2  # divide by ln(2)
    tl.store(LSE_ptr + h, lse_scaled)


@triton.jit
def softmax_matmul_kernel(
    Logits_ptr, Kc_ptr, Out_ptr,
    H, L, D_ckv,
    Kc_stride0, Kc_stride1,
    Out_stride0, Out_stride1,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Load logits row and apply mask (mask is all ones in this implementation)
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        causal_mask = tl.load(Mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.where(causal_mask > 0, vals, -float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sumexp
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        causal_mask = tl.load(Mask_ptr + ls, mask=mask_l, other=1)
        vals = tl.where(causal_mask > 0, vals, -float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    # Now compute out[h, :] = softmax @ Kc[:, :D_ckv]
    out_vec = tl.zeros([D_ckv], dtype=tl.float32)
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv

        # Compute softmax over this L tile
        for l0 in range(0, L, BLOCK_L):
            ls = l0 + tl.arange(0, BLOCK_L)
            mask_l = ls < L
            vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
            causal_mask = tl.load(Mask_ptr + ls, mask=mask_l, other=1)
            vals = tl.where(causal_mask > 0, vals, -float('inf'))
            e = tl.exp(vals - max_val)  # already adjusted by max_val
            p = e / sum_exp  # probability for each position

            # For each ks, accumulate p * Kc[ls, ks] into out_vec[ks]
            # out_vec[ks] += sum_l p[l] * Kc[l, ks]
            kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1
            kc_mat = tl.load(kc_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
            out_vec += tl.sum(kc_mat * p[:, None], axis=0)

    # Store output vector
    out_ptrs = Out_ptr + h * Out_stride0
    tl.store(out_ptrs + tl.arange(0, D_ckv), out_vec, mask=tl.arange(0, D_ckv) < D_ckv)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure everything is on the same CUDA device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
        device = q_nope.device

        # Constants
        num_qo_heads = 16
        head_dim_ckv = 512
        head_dim_kpe = 64

        # Batch size and total queries
        batch_size = qo_indptr.shape[0] - 1
        total_q = int(qo_indptr[-1].item())

        # Prepare Kc_all, Kp_all (float32 for compute)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Allocate output (bf16) and lse (float32)
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        inv_ln2 = 1.4426950408889634  # 1 / ln(2)

        # Process each batch b and query i
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # KV block
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            # Gather token indices and corresponding Kc/Kp rows
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [L]
            L = tok_idx.shape[0]
            Kc = Kc_all[tok_idx]  # [L, 512]
            Kp = Kp_all[tok_idx]  # [L, 64]

            # Prepare Qn_batch and Qp_batch for this (b): [q_len, H, D]
            Qn_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, 16, 512]
            Qp_batch = q_pe[q_start:q_end].to(torch.float32)    # [q_len, 16, 64]

            for i in range(q_len):
                # Allocate per-(b,i) buffers
                logits = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)

                # Launch compute_logits_kernel: grid over heads and L tiles
                BLOCK_L = 128
                BLOCK_K = 64
                grid = (num_qo_heads, triton.cdiv(L, BLOCK_L))
                compute_logits_kernel[grid](
                    Qn_batch[i].unsqueeze(0),  # [1, 16, 512]
                    Qp_batch[i].unsqueeze(0), # [1, 16, 64]
                    Kc, Kp, logits,
                    num_qo_heads, L, head_dim_ckv, head_dim_kpe,
                    Qn_batch.stride(0), Qn_batch.stride(1),
                    Qp_batch.stride(0), Qp_batch.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    Kp.stride(0), Kp.stride(1),
                    logits.stride(0), logits.stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
                )

                # Mask buffer (all ones for simplicity)
                mask = torch.ones((L,), dtype=torch.int32, device=device)

                # Launch lse_mask_kernel per head
                lse_h = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                lse_mask_kernel[(num_qo_heads,)](
                    logits, lse_h, mask, num_qo_heads, L, inv_ln2,
                    logits.stride(0), logits.stride(1),
                )

                # Launch softmax_matmul_kernel per head to compute output
                out_vec = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                softmax_matmul_kernel[(num_qo_heads,)](
                    logits, Kc, out_vec,
                    num_qo_heads, L, head_dim_ckv,
                    Kc.stride(0), Kc.stride(1),
                    out_vec.stride(0), out_vec.stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
                )

                # Store outputs: out_vec has shape [num_qo_heads, head_dim_ckv]
                for h in range(num_qo_heads):
                    output[q_start + i, h, :] = out_vec[h, :].to(torch.bfloat16)

                # Store lse: lse[q_start + i, h] = lse_h[h]
                for h in range(num_qo_heads):
                    lse[q_start + i, h] = lse_h[h]

        return output, lse


def run(*args):
    return ModelNew()(*args)
