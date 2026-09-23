import math
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

    # Load q_nope and q_pe for this head h (shape of Qn_ptr: [1, H, D], strides allow broadcasting over L)
    qn_base = Qn_ptr + h * Qn_stride1  # skip the leading "1" dimension in pointer arithmetic
    qn_vec = tl.load(qn_base + tl.arange(0, D_ckv), mask=tl.arange(0, D_ckv) < D_ckv, other=0.0)
    qp_vec = tl.load(Qp_ptr + h * Qp_stride1 + tl.arange(0, D_kpe), mask=tl.arange(0, D_kpe) < D_kpe, other=0.0)

    # Kc contributions: iterate over K in chunks
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv
        Kc_block = tl.load(Kc_ptr + ks * Kc_stride1, mask=mask_k, other=0.0)  # [BLOCK_K], each element is a row in Kc
        # Broadcast Kc_block over L tile and qn_vec over ks
        # Compute acc += sum over ks of qn_vec[ks] * Kc_block[ks]
        # acc += dot(qn_vec, Kc_block)
        acc += tl.sum(qn_vec[ks] * Kc_block, axis=0)

    # Kp contributions: similar
    for k0 in range(0, D_kpe, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_kpe
        Kp_block = tl.load(Kp_ptr + ks * Kp_stride1, mask=mask_k, other=0.0)  # [BLOCK_K]
        acc += tl.sum(qp_vec[ks] * Kp_block, axis=0)

    # Store acc to Logits[h, ls]
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def lse_mask_kernel(
    Logits_ptr, LSE_ptr, Mask_ptr, H, L, inv_ln2,
    Logits_stride0, Logits_stride1,
    Mask_stride0,
):
    # One program per head
    h = tl.program_id(0)

    max_val = -float('inf')
    for l0 in range(0, L, 128):
        ls = l0 + tl.arange(0, 128)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        mask = tl.load(Mask_ptr + ls * Mask_stride0, mask=mask_l, other=1).to(tl.int1)
        # Apply mask: non-causal -> -inf
        vals = tl.where(mask == 1, vals, -float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = 0.0
    for l0 in range(0, L, 128):
        ls = l0 + tl.arange(0, 128)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        mask = tl.load(Mask_ptr + ls * Mask_stride0, mask=mask_l, other=1).to(tl.int1)
        vals = tl.where(mask == 1, vals, -float('inf'))
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

    # Apply mask and compute softmax over L
    # First compute max
    max_val = -float('inf')
    for l0 in range(0, L, 128):
        ls = l0 + tl.arange(0, 128)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        # softmax does not need mask here; we will recompute logits_scaled in a second pass
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum_exp
    sum_exp = 0.0
    for l0 in range(0, L, 128):
        ls = l0 + tl.arange(0, 128)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    # Second pass: compute softmax and accumulate Out[h, :]
    out_vec = tl.zeros([D_ckv], dtype=tl.float32)
    for l0 in range(0, L, 128):
        ls = l0 + tl.arange(0, 128)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        p = tl.exp(vals - max_val) / sum_exp  # softmax at each position
        # For each k in D_ckv, accumulate p * Kc[k, :]
        for k0 in range(0, D_ckv, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            mask_k = ks < D_ckv
            Kc_block = tl.load(Kc_ptr + ks * Kc_stride1, mask=mask_k, other=0.0)  # [BLOCK_K]
            out_vec += tl.sum(p[None, :] * Kc_block[None, :], axis=1)

    # Store Out[h, :]
    out_ptrs = Out_ptr + h * Out_stride0
    tl.store(out_ptrs, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure device is CUDA
        assert q_nope.is_cuda, "Inputs must be on CUDA for Triton kernels."
        device = q_nope.device

        # Input shapes
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_qo_heads = q_nope.shape[1]
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Squeeze caches
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Compute batch size and launch
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        inv_ln2 = 1.4426950408889634  # 1 / ln(2)

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # KV block pointers
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())

            if page_beg >= page_end:
                continue

            # Gather K indices
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [L]
            L = tok_idx.shape[0]
            Kc = Kc_all[tok_idx]  # [L, 512]
            Kp = Kpe_all[tok_idx]  # [L, 64]

            # Prepare q batches for this (b)
            Qn_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, 16, 512]
            Qp_batch = q_pe[q_start:q_end].to(torch.float32)    # [q_len, 16, 64]

            # We run per i in q_len, per head h:
            for i in range(q_len):
                # Compute Logits buffer [H, L]
                logits = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)

                # Launch compute_logits_kernel: grid over H and L tiles
                H = num_qo_heads
                BLOCK_L = 128
                BLOCK_K = 64
                grid = (H, triton.cdiv(L, BLOCK_L))
                compute_logits_kernel[grid](
                    Qn_batch[i].unsqueeze(0),  # shape [1, H, D] with strides
                    Qp_batch[i].unsqueeze(0),  # shape [1, H, D]
                    Kc, Kp, logits,
                    H, L, head_dim_ckv, head_dim_kpe,
                    Qn_batch.stride(0), Qn_batch.stride(1),
                    Qp_batch.stride(0), Qp_batch.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    Kp.stride(0), Kp.stride(1),
                    logits.stride(0), logits.stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
                )

                # Build causal mask: ls > (L - q_len + i)
                # For head h, positions with (ls <= prefix_len + i) are causal; others are non-causal.
                prefix_len = L - q_len
                query_abs_pos = prefix_len + i
                mask = torch.ones((L,), dtype=torch.int32, device=device)
                mask[(query_abs_pos + 1):] = 0  # set non-causal to 0

                # Compute lse[h] for all heads
                grid_lse = (H,)
                lse_mask_kernel[grid_lse](
                    logits, lse[q_start + i], mask,
                    H, L, inv_ln2,
                    logits.stride(0), logits.stride(1),
                    mask.stride(0),
                )

                # Compute output[h, :] = softmax(logits_scaled) @ Kc[:, :]
                out_vec = torch.empty((H, head_dim_ckv), dtype=torch.float32, device=device)
                softmax_matmul_kernel[(H,)](
                    logits, Kc, out_vec,
                    H, L, head_dim_ckv,
                    Kc.stride(0), Kc.stride(1),
                    out_vec.stride(0), out_vec.stride(1),
                    BLOCK_L=128, BLOCK_K=64,
                )

                # Write outputs
                output[q_start + i] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
