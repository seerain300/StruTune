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

    # q vectors for head h: shape as [1, 1, D] via strides (stride0=H, stride1=D)
    qn_base = tl.load(Qn_ptr + h * Qn_stride0, mask=True, other=0.0)  # [D_ckv]
    qp_base = tl.load(Qp_ptr + h * Qp_stride0, mask=True, other=0.0)  # [D_kpe]

    # Iterate over Kc dimension in chunks
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv
        Kc_vals = tl.load(Kc_ptr + ks * Kc_stride1, mask=mask_k, other=0.0)  # [BLOCK_K]
        # Contribution from q_nope @ Kc: sum_k qn_base[k] * Kc_vals[k]
        acc += tl.sum(qn_base[None, :] * Kc_vals[None, :], axis=1)  # [BLOCK_L]

    # Iterate over Kp dimension in chunks
    for k0 in range(0, D_kpe, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_kpe
        Kp_vals = tl.load(Kp_ptr + ks * Kp_stride1, mask=mask_k, other=0.0)  # [BLOCK_K]
        # Contribution from q_pe @ Kp: sum_k' qp_base[k'] * Kp_vals[k']
        acc += tl.sum(qp_base[None, :] * Kp_vals[None, :], axis=1)  # [BLOCK_L]

    # Store acc to Logits[h, ls]
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def masked_lse_kernel(
    Logits_ptr, Mask_ptr, LSE_ptr,
    H, L, inv_ln2,
    Logits_stride0, Logits_stride1,
    Mask_stride0,
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
        # Apply causal mask: non-causal positions are -inf
        mask_vec = tl.load(Mask_ptr + ls * Mask_stride0, mask=mask_l, other=0)
        masked = tl.where(mask_vec != 0, vals, -float('inf'))
        block_max = tl.max(masked, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp of masked vals
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        mask_vec = tl.load(Mask_ptr + ls * Mask_stride0, mask=mask_l, other=0)
        masked = tl.where(mask_vec != 0, vals, -float('inf'))
        e = tl.exp(masked - max_val)
        sum_exp += tl.sum(e, axis=0)

    lse_scaled = tl.log(sum_exp) * inv_ln2  # divide by ln(2)
    tl.store(LSE_ptr + h, lse_scaled)


@triton.jit
def masked_softmax_matmul_kernel(
    Logits_ptr, Kc_ptr, Output_ptr,
    H, L, D_ckv,
    Kc_stride0, Kc_stride1,
    Output_stride0, Output_stride1,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per (i, h) computing output vector
    h = tl.program_id(0)
    i = tl.program_id(1)

    # Compute per-head max with mask (same as lse)
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        mask_vec = tl.load(Mask_ptr + ls * Mask_stride0, mask=mask_l, other=0)
        masked = tl.where(mask_vec != 0, vals, -float('inf'))
        block_max = tl.max(masked, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp of masked vals
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        mask_vec = tl.load(Mask_ptr + ls * Mask_stride0, mask=mask_l, other=0)
        masked = tl.where(mask_vec != 0, vals, -float('inf'))
        e = tl.exp(masked - max_val)
        sum_exp += tl.sum(e, axis=0)

    # Now compute out[h, :] = softmax @ Kc
    out_vec = tl.zeros([D_ckv], dtype=tl.float32)
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        mask_vec = tl.load(Mask_ptr + ls * Mask_stride0, mask=mask_l, other=0)
        masked = tl.where(mask_vec != 0, vals, -float('inf'))
        # Softmax values
        softmax = tl.exp(masked - max_val) / sum_exp  # [BLOCK_L]
        # Accumulate output: out += softmax[l] * Kc[l, :]
        for kk in range(0, D_ckv, BLOCK_K):
            ks = kk + tl.arange(0, BLOCK_K)
            mask_k = ks < D_ckv
            Kc_vals = tl.load(Kc_ptr + ks * Kc_stride1, mask=mask_k, other=0.0)  # [BLOCK_K]
            out_vec += tl.sum(softmax[:, None] * Kc_vals[None, :], axis=1)

    # Store output vector to Output[i, h, :]
    out_ptrs = Output_ptr + i * Output_stride0 + h * Output_stride1
    tl.store(out_ptrs, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shapes and constants
        head_dim_ckv = 512
        head_dim_kpe = 64
        device = q_nope.device
        # total_q is the last element of qo_indptr
        total_q = int(torch.sum(qo_indptr[1:] - qo_indptr[:-1]).item())  # infer total_q via lengths if needed; but qo_indptr[-1] is total_q
        # Directly read total_q from the last element
        total_q = int(qo_indptr[-1].item())
        num_qo_heads = 16
        # batch_size = len(kv_indptr) - 1
        batch_size = int((kv_indptr.shape[0] - 1).item())

        # Prepare Kc_all and Kp_all (since ckv_cache, kpe_cache have shape [num_pages, 1, D], squeeze dim=1)
        Kc_all = ckv_cache.squeeze(1)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1)  # [num_pages, 64]

        # Output tensors: float32 for compute
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # q indices for this batch
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            # KV indices for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len = int((kv_indptr[b + 1] - kv_indptr[b]).item())
            if page_beg >= page_end or kv_len == 0:
                continue

            tok_idx = kv_indices[page_beg:page_end]  # [kv_len]
            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            q_len = q_end - q_start

            for i in range(q_len):
                # q vectors for this query position (all heads)
                qn = q_nope[q_start + i].reshape(1, num_qo_heads, head_dim_ckv)  # [1, H, D_ckv]
                qp = q_pe[q_start + i].reshape(1, num_qo_heads, head_dim_kpe)  # [1, H, D_kpe]
                # Output vector for this query head
                Output_ptr = output[q_start + i]  # [H, D_ckv]

                # Allocate Logits [H, L] and mask vector [L]
                Logits = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                Logits_stride0 = num_qo_heads
                Logits_stride1 = 1  # contiguous along L

                # Causal mask: positions l > (kv_len - q_len + i) are non-causal
                abs_pos = kv_len - q_len + i
                l_range = torch.arange(kv_len, device=device)
                causal = l_range <= abs_pos
                mask_vec = causal.to(torch.int32)  # int32 mask for Triton
                Mask_stride0 = 1

                # Launch compute_logits_kernel
                BLOCK_L = 128
                grid_logits = (num_qo_heads, triton.cdiv(kv_len, BLOCK_L))
                compute_logits_kernel[grid_logits](
                    qn, qp, Kc, Kp, Logits,
                    num_qo_heads, kv_len, head_dim_ckv, head_dim_kpe,
                    qn.stride(0), qn.stride(1),  # strides for [1, H, D]
                    qp.stride(0), qp.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    Kp.stride(0), Kp.stride(1),
                    Logits_stride0, Logits_stride1,
                    BLOCK_L=BLOCK_L, BLOCK_K=64,
                )

                # Launch masked_lse_kernel
                lse_i = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                lse_kernel_grid = (num_qo_heads,)
                masked_lse_kernel[lse_kernel_grid](
                    Logits, mask_vec, lse_i,
                    num_qo_heads, kv_len, 1.0 / math.log(2.0),  # inv_ln2
                    Logits_stride0, Logits_stride1,
                    mask_vec.stride(0),
                    BLOCK_L=BLOCK_L,
                )
                lse[q_start + i, :] = lse_i  # store per query

                # Launch masked_softmax_matmul_kernel to compute output[i, :, :]
                Output_stride0 = num_qo_heads
                Output_stride1 = head_dim_ckv
                grid_softmax = (num_qo_heads,)
                masked_softmax_matmul_kernel[grid_softmax](
                    Logits, Kc, Output_ptr,
                    num_qo_heads, kv_len, head_dim_ckv,
                    Kc.stride(0), Kc.stride(1),
                    Output_stride0, Output_stride1,
                    BLOCK_L=BLOCK_L, BLOCK_K=64,
                )

        # Return outputs: cast output to bfloat16, lse stays float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
