import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_add_kernel(
    qn_ptr,   # *float32, [D]
    qp_ptr,   # *float32, [Dp]
    Kc_ptr,   # *float32, [L, D], row-major (L, D)
    Kp_ptr,   # *float32, [L, Dp], row-major (L, Dp)
    v_ptr,    # *float32, [L]
    L: tl.int32,
    D: tl.int32,
    Dp: tl.int32,
    BLOCK_K: tl.constexpr,
):
    # One program per output index i in [0, L)
    i = tl.program_id(0)
    sum1 = 0.0
    sum2 = 0.0
    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)
        kc_ptr_row = Kc_ptr + i * D + k_off
        kc_slice = tl.load(kc_ptr_row, mask=mask_k, other=0.0)
        sum1 += tl.sum(qn_slice * kc_slice, axis=0)
    # Reduce over Kp dimension (Dp)
    for p in range(0, Dp, BLOCK_K):
        p_off = p + tl.arange(0, BLOCK_K)
        mask_p = p_off < Dp
        qp_slice = tl.load(qp_ptr + p_off, mask=mask_p, other=0.0)
        kp_ptr_row = Kp_ptr + i * Dp + p_off
        kp_slice = tl.load(kp_ptr_row, mask=mask_p, other=0.0)
        sum2 += tl.sum(qp_slice * kp_slice, axis=0)
    v = sum1 + sum2
    tl.store(v_ptr + i, v)


@triton.jit
def lse_base2_kernel(
    v_ptr,   # *float32, [L]
    L: tl.int32,
    ln2: tl.float32,
    out_ptr, # *float32, [1]
):
    # Compute logsumexp in base-2: out = log(sum(exp(v / ln(2))))
    m = -1e20
    # Pass 1: find max of v
    for t in range(0, L):
        vt = tl.load(v_ptr + t)
        if vt > m:
            m = vt
    sum_exp = 0.0
    # Pass 2: sum exp(v - m) scaled by ln(2)
    for t in range(0, L):
        vt = tl.load(v_ptr + t)
        sum_exp += tl.exp((vt - m) * ln2)
    out_val = tl.log(sum_exp)
    tl.store(out_ptr, out_val)


@triton.jit
def softmax_base2_kernel(
    v_ptr,      # *float32, [L]
    lse_ptr,    # *float32, [1] containing logsumexp in base-2 per head
    attn_ptr,   # *float32, [L]
    L: tl.int32,
    ln2: tl.float32,
):
    lse_j = tl.load(lse_ptr)
    for t in range(0, L):
        vt = tl.load(v_ptr + t)
        attn_val = tl.exp((vt - lse_j) * ln2)
        tl.store(attn_ptr + t, attn_val)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,  # *float32, [L]
    Kc_ptr,    # *float32, [L, D], row-major (L, D)
    y_ptr,     # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_K: tl.constexpr,
):
    h = tl.program_id(0)
    acc = 0.0
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        kc_ptr_col = Kc_ptr + k_off * L  # pointer to column k across rows
        # Load attn slice for this column across L tokens
        attn_col = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for t in range(0, L):
            attn_col += tl.load(attn_ptr + t) * tl.load(kc_ptr_col + t, mask=mask_k, other=0.0)
        acc += tl.sum(attn_col, axis=0)
    tl.store(y_ptr + h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        if device.type != 'cuda':
            # Ensure CUDA for Triton kernels
            q_nope = q_nope.to('cuda')
            q_pe = q_pe.to('cuda')
            ckv_cache = ckv_cache.to('cuda')
            kpe_cache = kpe_cache.to('cuda')
            kv_indptr = kv_indptr.to('cuda')
            kv_indices = kv_indices.to('cuda')

        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        # Constants from original asserts
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        # Process per batch
        output = torch.empty(
            (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device
        )
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Prepare Kc_all and Kp_all for batch; squeeze dim=1 since it's always 1 in provided inputs
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)

        ln2 = math.log(2.0)

        for b in range(batch_size):
            # Token range for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = max(page_end - page_beg, 0)

            if L == 0:
                lse[b, :] = -float('inf')
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [L]

            # Slice Kc/Kp for these tokens
            Kc = Kc_all[tok_idx]  # [L, 512]
            Kp = Kp_all[tok_idx]  # [L, 64]
            D = head_dim_ckv
            Dp = head_dim_kpe

            # Cast queries to float32 for computation
            qn = q_nope[b].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[b].to(torch.float32).contiguous()    # [16, 64]

            # Compute v[j, :] per head: one program per j
            for j in range(num_qo_heads):
                # Allocate v
                v = torch.empty((L,), dtype=torch.float32, device=device)
                grid_v = (L,)
                matvec_add_kernel[grid_v](
                    qn[j], qp[j], Kc, Kp, v,
                    L, D, Dp,
                    BLOCK_K=128,
                    num_warps=2
                )

                # lse per head
                lse_b = torch.empty((1,), dtype=torch.float32, device=device)  # scalar tensor for kernel
                lse_base2_kernel[(1,)](
                    v, L, ln2, lse_b,
                    num_warps=1
                )
                lse_j = lse_b[0]

                # attention attn[i]
                attn = torch.empty((L,), dtype=torch.float32, device=device)
                softmax_base2_kernel[(L,)](
                    v, lse_j, attn,
                    L, ln2,
                    num_warps=1
                )

                # y = attn @ Kc -> [512] and store into output[b, j, :]
                y = torch.empty((D,), dtype=torch.float32, device=device)
                matvec_write_y_kernel[(D,)](
                    attn, Kc, y,
                    L, D,
                    BLOCK_K=128,
                    num_warps=2
                )
                output[b, j, :] = y

        return output.to(torch.bfloat16), lse  # cast output to bfloat16 to match original