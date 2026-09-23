import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_add_kernel(
    qn_ptr,           # *float32, [D]
    qp_ptr,           # *float32, [Dp]
    Kc_ptr,           # *float32, [L, D], row-major (L, D)
    Kp_ptr,           # *float32, [L, Dp], row-major (L, Dp)
    v_ptr,            # *float32, [L]
    L: tl.int32,      # number of tokens
    D: tl.int32,      # head_dim_ckv
    Dp: tl.int32,     # head_dim_kpe
    BLOCK_K: tl.constexpr
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
        kc_ptr = Kc_ptr + i * D + k_off
        kc_vals = tl.load(kc_ptr, mask=mask_k, other=0.0)
        sum1 += tl.sum(qn_slice * kc_vals, axis=0)

    # Reduce over Kp dimension (Dp)
    for k in range(0, Dp, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < Dp
        qp_slice = tl.load(qp_ptr + k_off, mask=mask_k, other=0.0)
        kp_ptr = Kp_ptr + i * Dp + k_off
        kp_vals = tl.load(kp_ptr, mask=mask_k, other=0.0)
        sum2 += tl.sum(qp_slice * kp_vals, axis=0)

    v = sum1 + sum2
    tl.store(v_ptr + i, v)


@triton.jit
def lse_kernel(
    v_ptr,            # *float32, [L]
    lse_out_ptr,      # *float32, scalar output
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK: tl.constexpr
):
    # Pass 1: compute max of v
    m = -float("inf")
    for i in range(0, L, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < L
        v_chunk = tl.load(v_ptr + idx, mask=mask, other=-float("inf"))
        m = tl.maximum(m, tl.max(v_chunk, axis=0))
    # Pass 2: compute sum exp(v - m)
    sum_exp = 0.0
    for i in range(0, L, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < L
        v_chunk = tl.load(v_ptr + idx, mask=mask, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(v_chunk - m), axis=0)
    lse_val = m + tl.log(sum_exp) * inv_ln2
    tl.store(lse_out_ptr, lse_val)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, scalar
    attn_ptr,         # *float32, [L]
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK: tl.constexpr
):
    # Compute attn[i] = exp(v[i] / ln(2) - lse)
    for i in range(0, L, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < L
        v_chunk = tl.load(v_ptr + idx, mask=mask, other=0.0)
        lse_val = tl.load(lse_ptr)  # scalar
        attn_chunk = tl.exp((v_chunk / tl.log(2.0)) - lse_val)
        tl.store(attn_ptr + idx, attn_chunk, mask=mask)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D]
    y_ptr,            # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK: tl.constexpr
):
    # One program per output dimension h in [0, D)
    h = tl.program_id(0)
    acc = 0.0
    for i in range(0, L, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < L
        attn_chunk = tl.load(attn_ptr + idx, mask=mask, other=0.0)
        kc_ptr = Kc_ptr + idx * D + h  # since Kc is [L, D] row-major, kc[i, h] = Kc_ptr + i*D + h
        # If i goes beyond L, mask attn_chunk; kc_ptr is valid since idx<L
        kc_vals = tl.load(kc_ptr, mask=mask, other=0.0)
        acc += tl.sum(attn_chunk * kc_vals, axis=0)
    tl.store(y_ptr + h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Cast inputs to float32 for computation
        q_nope = q_nope.to(torch.float32).contiguous()
        q_pe = q_pe.to(torch.float32).contiguous()
        ckv_cache = ckv_cache.to(torch.float32).contiguous()  # [num_pages, 1, D]
        kpe_cache = kpe_cache.to(torch.float32).contiguous()  # [num_pages, 1, Dp]

        # Output tensors
        output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv),
                             dtype=torch.bfloat16, device=device)
        lse = torch.full((batch_size, num_qo_heads), -float("inf"),
                         dtype=torch.float32, device=device)

        # Precompute constants
        inv_ln2 = 1.0 / math.log(2.0)

        # Process each batch element
        for b in range(batch_size):
            # Determine token range from kv_indptr
            L_tokens = int((kv_indptr[b + 1] - kv_indptr[b]).item())
            if L_tokens == 0:
                # No tokens for this batch, output zeros and lse = -inf
                for j in range(num_qo_heads):
                    lse[b, j] = -float("inf")
                continue

            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]]
            # Gather Kc and Kp for these tokens
            Kc = ckv_cache[tok_idx]  # [L_tokens, D]
            Kp = kpe_cache[tok_idx]  # [L_tokens, Dp]

            # Loop over heads j
            for j in range(num_qo_heads):
                # Extract qn, qp for this head
                qn = q_nope[b, j, :]  # [D]
                qp = q_pe[b, j, :]    # [Dp]
                qn = qn.contiguous()
                qp = qp.contiguous()
                Kc = Kc.contiguous()
                Kp = Kp.contiguous()

                # Kernel 1: compute v[j, :] for L_tokens
                v = torch.empty(L_tokens, dtype=torch.float32, device=device)
                matvec_add_kernel[(L_tokens,)](
                    qn, qp, Kc, Kp, v,
                    L=L_tokens, D=head_dim_ckv, Dp=head_dim_kpe,
                    BLOCK_K=64, num_warps=4
                )

                # Kernel 2: compute lse = logsumexp_base2(v)
                lse_b_j = torch.empty(1, dtype=torch.float32, device=device)
                lse_kernel[(1,)](
                    v, lse_b_j, L=L_tokens, inv_ln2=inv_ln2,
                    num_warps=1
                )
                lse_val = lse_b_j[0]

                # Kernel 3: compute attn[i] = exp(v[i] / ln(2) - lse)
                attn = torch.empty(L_tokens, dtype=torch.float32, device=device)
                softmax_base2_kernel[(L_tokens,)](
                    v, lse_b_j, attn, L=L_tokens, inv_ln2=inv_ln2,
                    num_warps=4
                )

                # Kernel 4: out[:, j, :] = attn @ Kc
                y = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
                matvec_write_y_kernel[(head_dim_ckv,)](
                    attn, Kc, y,
                    L=L_tokens, D=head_dim_ckv, BLOCK=64, num_warps=4
                )
                output[b, j, :] = y.to(torch.bfloat16)

                # Update lse
                lse[b, j] = lse_val

        return output, lse