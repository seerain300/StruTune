import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


# Triton kernels: must be invoked from ModelNew.forward.

@triton.jit
def compute_logits_and_lse_kernel(
    qn_row_ptr, qp_row_ptr, Kc_ptr, Kp_ptr,
    logits_ptr, lse_ptr,
    sm_scale: tl.float32,
    prefix_len: tl.int32,
    query_abs_pos: tl.int32,
    KV: tl.constexpr,         # number of KV tokens
    Dn: tl.constexpr,         # head_dim_ckv = 512
    Dp: tl.constexpr,         # head_dim_kpe = 64
    BLOCK_K: tl.constexpr,    # tile size along KV
):
    # Accumulate S = qn_row @ Kc.T and T = qp_row @ Kp.T
    acc_S = tl.zeros((Dn,), dtype=tl.float32)
    for k0 in range(0, KV, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < KV
        # Load Kc tile [BLOCK_K, Dn]
        Kc_tile = tl.load(Kc_ptr + k_idx[:, None] * Dn + tl.arange(0, Dn)[None, :], mask=mask_k[:, None], other=0.0)
        # Load qn_row [Dn]
        qn_row = tl.load(qn_row_ptr + tl.arange(0, Dn))
        # acc_S += sum over k of Kc_tile[j, :] * qn_row
        acc_S += tl.sum(Kc_tile * qn_row[None, :], axis=1)

    acc_T = tl.zeros((KV,), dtype=tl.float32)
    for k0 in range(0, KV, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < KV
        # Load Kp tile [BLOCK_K, Dp]
        Kp_tile = tl.load(Kp_ptr + k_idx[:, None] * Dp + tl.arange(0, Dp)[None, :], mask=mask_k[:, None], other=0.0)
        # Load qp_row [Dp]
        qp_row = tl.load(qp_row_ptr + tl.arange(0, Dp))
        # acc_T += sum over k of Kp_tile[:, j] * qp_row[j]
        acc_T += tl.sum(Kp_tile * qp_row[None, :], axis=1)

    # logits = (acc_S + acc_T) * sm_scale
    logits = acc_S + acc_T
    logits *= sm_scale

    # Apply causal mask: keep j if j > (prefix_len + i), else set to -inf
    m = -float("inf")
    for j in range(0, KV):
        val = tl.load(logits_ptr + j)
        if val > m:
            m = val

    sum_exp = 0.0
    for j in range(0, KV):
        val = tl.load(logits_ptr + j)
        sum_exp += tl.exp(val - m)

    inv_ln2 = 1.4426950408889634  # 1 / ln(2)
    lse_val = (m + tl.log(sum_exp)) * inv_ln2
    tl.store(lse_ptr, lse_val)

@triton.jit
def softmax_row_kernel(logits_ptr, attn_ptr, KV: tl.constexpr):
    # Softmax over a single row of length KV using a stable approach.
    m = -float("inf")
    for j in range(0, KV):
        val = tl.load(logits_ptr + j)
        if val > m:
            m = val

    sum_exp = 0.0
    for j in range(0, KV):
        val = tl.load(logits_ptr + j)
        sum_exp += tl.exp(val - m)

    inv_ln2 = 1.4426950408889634  # 1 / ln(2)
    for j in range(0, KV):
        val = tl.load(logits_ptr + j)
        attn_val = tl.exp(val - m) / sum_exp
        tl.store(attn_ptr + j, attn_val)

@triton.jit
def compute_out_row_kernel(attn_ptr, Kc_ptr, out_ptr, KV: tl.constexpr, Dn: tl.constexpr):
    # Compute out = attn @ Kc (GEMV). attn: [KV], Kc: [KV, Dn], out: [Dn].
    acc = tl.zeros((Dn,), dtype=tl.float32)
    for j in range(0, KV):
        a_j = tl.load(attn_ptr + j)  # scalar
        k_vec = tl.load(Kc_ptr + j * Dn + tl.arange(0, Dn))  # [Dn]
        acc += a_j * k_vec
    tl.store(out_ptr + tl.arange(0, Dn), acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # All computation done in Triton; forward only allocates and launches kernels.
        assert triton is not None and tl is not None, "Triton is required."

        total_q = int(qo_indptr[-1].item())
        num_qo_heads = 16
        head_dim_ckv = 512  # Dn
        head_dim_kpe = 64   # Dp

        # Output buffers (float32 for compute, cast later)
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q_nope.device)

        B = qo_indptr.numel() - 1
        for b in range(B):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            # Gather token indices and cache rows
            tok_idx = kv_indices[kv_start:kv_end].to(torch.int32)
            Kc_row = ckv_cache[tok_idx].contiguous().to(torch.float32)  # [kv_len, 512]
            Kp_row = kpe_cache[tok_idx].contiguous().to(torch.float32)  # [kv_len, 64]

            for i in range(q_len):
                query_abs_pos = (kv_len - q_len) + i

                # Load qn_row and qp_row for each head h
                for h in range(num_qo_heads):
                    qn_vec = q_nope[q_start + i, h, :].contiguous().to(torch.float32)  # [512]
                    qp_vec = q_pe[q_start + i, h, :].contiguous().to(torch.float32)    # [64]

                    # Prepare pointers
                    logits = torch.empty((kv_len,), dtype=torch.float32, device=q_nope.device)
                    lse_val = torch.empty((), dtype=torch.float32, device=q_nope.device)

                    # Launch compute_logits_and_lse_kernel for this head
                    compute_logits_and_lse_kernel[(1,)](
                        qn_vec, qp_vec, Kc_row, Kp_row,
                        logits, lse_val,
                        sm_scale,
                        kv_len - q_len, query_abs_pos,
                        KV=kv_len, Dn=head_dim_ckv, Dp=head_dim_kpe, BLOCK_K=128
                    )
                    lse[q_start + i, h] = lse_val

                    # Launch softmax_row_kernel
                    attn = torch.empty((kv_len,), dtype=torch.float32, device=q_nope.device)
                    softmax_row_kernel[(1,)](logits, attn, KV=kv_len)

                    # Compute out[h, :] = attn @ Kc_row => [512]
                    out_row = torch.empty((head_dim_ckv,), dtype=torch.float32, device=q_nope.device)
                    compute_out_row_kernel[(1,)](attn, Kc_row, out_row, KV=kv_len, Dn=head_dim_ckv)
                    output[q_start + i, h, :] = out_row

        # Return outputs: output in bfloat16, lse in float32
        output_bf = output.to(torch.bfloat16)
        return output_bf, lse


def run(*args):
    return ModelNew()(*args)
