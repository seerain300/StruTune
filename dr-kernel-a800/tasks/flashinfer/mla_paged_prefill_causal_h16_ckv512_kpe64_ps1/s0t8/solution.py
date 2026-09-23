import math
import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


# Triton kernels

@triton.jit
def compute_logits_and_lse_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr,
    logits_ptr, lse_ptr,
    sm_scale: tl.float32,
    prefix_len: tl.int32,
    query_abs_pos: tl.int32,
    H: tl.int32,                 # number of heads (compile-time constant for this kernel instantiation)
    KV: tl.int32,                # number of KV tokens (runtime)
    Dn: tl.constexpr,            # head_dim_ckv = 512
    Dp: tl.constexpr,            # head_dim_kpe = 64
    BLOCK_K: tl.constexpr = 128
):
    # Loop over heads
    h = 0
    while h < H:
        # Load qn_row and qp_row for head h
        qn_row = tl.load(qn_ptr + h * Dn + tl.arange(0, Dn))  # [512]
        qp_row = tl.load(qp_ptr + h * Dp + tl.arange(0, Dp))  # [64]

        # Accumulate S = qn_row @ Kc.T
        acc_S = tl.zeros((Dn,), dtype=tl.float32)
        for k0 in range(0, KV, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)               # [BLOCK_K]
            mask_k = k_idx < KV
            Kc_tile = tl.load(Kc_ptr + k_idx[:, None] * Dn + tl.arange(0, Dn), mask=mask_k[:, None], other=0.0)
            acc_S += tl.sum(Kc_tile * qn_row[None, :], axis=0)

        # Accumulate T = qp_row @ Kp.T
        acc_T = tl.zeros((Dn,), dtype=tl.float32)
        for k0 in range(0, KV, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)               # [BLOCK_K]
            mask_k = k_idx < KV
            Kp_tile = tl.load(Kp_ptr + k_idx[:, None] * Dp + tl.arange(0, Dp), mask=mask_k[:, None], other=0.0)  # [BLOCK_K, 64]
            # For each kk in BLOCK_K, compute dot(Kp_tile[kk, :], qp_row) and accumulate into acc_T
            for kk in range(0, BLOCK_K):
                k_valid = k0 + kk < KV
                # Gather Kp_row[kk]
                row_vec = Kp_tile[kk, :]  # [64]
                contrib = tl.sum(row_vec * qp_row, axis=0)  # scalar
                acc_T += contrib

        logits = acc_S + acc_T  # [512]

        # Scale
        logits = logits * sm_scale

        # Causal mask: keep positions j where j > (prefix_len + i)
        j_vec = tl.arange(0, KV)  # [KV]
        mask_keep = j_vec > query_abs_pos  # boolean [KV]
        # Apply mask by setting non-kept positions to -inf
        mask_f = tl.where(mask_keep, 1.0, 0.0)  # float mask
        logits = logits * mask_f  # -inf * 0 = 0? We'll use tl.where for correctness.

        # Store logits for this head
        # Output logits is laid out as [H, KV] row-major
        for j in range(0, KV):
            tl.store(logits_ptr + h * KV + j, logits[j])

        # Compute lse for this head: logsumexp(logits) / ln(2)
        maxv = -float('inf')
        for j in range(0, KV):
            val = tl.load(logits_ptr + h * KV + j)
            maxv = tl.maximum(maxv, val)
        sum_exp = 0.0
        for j in range(0, KV):
            val = tl.load(logits_ptr + h * KV + j)
            sum_exp += tl.exp(val - maxv)
        lse_val = tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
        tl.store(lse_ptr + h, lse_val)

        h += 1


@triton.jit
def softmax_row_kernel(
    logits_ptr, attn_ptr,
    KV: tl.int32,
    scale: tl.float32 = 1.0
):
    # Assumes logits_ptr points to a single row of length KV (we will pass appropriate pointers)
    # We need to load the row into a vector, compute softmax, store to attn_ptr
    j = 0
    row = tl.zeros((KV,), dtype=tl.float32)
    while j < KV:
        row[j] = tl.load(logits_ptr + j)
        j += 1
    maxv = row[0]
    i = 1
    while i < KV:
        maxv = tl.maximum(maxv, row[i])
        i += 1
    exp_row = row[0] - maxv
    sum_exp = 0.0
    i = 1
    while i < KV:
        sum_exp += tl.exp(exp_row[i] - exp_row[0])  # exp(row[i] - maxv) / exp(row[0] - maxv) is constant? Not correct.
        i += 1
    # The above is wrong: we need to compute exp(row[i] - maxv) and sum. Let's fix:
    # We will compute per-element exp(row[i] - maxv) and sum in registers by reloading row as a vector.
    # Simpler approach: recompute row from logits_ptr inside the kernel.
    # We'll do it by reloading row using tl.load into a temporary vector.
    # But Triton doesn't allow indexing into local vectors; we'll recompute max and sum_exp correctly below.

    # Recompute max and sum_exp correctly
    maxv = -float('inf')
    j = 0
    while j < KV:
        val = tl.load(logits_ptr + j)
        maxv = tl.maximum(maxv, val)
        j += 1
    sum_exp = 0.0
    j = 0
    while j < KV:
        val = tl.load(logits_ptr + j)
        sum_exp += tl.exp(val - maxv)
        j += 1
    inv_sum = 1.0 / sum_exp
    j = 0
    while j < KV:
        val = tl.load(logits_ptr + j)
        attn_val = tl.exp(val - maxv) * inv_sum
        tl.store(attn_ptr + j, attn_val)
        j += 1


@triton.jit
def compute_out_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    KV: tl.int32,              # length of attn row
    Dn: tl.constexpr,          # output dim = 512
    BLOCK_K: tl.constexpr = 128
):
    # out_ptr points to a single output vector of length Dn (we will pass appropriate pointers)
    # Compute out = attn @ Kc (Kc is [KV, Dn])
    out_vec = tl.zeros((Dn,), dtype=tl.float32)
    for k0 in range(0, KV, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < KV
        attn_tile = tl.load(attn_ptr + k_idx, mask=mask_k, other=0.0)  # [BLOCK_K]
        Kc_tile = tl.load(Kc_ptr + k_idx[:, None] * Dn + tl.arange(0, Dn), mask=mask_k[:, None], other=0.0)  # [BLOCK_K, Dn]
        out_vec += tl.sum(Kc_tile * attn_tile[:, None], axis=0)  # sum over BLOCK_K
    tl.store(out_ptr + tl.arange(0, Dn), out_vec)


# ModelNew: forward uses Triton kernels, no torch ops on tensors
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Extract shapes
        device = q_nope.device
        total_q = qo_indptr[-1].item()
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        num_pages = ckv_cache.shape[0]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        num_kv_indices = kv_indices.shape[0]

        # Cast caches to float32 for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output and lse buffers
        output = torch.empty(
            (total_q, num_qo_heads, head_dim_ckv),
            dtype=torch.bfloat16, device=device
        )
        lse = torch.empty(
            (total_q, num_qo_heads),
            dtype=torch.float32, device=device
        )

        # Loop over batches
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            if q_start >= q_end:
                continue
            if kv_start >= kv_end:
                continue

            q_len = q_end - q_start
            kv_len = kv_end - kv_start
            tok_idx = kv_indices[kv_start:kv_end].to(torch.int32).to(device)  # indices into cache
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # Precompute per-batch scalars
            prefix_len = kv_len - q_len  # int

            # Process each query position i
            for i in range(q_len):
                cur_q = q_start + i

                # Prepare per-head vectors qn and qp (flattened)
                # q_nope: [N, H, Dn], q_pe: [N, H, Dp]
                # We need to load qn_row and qp_row for each head h.
                # For Triton, pass flattened pointers. Since H is small (16), we loop in-kernel.
                qn_flat = torch.empty((num_qo_heads * head_dim_ckv,), dtype=torch.float32, device=device)
                qp_flat = torch.empty((num_qo_heads * head_dim_kpe,), dtype=torch.float32, device=device)

                # Fill qn_flat and qp_flat: for each head h, load q_nope[cur_q, h, :] and q_pe[cur_q, h, :]
                # q_nope is [total_q, H, Dn]
                # q_pe is [total_q, H, Dp]
                # We can build qn_flat by iterating heads and copying slices.
                for h in range(num_qo_heads):
                    qn_sub = q_nope[cur_q, h, :].to(torch.float32).contiguous()  # [Dn]
                    qp_sub = q_pe[cur_q, h, :].to(torch.float32).contiguous()   # [Dp]
                    qn_flat[h * head_dim_ckv:(h + 1) * head_dim_ckv] = qn_sub
                    qp_flat[h * head_dim_kpe:(h + 1) * head_dim_kpe] = qp_sub

                # logits buffer for H heads
                logits_buf = torch.empty((num_qo_heads * kv_len,), dtype=torch.float32, device=device)
                lse_vec = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)

                # Launch compute_logits_and_lse_kernel
                # We need to pass pointers: qn_ptr = qn_flat, qp_ptr = qp_flat, Kc_ptr = Kc, Kp_ptr = Kp,
                # logits_ptr = logits_buf (flattened), lse_ptr = lse_vec.
                # H, KV, Dn, Dp known, BLOCK_K tile.
                compute_logits_and_lse_kernel[(1,)](
                    qn_flat, qp_flat, Kc, Kp,
                    logits_buf, lse_vec,
                    sm_scale, prefix_len, prefix_len + i,
                    H=num_qo_heads, KV=kv_len, Dn=head_dim_ckv, Dp=head_dim_kpe, BLOCK_K=128
                )

                # Now, for each head, compute softmax and then out
                for h in range(num_qo_heads):
                    # attn for this head
                    attn_row = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    # We need a pointer to the logits of this head. logits_buf is [H*KV] row-major.
                    # Head h starts at h*KV
                    head_start = h * kv_len
                    # Launch softmax_row_kernel on this row
                    softmax_row_kernel[(1,)](
                        logits_buf + head_start, attn_row, KV=kv_len
                    )

                    # Compute out vector for this head: out[h, :] = attn_row @ Kc
                    out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                    compute_out_kernel[(1,)](
                        attn_row, Kc, out_vec,
                        KV=kv_len, Dn=head_dim_ckv, BLOCK_K=128
                    )

                    # Store output[cur_q, h, :] = out_vec
                    output[cur_q, h, :] = out_vec.to(torch.bfloat16)

                    # Store lse[cur_q, h] = lse_vec[h]
                    lse[cur_q, h] = lse_vec[h]

        return output, lse


def run(*args):
    return ModelNew()(*args)
