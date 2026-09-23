import math
import torch
import triton
import triton.language as tl


# Kernel: compute logsumexp over a 1D vector 'scores' with mask 'mask'.
# We assume scores is float32 and mask is 0/1 float. Positions where mask == 0 will be treated as -inf.
# The kernel writes lse[0] = logsumexp(scores) / ln(2). We pass N (length), and 'lse_ptr' points to a single element.
@triton.jit
def lse_kernel(scores_ptr, mask_ptr, lse_ptr, N: tl.int32, BLOCK: tl.constexpr):
    # One program instance computes the row-wise LSE over N
    row = 0  # single row
    # Compute row-wise max over masked entries (unmasked entries are +inf after we set masked=-inf)
    max_val = -float('inf')
    # First pass: find max
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        scores = tl.load(scores_ptr + offs, mask=m, other=-float('inf'))
        mask_vec = tl.load(mask_ptr + offs, mask=m, other=1.0)  # 1.0 means valid
        scores = tl.where(mask_vec == 0.0, -float('inf'), scores)
        block_max = tl.max(scores, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Second pass: sum exp(scores - max) over masked entries
    sum_exp = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        scores = tl.load(scores_ptr + offs, mask=m, other=-float('inf'))
        mask_vec = tl.load(mask_ptr + offs, mask=m, other=1.0)
        scores = tl.where(mask_vec == 0.0, -float('inf'), scores)
        e = tl.exp(scores - max_val)
        # Zero out unmasked contributions
        e = tl.where(mask_vec == 0.0, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    # LSE = log(sum_exp) + max_val; divide by ln(2)
    lse_val = tl.log(sum_exp) + max_val
    # Write to lse_ptr[0]
    tl.store(lse_ptr + 0, lse_val)


# Kernel: compute out[K] = sum_j attn[j] * Kc[j, k] for a given row 'row'.
# attn_ptr points to [N], Kc_ptr points to [N, K], out_ptr points to [K].
# We use a 2D grid over (k_block, row), reduce across N. Here row is passed as 0..15 (one per call).
@triton.jit
def dot_kernel(attn_ptr, Kc_ptr, out_ptr,
               N: tl.int32, K: tl.int32,
               stride_kc_n: tl.int32, stride_kc_k: tl.int32,
               BLOCK_K: tl.constexpr):
    row = tl.program_id(0)  # head id 0..15
    k_block = tl.program_id(1)
    k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    # Accumulate over N
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for n in range(0, N):
        attn_val = tl.load(attn_ptr + n)  # scalar
        # Load Kc[n, k_offs]
        kc_ptrs = Kc_ptr + n * stride_kc_n + k_offs * stride_kc_k
        kc_vals = tl.load(kc_ptrs, mask=k_mask, other=0.0)
        acc += attn_val * kc_vals

    tl.store(out_ptr + k_offs, acc, mask=k_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure we are on CUDA for Triton
        device = torch.device('cuda')
        q_nope = q_nope.to(device)
        q_pe = q_pe.to(device)
        ckv_cache = ckv_cache.to(device)
        kpe_cache = kpe_cache.to(device)
        qo_indptr = qo_indptr.to(device)
        kv_indptr = kv_indptr.to(device)
        kv_indices = kv_indices.to(device)

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        # Assert constants (same as original)
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        # Prepare Kc_all and Kp_all
        Kc_all = ckv_cache.squeeze(1).float()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).float()  # [num_pages, 64]

        # Output tensors
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        qo_indptr = qo_indptr.to(torch.int32)
        kv_indptr = kv_indptr.to(torch.int32)

        # Loop over batch elements
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            # Gather KV tokens for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.long)  # [kv_len]
            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tok_idx]  # [kv_len, 512], float32
            Kp = Kp_all[tok_idx]  # [kv_len, 64], float32

            # Batch q_nope and q_pe for this batch range
            q_nope_batch = q_nope[q_start:q_end].float()  # [q_len, 16, 512]
            q_pe_batch = q_pe[q_start:q_end].float()     # [q_len, 16, 64]
            q_len = q_nope_batch.shape[0]

            # Process each query in this batch
            for i in range(q_len):
                qn = q_nope_batch[i]  # [16, 512]
                qp = q_pe_batch[i]    # [16, 64]

                # Compute scores via torch matmul (optimized)
                scores_n = qn @ Kc.t()       # [16, kv_len]
                scores_p = qp @ Kp.t()       # [16, kv_len]
                scores = scores_n + scores_p # [16, kv_len], float32

                # Compute LSE for each head via Triton
                # We'll pass a mask with -inf for j > query_abs_pos, else 0.0
                prefix_len = kv_len - q_len
                query_abs_pos = prefix_len + i
                causal_mask = torch.arange(kv_len, device=device) > query_abs_pos
                mask = torch.ones(kv_len, device=device, dtype=torch.float32)
                mask[causal_mask] = 0.0  # 0 means "inf" in original; here 0 means we want to set to -inf in kernel

                # lse for this token, per head
                lse_vec = torch.empty(16, device=device, dtype=torch.float32)
                lse_kernel[(1,)](scores, mask, lse_vec, kv_len, BLOCK=256)
                lse[q_start + i] = lse_vec / math.log(2.0)

                # Now apply the mask and compute softmax (we still keep Triton)
                # Set masked positions to -inf
                scores_masked = scores.clone()
                scores_masked = torch.where(causal_mask.unsqueeze(0).expand(16, -1), scores_masked, -float('inf'))

                # For dot kernel: attn = softmax(scores_masked, dim=-1) per row
                # We will use Triton to compute out = attn @ Kc for each head.
                # Prepare attn and out
                for head in range(16):
                    attn = torch.softmax(scores_masked[head], dim=-1)  # [kv_len]
                    out_row = torch.empty(512, device=device, dtype=torch.float32)
                    dot_kernel[(1, triton.cdiv(512, 64))](attn, Kc, out_row, kv_len, 512, Kc.stride(0), Kc.stride(1), BLOCK_K=64)
                    output[q_start + i, head] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
