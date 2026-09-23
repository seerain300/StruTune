import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,       # *bf16, [total_q, 16, 512]
    q_pe_ptr,         # *bf16, [total_q, 16, 64]
    output_ptr,       # *bf16, [total_q, 16, 512]
    lse_ptr,          # *f32,  [total_q, 16]
    ckv_cache_ptr,    # *bf16, [num_pages, 1, 512]
    kpe_cache_ptr,    # *bf16, [num_pages, 1, 64]
    qo_indptr_ptr,    # *i32,  [len_indptr]
    kv_indptr_ptr,    # *i32,  [len_indptr]
    kv_indices_ptr,   # *i32,  [num_kv_indices]
    q_len_total,      # i32, total queries across all batches
    tok_len,          # i32, tokens per batch
    sm_scale,         # f32
):
    # program_id(0): batch element b, program_id(1): query index i
    b = tl.program_id(0)
    i = tl.program_id(1)

    # absolute query index
    q_start = tl.load(qo_indptr_ptr + b)  # i32
    q_abs = q_start + i
    q_len = tl.load(qo_indptr_ptr + b + 1) - q_start  # i32
    prefix_len = tok_len - q_len  # i32

    # Prepare output vectors per head
    for h in range(16):
        # load qn[h, :] (512) and qp[h, :] (64) as float32
        qn_offset = q_abs * (16 * 512) + h * 512
        qn_vec = tl.load(q_nope_ptr + qn_offset + tl.arange(0, 512), mask=(tl.arange(0, 512) < 512), other=0.0).to(tl.float32)
        qp_offset = q_abs * (16 * 64) + h * 64
        qp_vec = tl.load(q_pe_ptr + qp_offset + tl.arange(0, 64), mask=(tl.arange(0, 64) < 64), other=0.0).to(tl.float32)

        # accumulators
        logits = tl.full([tok_len], -1e30, dtype=tl.float32)
        out_vec = tl.zeros([512], dtype=tl.float32)

        # loop over tokens j in this batch
        for j in range(tok_len):
            # tok_idx = kv_indices[kv_indptr[b] + j]
            tok_idx_j = tl.load(kv_indices_ptr + (b * tok_len + j))  # i32

            # load Kc_sel[j, :] and Kp_sel[j, :]
            Kc_row = tl.load(ckv_cache_ptr + tok_idx_j * 512 + tl.arange(0, 512), mask=(tl.arange(0, 512) < 512), other=0.0).to(tl.float32)
            Kp_row = tl.load(kpe_cache_ptr + tok_idx_j * 64 + tl.arange(0, 64), mask=(tl.arange(0, 64) < 64), other=0.0).to(tl.float32)

            # dot products
            dot_qn = tl.sum(qn_vec * Kc_row, axis=0)
            dot_qp = tl.sum(qp_vec * Kp_row, axis=0)

            logits[j] = (dot_qn + dot_qp) * sm_scale

        # apply causal mask: j >= prefix_len + i + 1
        j_range = tl.arange(0, tok_len)
        mask_j = j_range >= (prefix_len + i + 1)
        logits = tl.where(mask_j, logits, -1e30)

        # logsumexp (natural log) then convert to log2
        m = tl.max(logits, axis=0)
        sum_exp = tl.sum(tl.exp(logits - m), axis=0)
        lse_val = (m + tl.log(sum_exp)) / math.log(2.0)
        lse_offset = q_abs * 16 + h
        tl.store(lse_ptr + lse_offset, lse_val)

        # softmax over logits
        exp_logits = tl.exp(logits - m)
        sum_exp = tl.sum(exp_logits, axis=0)
        softmax = exp_logits / sum_exp

        # output vector: out[h, :] = sum_j softmax[j] * Kc_sel[j, :]
        out_vec = tl.zeros([512], dtype=tl.float32)
        for j in range(tok_len):
            if mask_j[j]:
                Kc_row = tl.load(ckv_cache_ptr + tok_idx_j * 512 + tl.arange(0, 512), mask=(tl.arange(0, 512) < 512), other=0.0).to(tl.float32)
                out_vec += softmax[j] * Kc_row

        # store output as bfloat16
        out_offset = q_abs * (16 * 512) + h * 512
        tl.store(output_ptr + out_offset, out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # assertions to mirror original behavior
        assert q_nope.dim() == 3 and q_pe.dim() == 3
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        total_q2, num_qo_heads2, head_dim_kpe = q_pe.shape
        assert num_qo_heads == 16 and num_qo_heads == num_qo_heads2
        assert head_dim_ckv == 512 and head_dim_kpe == 64

        device = q_nope.device
        # ensure int32 indices
        qo_indptr = qo_indptr.to(torch.int32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        batch_size = qo_indptr.shape[0] - 1  # number of batch elements

        # allocate outputs (we can write per (b, i) and then return final concatenated tensors)
        output = torch.empty((0, 16, 512), dtype=torch.bfloat16, device=device)
        lse = torch.empty((0, 16), dtype=torch.float32, device=device)

        # per-batch launches to correctly pass tok_len
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            tok_start = int(kv_indptr[b].item())
            tok_end = int(kv_indptr[b + 1].item())
            tok_len = tok_end - tok_start

            # output buffers for this batch
            out_batch = torch.empty((q_len, 16, 512), dtype=torch.bfloat16, device=device)
            lse_batch = torch.empty((q_len, 16), dtype=torch.float32, device=device)

            # grid is (1, q_len) for this batch
            _forward_single_query_kernel[(1, q_len)](
                q_nope, q_pe, out_batch, lse_batch,
                ckv_cache, kpe_cache,
                qo_indptr, kv_indptr, kv_indices,
                q_len, tok_len, sm_scale,
            )

            # append to final outputs
            output = torch.cat([output, out_batch], dim=0)
            lse = torch.cat([lse, lse_batch], dim=0)

        return output, lse


def run(*args):
    return ModelNew()(*args)
