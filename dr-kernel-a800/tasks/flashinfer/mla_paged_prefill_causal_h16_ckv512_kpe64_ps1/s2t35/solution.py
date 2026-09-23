import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,       # *bf16, shape [Q_total, 16, 512]
    q_pe_ptr,         # *bf16, shape [Q_total, 16, 64]
    output_ptr,       # *bf16, shape [Q_total, 16, 512]
    lse_ptr,          # *f32,  shape [Q_total, 16]
    Kc_ptr,           # *f32,  shape [kv_len, 512]
    Kp_ptr,           # *f32,  shape [kv_len, 64]
    tok_len,          # int32: number of KV tokens for this batch element
    q_len,            # int32: number of queries in this batch element
    sm_scale,         # f32
):
    # program ids: one per (batch, query)
    pid_b = tl.program_id(axis=0)  # batch index b
    pid_i = tl.program_id(axis=1)  # query index i

    # absolute query index
    q_abs = pid_b * q_len + pid_i
    if q_abs < 0:
        return

    # Initialize per-head accumulators
    for h in range(16):
        # Compute logits vector for this head
        logits = tl.zeros([tok_len], dtype=tl.float32)

        # Compute qn[h, :] and qp[h, :] in fp32
        qn_offset = q_abs * (16 * 512) + h * 512
        qn = tl.load(q_nope_ptr + qn_offset).to(tl.float32)  # [512]

        qpe_offset = q_abs * (16 * 64) + h * 64
        qp = tl.load(q_pe_ptr + qpe_offset).to(tl.float32)  # [64]

        # Compute logits for all j
        for j in range(tok_len):
            Kc_row = tl.load(Kc_ptr + j * 512 + tl.arange(0, 512))  # [512]
            dot_qn = tl.sum(qn * Kc_row, axis=0)

            Kp_row = tl.load(Kp_ptr + j * 64 + tl.arange(0, 64))  # [64]
            dot_qp = tl.sum(qp * Kp_row, axis=0)

            logits[j] = dot_qn + dot_qp

        # Apply causal mask: j >= prefix_len + i + 1, where prefix_len = tok_len - q_len
        prefix_len = tok_len - q_len
        valid = tl.arange(0, tok_len) >= (prefix_len + pid_i + 1)
        logits = tl.where(valid, logits, -1e20)

        # Scale logits
        logits = logits * sm_scale

        # Stable logsumexp in fp32
        m = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - m)
        sum_exp = tl.sum(exp_logits, axis=0)
        lse_val = (m + tl.log(sum_exp)) / tl.log(2.0)  # logsumexp in log2
        tl.store(lse_ptr + q_abs * 16 + h, lse_val)

        # Softmax
        softmax = exp_logits / sum_exp  # [tok_len]

        # Accumulate attention output: out[h, :] += softmax[j] * Kc_sel[j, :]
        out_vec = tl.zeros([512], dtype=tl.float32)
        for j in range(tok_len):
            if valid[j]:
                Kc_row = tl.load(Kc_ptr + j * 512 + tl.arange(0, 512))  # [512]
                out_vec += softmax[j] * Kc_row

        # Store output in bfloat16
        out_offset = q_abs * (16 * 512) + h * 512
        out_bf16 = out_vec.to(tl.bfloat16)
        tl.store(output_ptr + out_offset, out_bf16)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Check shapes (constants from original asserts)
        assert q_nope.dim() == 3 and q_pe.dim() == 3
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        _, num_qo_heads2, head_dim_kpe = q_pe.shape
        assert num_qo_heads == 16 and num_qo_heads == num_qo_heads2
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Ensure device and contiguity
        qo_indptr = qo_indptr.to(torch.int32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        # Prepare output tensors
        # Compute total queries across all batches
        batch_size = qo_indptr.shape[0] - 1
        qo_indptr_sorted = qo_indptr.sort()[0]  # indices sorted ascending
        q_len_total = int(qo_indptr_sorted[-1].item())  # total queries
        output = torch.empty((q_len_total, 16, 512), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((q_len_total, 16), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel for each batch element's queries
        for b in range(batch_size):
            q_start = int(qo_indptr_sorted[b].item())
            q_end = int(qo_indptr_sorted[b + 1].item())
            q_len = q_end - q_start

            tok_start = int(kv_indptr[b].item())
            tok_end = int(kv_indptr[b + 1].item())
            tok_len = tok_end - tok


def run(*args):
    return ModelNew()(*args)
