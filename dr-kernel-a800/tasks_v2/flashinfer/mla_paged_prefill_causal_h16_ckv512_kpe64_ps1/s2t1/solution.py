import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_batch_kernel(
    # Input pointers
    q_nope_ptr,       # *bf16, shape [Q_total, 16, 512], row-major
    q_pe_ptr,         # *bf16, shape [Q_total, 16, 64], row-major
    Kc_sel_ptr,       # *bf16, shape [kv_len, 512], row-major (selected tokens)
    Kp_sel_ptr,       # *bf16, shape [kv_len, 64], row-major (selected tokens)
    output_ptr,       # *bf16, shape [Q_total, 16, 512], row-major
    lse_ptr,          # *fp32, shape [Q_total, 16]
    # runtime scalars
    q_len: tl.constexpr,   # number of queries in this batch element
    kv_len: tl.constexpr,  # number of selected KV tokens in this batch element
    q_start,               # int32, start query index in global q_nope
    sm_scale: tl.constexpr,       # fp32 scaling factor
    ln2_inv: tl.constexpr,        # fp32 = 1.0 / ln(2.0)
    NUM_HEADS: tl.constexpr,      # 16
    HEAD_DIM_CKV: tl.constexpr,   # 512
    HEAD_DIM_KPE: tl.constexpr,   # 64
):
    # Each program handles one batch element (b) implicitly via q_start; we still need to loop over queries i.
    # We assume grid size is 1 per launch; host will launch one program per batch element.

    # For each query i in this batch element
    for i in range(q_len):
        q_abs = q_start + i

        # Load qn[h, :] and qp[h, :] for all heads h in 0..NUM_HEADS-1, convert to fp32
        qn = tl.zeros((NUM_HEADS, HEAD_DIM_CKV), dtype=tl.float32)
        qp = tl.zeros((NUM_HEADS, HEAD_DIM_KPE), dtype=tl.float32)

        for h in range(NUM_HEADS):
            k_vec = tl.arange(0, HEAD_DIM_CKV)
            offset_qn = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV + k_vec
            qn[h, :] = tl.load(q_nope_ptr + offset_qn, mask=k_vec < HEAD_DIM_CKV, other=0.0).to(tl.float32)

            kpe_vec = tl.arange(0, HEAD_DIM_KPE)
            offset_qp = q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE + kpe_vec
            qp[h, :] = tl.load(q_pe_ptr + offset_qp, mask=kpe_vec < HEAD_DIM_KPE, other=0.0).to(tl.float32)

        # Compute logits per head: [NUM_HEADS, kv_len]
        logits = tl.zeros((NUM_HEADS, kv_len), dtype=tl.float32)

        # For each token j in [0, kv_len)
        for j in range(kv_len):
            k_vec = tl.arange(0, HEAD_DIM_CKV)
            # Load Kc_sel[j, :] in original dtype (bf16), cast to fp32 for dot
            Kc_j_bf = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + k_vec, mask=k_vec < HEAD_DIM_CKV, other=0.0)  # bf16
            Kc_j = Kc_j_bf.to(tl.float32)

            # dot(qn[h], Kc_j)
            dot_qn = tl.sum(qn * Kc_j[None, :], axis=1)  # [NUM_HEADS]

            kpe_vec = tl.arange(0, HEAD_DIM_KPE)
            Kp_j_bf = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + kpe_vec, mask=kpe_vec < HEAD_DIM_KPE, other=0.0)  # bf16
            Kp_j = Kp_j_bf.to(tl.float32)

            dot_qp = tl.sum(qp * Kp_j[None, :], axis=1)  # [NUM_HEADS]

            logits[:, j] = dot_qn + dot_qp

        # Scale
        logits = logits * sm_scale

        # Causal mask: for i-th query, only j >= prefix_len + i + 1 are valid, where prefix_len = kv_len - q_len
        prefix_len = kv_len - q_len
        valid_start = prefix_len + i + 1
        j_vec = tl.arange(0, kv_len)
        causal_mask = j_vec >= valid_start
        logits = tl.where(causal_mask, -float("inf"), logits)

        # logsumexp in log2
        m = tl.max(logits, axis=1)  # [NUM_HEADS]
        sumexp = tl.sum(tl.exp(logits - m[:, None]), axis=1)
        lse_val = m + tl.log(sumexp) * ln2_inv  # [NUM_HEADS]
        for h in range(NUM_HEADS):
            tl.store(lse_ptr + q_abs * NUM_HEADS + h, lse_val[h])

        # Softmax (stable)
        logits = logits - m[:, None]
        exp_logits = tl.exp(logits)
        denom = tl.sum(exp_logits, axis=1)[:, None]  # [NUM_HEADS, 1]
        softmax = exp_logits / denom  # [NUM_HEADS, kv_len]

        # Output: out[h, :] = sum_j softmax[h,j] * Kc_sel[j, :]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for h in range(NUM_HEADS):
            for j in range(kv_len):
                k_vec = tl.arange(0, HEAD_DIM_CKV)
                Kc_j_bf = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + k_vec, mask=k_vec < HEAD_DIM_CKV, other=0.0)  # bf16
                Kc_j = Kc_j_bf.to(tl.float32)
                out_vec += softmax[h, j] * Kc_j
            # Store to output[q_abs, h, :] in bf16
            k_out_vec = tl.arange(0, HEAD_DIM_CKV)
            out_offset = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV + k_out_vec
            tl.store(output_ptr + out_offset, out_vec.to(tl.bfloat16), mask=k_out_vec < HEAD_DIM_CKV)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Assertions (original code expects these)
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors"
        assert q_nope.dtype == torch.bfloat16 and q_pe.dtype == torch.bfloat16, "q_nope and q_pe must be bfloat16"
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "Cache must have shape [num_pages, 1, ...]"
        assert q_nope.shape[1] == 16, "num_qo_heads must be 16"
        assert q_nope.shape[2] == 512, "head_dim_ckv must be 512"
        assert q_pe.shape[2] == 64, "head_dim_kpe must be 64"
        num_q = q_nope.shape[0]
        device = q_nope.device

        total_q = int(qo_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        assert total_q == num_q, "total_q must equal q_nope.shape[0]"
        assert len_indptr >= 2, "len_indptr must be >= 2 to have at least one batch element"
        batch_size = len_indptr - 1

        # Prepare outputs
        output = torch.empty((num_q, 16, 512), dtype=torch.bfloat16, device=device)
        lse = torch.empty((num_q, 16), dtype=torch.float32, device=device)

        ln2_inv = 1.0 / math.log(2.0)

        # Launch one Triton program per batch element
        # We compute q_len and kv_len on host and pass them as scalar args; we also pass q_start.
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end:
                continue  # no queries in this batch element

            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            # tok_idx is the list of token indices for this batch element: kv_indices[kv_start:kv_end]
            tok_idx = kv_indices[kv_start:kv_end].to(torch.int32).to(device)  # ensure int32 on device

            # Select Kc and Kp rows for this batch element (keep original dtype bf16; cast in kernel)
            Kc_sel = ckv_cache[tok_idx]               # [kv_len, 512], bf16
            Kp_sel = kpe_cache[tok_idx]              # [kv_len, 64], bf16

            # Triton expects pointers; no .to() or dtype conversions in forward
            Kc_sel_ptr = Kc_sel.contiguous()
            Kp_sel_ptr = Kp_sel.contiguous()

            _forward_batch_kernel[(1,)](
                q_nope, q_pe, Kc_sel_ptr, Kp_sel_ptr, output, lse,
                q_len, kv_len, q_start,
                sm_scale, ln2_inv,
                NUM_HEADS=16, HEAD_DIM_CKV=512, HEAD_DIM_KPE=64,
                num_warps=4,
                num_stages=2,
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
