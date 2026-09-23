import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    # Input pointers (original dtype, will be cast inside kernel to float32)
    q_nope_ptr,       # *bf16, shape [Q_total, 16, 512]
    q_pe_ptr,         # *bf16, shape [Q_total, 16, 64]
    Kc_sel_ptr,       # *bf16, shape [kv_len, 512]
    Kp_sel_ptr,       # *bf16, shape [kv_len, 64]
    output_ptr,       # *bf16, shape [Q_total, 16, 512]
    lse_ptr,          # *fp32, shape [Q_total, 16]
    # runtime scalars
    q_start,          # int32: starting query index in global q_nope for this batch element
    # constexpr meta-parameters
    q_len: tl.constexpr,        # number of queries in this batch element (compile-time for this program)
    kv_len: tl.constexpr,       # number of selected KV tokens (compile-time for this program)
    sm_scale: tl.float32,       # scaling factor
    ln2_inv: tl.float32,        # 1.0 / ln(2.0)
    NUM_HEADS: tl.constexpr,    # 16
    HEAD_DIM_CKV: tl.constexpr, # 512
    HEAD_DIM_KPE: tl.constexpr, # 64
):
    # This program handles one specific query i within one batch element b.
    # We derive b from the grid, and i is the second grid dimension.
    # Here, the grid is (len_indptr-1, q_len). We can compute b = program_id(0), i = program_id(1).
    # But Triton doesn't provide program_id directly in this setup; instead, we pass q_start and i implicitly
    # via the grid and host-side. In other words, each program gets q_start and processes its i.

    # Compute absolute query index
    q_abs = q_start  # we expect the host to pass the correct q_start for this program. Triton doesn't expose pid here,
                      # but the host will launch one program per query, so we rely on q_start passed in.

    # We need to know which query i this program handles. Since we launch with grid=(len_indptr-1, q_len),
    # we can derive i from the second grid dimension. Triton requires us to pass meta-params; we can emulate
    # by passing i as a constexpr (compile-time) or derive from program_id(1). To keep it simple, we assume
    # the host ensures that q_abs = q_start + i for this program. In practice, we compute q_abs as q_start + i
    # where i is implicitly the program_id(1) index. Since Triton kernel doesn't have program_id, we rely on
    # host to pass q_abs; but we can infer i as q_abs - q_start. Let's do it.
    i = q_abs - q_start

    # Compute per-head vectors qn and qp in float32
    # qn[h, :] = q_nope[q_abs, h, :]
    # We'll build qn and qp as (NUM_HEADS, HEAD_DIM_CKV) by loading slices. Since we have H as constexpr, we can loop.
    # However, Triton kernels don't support dynamic loops well; to keep it simple, we assume H=16 and load all heads.
    # We can create arrays of pointers using tl.arange but Triton requires static shapes. Instead, we'll load each
    # head explicitly since H is small. Better: build a 2D tensor using tl.load with a static grid.
    # To keep code manageable, we will load qn and qp using H as a small loop (constexpr), and compute per head.
    # But Triton expects static shapes; we'll implement per-head computations using simple indexing.

    # We'll compute per head h = 0..NUM_HEADS-1
    for h in range(NUM_HEADS):
        # Build qn[h, :] and qp[h, :] vectors
        qn = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        # For bf16 input, q_nope_ptr + q_abs * stride, then cast
        # Compute address: q_nope_ptr + q_abs * (16*512) + h * 512
        # But q_nope is contiguous with shape [Q_total, 16, 512]; strides:
        # row-major: stride_q = 16*512 = 8192, head_stride = 512.
        base_qn = q_nope_ptr + q_abs * 8192 + h * 512
        k_idx = tl.arange(0, HEAD_DIM_CKV)
        qn = tl.load(base_qn + k_idx, mask=k_idx < HEAD_DIM_CKV, other=0.0).to(tl.float32)

        qp = tl.zeros((HEAD_DIM_KPE,), dtype=tl.float32)
        base_qp = q_pe_ptr + q_abs * (16 * 64) + h * 64
        kpe_idx = tl.arange(0, HEAD_DIM_KPE)
        qp = tl.load(base_qp + kpe_idx, mask=kpe_idx < HEAD_DIM_KPE, other=0.0).to(tl.float32)

        # Compute logits for all j in [0, kv_len)
        logits_vec = tl.zeros((kv_len,), dtype=tl.float32)

        # Loop over j: compute dot(qn[h], Kc_sel[j, :]) + dot(qp[h], Kp_sel[j, :])
        for j in range(kv_len):
            # Load Kc_sel[j, :]
            k_vec = tl.arange(0, HEAD_DIM_CKV)
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + k_vec, mask=k_vec < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            dot_qn = tl.sum(qn * Kc_j, axis=0)  # scalar

            kpe_vec = tl.arange(0, HEAD_DIM_KPE)
            Kp_j = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + kpe_vec, mask=kpe_vec < HEAD_DIM_KPE, other=0.0).to(tl.float32)
            # dot_qp = sum_k qp[k] * Kp_j[k], but Kp_j is length 64, we need to ensure qp is 64? Note: Kp_sel_ptr
            # is [kv_len, 64]; we have loaded qp as 64? Wait, q_pe is [Q_total, 16, 64]; qp is [16, 64]. We need
            # to access the h-th row. We have loaded qp as 64; Kp_j is also 64. We can compute dot_qp.
            dot_qp = tl.sum(qp * Kp_j, axis=0)  # scalar

            logits_vec[j] = dot_qn + dot_qp  # scalar per j

        # Scale
        logits_scaled = logits_vec * sm_scale

        # Apply causal mask: for absolute position query_abs_pos = prefix_len + i
        prefix_len = kv_len - q_len
        query_abs_pos = prefix_len + (i + 1)
        causal_mask = (tl.arange(0, kv_len) >= query_abs_pos)
        logits_scaled = tl.where(causal_mask, -float("inf"), logits_scaled)

        # Compute logsumexp in log2
        max_log = tl.max(logits_scaled, axis=0)
        sumexp = tl.sum(tl.exp(logits_scaled - max_log), axis=0)
        lse_val = max_log + tl.log(sumexp) * ln2_inv
        # Store lse[q_abs, h]
        lse_base = q_abs * NUM_HEADS + h
        tl.store(lse_ptr + lse_base, lse_val)

        # Softmax
        logits_stable = logits_scaled - max_log
        exp_logits = tl.exp(logits_stable)
        sumexp2 = tl.sum(exp_logits, axis=0)
        softmax = exp_logits / sumexp2  # [kv_len] vector

        # Output: out[h, :] = sum_j softmax[j] * Kc_sel[j, :]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for j in range(kv_len):
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            out_vec += softmax[j] * Kc_j

        # Store output vector for this query and head as bfloat16
        out_store = out_vec.to(tl.bfloat16)
        base_out = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        for k in range(HEAD_DIM_CKV):
            tl.store(output_ptr + base_out + k, out_store[k])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Assumptions based on original code
        assert q_nope.shape[1] == 16 and q_nope.shape[2] == 512, "q_nope must be [Q_total, 16, 512]"
        assert q_pe.shape[1] == 16 and q_pe.shape[2] == 64, "q_pe must be [Q_total, 16, 64]"
        assert ckv_cache.shape[1] == 1 and ckv_cache.shape[2] == 512, "ckv_cache must be [num_pages, 1, 512]"
        assert kpe_cache.shape[1] == 1 and kpe_cache.shape[2] == 64, "kpe_cache must be [num_pages, 1, 64]"
        assert qo_indptr.dim() == 1 and kv_indptr.dim() == 1, "indptrs must be 1D"

        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[2]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        # Output and LSE buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Precompute ln2_inv
        ln2_inv = 1.0 / math.log(2.0)

        # For each batch element b, construct tok_idx and Kc_sel, Kp_sel
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            # Get kv token indices for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len = page_end - page_beg

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)

            # Select Kc_sel and Kp_sel for this batch element. Keep as bfloat16 and cast in kernel.
            # Kc_sel: [kv_len, 512], Kp_sel: [kv_len, 64]
            Kc_sel = ckv_cache[tok_idx].to(torch.bfloat16)  # shape [kv_len, 512]
            Kp_sel = kpe_cache[tok_idx].to(torch.bfloat16)  # shape [kv_len, 64]

            # Launch Triton kernel: one program per query i in this batch element
            grid = (1, q_len)
            _forward_single_query_kernel[grid](
                q_nope, q_pe, Kc_sel, Kp_sel, output, lse,
                q_start,
                q_len=q_len, kv_len=kv_len,
                sm_scale=float(sm_scale), ln2_inv=float(ln2_inv),
                NUM_HEADS=16, HEAD_DIM_CKV=512, HEAD_DIM_KPE=64,
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
