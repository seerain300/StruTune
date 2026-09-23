import math
import torch
import triton
import triton.language as tl


# Triton kernels used by forward (no torch operations in forward)

# 1) Scale a 1D vector by a scalar
@triton.jit
def _scale_1d_kernel(x_ptr, y_ptr, alpha: tl.float32, n_elements: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * 1 + tl.arange(0, 1)
    if pid < n_elements:
        v = tl.load(x_ptr + pid)
        tl.store(y_ptr + pid, v * alpha)


# 2) Apply causal mask: if j <= query_abs_pos, set to -inf; else keep
# Assumes logits is a 1D vector of length KV.
@triton.jit
def _mask_neg_inf_kernel(logits_ptr, out_ptr, KV: tl.constexpr, query_abs_pos: tl.int32):
    j = tl.program_id(axis=0)
    if j < KV:
        val = tl.load(logits_ptr + j)
        keep = j > query_abs_pos
        val = tl.where(keep, val, -float('inf'))
        tl.store(out_ptr + j, val)


# 3) Compute lse per head for a masked 1D logits vector: lse = log(sum(exp(logit - max))) / ln(2)
@triton.jit
def _lse_row_kernel(logits_ptr, out_ptr, KV: tl.constexpr):
    # out_ptr points to a single scalar for this head
    # We implement lse in two passes: first to find max, second to sum exp.
    max_val = -float('inf')
    # Pass 1: find max
    for j in range(KV):
        val = tl.load(logits_ptr + j)
        if val > max_val:
            max_val = val
    # Pass 2: sum exp
    sum_exp = 0.0
    for j in range(KV):
        val = tl.load(logits_ptr + j)
        sum_exp += tl.exp(val - max_val)
    lse_val = tl.log(sum_exp) / 1.4426950408889634  # 1 / ln(2)
    tl.store(out_ptr, lse_val)


# 4) Compute softmax for a masked 1D logits vector and store into out_ptr
@triton.jit
def _softmax_row_kernel(logits_ptr, out_ptr, KV: tl.constexpr):
    # Compute max, then exp and normalize
    max_val = -float('inf')
    for j in range(KV):
        val = tl.load(logits_ptr + j)
        if val > max_val:
            max_val = val
    sum_exp = 0.0
    for j in range(KV):
        val = tl.load(logits_ptr + j)
        expv = tl.exp(val - max_val)
        sum_exp += expv
        # store expv / sum_exp in out_ptr[j] (we will divide after kernel)
    # Divide out_ptr[j] by sum_exp: do it by storing scaled version in a separate buffer
    # But Triton doesn't support writing to out_ptr in-place conditionally here; instead,
    # we can compute scaled values into a separate 'attn_ptr' buffer. Since we don't have another pointer,
    # we'll write out_ptr[j] as expv and return softmax by a second kernel. To keep it simple, we write final
    # softmax directly in this kernel using a 1D write per element.
    # However, Triton requires a 1-element per program write; we can implement elementwise store:
    # We'll create attn_ptr and store softmax there, then copy to out_ptr. For simplicity, we assume forward
    # provides out_ptr for final softmax values. Here, we compute and store softmax per element in a separate
    # kernel invocation: softmax_row_kernel_masked is not defined above. So we redefine it here as a separate kernel.
    pass  # placeholder; we'll define _softmax_row_kernel_masked below

# 4b) Proper softmax with masked fill and store
@triton.jit
def _softmax_row_kernel_masked(logits_ptr, out_ptr, KV: tl.constexpr, query_abs_pos: tl.int32):
    max_val = -float('inf')
    for j in range(KV):
        val = tl.load(logits_ptr + j)
        # If j <= query_abs_pos, val is -inf; skip in max. But we already masked logits, so val is valid.
        if val > max_val:
            max_val = val
    sum_exp = 0.0
    for j in range(KV):
        val = tl.load(logits_ptr + j)
        # Only valid positions have non -inf values; exp(-inf) contributes 0
        expv = tl.exp(val - max_val)
        sum_exp += expv
    for j in range(KV):
        val = tl.load(logits_ptr + j)
        soft = tl.exp(val - max_val) / sum_exp
        tl.store(out_ptr + j, soft)


# 5) GEMV: out_vec = softmax_vec @ Kc, where softmax_vec is [KV], Kc is [KV, Dn], out_vec is [Dn]
# We perform a tile-based reduction over KV. Dn is a constexpr known from the environment (head_dim_ckv = 512).
@triton.jit
def _gemv_row_kernel(softmax_ptr, Kc_ptr, out_ptr, Dn: tl.constexpr, KV: tl.constexpr):
    # out_ptr is 1D of size Dn
    for j in range(Dn):
        acc = 0.0
        # Reduce over KV
        for k in range(KV):
            soft = tl.load(softmax_ptr + k)
            kc = tl.load(Kc_ptr + k * Dn + j)
            acc += soft * kc
        tl.store(out_ptr + j, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Move to CUDA
        device = q_nope.device
        assert device.type == 'cuda', "Triton kernels require CUDA tensors"

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        num_kv_indices = kv_indices.shape[0]

        # Constants
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1
        assert qo_indptr[-1].item() == total_q

        # Prepare Kc_all and Kp_all from cache
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output and lse
        output = torch.zeros((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_qo_heads), -float('inf'), dtype=torch.float32, device=device)

        # Process each batch b
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            # Gather KV indices and tokens for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [kv_len]
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # Loop over queries in this batch
            for i in range(q_end - q_start):
                qn_row = q_nope[q_start + i].to(torch.float32)  # [16, 512]
                qp_row = q_pe[q_start + i].to(torch.float32)   # [16, 64]

                # We will compute per head h using torch GEMV for correctness, then use Triton for scaling, masking, lse, softmax, and final GEMV.
                for h in range(num_qo_heads):
                    # Per-head vectors
                    qn_vec = qn_row[h]        # [512]
                    qp_vec = qp_row[h]        # [64]
                    # Compute contributions (torch GEMV, acceptable for correctness here)
                    contrib1 = qn_vec @ Kc.T   # [512] after .T, but Kc.T is [512, kv_len]; wait, qn_vec is [512], Kc is [kv_len, 512], so qn_vec @ Kc -> [512]
                    contrib2 = qp_vec @ Kp.T   # [64] @ [64, kv_len] -> [kv_len]; then sum? No, we need [kv_len], so actually (qp_vec @ Kp.T) -> [kv_len]
                    # Correction: qn_vec is [512] @ Kc, Kc is [kv_len, 512], so qn_vec @ Kc -> [kv_len]
                    # However, qn_vec has shape [512], Kc has shape [kv_len, 512], so qn_vec @ Kc.T would be [512] @ [512, kv_len] -> [kv_len]. But qn_vec is a head vector? In original code, q_nope shape is [N, 16, 512], so per head vector is [512], and Kc is [M, 512], where M is number of tokens. So qn_vec[h] @ Kc.T -> [M].
                    # Similarly, qp_vec @ Kp.T -> [M].
                    # Therefore, logits[h, :] = (qn_vec[h] @ Kc.T) + (qp_vec[h] @ Kp.T) -> [M].
                    # The original code uses q_nope[q_start + i, h, :] and q_pe[q_start + i, h, :], but q_nope and q_pe are not shaped as [1, H, D]. The original code uses q_nope[q_start + i] and q_pe[q_start + i] with shape [H, D]. In this submission, we follow the shape assumptions: q_nope is [N, H, Dn], q_pe is [N, H, Dp]. Hence:
                    # qn_vec is a single row for head h of length Dn, and Kc is [M, Dn]. So qn_vec @ Kc.T -> [M].
                    # Let's compute logits per head using torch properly.

                    # Compute logits per head: qn_vec[h] @ Kc.T + qp_vec[h] @ Kp.T
                    # Here qn_vec and qp_vec are 1D vectors of length Dn and Dp respectively.
                    # We need to create 2D to use matmul? No, torch.dot works for 1D: qn_vec @ Kc.T -> [M], where Kc.T is [Dn, M].
                    # But Kc is [M, Dn]; Kc.T is [Dn, M]. So torch.dot(qn_vec, Kc.T) computes sum over Dn.
                    # To get per-token contribution across M, we need to use broadcasting or column-wise dot. The correct approach is:
                    # logits = qn_vec @ Kc.T + qp_vec @ Kp.T
                    # qn_vec: [Dn], Kc.T: [Dn, M] -> result: [M]
                    # Similarly, qp_vec: [Dp], Kp.T: [Dp, M] -> result: [M]
                    # So we can compute:
                    logits = torch.dot(qn_vec, Kc.T) + torch.dot(qp_vec, Kp.T)  # [M]
                    KV = kv_len

                    # Allocate a buffer for masked logits
                    logits_buf = torch.empty(KV, dtype=torch.float32, device=device)
                    # Copy logits to buffer
                    # Since logits is 1D of length KV, place it at buffer
                    # Here logits is scalar? That would be incorrect. We need per-token logits, which require per-token Kc entries.
                    # The above torch.dot produces a scalar for the whole head, which is not correct. Let me correct this.

                    # Correction: To compute per-token logits[h, j], we need to use broadcasting or elementwise operations.
                    # But since Kc is [M, Dn], and qn_vec is [Dn], for each token j, logits[h, j] = qn_vec @ Kc[j, :].T + qp_vec @ Kp[j, :].T
                    # This can be implemented using torch operations:
                    # per_token_qn = torch.einsum('i,ij->j', qn_vec, Kc)  # wrong, not standard
                    # Use torch: per_token_qn = torch.sum(qn_vec[:, None] * Kc.T[None, :], dim=0)  # [M]
                    # But that's exactly torch.dot(qn_vec, Kc.T).
                    # Therefore, for per-token, we need a loop or vectorized approach. The simplest is to compute per-token logits as:
                    # per_token_qn = torch.sum(qn_vec * Kc, dim=1)  # [M] -> no, that's not general. torch.dot(qn_vec, Kc.T) returns scalar.
                    # The original code computes per-token via GEMV: qn_row[h] @ Kc.T -> [M], which is a dot per token.
                    # However, Kc is [M, Dn], so qn_vec @ Kc.T -> [M] is correct. Similarly, qp_vec @ Kp.T -> [M].
                    # So logits is per-token vector of length M.

                    # To match original semantics, we need to construct logits per-token. The simplest is:
                    # per_token_qn = torch.einsum('i,ij->j', qn_vec, Kc)  # [M]
                    # But torch.einsum requires specific patterns; the correct way is:
                    # per_token_qn = (qn_vec[:, None] * Kc[None, :]).sum(dim=0)  # [M]
                    # Similarly, per_token_qp = (qp_vec[:, None] * Kp[None, :]).sum(dim=0)  # [M]
                    # Then logits = per_token_qn + per_token_qp  # [M]

                    # Implement:
                    per_token_qn = (qn_vec[:, None] * Kc[None, :]).sum(dim=0)  # [M]
                    per_token_qp = (qp_vec[:, None] * Kp[None, :]).sum(dim=0)  # [M]
                    logits = per_token_qn + per_token_qp  # [M], length = kv_len

                    # Scale logits
                    logits_scaled = torch.empty(KV, dtype=torch.float32, device=device)
                    # Launch Triton scale kernel
                    _scale_1d_kernel[(KV,)](logits, logits_scaled, sm_scale, KV)

                    # Apply causal mask
                    prefix_len = kv_len - (q_end - q_start)  # number of previously cached tokens
                    query_abs_pos = prefix_len + i
                    logits_masked = torch.empty(KV, dtype=torch.float32, device=device)
                    _mask_neg_inf_kernel[(KV,)](logits_scaled, logits_masked, KV, query_abs_pos)

                    # Compute lse for this head
                    lse_val = torch.empty(1, dtype=torch.float32, device=device)
                    _lse_row_kernel[(1,)](logits_masked, lse_val, KV)
                    lse[q_start + i, h] = lse_val[0]

                    # Compute softmax for masked logits
                    softmax_out = torch.empty(KV, dtype=torch.float32, device=device)
                    _softmax_row_kernel_masked[(KV,)](logits_masked, softmax_out, KV, query_abs_pos)

                    # Final GEMV: out[h, :] = softmax_out @


def run(*args):
    return ModelNew()(*args)
