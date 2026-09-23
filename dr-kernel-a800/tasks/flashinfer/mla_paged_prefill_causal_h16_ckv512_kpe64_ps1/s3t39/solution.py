import torch
import triton
import triton.language as tl


# Triton kernel: compute logits_scaled for a single head h over L tokens
# Arguments:
#   qn_ptr: *fp32, length H*K, flattened q_nope[q_abs] as fp32
#   qp_ptr: *fp32, length H*Kp, flattened q_pe[q_abs] as fp32
#   Kc_ptr: *fp32, flattened ckv_cache.squeeze(1) -> [P*K] float32
#   Kp_ptr: *fp32, flattened kpe_cache.squeeze(1) -> [P*Kp] float32
#   tok_idx_ptr: *int32, length L
#   L: int32, number of tokens
#   logits_scaled_ptr: *fp32, output vector [L], to be filled for head h
# Constants (meta-parameters):
#   K: tl.constexpr, head_dim_ckv (512)
#   Kp: tl.constexpr, head_dim_kpe (64)
#   head: tl.constexpr, head index
@triton.jit
def compute_logits_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, tok_idx_ptr, logits_scaled_ptr,
    L, sm_scale,
    K: tl.constexpr, Kp: tl.constexpr, head: tl.constexpr
):
    # Accumulate qn and qp dot products for head 'head'
    sum_qn = 0.0
    sum_qp = 0.0

    # Loop over tokens l = 0..L-1
    for l in range(0, L):
        tok_idx_val = tl.load(tok_idx_ptr + l)  # int32 token index
        # qn contribution: qn_vec[head*K + k] * Kc[tok_idx_val, k]
        for k in range(0, K):
            qnk = tl.load(qn_ptr + head * K + k)  # fp32
            kc = tl.load(Kc_ptr + tok_idx_val * K + k)  # fp32
            sum_qn += qnk * kc
        # qp contribution: qp_vec[head*Kp + kp] * Kp[tok_idx_val, kp]
        for kp in range(0, Kp):
            qpk = tl.load(qp_ptr + head * Kp + kp)  # fp32
            kpval = tl.load(Kp_ptr + tok_idx_val * Kp + kp)  # fp32
            sum_qp += qpk * kpval

        # Store scaled logits for this head and token
        tl.store(logits_scaled_ptr + l, (sum_qn + sum_qp) * sm_scale)

        # Reset accumulators for next l (optional: they are scalar, cheap to keep)
        sum_qn = 0.0
        sum_qp = 0.0


# Triton kernel: compute lse for a single head using logits_scaled_ptr and L
# Arguments:
#   logits_scaled_ptr: *fp32, vector [L]
#   lse_out_ptr: *fp32, scalar output for this head
#   L: int32
@triton.jit
def compute_lse_row_kernel(
    logits_scaled_ptr, lse_out_ptr, L
):
    # Compute logsumexp over L elements, then divide by ln(2)
    # Initialize max
    max_val = -float("inf")
    for l in range(0, L):
        val = tl.load(logits_scaled_ptr + l)
        if val > max_val:
            max_val = val
    # Compute sum(exp(x - max))
    sum_exp = 0.0
    for l in range(0, L):
        val = tl.load(logits_scaled_ptr + l)
        sum_exp += tl.exp(val - max_val)
    lse = tl.log(sum_exp) + max_val
    tl.store(lse_out_ptr, lse / 0.6931471805599453)  # 1 / ln(2)


# Triton kernel: compute softmax (scaled) for a single head and write attn vector
# Arguments:
#   logits_scaled_ptr: *fp32, vector [L]
#   lse_ptr: *fp32, scalar lse for this head
#   attn_ptr: *fp32, output vector [L]
#   L: int32
@triton.jit
def compute_softmax_row_kernel(
    logits_scaled_ptr, lse_ptr, attn_ptr, L
):
    lse = tl.load(lse_ptr)
    for l in range(0, L):
        val = tl.load(logits_scaled_ptr + l)
        attn = tl.exp(val - lse)  # softmax
        tl.store(attn_ptr + l, attn)


# Triton kernel: compute output vector for a single head: attn @ Kc[tok_idx, :]
# Inputs:
#   attn_ptr: *fp32, vector [L]
#   Kc_ptr: *fp32, flattened Kc_all -> [P*K]
#   tok_idx_ptr: *int32, length L
#   out_vec_ptr: *fp32, output vector [K]
# Constants:
#   K: tl.constexpr
@triton.jit
def gemv_row_kernel(
    attn_ptr, Kc_ptr, tok_idx_ptr, out_vec_ptr, L, K: tl.constexpr
):
    # For each feature k in 0..K-1, compute dot product of attn over L tokens with Kc[tok_idx[l], k]
    for k in range(0, K):
        dot = 0.0
        for l in range(0, L):
            attn_val = tl.load(attn_ptr + l)  # fp32
            tok_idx_val = tl.load(tok_idx_ptr + l)  # int32
            kc_val = tl.load(Kc_ptr + tok_idx_val * K + k)  # fp32
            dot += attn_val * kc_val
        tl.store(out_vec_ptr + k, dot)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        # Ensure all inputs are on the same device
        assert q_pe.device == device and ckv_cache.device == device and kpe_cache.device == device \
               and qo_indptr.device == device and kv_indptr.device == device and kv_indices.device == device

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"
        P = ckv_cache.shape[0]  # num_pages

        # Squeeze caches to [P, K] and [P, Kp]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, 64]

        # Output tensors
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        B = qo_indptr.shape[0]
        for b in range(B):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue
            q_len = q_end - q_start

            # Token indices for this batch
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]]
            tok_idx = tok_idx.to(torch.int32).to(device)  # ensure int32, on device
            L = tok_idx.shape[0]

            # Allocate per-query intermediates
            for i in range(q_len):
                q_abs = q_start + i

                # Prepare qn_vec and qp_vec: flatten to fp32
                qn = q_nope[q_abs]  # [16, 512], bfloat16
                qp = q_pe[q_abs]    # [16, 64], bfloat16
                qn_vec = qn.to(torch.float32).reshape(-1)  # [16*512]
                qp_vec = qp.to(torch.float32).reshape(-1)  # [16*64]

                # We will compute per head
                for h in range(16):
                    # 1) Compute logits_scaled for this head
                    logits_scaled = torch.empty((L,), dtype=torch.float32, device=device)
                    # Launch Triton kernel: compute_logits_row_kernel
                    compute_logits_row_kernel[(1,)](
                        qn_vec, qp_vec, Kc_all, Kp_all, tok_idx, logits_scaled,
                        L, float(sm_scale),
                        K=512, Kp=64, head=h
                    )

                    # 2) Compute lse for this head
                    lse_entry = torch.empty((1,), dtype=torch.float32, device=device)  # scalar output
                    compute_lse_row_kernel[(1,)](
                        logits_scaled, lse_entry, L
                    )
                    lse[q_abs, h] = lse_entry[0]

                    # 3) Compute attn vector for this head
                    attn = torch.empty((L,), dtype=torch.float32, device=device)
                    compute_softmax_row_kernel[(1,)](
                        logits_scaled, lse[q_abs, h].to(torch.float32), attn, L
                    )

                    # 4) Compute output vector for this head: attn @ Kc[tok_idx, :]
                    out_vec = torch.empty((512,), dtype=torch.float32, device=device)
                    gemv_row_kernel[(1,)](
                        attn, Kc_all, tok_idx, out_vec, L, K=512
                    )

                    # Store output as bfloat16
                    output[q_abs, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
