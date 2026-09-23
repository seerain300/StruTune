import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits per head for a single query i in a batch b
# It expects flattened qn_vec (H*K elements) and qp_vec (H*Kp elements).
# It expects tok_idx_ptr (int32, length L). It will index into Kc_all and Kp_all using tok_idx[l].
# It writes a per-head row of logits_scaled into logits_scaled_ptr (fp32, length L).
@triton.jit
def compute_logits_kernel(
    qn_vec_ptr,      # *fp32, length = H * K
    qp_vec_ptr,      # *fp32, length = H * Kp
    Kc_ptr,          # *fp32, base pointer to Kc_all (P*K)
    Kp_ptr,          # *fp32, base pointer to Kp_all (P*Kp)
    logits_scaled_ptr,  # *fp32, output for this head, length L
    H: tl.constexpr,     # number of heads used to interpret qn_vec_ptr
    K: tl.constexpr,     # head_dim_ckv
    Kp: tl.constexpr,    # head_dim_kpe
    L: tl.constexpr,     # number of tokens in this segment
    tok_idx_ptr,         # *int32, length L
    sm_scale,            # fp32 scalar
):
    head = tl.program_id(0)  # which head to compute, grid (H,)
    base_qn = head * K
    base_qp = head * Kp

    for l in tl.static_range(0, L):
        # Load qn[head, :] and qp[head, :]
        qn_row = [0.0] * K
        qn_qp_row = [0.0] * Kp
        for k in tl.static_range(0, K):
            qn_row[k] = tl.load(qn_vec_ptr + base_qn + k)
        for kp in tl.static_range(0, Kp):
            qn_qp_row[kp] = tl.load(qp_vec_ptr + base_qp + kp)

        tok = tl.load(tok_idx_ptr + l)
        dot_qn = 0.0
        dot_qp = 0.0
        for k in tl.static_range(0, K):
            dot_qn += qn_row[k] * tl.load(Kc_ptr + tok * K + k)
        for kp in tl.static_range(0, Kp):
            dot_qp += qn_qp_row[kp] * tl.load(Kp_ptr + tok * Kp + kp)

        logits = dot_qn + dot_qp
        tl.store(logits_scaled_ptr + l, logits * sm_scale)


# Triton kernel: compute lse = logsumexp(logits_scaled) / ln(2) for a single head.
# It expects logits_scaled_ptr (length L) and stores into lse_ptr (single fp32).
@triton.jit
def compute_lse_kernel(
    logits_scaled_ptr,  # *fp32, length L
    lse_ptr,            # *fp32, single element
    L: tl.constexpr,
):
    max_val = -1e30
    for l in tl.static_range(0, L):
        val = tl.load(logits_scaled_ptr + l)
        if val > max_val:
            max_val = val

    sum_exp = 0.0
    for l in tl.static_range(0, L):
        val = tl.load(logits_scaled_ptr + l)
        sum_exp += tl.exp(val - max_val)

    lse = tl.log(sum_exp) + max_val
    ln2 = 0.6931471805599453  # math.log(2.0)
    tl.store(lse_ptr, lse / ln2)


# Triton kernel: compute softmax over logits_scaled for a single head.
# It expects logits_scaled_ptr (length L), lse_ptr (single fp32), and writes attn_ptr (length L).
@triton.jit
def compute_softmax_kernel(
    logits_scaled_ptr,  # *fp32, length L
    lse_ptr,            # *fp32, single element
    attn_ptr,           # *fp32, length L
    L: tl.constexpr,
):
    lse = tl.load(lse_ptr)
    for l in tl.static_range(0, L):
        val = tl.load(logits_scaled_ptr + l)
        soft = tl.exp(val - lse)
        tl.store(attn_ptr + l, soft)


# Triton kernel: GEMV out = attn @ Kc_rows[tok_idx[:]] where Kc_rows[tok_idx[:]] is [L, K]
# It expects attn_ptr (length L), Kc_ptr (P*K), tok_idx_ptr (int32, length L), K (512), and writes out_ptr (length K).
@triton.jit
def gemv_out_kernel(
    attn_ptr,           # *fp32, length L
    Kc_ptr,             # *fp32, (P*K)
    tok_idx_ptr,        # *int32, length L
    out_ptr,            # *fp32, length K
    L: tl.constexpr,
    K: tl.constexpr,
):
    for k in tl.static_range(0, K):
        acc = 0.0
        for l in tl.static_range(0, L):
            attn_l = tl.load(attn_ptr + l)
            tok = tl.load(tok_idx_ptr + l)
            acc += attn_l * tl.load(Kc_ptr + tok * K + k)
        tl.store(out_ptr + k, acc)


def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    # Ensure dtype and device for caches
    ckv_cache = ckv_cache.squeeze(1).to(device=device, dtype=torch.float32)
    kpe_cache = kpe_cache.squeeze(1).to(device=device, dtype=torch.float32)

    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    assert num_qo_heads == 16, "num_qo_heads must be 16"
    assert head_dim_ckv == 512, "head_dim_ckv must be 512"
    assert head_dim_kpe == 64, "head_dim_kpe must be 64"

    output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

    # Only one batch element as per provided get_inputs
    batch_size = qo_indptr.shape[0] - 1
    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        if q_start >= q_end:
            continue

        q_len = q_end - q_start

        # KV indices for this batch
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        if page_beg >= page_end:
            continue
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # [L]
        L = tok_idx.shape[0]

        # We implement per-head with H=16
        H = 16
        K = head_dim_ckv  # 512
        Kp = head_dim_kpe  # 64

        for i in range(q_len):
            q_abs = q_start + i

            # Process each head h
            for h in range(H):
                # Flattened q vectors for this head
                qn_vec = q_nope[q_abs, h, :].to(device=device, dtype=torch.float32).contiguous().view(-1)  # [K]
                qp_vec = q_pe[q_abs, h, :].to(device=device, dtype=torch.float32).contiguous().view(-1)  # [Kp]

                # Logits for this head
                logits_scaled = torch.empty((L,), dtype=torch.float32, device=device)
                compute_logits_kernel[(1,)](
                    qn_vec, qp_vec, ckv_cache, kpe_cache, logits_scaled,
                    H=H, K=K, Kp=Kp, L=L,
                    tok_idx_ptr=tok_idx,
                    sm_scale=float(sm_scale),
                )

                # lse for this head
                lse_per_head = torch.empty((1,), dtype=torch.float32, device=device)
                compute_lse_kernel[(1,)](
                    logits_scaled, lse_per_head,
                    L=L,
                )

                # softmax
                attn = torch.empty((L,), dtype=torch.float32, device=device)
                compute_softmax_kernel[(1,)](
                    logits_scaled, lse_per_head, attn,
                    L=L,
                )

                # GEMV to produce output vector for this head
                out_vec = torch.empty((K,), dtype=torch.float32, device=device)
                gemv_out_kernel[(1,)](
                    attn, ckv_cache, tok_idx, out_vec,
                    L=L, K=K,
                )

                # Store output and lse
                output[q_abs, h, :] = out_vec.to(torch.bfloat16)
                lse[q_abs, h] = lse_per_head[0]

    return output, lse


# For completeness, provide ModelNew as requested
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        return run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
