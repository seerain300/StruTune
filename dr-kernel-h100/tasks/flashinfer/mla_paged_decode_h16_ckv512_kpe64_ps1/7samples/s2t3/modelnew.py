import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute output and lse for a single batch element b and head h.
# Assumes:
# - q_nope_ptr points to q_nope[b, h, :] contiguous
# - q_pe_ptr points to q_pe[b, h, :] contiguous
# - Kc_ptr points to Kc_all[indices[0:L_tokens], :] contiguous (shape [L_tokens, Dc])
# - Kp_ptr points to Kp_all[indices[0:L_tokens], :] contiguous (shape [L_tokens, Dp])
# - out_ptr points to out[b, h, :] contiguous (we store bfloat16)
# - lse_ptr points to lse[b, h] contiguous (we store float32)
@triton.jit
def _compute_single_head(
    q_nope_ptr, q_pe_ptr,
    Kc_ptr, Kp_ptr,
    out_ptr, lse_ptr,
    B: tl.constexpr, H: tl.constexpr,
    Dc: tl.constexpr, Dp: tl.constexpr,
    L_tokens: tl.constexpr,
    sm_scale: tl.float32,
    LN2: tl.float32  # natural log of 2
):
    # We don't need B/H here since we launch one program per (b,h), but we keep args for clarity.

    # Load qn and qp (each of length Dc or Dp)
    # qn: [Dc]
    qn = tl.zeros([Dc], dtype=tl.float32)
    # qn is contiguous at q_nope_ptr, stride 1 across dimension
    for i in tl.static_range(0, Dc):
        qn[i] = tl.load(q_nope_ptr + i)

    # qp: [Dp]
    qp = tl.zeros([Dp], dtype=tl.float32)
    for i in tl.static_range(0, Dp):
        qp[i] = tl.load(q_pe_ptr + i)

    # Compute logits for each token
    # logits: [L_tokens]
    logits = tl.zeros([L_tokens], dtype=tl.float32)
    # sum_qn_Kc: accumulate dot product qn @ Kc.T per token
    # We loop over Dc in chunks to reduce register pressure (though Dc=512 here)
    for t in tl.static_range(0, L_tokens):
        # sum_qn_Kc[t] = sum_i qn[i] * Kc[t, i]
        sum_qn_Kc = 0.0
        for i in tl.static_range(0, Dc):
            # Kc[t, i] at row t, column i. We can construct pointer by:
            # Kc_ptr offset = t * Dc + i
            k_ci = tl.load(Kc_ptr + t * Dc + i)
            sum_qn_Kc += qn[i] * k_ci

        # sum_qp_Kp[t] = sum_j qp[j] * Kp[t, j]
        sum_qp_Kp = 0.0
        for j in tl.static_range(0, Dp):
            k_pj = tl.load(Kp_ptr + t * Dp + j)
            sum_qp_Kp += qp[j] * k_pj

        logits[t] = sum_qn_Kc + sum_qp_Kp

    # Scale logits
    logits_scaled = logits * sm_scale

    # Compute lse = logsumexp(logits_scaled) / ln(2)
    # Numerically stable: max, then sum exp, then lse
    m = logits_scaled[0]
    for t in tl.static_range(1, L_tokens):
        m = tl.maximum(m, logits_scaled[t])

    sum_exp = 0.0
    for t in tl.static_range(0, L_tokens):
        sum_exp += tl.exp(logits_scaled[t] - m)

    lse_val = m + tl.log(sum_exp) / LN2  # base-2 logsumexp

    # Store lse
    tl.store(lse_ptr, lse_val)  # single scalar per (b,h)

    # Compute attention vector
    attn = tl.zeros([L_tokens], dtype=tl.float32)
    for t in tl.static_range(0, L_tokens):
        attn[t] = tl.exp(logits_scaled[t] - lse_val) / LN2

    # Compute output vector for this head: out[h, :] = sum_t attn[t] * Kc[t, :]
    out_vec = tl.zeros([Dc], dtype=tl.float32)
    for i in tl.static_range(0, Dc):
        # out_vec[i] = sum_t attn[t] * Kc[t, i]
        sum_attn_times_Kci = 0.0
        for t in tl.static_range(0, L_tokens):
            k_ci = tl.load(Kc_ptr + t * Dc + i)
            sum_attn_times_Kci += attn[t] * k_ci
        out_vec[i] = sum_attn_times_Kci

    # Store output in bfloat16
    # out_ptr points to out[b, h, :], contiguous of length Dc
    for i in tl.static_range(0, Dc):
        tl.store(out_ptr + i, tl.cast(out_vec[i], tl.bfloat16))


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    """
    Triton-only forward that returns (output, lse).
    """
    assert q_nope.ndim == 3 and q_pe.ndim == 3
    B, H, Dc = q_nope.shape
    assert H == 16, "num_qo_heads must be 16"
    assert Dc == 512, "head_dim_ckv must be 512"
    Dp = q_pe.shape[-1]
    assert Dp == 64, "head_dim_kpe must be 64"

    # Prepare Kc_all and Kp_all (shape [num_pages, Dc] and [num_pages, Dp])
    # They are already squeezed in the original run. We'll use their full arrays indexed by tokens.
    Kc_all = ckv_cache  # [num_pages, 1, Dc] squeezed => [num_pages, Dc]
    Kp_all = kpe_cache  # [num_pages, 1, Dp] squeezed => [num_pages, Dp]
    # Ensure they are contiguous on the right device
    Kc_all = Kc_all.contiguous().to(torch.float32)
    Kp_all = Kp_all.contiguous().to(torch.float32)

    # Output buffers
    out = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
    lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=q_nope.device)

    # Process each batch element and head
    LN2 = math.log(2.0)
    # We'll launch one program per (b, h)
    for b in range(B):
        # Determine token range
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L_tokens = page_end - page_beg

        if L_tokens <= 0:
            # No valid tokens for this batch element
            # Leave out[b, :] zeros and lse[b] = -inf (already initialized)
            continue

        # Gather token indices for this batch element: tok_idx = kv_indices[page_beg:page_end]
        tok_idx = kv_indices[page_beg:page_end].to(torch.long).contiguous()

        # Gather Kc and Kp rows
        Kc = Kc_all[tok_idx]  # [L_tokens, Dc]
        Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

        # Pointers for q_nope[b, h, :] and q_pe[b, h, :]
        # We pass contiguous vectors for each head h
        for h in range(H):
            # q_nope[b, h, :]
            qn_ptr = q_nope[b, h, :].contiguous().to(torch.float32)
            # q_pe[b, h, :]
            qp_ptr = q_pe[b, h, :].contiguous().to(torch.float32)

            # Output vector pointer: out[b, h, :]
            out_ptr = out[b, h, :].contiguous()

            # lse scalar pointer: lse[b, h]
            lse_ptr = lse[b, h]

            # Kc and Kp rows as contiguous (already contiguous due to gather)
            Kc_ptr = Kc  # [L_tokens, Dc] contiguous
            Kp_ptr = Kp  # [L_tokens, Dp] contiguous

            # Launch Triton kernel for this (b, h)
            _compute_single_head[
                (1, 1)
            ](
                qn_ptr, qp_ptr,
                Kc_ptr, Kp_ptr,
                out_ptr, lse_ptr,
                B, H,
                Dc, Dp,
                L_tokens,
                sm_scale,
                LN2,
                num_warps=4,  # heuristic; can be tuned
                num_stages=2  # heuristic; can be tuned
            )

    return out, lse


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: no torch ops in the kernel-launch path.
        return _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)

    # Optional: keep original Model for reference/tests
    class Model(torch.nn.Module):
        @torch.no_grad()
        def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
            batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
            head_dim_kpe = q_pe.shape[-1]
            assert num_qo_heads == 16
            assert head_dim_ckv == 512
            assert head_dim_kpe == 64

            Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, Dc]
            Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, Dp]

            output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv),
                                 dtype=torch.bfloat16, device=q_nope.device)
            lse = torch.full((batch_size, num_qo_heads), -float("inf"),
                             dtype=torch.float32, device=q_nope.device)

            for b in range(batch_size):
                page_beg = int(kv_indptr[b].item())
                page_end = int(kv_indptr[b + 1].item())

                if page_beg >= page_end:
                    output[b].zero_()
                    continue

                L_tokens = page_end - page_beg
                tok_idx = kv_indices[page_beg:page_end].to(torch.long)

                Kc = Kc_all[tok_idx]  # [L_tokens, Dc]
                Kp = Kp_all[tok_idx]  # [L_tokens, Dp]
                qn = q_nope[b].to(torch.float32)  # [H, Dc]
                qp = q_pe[b].to(torch.float32)    # [H, Dp]

                # logits per head
                for h in range(num_qo_heads):
                    logits = qn[h] @ Kc.T + qp[h] @ Kp.T  # [L_tokens]
                    logits_scaled = logits * sm_scale
                    lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                    attn = torch.softmax(logits_scaled, dim=-1)  # [L_tokens]
                    out[b, h, :] = attn @ Kc  # [Dc]
                    out[b, h, :] = out[b, h, :].to(torch.bfloat16)

            return output, lse