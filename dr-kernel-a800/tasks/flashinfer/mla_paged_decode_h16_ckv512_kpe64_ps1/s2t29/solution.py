@triton.jit
def _compute_output_and_lse_per_bh(
    q_nope_ptr,            # *bfloat16, flattened [B*H*D1]
    q_pe_ptr,              # *bfloat16, flattened [B*H*D2]
    ckv_cache_ptr,         # *bfloat16, flattened [N*D1]
    kpe_cache_ptr,         # *bfloat16, flattened [N*D2]
    kv_indices,            # *int32, shape [L_tokens]
    lse_ptr,               # *float32, flattened [B*H]
    out_ptr,               # *float32, flattened [B*H*D1]
    H: tl.constexpr,       # num heads
    D1: tl.constexpr,      # head_dim_ckv
    D2: tl.constexpr,      # head_dim_kpe
    L_tokens: tl.constexpr,# number of tokens for this batch element
    sm_scale: tl.constexpr,# scaling factor
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    qn_base = q_nope_ptr + h * D1
    qp_base = q_pe_ptr + h * D2
    out_base = out_ptr + b * H * D1 + h * D1
    lse_offset = b * H + h

    # Initialize scalar accumulators for lse
    token_max = -float("inf")
    token_sum = 0.0

    # Output initialization
    for d in tl.static_range(0, D1):
        tl.store(out_base + d, 0.0)

    for t in range(0, L_tokens):
        qn = tl.load(qn_base + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(qp_base + tl.arange(0, D2)).to(tl.float32)  # [D2]

        idx = tl.load(kv_indices + t).to(tl.int32)
        Kc_row = tl.load(ckv_cache_ptr + idx * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        Kp_row = tl.load(kpe_cache_ptr + idx * D2 + tl.arange(0, D2)).to(tl.float32)  # [D2]

        dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
        dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
        logits_scalar = (dot1 + dot2) * sm_scale

        # Accumulate output
        for d in tl.static_range(0, D1):
            tl.store(out_base + d, tl.load(out_base + d) + logits_scalar * Kc_row[d])

        # Update scalar lse
        token_max = tl.maximum(token_max, logits_scalar)
        token_sum += tl.exp(logits_scalar - token_max)

    # Compute lse = logsumexp(logits_scaled) / ln(2)
    # torch.nn.functional.logsumexp uses natural log, so we compute log(token_sum) + token_max
    lse_val = tl.log(token_sum) + token_max
    # Write lse
    tl.store(lse_ptr + lse_offset, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        D1 = q_nope.shape[2]
        D2 = q_pe.shape[2]
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"

        # Compute L_tokens per batch element: len_indptr[b+1] - len_indptr[b]
        L_tokens_list = (kv_indptr[1:] - kv_indptr[:-1]).tolist()
        assert len(L_tokens_list) == B, "kv_indptr length must be B+1"

        # Allocate output buffer (float32 for compute)
        out_flat = torch.empty(B * H * D1, dtype=torch.float32, device=device)
        lse = torch.empty((B * H), dtype=torch.float32, device=device)  # [B*H]

        grid = (B, H)
        _compute_output_and_lse_per_bh[grid](
            q_nope.to(torch.bfloat16).view(-1),          # [B*H*D1]
            q_pe.to(torch.bfloat16).view(-1),            # [B*H*D2]
            ckv_cache.to(torch.bfloat16).view(-1),       # [N*D1]
            kpe_cache.to(torch.bfloat16).view(-1),       # [N*D2]
            kv_indices,                                   # [L_tokens_total] -- environment provides single kv_indices tensor for all batches
            lse,                                          # [B*H]
            out_flat,                                     # [B*H*D1]
            H=H, D1=D1, D2=D2, L_tokens=L_tokens_list[0], sm_scale=float(sm_scale),
        )

        output = out_flat.view(B, H, D1).to(torch.bfloat16)
        lse = lse.view(B, H)
        return output, lse


def run(*args):
    return ModelNew()(*args)
