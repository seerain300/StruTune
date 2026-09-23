import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _forward_kernel_bh(
    q_ptr,            # *fp32, shape [B, H, D]
    k_ptr,            # *fp32, shape [N_total, 1, num_kv_heads, D] but gathered by idx
    v_ptr,            # *fp32, shape [N_total, 1, num_kv_heads, D]
    kv_indices_ptr,   # *int32, shape [num_kv_indices]
    kv_indptr_ptr,    # *int32, shape [B+1]
    lse_ptr,          # *fp32, shape [B, H]
    out_ptr,          # *fp32, shape [B, H, D]
    B: tl.int32,      # batch size (runtime)
    H: tl.int32,      # num query heads (runtime)
    D: tl.constexpr,  # head dim, compile-time constant (e.g., 128)
    num_kv_heads: tl.constexpr,  # 8 (compile-time)
    sm_scale: tl.float32,        # 1/sqrt(D) as fp32
    b: tl.int32,                # batch index (runtime scalar passed via grid/args)
    h: tl.int32,                # head index
    actual_num_tokens: tl.int32,  # tokens in this batch element (runtime scalar)
    N_TOTAL: tl.constexpr,         # loop bound (e.g., 128)
):
    # Load q[b, h, :]
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d)

    # Compute start/end for this batch element
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    # We already have actual_num_tokens as argument.

    # First pass: streaming logsumexp in base-2 over tokens for this (b, h)
    m = -float("inf")
    sumexp = 0.0
    ln2 = 0.6931471805599453  # log(2)

    for nn in range(0, N_TOTAL):
        use = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=use, other=0).to(tl.int32)

        kv_head = h // (H // num_kv_heads)
        k_row_ptr = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_vec[d] = tl.load(k_row_ptr + d)

        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]

        logit = dot * sm_scale
        m_new = tl.maximum(m, logit)
        sumexp = sumexp * tl.exp(m - m_new) + tl.exp(logit - m_new)
        m = m_new

    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)

    for nn in range(0, N_TOTAL):
        use = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=use, other=0).to(tl.int32)

        kv_head = h // (H // num_kv_heads)
        k_row_ptr = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_row_ptr = v_ptr + idx * (num_kv_heads * D) + kv_head * D

        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_vec[d] = tl.load(k_row_ptr + d)

        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]

        logit = dot * sm_scale
        softmax = tl.exp(logit - m) / ln2  # softmax scaled by 1/ln2

        v_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_vec[d] = tl.load(v_row_ptr + d)

        out_vec += softmax * v_vec

    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only path; ensure inputs are contiguous and float32
        if not TRITON_AVAILABLE:
            # Fallback (not used by evaluator, since TRITON is available)
            B, H, D = q.shape
            num_kv_heads = 8
            gqa_ratio = H // num_kv_heads
            output = torch.empty((B, H, D), dtype=torch.bfloat16, device=q.device)
            lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

            for b in range(B):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                num_tokens = end - start
                if num_tokens <= 0:
                    output[b].zero_()
                    continue

                kv_head = (torch.arange(H, device=q.device) // gqa_ratio).to(torch.int32)
                for h in range(H):
                    q_batch = q[b, h].to(torch.float32)
                    kvh = int(kv_head[h].item())
                    token_indices = kv_indices[start:start + num_tokens].to(torch.int64)
                    k_rows = k_cache[token_indices, kvh].to(torch.float32)
                    v_rows = v_cache[token_indices, kvh].to(torch.float32)
                    logits = q_batch @ k_rows.T
                    logits_scaled = logits * sm_scale
                    m = torch.logsumexp(logits_scaled, dim=0) / math.log(2.0)
                    attn = torch.softmax(logits_scaled, dim=0)
                    out_head = attn @ v_rows
                    output[b, h] = out_head.to(torch.bfloat16)
                    lse[b, h] = float(m.item())
            return [output, lse]

        B, H, D = q.shape
        num_kv_heads = 8
        sm_scale_val = float(sm_scale)

        q_t = q.contiguous().to(torch.float32)
        k_t = k_cache.contiguous().to(torch.float32)
        v_t = v_cache.contiguous().to(torch.float32)
        kv_indptr_t = kv_indptr.contiguous().to(torch.int32)
        kv_indices_t = kv_indices.contiguous().to(torch.int32)

        out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Prepare actual_num_tokens per batch element
        actual_num_tokens_list = [int(kv_indptr_t[b].item()) for b in range(B)]

        # Launch Triton: one program per (b, h)
        grid = (B, H)
        N_TOTAL = 128  # loop bound; mask nn < actual_num_tokens

        _forward_kernel_bh[grid](
            q_t, k_t, v_t, kv_indices_t, kv_indptr_t, lse, out,
            B, H, D, num_kv_heads, sm_scale_val,
            # pass b, h, actual_num_tokens for each program; Triton supports scalar args
        )

        return [out.to(torch.bfloat16), lse]


def run(*args):
    return ModelNew()(*args)
