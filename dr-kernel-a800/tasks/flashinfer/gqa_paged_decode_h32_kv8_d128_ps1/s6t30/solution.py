import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (b, h)
@triton.jit
def _forward_kernel_bh(
    q_ptr,             # *fp16, shape [B, H, D]
    k_ptr,             # *fp16, shape [N_total, num_kv_heads, D]
    v_ptr,             # *fp16, shape [N_total, num_kv_heads, D]
    kv_indices_ptr,    # *int32, shape [num_kv_indices]
    kv_indptr_ptr,     # *int32, shape [B+1]
    lse_ptr,           # *fp32, shape [B, H]
    out_ptr,           # *fp32, shape [B, H, D]
    B: tl.constexpr,   # batch size
    H: tl.constexpr,   # num query heads
    D: tl.constexpr,   # head dim
    num_kv_heads: tl.constexpr,  # number of kv heads (8)
    sm_scale,          # scalar float32
    N_TOTAL: tl.constexpr,        # loop bound (e.g., 128)
):
    b = tl.program_id(0)  # batch index
    h = tl.program_id(1)  # head index

    # Load q vector for (b, h)
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d).to(tl.float32)

    # Compute token range for this batch element
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start

    # GQA mapping: kv_head = h // 4
    kv_head = h // 4

    # First pass: compute logsumexp in base-2
    m = -float('inf')
    sumexp = 0.0
    ln2 = 0.6931471805599453

    for nn in range(0, N_TOTAL):
        mask = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask, other=0).to(tl.int32)

        # k_row = k[idx, kv_head, :]
        k_row_ptr = k_ptr + idx * num_kv_heads * D + kv_head * D
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_row_ptr + d).to(tl.float32)
            k_row[d] = k_val

        # dot = q_vec · k_row
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]

        logit = dot * sm_scale
        new_m = tl.maximum(m, logit)
        sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

    # lse for this (b, h) in base-2
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        mask = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask, other=0).to(tl.int32)

        v_row_ptr = v_ptr + idx * num_kv_heads * D + kv_head * D
        v_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_val = tl.load(v_row_ptr + d).to(tl.float32)
            v_row[d] = v_val

        k_row_ptr = k_ptr + idx * num_kv_heads * D + kv_head * D
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_row_ptr + d).to(tl.float32)
            k_row[d] = k_val

        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]
        logit = dot * sm_scale

        softmax = tl.exp(logit - m) / (sumexp * ln2)
        for d in range(0, D):
            out_vec[d] += softmax * v_row[d]

    # Store final output for this (b, h)
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Fallback to PyTorch if Triton unavailable or tensors not on CUDA
        if not TRITON_AVAILABLE or q.device.type != "cuda":
            # Pure PyTorch implementation (matches original)
            B, H, D = q.shape
            assert H == 32, "This implementation assumes num_qo_heads == 32"
            assert D == 128, "This implementation assumes head_dim == 128"
            num_kv_heads = 8

            output = torch.zeros((B, H, D), dtype=torch.bfloat16, device=q.device)
            lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=q.device)

            gqa_ratio = H // num_kv_heads

            # Compute per (b, h)
            for b in range(B):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                if start >= end:
                    output[b].zero_()
                    continue

                token_indices = kv_indices[start:end].to(torch.long)
                num_tokens = token_indices.shape[0]
                if num_tokens == 0:
                    output[b].zero_()
                    continue

                # GQA mapping
                q_b = q[b].to(torch.float32)  # [H, D]
                for h in range(H):
                    kv_head = h // gqa_ratio

                    q_vec = q_b[h]  # [D]
                    k_rows = k_cache[token_indices, kv_head]  # [num_tokens, D]
                    v_rows = v_cache[token_indices, kv_head]  # [num_tokens, D]

                    logits = torch.matmul(q_vec, k_rows.transpose(0, 1))  # [num_tokens]
                    logits_scaled = logits * sm_scale

                    # base-2 LSE
                    m = torch.max(logits_scaled)
                    sumexp = torch.sum(torch.exp(logits_scaled - m))
                    lse[b, h] = (m + torch.log(sumexp)) / 0.6931471805599453

                    attn = torch.softmax(logits_scaled, dim=0)
                    out_vec = torch.matmul(attn, v_rows)  # [D]
                    output[b, h] = out_vec.to(torch.bfloat16)

            return output, lse

        # Triton path: ensure inputs are contiguous and on CUDA
        B, H, D = q.shape
        # We can only run Triton kernel for the expected config
        if H != 32 or D != 128:
            # Fallback to PyTorch for unsupported shapes
            # (You can relax this by adding more kernels, but here we keep it strict.)
            output = torch.zeros((B, H, D), dtype=torch.bfloat16, device=q.device)
            lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=q.device)
            # Implement the same PyTorch logic as above (see above for code)
            # For brevity, we reuse the PyTorch block here.
            gqa_ratio = H // 8
            for b in range(B):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                token_indices = kv_indices[start:end].to(torch.long)
                num_tokens = token_indices.shape[0]
                q_b = q[b].to(torch.float32)
                for h in range(H):
                    kv_head = h // gqa_ratio
                    q_vec = q_b[h]
                    k_rows = k_cache[token_indices, kv_head].to(torch.float32)
                    v_rows = v_cache[token_indices, kv_head].to(torch.float32)
                    logits = torch.matmul(q_vec, k_rows.transpose(0, 1))
                    logits_scaled = logits * sm_scale
                    m = torch.max(logits_scaled)
                    sumexp = torch.sum(torch.exp(logits_scaled - m))
                    lse[b, h] = (m + torch.log(sumexp)) / 0.6931471805599453
                    attn = torch.softmax(logits_scaled, dim=0)
                    out_vec = torch.matmul(attn, v_rows)
                    output[b, h] = out_vec.to(torch.bfloat16)
            return output, lse

        # Prepare pointers
        q_fp16 = q  # keep dtype bfloat16; kernel loads and converts to fp32 internally
        k_cache_fp16 = k_cache  # [N_total, 1, num_kv_heads, D]
        v_cache_fp16 = v_cache  # same shape
        # Flatten kv indices to 1D in original order (already provided)
        kv_indices_i32 = kv_indices
        kv_indptr_i32 = kv_indptr

        # Output buffers (fp32 for accumulation, will cast to bfloat16 at end)
        out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q_fp16, k_cache_fp16, v_cache_fp16, kv_indices_i32, kv_indptr_i32, lse, out,
            B, H, D, 8, float(sm_scale), 128,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 as original code returns bfloat16
        out_bf16 = out.to(torch.bfloat16)
        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
