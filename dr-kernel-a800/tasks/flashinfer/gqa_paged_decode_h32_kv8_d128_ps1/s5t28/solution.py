import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def gqa_row_kernel_scalar(
    q_ptr,                 # *bf16, shape [B, H, D]
    k_ptr,                 # *bf16, shape [Np, 1, D] (Np is number of cache slots; here treated as a flat buffer)
    v_ptr,                 # *bf16, shape [Np, 1, D]
    token_indices_ptr,     # *int32, shape [B * MAX_TOKENS], but we index using (b * MAX_TOKENS + t)
    output_ptr,            # *float32, shape [B, H, D]
    lse_ptr,               # *float32, shape [B, H]
    sm_scale,              # float32 scalar
    B: tl.int32,           # batch size
    H: tl.int32,           # num query heads
    D: tl.int32,           # head dim (128)
    K: tl.int32,           # num kv heads (8)
):
    # program id maps to (b, h)
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H
    if b >= B or h >= H:
        return

    # Load q[b, h, :]
    q_offset = b * H * D + h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

    # Compute kv_head for GQA: kv_head = h // (H // K) = h // 4
    # K is passed as scalar; compute ratio via integer division.
    gqa_ratio = H // K
    kv_head = h // gqa_ratio  # integer division

    # Pass 1: compute sum_exp = sum(exp((q·k) * sm_scale)) across tokens for this (b, h)
    sum_exp = 0.0
    # Fixed iteration; guard with num_tokens[b]. We precompute num_tokens on host and pass num_tokens_ptr.
    # Note: Triton does not support dynamic loops; use fixed iterations with scalar guard.
    for t in range(1024):
        # Load num_tokens for this batch
        num_tokens_offset = b  # num_tokens is a 1D tensor of length B
        num_tokens = tl.load(num_tokens_offset + token_indices_ptr).to(tl.int32)
        if t < num_tokens:
            # Load token index
            idx = tl.load(token_indices_ptr + b * 1024 + t).to(tl.int32)
            # Load k_row[idx, kv_head, :] and v_row[idx, kv_head, :]
            k_base = idx * K * D + kv_head * D
            v_base = idx * K * D + kv_head * D
            k_row = tl.load(k_ptr + k_base, mask=True).to(tl.float32)  # [D]
            v_row = tl.load(v_ptr + v_base, mask=True).to(tl.float32)  # [D]

            # Compute dot product
            dot = 0.0
            for d in range(D):
                dot += q_vec[d] * k_row[d]
            # Scale logits
            logits_scaled = dot * sm_scale
            sum_exp += tl.exp(logits_scaled)

    # Compute lse in base-2: logsumexp / ln(2)
    lse_val = tl.log(sum_exp) / 1.4426950408889634  # 1 / ln(2)

    # Pass 2: accumulate output vector out_vec = sum_t attn_t * v_row_t
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(1024):
        num_tokens = tl.load(num_tokens_offset + token_indices_ptr).to(tl.int32)
        if t < num_tokens:
            idx = tl.load(token_indices_ptr + b * 1024 + t).to(tl.int32)
            k_base = idx * K * D + kv_head * D
            v_base = idx * K * D + kv_head * D
            k_row = tl.load(k_ptr + k_base, mask=True).to(tl.float32)  # [D]
            v_row = tl.load(v_ptr + v_base, mask=True).to(tl.float32)  # [D]

            dot = 0.0
            for d in range(D):
                dot += q_vec[d] * k_row[d]
            logits_scaled = dot * sm_scale
            attn = tl.exp((logits_scaled - lse_val) * sm_scale)

            for d in range(D):
                out_vec[d] += attn * v_row[d]

    # Store output and lse
    out_offset = b * H * D + h * D
    tl.store(output_ptr + out_offset, out_vec)
    tl.store(lse_ptr + b * H + h, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton is available and tensors are on CUDA
        if not TRITON_AVAILABLE or not q.is_cuda:
            # Fallback: original PyTorch behavior (kept for robustness, but Triton path is preferred)
            batch_size, num_qo_heads, head_dim = q.shape
            _, _, num_kv_heads, _ = k_cache.shape
            device = q.device
            output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
            lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)
            gqa_ratio = num_qo_heads // num_kv_heads
            q32 = q.to(torch.float32)
            k32 = k_cache.squeeze(1).to(torch.float32)
            v32 = v_cache.squeeze(1).to(torch.float32)
            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                if start >= end:
                    output[b].zero_()
                    lse[b].fill_(-float("inf"))
                    continue
                token_indices = kv_indices[start:end].to(torch.long)
                num_tokens = token_indices.shape[0]
                if num_tokens == 0:
                    output[b].zero_()
                    lse[b].fill_(-float("inf"))
                    continue
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    qh = q32[b, h]
                    kbatch = k32[token_indices, kv_head]
                    vbatch = v32[token_indices, kv_head]
                    logits = torch.matmul(qh, kbatch.transpose(0, 1))
                    lse[b, h] = torch.logsumexp(logits * sm_scale, dim=-1) / math.log(2.0)
                    attn = torch.softmax(logits * sm_scale, dim=-1)
                    out_vec = torch.matmul(attn, vbatch)
                    output[b, h] = out_vec.to(torch.bfloat16)
            return output, lse

        # Triton path: compute everything in kernels
        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape

        # Precompute on host (PyTorch): num_tokens and token_indices per batch
        num_tokens_per_batch = (kv_indptr[1:] - kv_indptr[:batch_size]).to(torch.int32)
        # Flatten token indices per batch segment
        # Build a single token_indices tensor of length B*MAX_TOKENS; we will index it via (b*MAX_TOKENS + t)
        # but guard with num_tokens[b]. To do this, we need to compute the offsets for each batch:
        # We allocate token_indices_flat and fill with zeros; then fill valid entries.
        token_indices_flat = torch.empty(batch_size * 1024, dtype=torch.int32, device=q.device)
        start = 0
        for b in range(batch_size):
            end = kv_indptr[b + 1].item()
            start_b = kv_indptr[b].item()
            # Number of valid tokens for this batch
            num_tok = (end - start_b)
            # Fill the valid positions for this batch in the flattened array
            if num_tok > 0:
                # Write kv_indices[start_b:start_b+num_tok] into positions b*1024 + 0..num_tok-1
                token_indices_flat[b * 1024 : (b * 1024 + num_tok)] = kv_indices[start_b : start_b + num_tok]
            # The rest are invalid; kernel guards with num_tokens[b]
        # Now launch Triton kernel

        # Ensure inputs are contiguous
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Output buffers: fp32 for compute, convert to bfloat16 after kernel
        output_fp32 = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse_fp32 = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

        # Grid: one program per (b, h)
        grid = (batch_size * num_qo_heads,)

        gqa_row_kernel_scalar[grid](
            q, k_cache, v_cache, token_indices_flat, output_fp32, lse_fp32, float(sm_scale),
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads,  # no tl.constexpr args
            num_warps=2, num_stages=1,
        )

        # Convert to requested dtype
        output = output_fp32.to(torch.bfloat16)
        return output, lse_fp32


def run(*args):
    return ModelNew()(*args)
