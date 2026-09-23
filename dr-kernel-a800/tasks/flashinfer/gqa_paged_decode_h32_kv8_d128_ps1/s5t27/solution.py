import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def gqa_row_scalar_kernel(
    q_ptr,                # *bf16, [B, H, D]
    k_ptr,                # *bf16, [Np, 1, D]
    v_ptr,                # *bf16, [Np, 1, D]
    token_indices_ptr,    # *int32, [B, max_tokens]
    num_tokens_ptr,       # *int32, [B]
    out_ptr,              # *fp32, [B, H, D]
    lse_ptr,              # *fp32, [B, H]
    sm_scale,             # fp32 scalar
    B, H, D, K,           # runtime ints
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Load q vector for this (b, h) as fp32
    q_offset = pid_b * H * D + pid_h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

    # Number of tokens for this batch b
    num_tokens = tl.load(num_tokens_ptr + pid_b).to(tl.int32)

    # GQA mapping: kv_head = h // (H // K)
    gqa_ratio = H // K
    kv_head = pid_h // gqa_ratio

    # Pass 1: compute sum_exp = sum(exp((q·k) * sm_scale)) across tokens
    sum_exp = 0.0
    for t in range(0, 1024):
        if t < num_tokens:
            idx = tl.load(token_indices_ptr + pid_b * num_tokens + t).to(tl.int32)
            k_offset = idx * K * D + kv_head * D
            k_row = tl.load(k_ptr + k_offset).to(tl.float32)  # [D]
            dot = 0.0
            for d in range(0, 128):
                dot += q_vec[d] * k_row[d]
            logits_scaled = dot * sm_scale
            sum_exp += tl.exp(logits_scaled)
        else:
            break

    # LSE in base-2
    ln2 = 0.6931471805599453
    lse_base2 = tl.log(sum_exp) / ln2
    tl.store(lse_ptr + pid_b * H + pid_h, lse_base2)

    # Pass 2: accumulate output vector
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(0, 1024):
        if t < num_tokens:
            idx = tl.load(token_indices_ptr + pid_b * num_tokens + t).to(tl.int32)
            k_offset = idx * K * D + kv_head * D
            k_row = tl.load(k_ptr + k_offset).to(tl.float32)  # [D]
            v_offset = idx * K * D + kv_head * D
            v_row = tl.load(v_ptr + v_offset).to(tl.float32)  # [D]

            dot = 0.0
            for d in range(0, 128):
                dot += q_vec[d] * k_row[d]
            logits_scaled = dot * sm_scale
            attn = tl.exp((logits_scaled - lse_base2) * sm_scale)
            for d in range(0, 128):
                out_vec[d] += attn * v_row[d]
        else:
            break

    # Store output
    out_offset = pid_b * H * D + pid_h * D
    tl.store(out_ptr + out_offset, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Fallback to PyTorch if Triton/CUDA not available
        if (not TRITON_AVAILABLE) or (not q.is_cuda):
            batch_size, num_qo_heads, head_dim = q.shape
            _, _, num_kv_heads, _ = k_cache.shape
            device = q.device

            output = torch.empty(
                (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
            )
            lse = torch.empty(
                (batch_size, num_qo_heads), dtype=torch.float32, device=device
            )

            assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

            k_cache_flat = k_cache.squeeze(1).to(torch.float32)
            v_cache_flat = v_cache.squeeze(1).to(torch.float32)

            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                num_tokens = end - start
                if num_tokens <= 0:
                    output[b].zero_()
                    lse[b] = -float("inf")
                    continue

                token_indices = kv_indices[start:end].to(torch.long)

                q_batch = q[b].to(torch.float32)  # [H, D] via q[b, h, :]
                for h in range(num_qo_heads):
                    kv_head = h // (num_qo_heads // num_kv_heads)
                    q_head = q_batch[h]  # [D]
                    k_head = k_cache_flat[token_indices, kv_head]  # [num_tokens, D]
                    v_head = v_cache_flat[token_indices, kv_head]  # [num_tokens, D]

                    logits = torch.matmul(q_head, k_head.T)  # [num_tokens]
                    logits_scaled = logits * sm_scale

                    lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                    attn = torch.softmax(logits_scaled, dim=-1)  # [num_tokens]
                    out_head = torch.matmul(attn, v_head)  # [D]
                    output[b, h] = out_head.to(torch.bfloat16)

            return output, lse

        # Triton path
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        device = q.device

        # Precompute num_tokens per batch and token indices per batch (PyTorch)
        num_tokens_per_batch = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)  # [B]
        token_indices_list = [kv_indices[start:end].to(torch.int32) for start, end in
                              zip(kv_indptr[:-1].cpu(), kv_indptr[1:].cpu())]

        # Flatten token indices to [B, max_tokens], padding per batch to max_tokens
        max_tokens = int(max(num_tokens_per_batch).item())
        token_indices_flat = []
        for b in range(batch_size):
            idx_b = token_indices_list[b]
            pad = max_tokens - idx_b.numel()
            if pad > 0:
                idx_b = torch.cat([idx_b, torch.full((pad,), -1, dtype=torch.int32)], dim=0)
            token_indices_flat.append(idx_b)
        token_indices_flat = torch.stack(token_indices_flat, dim=0)  # [B, max_tokens]

        # Allocate outputs
        output_fp32 = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse_fp32 = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel
        grid = (batch_size, num_qo_heads)
        gqa_row_scalar_kernel[grid](
            q, k_cache, v_cache,
            token_indices_flat, num_tokens_per_batch,
            output_fp32, lse_fp32,
            float(sm_scale),
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads,
            num_warps=1, num_stages=1,
        )

        # Cast output to bfloat16 to match original
        output = output_fp32.to(torch.bfloat16)
        return output, lse_fp32


def run(*args):
    return ModelNew()(*args)
