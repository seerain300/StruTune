import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def fused_gqa_row_scalar_kernel(
    q_ptr,          # *bf16, [B, H, D]
    k_ptr,          # *bf16, [Np, 1, K, D]
    v_ptr,          # *bf16, [Np, 1, K, D]
    indptr_ptr,     # *int32, [B+1]
    kv_indices_ptr, # *int32, [num_kv_indices]
    out_ptr,        # *float32, [B, H, D]
    lse_ptr,        # *float32, [B, H]
    sm_scale,       # float32 scalar
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,     # head_dim
    K: tl.constexpr,     # num_kv_heads
    gqa_ratio: tl.constexpr,  # = H // K = 4
    MAX_TOKENS: tl.constexpr, # fixed iteration cap for tokens
):
    pid_b = tl.program_id(0)  # batch id
    pid_h = tl.program_id(1)  # query head id

    if (pid_b >= B) or (pid_h >= H):
        return

    # Compute q vector for this (b, h)
    q_offset = pid_b * H * D + pid_h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

    # GQA mapping: kv head per query head
    kv_head = pid_h // gqa_ratio

    # Load indptr[start, end] for this batch element
    start = tl.load(indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start  # int32

    # Pass 1: compute sum_exp = sum(exp((q·k) * sm_scale)) across tokens
    sum_exp = 0.0
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        tok_idx = tl.load(kv_indices_ptr + start + t).to(tl.int32)
        k_offset = tok_idx * K * D + kv_head * D
        k_vec = tl.load(k_ptr + k_offset).to(tl.float32)  # [D]
        dot = 0.0
        for d in range(D):
            dot += q_vec[d] * k_vec[d]
        sum_exp += tl.exp(dot * sm_scale)

    # lse in base-2
    lse = tl.log(sum_exp) / math.log(2.0)
    # Store lse for this (b, h)
    tl.store(lse_ptr + pid_b * H + pid_h, lse)

    # Pass 2: compute output vector for this (b, h)
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        tok_idx = tl.load(kv_indices_ptr + start + t).to(tl.int32)
        k_offset = tok_idx * K * D + kv_head * D
        k_vec = tl.load(k_ptr + k_offset).to(tl.float32)  # [D]
        v_offset = tok_idx * K * D + kv_head * D
        v_vec = tl.load(v_ptr + v_offset).to(tl.float32)  # [D]

        dot = 0.0
        for d in range(D):
            dot += q_vec[d] * k_vec[d]
        attn = tl.exp((dot - lse) * sm_scale)  # scaled attention

        for d in range(D):
            out_vec[d] += attn * v_vec[d]

    # Store output vector for this (b, h)
    out_offset = pid_b * H * D + pid_h * D
    tl.store(out_ptr + out_offset, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-optimized forward:
        - Computes per-(b, h) attention using Triton kernels (no PyTorch matmul/softmax on tensors).
        - Ensures Triton kernels are actually launched.
        """
        if not TRITON_AVAILABLE:
            # Fallback to original PyTorch path if Triton is unavailable
            # (This function was originally provided; we keep behavior here)
            batch_size, num_qo_heads, head_dim = q.shape
            _, page_size, num_kv_heads, _ = k_cache.shape
            device = q.device
            output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
            lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)
            gqa_ratio = num_qo_heads // num_kv_heads
            k_cache_flat = k_cache.squeeze(1).to(torch.float32)
            v_cache_flat = v_cache.squeeze(1).to(torch.float32)
            for b in range(batch_size):
                if kv_indptr[b + 1] - kv_indptr[b] == 0:
                    output[b].zero_()
                    continue
                token_indices = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(torch.long)
                num_tokens = token_indices.shape[0]
                k_batch = k_cache_flat[token_indices]  # [num_tokens, num_kv_heads, head_dim]
                v_batch = v_cache_flat[token_indices]  # [num_tokens, num_kv_heads, head_dim]
                q_batch = q[b].to(torch.float32)  # [num_qo_heads, head_dim]
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_head = q_batch[h]  # [head_dim]
                    k_head = k_batch[:, kv_head]  # [num_tokens, head_dim]
                    v_head = v_batch[:, kv_head]  # [num_tokens, head_dim]
                    logits = torch.matmul(q_head, k_head.T)  # [num_tokens]
                    logits_scaled = logits * sm_scale
                    lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
                    attn = torch.softmax(logits_scaled, dim=-1)  # [num_tokens]
                    out_head = torch.matmul(attn, v_head)  # [head_dim]
                    output[b, h] = out_head.to(torch.bfloat16)
            return output, lse

        # Triton path
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "Inputs must be CUDA tensors"
        B, H, D = q.shape
        _, Np, K, _ = k_cache.shape
        assert H == 32 and K == 8 and D == 128, "Fixed shape constraints: H=32, K=8, D=128"

        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Outputs (float32 for compute, convert later)
        out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        fused_gqa_row_scalar_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, out, lse,
            sm_scale,
            B=B, H=H, D=D, K=K, gqa_ratio=4, MAX_TOKENS=2048,
            num_warps=2, num_stages=1,
        )

        # Convert output to bfloat16 as expected by original function
        output = out.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
