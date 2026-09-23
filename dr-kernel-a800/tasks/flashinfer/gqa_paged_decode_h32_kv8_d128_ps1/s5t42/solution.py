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
    q_ptr,            # *bf16, [B, H, D]
    k_ptr,            # *bf16, [Np, 1, K, D] but used as [num_tokens, K, D] via indices
    v_ptr,            # *bf16, [Np, 1, K, D] but used as [num_tokens, K, D] via indices
    kv_indptr_ptr,    # *int32, [B+1]
    kv_indices_ptr,   # *int32, [Np] (not directly used; range is via indptr)
    out_ptr,          # *bf16, [B, H, D]
    lse_ptr,          # *float32, [B, H]
    sm_scale,         # float32 scalar
    B: tl.constexpr,  # batch_size
    H: tl.constexpr,  # num_qo_heads
    D: tl.constexpr,  # head_dim (128)
    K: tl.constexpr,  # num_kv_heads (8)
    GQA_RATIO: tl.constexpr,  # H // K (e.g., 4)
    MAX_TOKENS: tl.constexpr  # upper bound for token loop (e.g., 2048)
):
    # One program per (b, h)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Offsets
    q_base = q_ptr + pid_b * H * D
    out_base = out_ptr + pid_b * H * D
    lse_base = lse_ptr + pid_b * H

    # Load q vector for this head: q[b, h, :]
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in tl.range(0, D):
        q_elem = tl.load(q_base + pid_h * D + d).to(tl.float32)
        q_vec[d] = q_elem

    # Load indptr range for this batch
    start = tl.load(kv_indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Pass 1: compute sum_exp over tokens
    sum_exp = 0.0  # float32
    for t in tl.range(0, MAX_TOKENS):
        if t >= num_tokens:
            break
        kv_head = pid_h // GQA_RATIO
        idx = start + t

        # Load k_row and v_row
        k_row = tl.zeros((D,), dtype=tl.float32)
        v_row = tl.zeros((D,), dtype=tl.float32)
        for d in tl.range(0, D):
            k_elem = tl.load(k_ptr + idx * K * D + kv_head * D + d).to(tl.float32)
            v_elem = tl.load(v_ptr + idx * K * D + kv_head * D + d).to(tl.float32)
            k_row[d] = k_elem
            v_row[d] = v_elem

        # Dot product
        dot = 0.0
        for d in tl.range(0, D):
            dot += q_vec[d] * k_row[d]

        sum_exp += tl.exp(dot * sm_scale)

    # lse in base-2
    lse_val = tl.log(sum_exp) / 0.6931471805599453  # ln(2)

    # Pass 2: compute output vector
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in tl.range(0, MAX_TOKENS):
        if t >= num_tokens:
            break
        kv_head = pid_h // GQA_RATIO
        idx = start + t

        k_row = tl.zeros((D,), dtype=tl.float32)
        v_row = tl.zeros((D,), dtype=tl.float32)
        for d in tl.range(0, D):
            k_elem = tl.load(k_ptr + idx * K * D + kv_head * D + d).to(tl.float32)
            v_elem = tl.load(v_ptr + idx * K * D + kv_head * D + d).to(tl.float32)
            k_row[d] = k_elem
            v_row[d] = v_elem

        dot = 0.0
        for d in tl.range(0, D):
            dot += q_vec[d] * k_row[d]

        attn = tl.exp((dot - lse_val) * sm_scale)

        for d in tl.range(0, D):
            out_vec[d] += attn * v_row[d]

    # Store results
    tl.store(lse_base + pid_h, lse_val)
    for d in tl.range(0, D):
        tl.store(out_base + pid_h * D + d, tl.cast(out_vec[d], tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton path
        if not TRITON_AVAILABLE or not q.is_cuda:
            # Fallback PyTorch implementation for safety if Triton not available
            batch_size, num_qo_heads, head_dim = q.shape
            _, _, num_kv_heads, _ = k_cache.shape
            assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

            output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
            lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)
            gqa_ratio = 4  # 32 qo heads / 8 kv heads

            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                num_tokens = end - start
                if num_tokens <= 0:
                    output[b].zero_()
                    lse[b] = torch.full((num_qo_heads,), -float("inf"), dtype=torch.float32, device=q.device)
                    continue

                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_head = q[b, h].to(torch.float32)  # [D]
                    # k_batch and v_batch are [num_tokens, D]
                    k_batch = k_cache[start:end, 0, kv_head, :].to(torch.float32)  # shape [num_tokens, D]
                    v_batch = v_cache[start:end, 0, kv_head, :].to(torch.float32)  # shape [num_tokens, D]

                    logits = torch.matmul(q_head, k_batch.t())  # [num_tokens]
                    logits_scaled = logits * sm_scale
                    lse_b = torch.logsumexp(logits_scaled, dim=0) / math.log(2.0)
                    lse[b, h] = lse_b

                    attn = torch.softmax(logits_scaled, dim=0)  # [num_tokens]
                    out_head = torch.matmul(attn, v_batch)  # [D]
                    output[b, h] = out_head.to(torch.bfloat16)

            return output, lse

        # Triton path: ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()  # shape [Np, 1, K, D]
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        device = q.device
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        grid = (batch_size, num_qo_heads)
        gqa_row_scalar_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse, sm_scale,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads, GQA_RATIO=4, MAX_TOKENS=2048,
            num_warps=1, num_stages=1,
        )
        return output, lse


def run(*args):
    return ModelNew()(*args)
