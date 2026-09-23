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
    q_ptr,                 # *bf16, [B, H, D]
    k_ptr,                 # *bf16, [Np, 1, D]
    v_ptr,                 # *bf16, [Np, 1, D]
    token_indices_ptr,     # *int32, [B * num_tokens_max]
    num_tokens_ptr,        # *int32, [B]
    output_ptr,            # *fp32, [B, H, D]
    lse_ptr,               # *fp32, [B, H]
    B: tl.int32,           # batch size
    H: tl.int32,           # num query heads
    D: tl.int32,           # head_dim
    K: tl.int32,           # num kv heads
    SM_SCALE: tl.float32,  # scaling factor
    MAX_TOKENS: tl.int32,  # max tokens per batch (guard iterations)
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Compute q vector for this (b, h) in fp32
    # q_ptr is [B, H, D] contiguous
    q_offset = pid_b * H * D + pid_h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

    # Compute kv_head for GQA: kv_head = h // (H // K) = h // 4
    kv_head = pid_h // 4  # integer division; H=32, K=8, so gqa_ratio=4

    # Pass 1: accumulate sum_exp = sum(exp(dot(q, k) * sm_scale)) across tokens
    sum_exp = 0.0  # fp32 scalar
    for t in range(0, MAX_TOKENS):
        if t < tl.load(num_tokens_ptr + pid_b):
            # Load token index
            idx = tl.load(token_indices_ptr + pid_b * MAX_TOKENS + t).to(tl.int32)
            # k_row = k_cache[idx, kv_head, :]  # [D]
            k_offset = idx * K * D + kv_head * D
            k_row = tl.load(k_ptr + k_offset).to(tl.float32)  # [D]
            # dot product
            dot = 0.0
            # iterate over D with constexpr loop
            for d in range(0, D):
                dot += q_vec[d] * k_row[d]
            # accumulate
            sum_exp += tl.exp(dot * SM_SCALE)

    # Compute lse in base-2: lse = log(sum_exp) / ln(2)
    lse = tl.log(sum_exp) / 1.4426950408889634  # 1 / ln(2)

    # Pass 2: compute output vector
    out_vec = tl.zeros([D], dtype=tl.float32)
    for t in range(0, MAX_TOKENS):
        if t < tl.load(num_tokens_ptr + pid_b):
            idx = tl.load(token_indices_ptr + pid_b * MAX_TOKENS + t).to(tl.int32)
            k_offset = idx * K * D + kv_head * D
            k_row = tl.load(k_ptr + k_offset).to(tl.float32)  # [D]
            dot = 0.0
            for d in range(0, D):
                dot += q_vec[d] * k_row[d]
            attn = tl.exp((dot - lse) * SM_SCALE)  # softmax weight for this token
            v_offset = idx * K * D + kv_head * D
            v_row = tl.load(v_ptr + v_offset).to(tl.float32)  # [D]
            for d in range(0, D):
                out_vec[d] += attn * v_row[d]

    # Store output and lse
    out_offset = pid_b * H * D + pid_h * D
    tl.store(output_ptr + out_offset, out_vec)
    tl.store(lse_ptr + pid_b * H + pid_h, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity and device
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Shapes
        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Fixed shapes required"

        device = q.device
        # Compute num_tokens and token_indices per batch element (on host)
        # num_tokens[b] = kv_indptr[b+1] - kv_indptr[b]
        # token_indices[b] = kv_indices[kv_indptr[b]:kv_indptr[b+1]]
        # We assume len_indptr == batch_size + 1
        num_tokens_list = []
        token_indices_list = []
        for b in range(batch_size):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens = end - start
            num_tokens_list.append(num_tokens)
            # collect indices
            idx = kv_indices[start:end].to(torch.int32)
            token_indices_list.append(idx)
        # Pad token_indices to max tokens for the grid; but we'll pass per-batch num_tokens and per-batch pointer offsets.
        # To keep it simple and avoid dynamic per-program offsets, flatten and keep per-batch offset using pointers.
        # However, Triton kernels expect contiguous offsets. We'll build a single token_indices buffer by flattening all b,
        # and num_tokens per b remains a [B] tensor. For t loop, we compute per-b b's range using pid_b * MAX_TOKENS + t and mask.

        # Flatten token_indices: total_tokens = sum(num_tokens_list), build single tensor
        total_tokens = sum(num_tokens_list)
        token_indices_flat = torch.empty(total_tokens, dtype=torch.int32, device=device)
        offset = 0
        for b, num in enumerate(num_tokens_list):
            if num > 0:
                token_indices_flat[offset:offset + num] = token_indices_list[b]
                offset += num
            else:
                # fill with dummy to keep length; not used when num==0 anyway
                token_indices_flat[offset:offset + 1] = torch.tensor([0], dtype=torch.int32, device=device)
                offset += 1

        # num_tokens per batch element
        num_tokens_tensor = torch.tensor(num_tokens_list, dtype=torch.int32, device=device)

        # Allocate outputs
        output_fp32 = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse_fp32 = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: grid = (B, H)
        grid = (batch_size, num_qo_heads)
        # We need to pass MAX_TOKENS; we choose 1024 as an upper bound. Triton will guard with t < num_tokens[b].
        gqa_row_scalar_kernel[grid](
            q, k_cache, v_cache, token_indices_flat, num_tokens_tensor, output_fp32, lse_fp32,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads, SM_SCALE=float(sm_scale), MAX_TOKENS=1024,
            num_warps=2, num_stages=1,
        )

        # Convert output to bfloat16 to match original
        output = output_fp32.to(torch.bfloat16)
        return output, lse_fp32


def run(*args):
    return ModelNew()(*args)
