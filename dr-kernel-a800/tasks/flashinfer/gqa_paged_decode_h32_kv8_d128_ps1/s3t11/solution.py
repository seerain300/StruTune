import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_gqa_kernel(
    q_ptr,           # *bfloat16, [B, H, D]
    k_ptr,           # *float32,  [N, D] (k_cache.squeeze(1).to(float32))
    v_ptr,           # *float32,  [N, D] (v_cache.squeeze(1).to(float32))
    kv_indptr_ptr,   # *int32,    [B+1]
    kv_indices_ptr,  # *int32,    [num_kv_indices]
    output_ptr,      # *bfloat16, [B, H, D]
    lse_ptr,         # *float32,  [B, H]
    sm_scale,        # float32
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # num query heads
    D: tl.constexpr,          # head dim
    N: tl.constexpr,          # num kv heads (e.g., 8)
    gqa_ratio: tl.constexpr,  # H // N, e.g., 4
    T_MAX: tl.constexpr,      # maximum number of tokens per batch (we set to num_kv_indices)
):
    # Grid: (B, H)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Compute range [start, end) for tokens of this batch
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    num_tokens_b = end - start  # number of tokens for this batch

    # Base pointer to q[b, h, :]
    q_base = q_ptr + b * (H * D) + h * D
    q_vec = tl.load(q_base).to(tl.float32)  # [D]

    # GQA mapping: kv head for query head h
    kvh = h // gqa_ratio

    # First pass: compute max over tokens of logits_scaled for numerical stability
    l_max = -float("inf")
    for t in range(T_MAX):
        if t >= num_tokens_b:
            continue
        tok_id = tl.load(kv_indices_ptr + start + t).to(tl.int32)
        # Load k_vec for kvh and tok_id
        k_base = k_ptr + kvh * D
        k_vec = tl.load(k_base).to(tl.float32)  # [D]
        logits = tl.dot(q_vec, k_vec)  # scalar
        logits_scaled = logits * sm_scale
        l_max = tl.maximum(l_max, logits_scaled)

    # Second pass: compute sum of exp(logits_scaled - l_max) and accumulate output
    lse_sum = 0.0
    for t in range(T_MAX):
        if t >= num_tokens_b:
            continue
        tok_id = tl.load(kv_indices_ptr + start + t).to(tl.int32)
        k_base = k_ptr + kvh * D
        k_vec = tl.load(k_base).to(tl.float32)  # [D]
        logits = tl.dot(q_vec, k_vec)
        logits_scaled = logits * sm_scale
        exp_term = tl.exp(logits_scaled - l_max)
        lse_sum += exp_term

    # Compute lse = logsumexp(logits_scaled) / log(2)
    inv_log2 = 1.0 / math.log(2.0)
    lse_b_h = l_max + tl.log(lse_sum) * inv_log2
    tl.store(lse_ptr + b * H + h, lse_b_h)

    # Accumulate output: output[b, h, :] += attn[t] * v[t]
    acc = tl.zeros((D,), dtype=tl.float32)
    for t in range(T_MAX):
        if t >= num_tokens_b:
            continue
        tok_id = tl.load(kv_indices_ptr + start + t).to(tl.int32)
        k_base = k_ptr + kvh * D
        k_vec = tl.load(k_base).to(tl.float32)  # [D]
        logits = tl.dot(q_vec, k_vec)
        logits_scaled = logits * sm_scale
        attn = tl.exp(logits_scaled - l_max) / lse_sum
        v_base = v_ptr + kvh * D
        v_vec = tl.load(v_base).to(tl.float32)  # [D]
        acc += attn * v_vec

    # Store result as bfloat16
    out_base = output_ptr + b * (H * D) + h * D
    tl.store(out_base, acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "Inputs must be on CUDA"
        device = q.device

        B, H, D = q.shape
        N = 8  # num_kv_heads
        gqa_ratio = H // N  # 4 for H=32

        # Squeeze cache dims to [N, D] and cast to float32 for stable computation
        k_squeezed = k_cache.squeeze(1).to(torch.float32)  # [P, 1, N, D] -> [N, D]
        v_squeezed = v_cache.squeeze(1).to(torch.float32)  # [N, D]

        # Allocate outputs
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

        # Triton kernel launch: one program per (b, h)
        grid = (B, H)
        # Use T_MAX equal to num_kv_indices (not padded), since each b has its own end
        num_tokens_per_b = (kv_indptr[1:] - kv_indptr[:B]).to(torch.int32).to(device)  # [B]
        # We pass num_tokens_per_b to kernel via kernel args? Triton doesn't read tensors, so pass T_MAX as a constexpr-like meta parameter.
        # Here, we set T_MAX to max(num_tokens_per_b), but we can simply iterate up to num_tokens_per_b in runtime via the loop with 'continue' when t >= num_tokens_per_b.
        # However, Triton requires loop bound to be constexpr. So we set T_MAX to num_kv_indices and mask t >= num_tokens_per_b.
        # Compute T_MAX on host
        # Note: Triton requires T_MAX as tl.constexpr. We pass T_MAX as num_kv_indices. We also pass num_tokens_per_b as tensor? Triton can't read tensors, so we'll just use a single T_MAX based on the largest value.
        # To keep it simple and correct, we set T_MAX equal to the number of elements in kv_indices (num_kv_indices) and rely on 'continue' when t >= num_tokens_per_b.
        # But Triton needs constexpr. We'll compute T_MAX as int(num_kv_indices.item()) since kv_indices.numel() is known.

        # Launch kernel. Triton will compile with T_MAX equal to num_kv_indices. Inside kernel, we guard with 'continue' when t >= num_tokens_per_b.
        # However, Triton kernel args cannot use tensor variables in loop bounds. We'll set T_MAX equal to num_kv_indices via .item() and rely on masking in kernel.
        T_MAX = int(kv_indices.numel())

        attention_gqa_kernel[grid](
            q, k_squeezed, v_squeezed, kv_indptr, kv_indices, output, lse, sm_scale,
            B, H, D, N, gqa_ratio, T_MAX,
            num_warps=4, num_stages=2
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
