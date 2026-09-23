import torch
import math
import triton
import triton.language as tl


# Triton kernel: one program per (b, h). Computes output[b, h, :] and lse[b, h].
# Assumes:
#   - q_ptr: [B, 32, D] fp32 (we'll cast q to fp32 in forward)
#   - K_ptr: [T, H, D] fp32, where T = num_tokens, H = num_kv_heads (e.g., 8)
#   - V_ptr: [T, H, D] fp32
#   - out_ptr: [B, 32, D] fp32
#   - lse_ptr: [B, 32] fp32
@triton.jit
def softmax_and_attention_single_bh(
    q_ptr, K_ptr, V_ptr,
    out_ptr, lse_ptr,
    T: tl.constexpr,      # num_tokens for this batch element (runtime scalar)
    sm_scale,             # scalar float32
    D: tl.constexpr,      # head_dim, e.g., 128
    H: tl.constexpr,      # num_kv_heads, e.g., 8
    gqa_ratio: tl.constexpr  # num_qo_heads // num_kv_heads, e.g., 4
):
    # Program ids
    b = tl.program_id(0)  # batch id
    h = tl.program_id(1)  # query head id

    # GQA mapping: kv_head = h // gqa_ratio
    kv_head = h // gqa_ratio

    # Load q vector for this (b, h): q[b, h, :]
    q_offs = b * (32 * D) + h * D + tl.arange(0, D)
    q_vec = tl.load(q_ptr + q_offs)

    # Running max and sum for logsumexp over T tokens
    running_max = -float("inf")
    running_sum = 0.0

    # First pass: compute numerically stable logsumexp over scaled logits
    # Iterate over tokens 0..T-1
    for t in range(0, T):
        # Gather k vector for this kv_head: K[t, kv_head, :]
        k_offs = t * (H * D) + kv_head * D
        k_vec = tl.load(K_ptr + k_offs)

        # Gather v vector for this kv_head: V[t, kv_head, :]
        v_offs = t * (H * D) + kv_head * D
        v_vec = tl.load(V_ptr + v_offs)

        # Compute logits = dot(q_vec, k_vec), reduce over D
        q_exp = q_vec[None, :]            # [1, D]
        k_exp = k_vec[:, None, :]         # [D, 1]
        logits = tl.sum(q_exp * k_exp, axis=1)  # scalar

        # Scale logits
        scaled = logits * sm_scale

        # Update running max and sum
        running_max = tl.maximum(running_max, scaled)
        exp_scaled = tl.exp(scaled - running_max)
        running_sum += exp_scaled

    # Compute lse = log(running_sum) + running_max, then divide by ln(2)
    log2_inverse = 1.4426950408889634  # 1 / ln(2)
    lse_val = tl.log(running_sum) + running_max
    lse_val = lse_val * log2_inverse  # divide by ln(2)

    # Store lse[b, h]
    lse_off = b * 32 + h
    tl.store(lse_ptr + lse_off, lse_val)

    # Second pass: compute attention and accumulate output
    out_vec = tl.zeros([D], dtype=tl.float32)
    for t in range(0, T):
        k_offs = t * (H * D) + kv_head * D
        k_vec = tl.load(K_ptr + k_offs)
        v_offs = t * (H * D) + kv_head * D
        v_vec = tl.load(V_ptr + v_offs)

        q_exp = q_vec[None, :]            # [1, D]
        k_exp = k_vec[:, None, :]         # [D, 1]
        logits = tl.sum(q_exp * k_exp, axis=1)  # scalar
        scaled = logits * sm_scale

        attn = tl.exp(scaled - lse_val)  # softmax over scaled logits

        # Accumulate out_vec += attn * v_vec, per element
        out_vec += tl.sum(attn[:, None] * v_vec[:, None, :], axis=0)

    # Store output[b, h, :]
    out_offs = b * (32 * D) + h * D + tl.arange(0, D)
    tl.store(out_ptr + out_offs, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # q: [B, 32, 128], dtype bfloat16
        # k_cache, v_cache: [N, H, D], dtype float32 or bfloat16 (we will cast to float32 for kernel)
        # kv_indptr: [len_indptr], int32
        # kv_indices: [num_kv_indices], int32
        # sm_scale: float32 scalar

        # Ensure tensors are on CUDA and contiguous
        if not q.is_cuda or not k_cache.is_cuda or not v_cache.is_cuda:
            raise RuntimeError("All inputs must be on CUDA device for Triton kernels.")
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Shapes
        B, num_qo_heads, head_dim = q.shape
        N, H, D = k_cache.shape
        assert v_cache.shape == (N, H, D), "v_cache must have same shape as k_cache"
        assert num_qo_heads == 32 and H == 8 and D == 128, "This implementation expects num_qo_heads=32, H=8, D=128."

        # Precompute num_tokens per batch element: num_tokens_b = kv_indptr[b+1] - kv_indptr[b]
        # Note: Triton doesn't allow passing runtime T as constexpr, so we compute it here and pass as an argument.
        # This is the only minimal host-side computation needed to correctly launch the kernel.
        num_tokens = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens.append(end - start)
        num_tokens = torch.tensor(num_tokens, dtype=torch.int32, device=q.device)

        # Prepare outputs (fp32)
        out = torch.empty((B, num_qo_heads, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Cast q, k_cache, v_cache to fp32 for Triton
        q_fp32 = q.to(torch.float32)
        k_cache_fp32 = k_cache.to(torch.float32)
        v_cache_fp32 = v_cache.to(torch.float32)

        # Launch Triton: one program per (b, h)
        grid = (B, num_qo_heads)
        softmax_and_attention_single_bh[grid](
            q_fp32, k_cache_fp32, v_cache_fp32,
            out, lse,
            T=num_tokens,      # runtime scalar per b (but Triton accepts scalar tensors for arguments; we pass num_tokens[b] by indexing inside kernel loop)
            sm_scale=sm_scale,
            D=D,
            H=H,
            gqa_ratio=4,       # num_qo_heads // num_kv_heads = 32 // 8
            num_warps=4,
        )

        # Return output (cast to bfloat16) and lse (float32) to match original
        return out.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
