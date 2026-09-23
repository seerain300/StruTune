import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute sum_exp for each (b, h) across tokens in [start, end).
@triton.jit
def reduce_lse_kernel(
    q_ptr,            # *bf16, [B, H, D]
    k_ptr,            # *bf16, [Np, K, D]  (after squeeze from [Np, 1, K, D])
    lse_ptr,          # *f32,  [B, H]
    kv_indptr_ptr,    # *i32,  [B+1]
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr, gqa_ratio: tl.constexpr, MAX_TOKENS: tl.constexpr, sm_scale: tl.float32
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Load token range for this batch element
    start = tl.load(kv_indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    sum_exp = 0.0
    # Fixed iteration loop with guard; Triton supports scalar control flow.
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = start + t  # token index for this batch
        kvh = pid_h // gqa_ratio  # select corresponding kv head

        # Load q vector for this head
        q_offset = pid_b * H * D + pid_h * D
        q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

        # Load k[idx, kvh, :]
        k_offset = idx * K * D + kvh * D
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(D):
            k_vec[d] = tl.load(k_ptr + k_offset + d).to(tl.float32)

        # Dot product q · k
        dot = 0.0
        for d in range(D):
            dot += q_vec[d] * k_vec[d]

        # Accumulate exp(dot * sm_scale)
        sum_exp += tl.exp(dot * sm_scale)

    # Compute lse in base-2: lse = log(sum_exp) / ln(2)
    lse_b2 = tl.log(sum_exp) * 1.4426950408889634  # 1/ln(2) = log2(e)
    tl.store(lse_ptr + pid_b * H + pid_h, lse_b2)


# Triton kernel: update output for each (b, h) across tokens in [start, end).
# Note: output is a bfloat16 tensor of shape [B, H, D]. We'll treat it as f32 for accumulation, then cast at the end in host.
@triton.jit
def update_output_kernel(
    q_ptr,            # *bf16, [B, H, D]
    k_ptr,            # *bf16, [Np, K, D]
    v_ptr,            # *bf16, [Np, K, D]
    output_ptr,       # *f32,  [B, H, D]  (accumulator in f32)
    kv_indptr_ptr,    # *i32,  [B+1]
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr, gqa_ratio: tl.constexpr, MAX_TOKENS: tl.constexpr, sm_scale: tl.float32
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Load token range for this batch element
    start = tl.load(kv_indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Load q vector for this head (used to recompute dot per token)
    q_offset = pid_b * H * D + pid_h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = start + t  # token index for this batch
        kvh = pid_h // gqa_ratio  # select corresponding kv head

        # Load k[idx, kvh, :] and v[idx, kvh, :]
        k_offset = idx * K * D + kvh * D
        v_offset = idx * K * D + kvh * D
        k_vec = tl.zeros((D,), dtype=tl.float32)
        v_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(D):
            k_vec[d] = tl.load(k_ptr + k_offset + d).to(tl.float32)
            v_vec[d] = tl.load(v_ptr + v_offset + d).to(tl.float32)

        # Dot product q · k
        dot = 0.0
        for d in range(D):
            dot += q_vec[d] * k_vec[d]

        # attn = exp((dot - lse) * sm_scale)
        lse_b2 = tl.load(output_ptr + pid_b * H * D + pid_h * D)  # placeholder, not used here
        # We don't have lse stored; we can recompute lse in this kernel? Not feasible since lse depends on all tokens.
        # Instead, we assume output_ptr was zero-initialized and we only write per-token output vector (not used here),
        # but since the original PyTorch code produces a final output vector per token, we need to mimic that behavior.
        # To keep correctness, we'll implement per-token output vector, not accumulating across tokens. The original code
        # returns a [B, H, D] tensor. We will write the final per-head output vector into output_ptr by reassigning
        # the entire vector for this head. However, Triton does not allow dynamic address arithmetic to write the whole
        # vector unless we know the index. The safest approach is to return only the final aggregated output, which
        # requires knowing lse. Therefore, we instead compute final output directly in a separate kernel that reads lse
        # from the first kernel. For simplicity, we implement a lighter kernel that assigns per-token output vector
        # into a separate buffer if needed. Here, we will implement the final assignment using a separate final kernel.
        pass  # Placeholder to satisfy Triton compilation (will be replaced by a real write below)

    # Note: We will not write here because we cannot maintain a [B, H, D] output in this kernel without
    # storing to a 3D tensor pointer which Triton doesn't support directly in a scalar loop this way.
    # The actual final output will be computed by a separate kernel that reads lse.


# We will perform final output assignment in a third kernel (final_output_kernel) after we have lse.

# Triton kernel: compute final output[b, h, :] given q, k, v, lse_b2 for each (b, h).
@triton.jit
def final_output_kernel(
    q_ptr,            # *bf16, [B, H, D]
    k_ptr,            # *bf16, [Np, K, D]
    v_ptr,            # *bf16, [Np, K, D]
    output_ptr,       # *bf16, [B, H, D]
    lse_ptr,          # *f32,  [B, H]
    kv_indptr_ptr,    # *i32,  [B+1]
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr, gqa_ratio: tl.constexpr, MAX_TOKENS: tl.constexpr, sm_scale: tl.float32
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Load token range for this batch element
    start = tl.load(kv_indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Load q vector for this head (used to recompute dot per token)
    q_offset = pid_b * H * D + pid_h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

    # lse for this (b, h) in base-2
    lse_b2 = tl.load(lse_ptr + pid_b * H + pid_h)  # f32

    # We will write the final output vector for this (b, h) into output_ptr as bfloat16
    out_offset = pid_b * H * D + pid_h * D
    for d in range(D):
        # Initialize output vector to zeros (already zero-initialized in host)
        # Now accumulate or assign per-token contribution. Since the original code returns a final aggregated output,
        # per token it assigns output[b, h, :] = attn * v[token, kvh, :], not accumulated across tokens.
        # To match behavior, we will recompute per-token contribution and write it into output[b, h, :].
        # However, writing directly into output_ptr requires knowing the exact address. Triton allows storing scalar values
        # to pointer expressions, but we need to ensure output_ptr is a pointer to a [B, H, D] contiguous tensor and we
        # compute the linear index properly. We'll store each element at linear index out_offset + d.
        # Compute attn per token and write it.
        for t in range(MAX_TOKENS):
            if t >= num_tokens:
                break
            idx = start + t  # token index for this batch
            kvh = pid_h // gqa_ratio  # select corresponding kv head

            # Load k[idx, kvh, :] and v[idx, kvh, :]
            k_offset = idx * K * D + kvh * D
            v_offset = idx * K * D + kvh * D
            k_vec = tl.zeros((D,), dtype=tl.float32)
            v_vec = tl.zeros((D,), dtype=tl.float32)
            for dd in range(D):
                k_vec[dd] = tl.load(k_ptr + k_offset + dd).to(tl.float32)
                v_vec[dd] = tl.load(v_ptr + v_offset + dd).to(tl.float32)

            # Dot product q · k
            dot = 0.0
            for dd in range(D):
                dot += q_vec[dd] * k_vec[dd]

            # attn = exp((dot - lse_b2) * sm_scale)
            attn = tl.exp((dot - lse_b2) * sm_scale)

            # Assign output[b, h, d] = attn * v[d]
            out_val = attn * v_vec[d]
            # Store as bfloat16 (cast from f32)
            tl.store(output_ptr + out_offset + d, out_val.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are contiguous
        q = q.contiguous()
        # k_cache and v_cache are [Np, 1, K, D] in the original; squeeze dim=1 to [Np, K, D]
        k_cache = k_cache.squeeze(1).contiguous()
        v_cache = v_cache.squeeze(1).contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Shapes
        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape  # should be [Np, K, D], we don't use Np here
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        device = q.device

        # Allocate buffers
        # lse in base-2: [B, H] float32
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)
        # Output as bfloat16: [B, H, D]
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        # For accumulation in Triton kernels, we keep output as f32 and cast at the end. However, final_output_kernel
        # writes bfloat16 directly. To keep types correct, we'll produce output in bfloat16 via final_output_kernel.

        # Launch kernel 1: reduce_lse_kernel to compute lse per (b, h)
        grid = (batch_size, num_qo_heads)
        reduce_lse_kernel[grid](
            q, k_cache, lse, kv_indptr,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads, gqa_ratio=4,
            MAX_TOKENS=1024, sm_scale=sm_scale,
            num_warps=2, num_stages=1,
        )

        # Launch kernel 3: final_output_kernel to write final output[b, h, :] as bfloat16
        # Note: This kernel computes per-token contribution but the original logic assigns per token, not accumulates.
        # Therefore, we write per-token contribution. Since we cannot aggregate per (b, h), we implement per-token
        # output assignment as the original code does. The result is a [B, H, D] tensor where each element is
        # assigned by the last token (which matches the original assignment behavior, not accumulation).
        final_output_kernel[grid](
            q, k_cache, v_cache, output, lse, kv_indptr,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads, gqa_ratio=4,
            MAX_TOKENS=1024, sm_scale=sm_scale,
            num_warps=2, num_stages=1,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
