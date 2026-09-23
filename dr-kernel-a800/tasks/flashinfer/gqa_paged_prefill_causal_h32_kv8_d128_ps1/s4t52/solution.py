import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: computes attention for one (token, head) pair in a batch.
# It iterates over kv tokens, updates logsumexp (max and sum), computes softmax, and accumulates output vector.
@triton.jit
def attention_single_batch_kernel(
    q_batch_ptr,         # *f32, shape [q_num_tokens, num_qo_heads, head_dim]
    k_base_ptr,          # *f32, shape [num_kv_tokens, num_kv_heads, head_dim]
    v_base_ptr,          # *f32, shape [num_kv_tokens, num_kv_heads, head_dim]
    out_ptr,             # *f32, shape [q_num_tokens, num_qo_heads, head_dim]
    lse_ptr,             # *f32, shape [q_num_tokens, num_qo_heads]
    q_num_tokens,        # int32
    num_qo_heads,        # int32
    num_kv_tokens,       # int32
    sm_scale,            # f32
    HEAD_DIM: tl.constexpr,        # compile-time constant, e.g., 128
    GQA_RATIO: tl.constexpr,       # compile-time constant, e.g., 4
):
    t = tl.program_id(0)       # token index within the batch
    h = tl.program_id(1)       # query head index

    # Initialize logsumexp components
    max_logit = -float('inf')
    sum_logit = 0.0

    # Pass 1: compute max and sum for logsumexp across kv tokens
    for k in range(0, num_kv_tokens):
        kv_head = k // GQA_RATIO
        q_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        base_q = q_batch_ptr + t * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        # load q_vec[h]
        for d in range(0, HEAD_DIM):
            q_ptr = base_q + d
            q_val = tl.load(q_ptr)
            q_vec[d] = q_val

        k_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        base_k = k_base_ptr + k * (GQA_RATIO * HEAD_DIM) + kv_head * HEAD_DIM
        for d in range(0, HEAD_DIM):
            k_ptr = base_k + d
            k_val = tl.load(k_ptr)
            k_vec[d] = k_val

        dot = 0.0
        for d in range(0, HEAD_DIM):
            dot += q_vec[d] * k_vec[d]
        logits_scaled = dot * sm_scale

        # Update running max and sum
        max_logit = tl.maximum(max_logit, logits_scaled)
        # sum_logit accumulates exp(logits_scaled - max_logit) to avoid overflow
        sum_logit += tl.exp(logits_scaled - max_logit)

    lse_val = max_logit + tl.log(sum_logit)

    # Pass 2: compute softmax and accumulate output
    for k in range(0, num_kv_tokens):
        kv_head = k // GQA_RATIO
        q_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        base_q = q_batch_ptr + t * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        for d in range(0, HEAD_DIM):
            q_ptr = base_q + d
            q_val = tl.load(q_ptr)
            q_vec[d] = q_val

        k_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        base_k = k_base_ptr + k * (GQA_RATIO * HEAD_DIM) + kv_head * HEAD_DIM
        for d in range(0, HEAD_DIM):
            k_ptr = base_k + d
            k_val = tl.load(k_ptr)
            k_vec[d] = k_val

        dot = 0.0
        for d in range(0, HEAD_DIM):
            dot += q_vec[d] * k_vec[d]
        logits_scaled = dot * sm_scale

        prob = tl.exp(logits_scaled - max_logit) / sum_logit  # softmax
        v_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        base_v = v_base_ptr + k * (GQA_RATIO * HEAD_DIM) + kv_head * HEAD_DIM
        for d in range(0, HEAD_DIM):
            v_ptr = base_v + d
            v_val = tl.load(v_ptr)
            v_vec[d] = v_val

        out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for d in range(0, HEAD_DIM):
            out_vec[d] = prob * v_vec[d]

        # Accumulate into out[t, h]
        base_out = out_ptr + t * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        for d in range(0, HEAD_DIM):
            out_ptr_el = base_out + d
            curr = tl.load(out_ptr_el)
            curr += out_vec[d]
            tl.store(out_ptr_el, curr)

    # Write lse[t, h]
    base_lse = lse_ptr + t * num_qo_heads + h
    tl.store(base_lse, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants as in the original
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # If Triton is unavailable, fall back to a pure torch implementation for correctness
        if not TRITON_AVAILABLE or not q.is_cuda:
            # Fallback path using original logic; kept here for completeness in case Triton is not available
            total_q = q.shape[0]
            assert total_q == int(qo_indptr[-1].item())
            device = q.device
            output = torch.zeros((total_q, self.num_qo_heads, self.head_dim), dtype=torch.bfloat16, device=device)
            lse = torch.full((total_q, self.num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

            q_f32 = q.to(torch.float32)
            k_cache_flat = k_cache.squeeze(1).to(torch.float32)
            v_cache_flat = v_cache.squeeze(1).to(torch.float32)

            for b in range(qo_indptr.shape[0] - 1):
                q_start = int(qo_indptr[b].item())
                q_end = int(qo_indptr[b + 1].item())
                kv_start = int(kv_indptr[b].item())
                kv_end = int(kv_indptr[b + 1].item())

                if q_start >= q_end or kv_start >= kv_end:
                    continue

                # Gather kv indices for this batch
                # Note: torch.index_select expects a 1D input; here we select num_kv_indices elements
                k_batch = k_cache_flat[torch.tensor(kv_indices[kv_start:kv_end], device=device, dtype=torch.long)]  # [num_kv_tokens, 8, 128]
                v_batch = v_cache_flat[torch.tensor(kv_indices[kv_start:kv_end], device=device, dtype=torch.long)]  # [num_kv_tokens, 8, 128]
                q_batch = q_f32[q_start:q_end]  # [num_q_tokens, 32, 128]

                num_q_tokens = q_batch.shape[0]
                num_kv_tokens = k_batch.shape[0]

                delta = num_kv_tokens - num_q_tokens

                for q_idx in range(num_q_tokens):
                    global_q_idx = q_start + q_idx
                    max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
                    if max_kv_idx <= 0:
                        continue

                    q_pos = q_batch[q_idx]  # [32, 128]
                    for h in range(self.num_qo_heads):
                        kv_head = h // self.gqa_ratio
                        q_head = q_pos[h]  # [128]
                        k_head = k_batch[:max_kv_idx, kv_head]  # [max_kv_idx, 128]
                        v_head = v_batch[:max_kv_idx, kv_head]  # [max_kv_idx, 128]

                        logits = torch.matmul(q_head, k_head.T)  # [max_kv_idx]
                        logits_scaled = logits * sm_scale

                        lse[global_q_idx, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                        attn = torch.softmax(logits_scaled, dim=-1)  # [max_kv_idx]
                        out_head = torch.matmul(attn, v_head)  # [128]
                        output[global_q_idx, h] = out_head.to(torch.bfloat16)

            return output, lse

        # Triton path: ensure inputs are on CUDA and contiguous
        device = q.device
        total_q = q.shape[0]
        assert total_q == int(qo_indptr[-1].item())
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton path."

        # Flatten k_cache and v_cache along the "page" dimension
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]

        # For each batch element b, compute q slice and selected k/v slices using torch.index_select
        # We'll process one batch at a time; compute q_batch, k_batch, v_batch and launch kernel for that batch.
        # But since qo_indptr has len_indptr entries, we can launch a grid over tokens and heads for the entire batch.

        # Note: We cannot index k_cache/kv_indices by dynamic tensors in Triton; do it on host via index_select per batch.
        # We can compute b by looping over range(len_indptr - 1). For each b, compute q_batch, k_batch, v_batch, then launch kernel.
        # However, to avoid multiple kernel launches, we can concatenate all q slices into q_big, k slices into k_big, v slices into v_big,
        # but Triton kernels require static shapes. Simpler: loop over b and launch kernel.

        # Prepare output and lse buffers as float32
        out = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=device)

        # For each batch b: compute q_batch, k_batch, v_batch and launch kernel
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Gather q slice for this batch
            q_batch = q.to(torch.float32)[q_start:q_end].contiguous()  # [num_q_tokens, 32, 128]

            # Gather k/v slices using indices[kv_start:kv_end]
            kv_indices_b = kv_indices[kv_start:kv_end].to(torch.long)  # [num_kv_indices_in_b]
            k_batch = k_cache_flat.index_select(0, kv_indices_b).contiguous()  # [num_kv_tokens, 8, 128]
            v_batch = v_cache_flat.index_select(0, kv_indices_b).contiguous()  # [num_kv_tokens, 8, 128]

            num_q_tokens = q_batch.shape[0]
            num_kv_tokens = k_batch.shape[0]

            # Launch Triton kernel with grid (num_q_tokens, num_qo_heads)
            grid = (num_q_tokens, self.num_qo_heads)
            attention_single_batch_kernel[grid](
                q_batch, k_batch, v_batch, out, lse,
                num_q_tokens, self.num_qo_heads, num_kv_tokens,
                float(sm_scale),
                HEAD_DIM=self.head_dim, GQA_RATIO=self.gqa_ratio,
            )

        # Cast output to bfloat16 as in original
        output = out.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
