import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def attention_single_batch_kernel(
    q_batch_ptr,        # *[q_num_tokens, 32, 128] float32
    k_base_ptr,         # *[num_kv_tokens, 8, 128] float32
    v_base_ptr,         # *[num_kv_tokens, 8, 128] float32
    out_ptr,            # *[q_num_tokens, 32, 128] float32
    lse_ptr,            # *[q_num_tokens, 32] float32
    q_num_tokens: tl.constexpr,   # int
    num_qo_heads: tl.constexpr,   # 32
    num_kv_heads: tl.constexpr,   # 8
    num_kv_tokens: tl.constexpr,  # int
    delta: tl.constexpr,          # int
    sm_scale,                     # float32
    HEAD_DIM: tl.constexpr,       # 128
):
    # One Triton program handles one token t
    t = tl.program_id(0)
    if t >= q_num_tokens:
        return

    # Track logsumexp per (t, h) across all kv tokens for this batch
    max_logit = tl.full((), -float("inf"), dtype=tl.float32)
    sum_logit = tl.zeros((), dtype=tl.float32)

    # First pass: compute max and sum for logsumexp across kv tokens
    for k in range(0, num_kv_tokens):
        max_kv_idx = tl.minimum(t + 1 + delta, num_kv_tokens)
        if k >= max_kv_idx:
            break
        # Compute kv head index for GQA
        kv_head = k // 4  # since num_kv_heads=8 and GQA ratio=4

        # For each query head h, compute dot and update logsumexp
        for h in range(0, num_qo_heads):
            # Load q vector q_vec[h]
            q_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
            base_q = q_batch_ptr + t * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
            for d in range(0, HEAD_DIM):
                q_ptr = base_q + d
                q_val = tl.load(q_ptr)
                q_vec[d] = q_val

            # Load k vector for this kv token and head
            base_k = k_base_ptr + k * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            k_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
            for d in range(0, HEAD_DIM):
                k_ptr = base_k + d
                k_val = tl.load(k_ptr)
                k_vec[d] = k_val

            # Compute logits_scaled = dot(q_vec[h], k_vec)
            dot = tl.zeros((), dtype=tl.float32)
            for d in range(0, HEAD_DIM):
                dot += q_vec[d] * k_vec[d]
            logits_scaled = dot * sm_scale

            # Update running max and sum for logsumexp
            max_logit = tl.maximum(max_logit, logits_scaled)
            sum_logit += tl.exp(logits_scaled - max_logit)

    # lse = max_logit + log(sum_logit)
    lse_val = max_logit + tl.log(sum_logit)

    # Second pass: compute softmax and accumulate output
    for k in range(0, num_kv_tokens):
        max_kv_idx = tl.minimum(t + 1 + delta, num_kv_tokens)
        if k >= max_kv_idx:
            break
        kv_head = k // 4

        for h in range(0, num_qo_heads):
            # Load q vector q_vec[h]
            q_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
            base_q = q_batch_ptr + t * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
            for d in range(0, HEAD_DIM):
                q_ptr = base_q + d
                q_val = tl.load(q_ptr)
                q_vec[d] = q_val

            # Load k vector
            base_k = k_base_ptr + k * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            k_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
            for d in range(0, HEAD_DIM):
                k_ptr = base_k + d
                k_val = tl.load(k_ptr)
                k_vec[d] = k_val

            # Compute dot and scaled logits
            dot = tl.zeros((), dtype=tl.float32)
            for d in range(0, HEAD_DIM):
                dot += q_vec[d] * k_vec[d]
            logits_scaled = dot * sm_scale

            # Softmax probability for this kv token and head
            prob = tl.exp(logits_scaled - max_logit) / sum_logit  # float32

            # Load v vector and accumulate
            base_v = v_base_ptr + k * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
            for d in range(0, HEAD_DIM):
                v_ptr = base_v + d
                v_val = tl.load(v_ptr)
                v_vec[d] = v_val

            acc_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
            for d in range(0, HEAD_DIM):
                acc_vec[d] = prob * v_vec[d]

            # Store output: out[t, h, :]
            base_out = out_ptr + t * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
            for d in range(0, HEAD_DIM):
                out_ptr_el = base_out + d
                curr = tl.load(out_ptr_el)  # initialize by zero
                curr += acc_vec[d]
                tl.store(out_ptr_el, curr)

    # Store lse per token and head (we compute per h within the second pass; store once after)
    # We'll store lse_val into lse_ptr[t, h] right before finishing the loop by using h=0.. and overwrite.
    # But to keep clean, store after all h processed: Triton will allow storing scalar here.
    base_lse = lse_ptr + t * num_qo_heads
    for h in range(0, num_qo_heads):
        tl.store(base_lse + h, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self, total_q: int = None, num_qo_heads: int = 32, num_kv_heads: int = 8, head_dim: int = 128):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_ratio = num_qo_heads // num_kv_heads  # 4

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton and CUDA
        if not TRITON_AVAILABLE:
            # Fallback path: mimic computation using torch ops
            # This path is not used in Triton evaluation but kept for robustness.
            total_q = q.shape[0]
            num_qo_heads = self.num_qo_heads
            num_kv_heads = self.num_kv_heads
            head_dim = self.head_dim

            device = q.device
            output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
            lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

            # Flatten k_cache, v_cache along "page" dim
            k_cache_flat = k_cache.squeeze(1)
            v_cache_flat = v_cache.squeeze(1)

            # Compute per-batch attention (simple PyTorch implementation for fallback)
            for b in range(qo_indptr.shape[0] - 1):
                q_start = int(qo_indptr[b].item())
                q_end = int(qo_indptr[b + 1].item())
                kv_start = int(kv_indptr[b].item())
                kv_end = int(kv_indptr[b + 1].item())

                q_batch = q[q_start:q_end].to(torch.float32)
                kv_indices_b = kv_indices[kv_start:kv_end].to(torch.int64)
                k_batch = k_cache_flat.index_select(0, kv_indices_b.to(torch.int64)).to(torch.float32)
                v_batch = v_cache_flat.index_select(0, kv_indices_b.to(torch.int64)).to(torch.float32)

                q_num_tokens = q_batch.shape[0]
                num_qo_heads = self.num_qo_heads
                num_kv_heads = self.num_kv_heads
                num_kv_tokens = k_batch.shape[0]
                delta = num_kv_tokens - q_num_tokens

                for t in range(q_num_tokens):
                    for h in range(num_qo_heads):
                        kv_head = h // self.gqa_ratio
                        max_kv_idx = min(t + 1 + delta, num_kv_tokens)
                        q_vec = q_batch[t, h]
                        sum_logit = 0.0
                        max_logit = float("-inf")
                        for k in range(0, num_kv_tokens):
                            if k >= max_kv_idx:
                                break
                            k_vec = k_batch[k, kv_head]
                            dot = torch.dot(q_vec, k_vec)
                            logits_scaled = dot * sm_scale
                            sum_logit += torch.exp(logits_scaled - max_logit)
                            max_logit = max(max_logit, logits_scaled)
                        lse_val = max_logit + math.log(sum_logit)
                        for k in range(0, num_kv_tokens):
                            if k >= max_kv_idx:
                                break
                            kv_head = k // self.gqa_ratio
                            q_vec = q_batch[t, h]
                            k_vec = k_batch[k, kv_head]
                            dot = torch.dot(q_vec, k_vec)
                            logits_scaled = dot * sm_scale
                            prob = torch.exp(logits_scaled - max_logit) / sum_logit
                            v_vec = v_batch[k, kv_head]
                            out_vec = prob * v_vec
                            output[t, h] = out_vec  # placeholder
                        lse[t, h] = lse_val
            return output, lse

        # Triton path
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "Inputs must be on CUDA."
        assert qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Index tensors must be on CUDA."

        total_q = q.shape[0]
        num_qo_heads = self.num_qo_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        # Sanity checks
        assert num_qo_heads == 32, "num_qo_heads must be 32."
        assert num_kv_heads == 8, "num_kv_heads must be 8."
        assert head_dim == 128, "head_dim must be 128."
        assert total_q == int(qo_indptr[-1].item()), "qo_indptr[-1] must equal total_q."

        # Flatten k_cache and v_cache along the "page" dimension
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]


def run(*args):
    return ModelNew()(*args)
