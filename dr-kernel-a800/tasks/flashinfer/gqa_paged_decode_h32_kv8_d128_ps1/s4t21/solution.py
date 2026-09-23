import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh(
    q_ptr,                 # *fp32, pointer to q[b, h] vector, length = HEAD_DIM
    K_ptr,                 # *fp32, pointer to K tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    V_ptr,                 # *fp32, pointer to V tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    OUT_ptr,               # *fp32, output vector for this head [HEAD_DIM]
    LSE_ptr,               # *fp32, single scalar lse for this (b, h)
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.float32,      # 1.0 / sqrt(HEAD_DIM) (here 1/sqrt(128))
    LOG2_INVERSE: tl.float32,  # 1.0 / ln(2.0) ≈ 1.4426950408889634
):
    # One program instance per head h (grid dimension). We also have batch index in host code.
    h = tl.program_id(0)

    # First pass: compute lse = logsumexp(logits_scaled) / ln(2)
    running_max = -float("inf")
    running_sum = 0.0

    for t in range(0, NUM_TOKENS):
        # Load q vector for this head: [HEAD_DIM]
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        # Load k vector for token t: [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        # Compute dot product (scalar)
        logits_t = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits_t * SM_SCALE
        # numerically stable update
        running_max = tl.maximum(running_max, scaled)
        running_sum = running_sum * tl.exp(running_max - running_max) + tl.exp(scaled - running_max)

    lse_val = tl.log(running_sum) + running_max
    lse_val = lse_val * LOG2_INVERSE  # divide by ln(2)
    # Store scalar lse at LSE_ptr[b * num_qo_heads + h] (host sets base pointer for b)
    # Note: we receive LSE_ptr already offset for this b; it's a 1-element fp32 tensor
    tl.store(LSE_ptr, lse_val)


@triton.jit
def softmax_and_attention_single_bh2(
    q_ptr,                 # *fp32, pointer to q[b, h] vector, length = HEAD_DIM
    K_ptr,                 # *fp32, pointer to K tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    V_ptr,                 # *fp32, pointer to V tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    OUT_ptr,               # *fp32, output vector for this head [HEAD_DIM]
    LSE_ptr,               # *fp32, scalar lse for this (b, h)
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.float32,      # 1.0 / sqrt(HEAD_DIM)
    LOG2_INVERSE: tl.float32,  # 1.0 / ln(2.0)
):
    h = tl.program_id(0)
    # Load lse scalar for this (b, h)
    lse_val = tl.load(LSE_ptr)
    # Second pass: compute output vector
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        v_vec = tl.load(V_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        logits_t = tl.sum(q_vec * k_vec, axis=0)                        # scalar
        scaled = logits_t * SM_SCALE
        attn = tl.exp(scaled - lse_val)  # softmax over tokens
        out_vec += attn * v_vec

    # Store output
    tl.store(OUT_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        self.log2_inverse = 1.0 / math.log(2.0)

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Assertions and shapes
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be on CUDA device"
        assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16
        B, num_qo_heads, head_dim = q.shape
        # k_cache and v_cache should be [num_pages, 1, num_kv_heads, head_dim]
        assert k_cache.shape == v_cache.shape
        assert k_cache.shape[1] == 1 and k_cache.shape[3] == head_dim
        assert num_qo_heads == 32 and head_dim == 128 and k_cache.shape[2] == 8

        # Prepare outputs (fp32 for compute, cast later)
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Precompute per-batch num_tokens using kv_indptr (host-side)
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_b = end - start
            # Gather token indices for this batch
            token_indices = [start + t for t in range(num_tokens_b)]
            # Gather K_t and V_t per token: [num_tokens_b, 1, 8, 128]
            K_t = k_cache[token_indices]  # [num_tokens_b, 1, 8, 128]
            V_t = v_cache[token_indices]  # [num_tokens_b, 1, 8, 128]
            K_t = K_t.to(torch.float32).contiguous()  # [num_tokens_b, 1, 8, 128]
            V_t = V_t.to(torch.float32).contiguous()

            # Launch Triton kernels per head
            for h in range(num_qo_heads):
                kv_head = h // self.gqa_ratio  # 0..7
                # q vector for this head: [head_dim]
                q_vec = q[b, h].to(torch.float32).contiguous()  # [128]
                # Slice per head: [1, 128] and then squeeze
                K_t_h = K_t[:, 0, kv_head, :]  # [num_tokens_b, 128]
                V_t_h = V_t[:, 0, kv_head, :]  # [num_tokens_b, 128]
                K_t_h = K_t_h.contiguous().view(num_tokens_b, head_dim)  # [num_tokens_b, 128]
                V_t_h = V_t_h.contiguous().view(num_tokens_b, head_dim)  # [num_tokens_b, 128]

                # Prepare pointers
                # We will allocate a 1-element tensor for lse[b, h]
                lse_bh = torch.empty(1, dtype=torch.float32, device=q.device)
                out_vec = torch.empty((head_dim,), dtype=torch.float32, device=q.device)

                # Launch kernel 1: compute lse
                grid = (1,)  # one program per head
                softmax_and_attention_single_bh[grid](
                    q_vec, K_t_h, V_t_h, out_vec, lse_bh,
                    NUM_TOKENS=num_tokens_b, HEAD_DIM=head_dim,
                    SM_SCALE=self.sm_scale, LOG2_INVERSE=self.log2_inverse
                )

                # Store lse for this (b, h)
                lse[b, h] = lse_bh[0]

                # Launch kernel 2: compute output vector using lse
                softmax_and_attention_single_bh2[grid](
                    q_vec, K_t_h, V_t_h, out_vec, lse_bh,
                    NUM_TOKENS=num_tokens_b, HEAD_DIM=head_dim,
                    SM_SCALE=self.sm_scale, LOG2_INVERSE=self.log2_inverse
                )

                # Store output vector for this head
                output[b, h] = out_vec

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
