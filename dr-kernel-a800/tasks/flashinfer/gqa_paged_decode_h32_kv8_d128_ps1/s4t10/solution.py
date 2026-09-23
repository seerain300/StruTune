import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh(
    Q_ptr,                 # *fp32, q vector for head h, length = HEAD_DIM (1D)
    K_ptr,                 # *fp32, K tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    V_ptr,                 # *fp32, V tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    OUT_ptr,               # *fp32, output vector for this head [HEAD_DIM]
    LSE_ptr,               # *fp32, single scalar lse for this (b, h)
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.float32,      # 1.0 / sqrt(HEAD_DIM) (here 1/sqrt(128))
    LOG2_INVERSE: tl.float32,  # 1.0 / ln(2.0) ≈ 1.4426950408889634
):
    # One program instance per (b, h), where h = program_id(0)
    h = tl.program_id(0)

    # First pass: compute lse = logsumexp(logits_scaled) / ln(2)
    running_max = -float("inf")
    running_sum = 0.0

    for t in range(0, NUM_TOKENS):
        # Load q vector for this head: [HEAD_DIM]
        q_vec = tl.load(Q_ptr + tl.arange(0, HEAD_DIM))
        # Load k vector for token t: [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        # Dot product (scalar)
        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * SM_SCALE
        # Stable logsumexp update
        running_max = tl.maximum(running_max, scaled)
        # running_sum = sum(exp(scaled - running_max))
        running_sum = running_sum * tl.exp(running_max - scaled) + 1.0

    lse = tl.log(running_sum) + running_max
    lse = lse * LOG2_INVERSE  # divide by ln(2)

    # Store lse as scalar at LSE_ptr[0]
    tl.store(LSE_ptr, lse)

    # Second pass: compute output vector
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        q_vec = tl.load(Q_ptr + tl.arange(0, HEAD_DIM))
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        v_vec = tl.load(V_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        logits = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits * SM_SCALE
        attn = tl.exp(scaled - lse)  # softmax over tokens
        out_vec += attn * v_vec

    # Store output vector
    tl.store(OUT_ptr + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.head_dim = 128
        self.gqa_ratio = 32 // 8  # 4
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        self.log2_inverse = 1.0 / math.log(2.0)

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Extract shapes
        B = q.shape[0]
        _, _, num_kv_heads, head_dim = k_cache.shape
        assert head_dim == self.head_dim, f"head_dim must be {self.head_dim}, got {head_dim}"
        num_qo_heads = q.shape[1]
        num_pages = k_cache.shape[0]
        # Ensure devices
        device = q.device
        assert k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be on CUDA device"
        assert q.is_cuda, "Q must be on CUDA device"

        # Compute per-batch number of tokens: num_tokens[b] = kv_indptr[b+1] - kv_indptr[b]
        len_indptr = kv_indptr.shape[0]
        assert len_indptr == B + 1, "kv_indptr must have length batch_size + 1"

        # Prepare output and lse
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=device)  # will cast to bfloat16 later
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        # List of tokens per batch
        num_tokens_list = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens = end - start
            num_tokens_list.append(num_tokens)

        # For each batch b and head h, gather per-head K/V rows and launch Triton kernel
        for b in range(B):
            num_tokens = num_tokens_list[b]
            if num_tokens <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # Determine token range for this batch using kv_indptr
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())

            # GQA mapping
            gqa_ratio = self.gqa_ratio
            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio  # 0..7

                # Slice k_cache and v_cache for this batch and head: [num_tokens, 1, 1, head_dim]
                K_t_h = k_cache[start:end, 0, kv_head, :]  # [num_tokens, 128]
                V_t_h = v_cache[start:end, 0, kv_head, :]  # [num_tokens, 128]
                # Ensure contiguous fp32
                K_t_h = K_t_h.to(torch.float32).contiguous()  # [num_tokens, 128]
                V_t_h = V_t_h.to(torch.float32).contiguous()  # [num_tokens, 128]

                # Prepare q vector for this head: Q[b, h] as 1D fp32 tensor
                q_vec = q[b, h].to(torch.float32).contiguous()  # [1


def run(*args):
    return ModelNew()(*args)
