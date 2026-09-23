import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_scaled_logits_vec(
    Q_ptr,          # *fp32, q vector for head h, length = HEAD_DIM
    K_ptr,          # *fp32, K tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    OUT_ptr,        # *fp32, output scaled_logits vector [NUM_TOKENS]
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.float32,
):
    t = tl.program_id(0)
    q_vec = tl.load(Q_ptr + tl.arange(0, HEAD_DIM))
    k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
    dot = tl.sum(q_vec * k_vec, axis=0)
    scaled = dot * SM_SCALE
    tl.store(OUT_ptr + t, scaled)


@triton.jit
def reduce_lse_div_log2(
    LOGS_ptr,       # *fp32, scaled_logits vector [NUM_TOKENS]
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOG2_INVERSE: tl.float32,  # 1.0 / ln(2.0) ≈ 1.4426950408889634
):
    # One program instance; reduce over NUM_TOKENS using block-wise chunks
    # For small NUM_TOKENS (e.g., <= 1000), this single instance is fine.
    running_max = -float("inf")
    running_sum = 0.0
    for t in range(0, NUM_TOKENS):
        v = tl.load(LOGS_ptr + t)
        running_max = tl.maximum(running_max, v)
        running_sum = running_sum * tl.exp(running_max - running_max) + tl.exp(v - running_max)
    lse = tl.log(running_sum) + running_max  # logsumexp
    # divide by ln(2) via multiply with LOG2_INVERSE
    lse = lse * LOG2_INVERSE
    # store to OUT[0]
    tl.store(OUT_ptr, lse)


@triton.jit
def compute_attention_and_out(
    LOGS_ptr,       # *fp32, scaled_logits vector [NUM_TOKENS]
    V_ptr,          # *fp32, V tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    OUT_ptr,        # *fp32, output vector for this head [HEAD_DIM]
    LSE_scalar,     # scalar fp32, lse for this (b, h)
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Compute output = sum_j exp(scaled_j - lse) * V_j
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        v_vec = tl.load(V_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        scaled_j = tl.load(LOGS_ptr + t)
        attn = tl.exp(scaled_j - LSE_scalar)
        out_vec += attn * v_vec
    tl.store(OUT_ptr + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA"
        B = q.shape[0]
        device = q.device
        head_dim = self.head_dim
        num_qo_heads = self.num_qo_heads
        num_kv_heads = self.num_kv_heads

        # Output and lse (float32 for lse)
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        # Prepare constants
        LOG2_INVERSE = 1.4426950408889634  # 1 / ln(2)

        # Process each batch element
        for b in range(B):
            # Determine number of tokens for this batch from kv_indptr
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens = end - start
            if num_tokens <= 0:
                # No tokens for this batch element; set output and lse to zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            # For each query head
            for h in range(num_qo_heads):
                kv_head = h // self.gqa_ratio  # 0..7

                # Gather K and V rows for this batch and head: slice along num_pages dim
                # k_cache/v_cache are [num_pages, 1, num_kv_heads, head_dim]
                # We need tokens t in [start, end). Using k_cache[start:end, 0, kv_head, :] gives [num_tokens, 128]
                K_t_h = k_cache[start:end, 0, kv_head, :].contiguous()  # [num_tokens, 128]
                V_t_h = v_cache[start:end, 0, kv_head, :].contiguous() # [num_tokens, 128]
                # Cast to fp32 for compute
                K_t_h = K_t_h.to(torch.float32)
                V_t_h = V_t_h.to(torch.float32)

                # Prepare q vector for this head: [128]
                q_vec = q[b, h].to(torch.float32).contiguous()  # [128]

                # Allocate scaled_logits vector
                scaled_logits = torch.empty((num_tokens,), dtype=torch.float32, device=device)

                # Launch kernel to compute scaled_logits for this (b, h)
                grid = (num_tokens,)
                compute_scaled_logits_vec[grid](
                    q_vec, K_t_h.view(-1, head_dim), scaled_logits,
                    NUM_TOKENS=num_tokens,
                    HEAD_DIM=head_dim,
                    SM_SCALE=sm_scale,
                )

                # Launch kernel to compute lse = logsumexp(scaled_logits) / ln(2) for this (b, h)
                lse_scalar = torch.empty((1,), dtype=torch.float32, device=device)
                reduce_lse_div_log2[(1,)](
                    scaled_logits,
                    NUM_TOKENS=num_tokens,
                    HEAD_DIM=head_dim,
                    LOG2_INVERSE=LOG2_INVERSE,
                )
                lse_scalar = lse_scalar[0]  # scalar tensor

                # Launch kernel to compute attention and output for this (b, h)
                out_vec = torch.empty((head_dim,), dtype=torch.float32, device=device)
                compute_attention_and_out[(1,)](
                    scaled_logits,
                    V_t_h, out_vec,
                    LSE_scalar=lse_scalar,
                    NUM_TOKENS=num_tokens,
                    HEAD_DIM=head_dim,
                )

                # Store result
                output[b, h] = out_vec
                lse[b, h] = lse_scalar

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
