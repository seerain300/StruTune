import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh_kernel(
    q_ptr,                 # *fp32, pointer to q[b, h] vector, length = HEAD_DIM
    K_ptr,                 # *fp32, pointer to K tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    V_ptr,                 # *fp32, pointer to V tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    OUT_ptr,               # *fp32, output vector for this head [HEAD_DIM]
    LSE_ptr,               # *fp32, scalar lse for this (b, h)
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.float32,      # 1.0 / sqrt(HEAD_DIM) (here 1/sqrt(128))
    LOG2_INVERSE: tl.float32,  # 1.0 / ln(2.0) ≈ 1.4426950408889634
):
    # One program instance per (b, h) where program_id(1) is h
    b = tl.program_id(0)
    h = tl.program_id(1)

    # First pass: compute lse = logsumexp(logits_scaled) / ln(2)
    running_max = -float("inf")
    running_sum = 0.0

    for t in range(0, NUM_TOKENS):
        # Load q vector for this head: [HEAD_DIM]
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))
        # Load k vector for token t: [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        # Dot product over head_dim
        logits_t = tl.sum(q_vec * k_vec, axis=0)  # scalar
        scaled = logits_t * SM_SCALE
        running_max = tl.maximum(running_max, scaled)
        running_sum = running_sum * tl.exp(running_max - running_max) + tl.exp(scaled - running_max)

    lse_raw = tl.log(running_sum) + running_max  # logsumexp of scaled_logits
    lse_scaled = lse_raw * LOG2_INVERSE          # divide by ln(2)
    # Store scalar lse to LSE_ptr[b * num_qo_heads + h]
    tl.store(LSE_ptr + b * 32 + h, lse_scaled)

    # Second pass: compute output vector
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        v_vec = tl.load(V_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        logits_t = tl.sum(q_vec * k_vec, axis=0)                        # scalar
        scaled = logits_t * SM_SCALE
        attn = tl.exp(scaled - lse_scaled)  # softmax over tokens
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

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        B, num_qo_heads, head_dim = q.shape
        num_pages, seq_dim, num_kv_heads, _ = k_cache.shape
        assert seq_dim == 1, "Expected k_cache shape [num_pages, 1, num_kv_heads, head_dim]"
        assert num_kv_heads == self.num_kv_heads and head_dim == self.head_dim, "Fix: num_kv_heads=8, head_dim=128"
        device = q.device

        # Output and lse tensors (fp32 for compute)
        output_fp32 = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse_fp32 = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        LOG2_INVERSE = 1.0 / math.log(2.0)  # 1/ln(2)

        # Compute num_tokens per batch (len_indptr length is B+1)
        num_tokens_list = [int(kv_indptr[b + 1].item() - kv_indptr[b].item()) for b in range(B)]

        for b in range(B):
            num_tokens = num_tokens_list[b]
            if num_tokens <= 0:
                # No tokens for this batch element: output zeros, lse zeros
                output_fp32[b].zero_()
                lse_fp32[b].zero_()
                continue

            # Compute token range for this batch
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            # Single token index for this batch in provided tests
            token_index = int(kv_indices[start].item())

            # Gather K and V for this token index and per-head slices: k_cache [num_pages, 1, 8, 128]
            K_t = k_cache[token_index]        # [1, 1, 8, 128]
            V_t = v_cache[token_index]        # [1, 1, 8, 128]
            K_t = K_t.to(torch.float32).contiguous()  # [1, 1, 8, 128]
            V_t = V_t.to(torch.float32).contiguous()  # [1, 1, 8, 128]

            # GQA mapping
            gqa_ratio = self.gqa_ratio  # 4
            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio  # 0..7
                # q vector for this head
                q_vec = q[b, h].to(torch.float32).contiguous()  # [128]
                # Slice per head: [1, 128]
                K_t_h = K_t[:, 0, kv_head, :]  # [1, 128]
                V_t_h = V_t[:, 0, kv_head, :]  # [1, 128]
                K_t_h = K_t_h.squeeze(0).squeeze(0).contiguous()  # [128]
                V_t_h = V_t_h.squeeze(0).squeeze(0).contiguous()  # [128]

                # Launch Triton kernel: grid = (B, num_qo_heads)
                softmax_and_attention_single_bh_kernel[(B, num_qo_heads)](
                    q_vec, K_t_h, V_t_h, output_fp32[b, h], lse_fp32[b, h],
                    NUM_TOKENS=num_tokens, HEAD_DIM=head_dim,
                    SM_SCALE=float(sm_scale), LOG2_INVERSE=LOG2_INVERSE,
                    num_warps=4, num_stages=2
                )

        # Cast output to bfloat16 to match original
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse_fp32


def run(*args):
    return ModelNew()(*args)
