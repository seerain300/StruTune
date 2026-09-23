import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_attention_bh(
    Q_ptr,                # *fp32, pointer to q[b, h] vector: length = HEAD_DIM
    K_ptr,                # *fp32, pointer to K tokens for this batch: shape [NUM_TOKENS, HEAD_DIM], contiguous
    V_ptr,                # *fp32, pointer to V tokens for this batch: shape [NUM_TOKENS, HEAD_DIM], contiguous
    OUT_ptr,              # *fp32, output vector for this head [HEAD_DIM]
    LSE_ptr,              # *fp32, single scalar lse for this (b, h)
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.float32,      # 1.0 / sqrt(HEAD_DIM) (here 1/sqrt(128))
    LOG2_INVERSE: tl.float32,  # 1.0 / ln(2.0) ≈ 1.4426950408889634
):
    # One program instance per (b, h)
    h = tl.program_id(0)

    # First pass: compute lse = logsumexp(logits_scaled) / ln(2) => logsumexp * (1/ln(2))
    running_max = -float("inf")
    running_sum = 0.0

    for t in range(0, NUM_TOKENS):
        # Load q vector for this head: [HEAD_DIM]
        q_vec = tl.load(Q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        # Load k vector for token t: [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        # Dot product: scalar
        logits_t = tl.sum(q_vec * k_vec, axis=0)  # scalar
        scaled = logits_t * SM_SCALE
        # Numerically stable update
        running_max = tl.maximum(running_max, scaled)
        running_sum = running_sum * tl.exp(running_max - running_max) + tl.exp(scaled - running_max)

    lse = tl.log(running_sum) + running_max
    lse = lse * LOG2_INVERSE
    # Store lse as single scalar at LSE_ptr
    tl.store(LSE_ptr, lse)

    # Second pass: compute output vector = sum_j exp(scaled_j - lse) * V_j
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        q_vec = tl.load(Q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        v_vec = tl.load(V_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        logits_t = tl.sum(q_vec * k_vec, axis=0)                        # scalar
        scaled = logits_t * SM_SCALE
        attn = tl.exp(scaled - lse)  # softmax over tokens for this head
        out_vec += attn * v_vec

    # Store output vector
    tl.store(OUT_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, 32, 128], bfloat16
        k_cache: [N, 1, 8, 128], bfloat16
        v_cache: [N, 1, 8, 128], bfloat16
        kv_indptr: [B+1], int32
        kv_indices: [T], int32
        sm_scale: float32 scalar, e.g., 1.0 / sqrt(128)
        Returns: (output [B, 32, 128] bfloat16, lse [B, 32] float32)
        """
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton."
        B, num_qo_heads, head_dim = q.shape
        N, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Fixed constants: num_qo_heads=32, num_kv_heads=8, head_dim=128."

        # Ensure contiguous and cast to fp32 for compute
        q = q.contiguous().to(torch.float32)  # [B, 32, 128]
        k_cache = k_cache.contiguous().to(torch.float32)  # [N, 1, 8, 128]
        v_cache = v_cache.contiguous().to(torch.float32)  # [N, 1, 8, 128]

        # Prepare output and lse
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Compute per-batch number of tokens (len_indptr[b+1] - len_indptr[b])
        num_tokens_list = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_b = end - start
            num_tokens_list.append(num_tokens_b)

        # GQA ratio
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # For each batch b and head h, run Triton kernel
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_b = end - start
            token_indices = kv_indices[start:end].to(torch.long).cuda()  # [num_tokens_b]

            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio  # 0..7

                # Prepare K_t_h and V_t_h as contiguous [num_tokens_b, HEAD_DIM] fp32
                K_t_h = torch.empty((num_tokens_b, head_dim), dtype=torch.float32, device=q.device)
                V_t_h = torch.empty((num_tokens_b, head_dim), dtype=torch.float32, device=q.device)
                for i, idx in enumerate(token_indices):
                    row = int(idx.item())  # valid because end <= N (kv_indptr defines bounds)
                    # k_cache/v_cache: [N, 1, num_kv_heads, head_dim]
                    K_t_h[i] = k_cache[row, 0, kv_head, :].to(torch.float32).contiguous()
                    V_t_h[i] = v_cache[row, 0, kv_head, :].to(torch.float32).contiguous()

                # q[b, h]
                q_bh = q[b, h].to(torch.float32).contiguous()  # [128]

                # Launch Triton kernel for this (b, h)
                grid = (1,)
                softmax_attention_bh[grid](
                    q_bh,                      # Q_ptr
                    K_t_h,                     # K_ptr, shape [NUM_TOKENS, HEAD_DIM], contiguous
                    V_t_h,                     # V_ptr, shape [NUM_TOKENS, HEAD_DIM], contiguous
                    output[b, h],              # OUT_ptr, vector [HEAD_DIM]
                    lse[b, h],                 # LSE_ptr, scalar
                    NUM_TOKENS=num_tokens_b,
                    HEAD_DIM=head_dim,
                    SM_SCALE=sm_scale,
                    LOG2_INVERSE=1.4426950408889634,  # 1 / ln(2)
                )

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
