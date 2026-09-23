import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute per (b, h) attention output and lse
# Arguments:
#   OUT_ptr: float32[head_dim] output vector for (b, h)
#   LSE_ptr: float32[1] scalar lse for (b, h)
#   Q_ptr: float32[head_dim] Q vector for this head
#   K_ptr: float32[num_tokens*head_dim] flattened K rows for this batch
#   V_ptr: float32[num_tokens*head_dim] flattened V rows for this batch
#   num_tokens: int runtime
#   sm_scale: float
#   LOG2_INVERSE: float (1 / ln(2))
#   head_dim: constexpr (compile-time)
#   kv_head: constexpr (GQA mapping h // 4)
@triton.jit
def softmax_and_attention_single_bh(
    OUT_ptr, LSE_ptr,
    Q_ptr, K_ptr, V_ptr,
    num_tokens: tl.int32,
    sm_scale: tl.float32,
    LOG2_INVERSE: tl.float32,
    head_dim: tl.constexpr,
    kv_head: tl.constexpr
):
    # Load Q vector [head_dim]
    i = tl.arange(0, head_dim)
    q_vec = tl.load(Q_ptr + i)  # [head_dim]

    # First pass: compute logsumexp of scaled logits over all tokens
    running_max = -float("inf")
    running_sum = 0.0

    t = 0
    while t < num_tokens:
        # Load k_row [head_dim] from flattened K_ptr
        k_row = tl.load(K_ptr + t * head_dim + i)  # [head_dim]
        # Dot product
        logits = tl.sum(q_vec * k_row, axis=0)     # scalar
        scaled = logits * sm_scale
        # Numerically stable update
        if scaled > running_max:
            running_sum = running_sum * tl.exp(running_max - scaled) + 1.0
            running_max = scaled
        else:
            running_sum = running_sum + tl.exp(scaled - running_max)
        t += 1

    # lse = log(running_sum) + running_max, scaled by 1/ln(2)
    lse_val = tl.log(running_sum) + running_max
    lse_val = lse_val * LOG2_INVERSE

    # Store lse scalar
    tl.store(LSE_ptr, lse_val)

    # Second pass: compute attention and accumulate output vector
    out_vec = tl.zeros((head_dim,), dtype=tl.float32)
    t = 0
    while t < num_tokens:
        k_row = tl.load(K_ptr + t * head_dim + i)  # [head_dim]
        v_row = tl.load(V_ptr + t * head_dim + i)  # [head_dim]
        logits = tl.sum(q_vec * k_row, axis=0)     # scalar
        scaled = logits * sm_scale
        attn = tl.exp(scaled - lse_val)            # scalar
        out_vec += attn * v_row                    # [head_dim]
        t += 1

    # Store output vector
    tl.store(OUT_ptr + i, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, 32, 128], dtype=bfloat16
        k_cache: [num_pages, num_kv_heads, head_dim], dtype=bfloat16
        v_cache: [num_pages, num_kv_heads, head_dim], dtype=bfloat16
        kv_indptr: [len_indptr], int32, cumulative starts per batch
        kv_indices: [num_kv_indices], int32, token indices
        sm_scale: float
        returns: (output: [B, 32, 128], bfloat16), (lse: [B, 32], float32)
        """
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Triton kernels require CUDA tensors"
        B, num_qo_heads, head_dim = q.shape
        num_kv_heads = k_cache.shape[1]
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        device = q.device
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        # Per-batch token range: [kv_indptr[b], kv_indptr[b+1]]
        for b in range(B):
            start_b = int(kv_indptr[b].item())
            end_b = int(kv_indptr[b + 1].item())
            num_tokens_b = end_b - start_b
            if num_tokens_b == 0:
                lse[b].fill_(-float("inf"))
                output[b].zero_()
                continue

            # Gather token indices for this batch
            token_indices = kv_indices[start_b:end_b].to(torch.long).contiguous()  # [num_tokens_b]

            # For each query head
            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio

                # Gather K and V rows: [num_tokens_b, head_dim] for this kv_head
                k_rows = k_cache[token_indices, kv_head, :]  # bfloat16
                v_rows = v_cache[token_indices, kv_head, :]  # bfloat16

                # Cast to float32 for Triton computation
                K_t = k_rows.to(torch.float32).contiguous().view(-1)  # flatten to [num_tokens_b * head_dim]
                V_t = v_rows.to(torch.float32).contiguous().view(-1)

                # Q vector for this head
                q_vec = q[b, h, :].to(torch.float32).contiguous()  # [head_dim]

                # Output vector for this (b, h)
                out_vec = torch.empty((head_dim,), dtype=torch.float32, device=device)

                # Launch Triton kernel: one program per (b, h)
                grid = (1, 1)
                softmax_and_attention_single_bh[grid](
                    out_vec,                 # OUT_ptr
                    lse[b, h],               # LSE_ptr (scalar)
                    q_vec,                   # Q_ptr
                    K_t,                     # K_ptr flattened
                    V_t,                     # V_ptr flattened
                    num_tokens_b,            # runtime: number of tokens in this batch
                    float(sm_scale),         # sm_scale
                    1.4426950408889634,      # LOG2_INVERSE (1 / ln(2))
                    head_dim,                # constexpr
                    kv_head                  # constexpr GQA mapping
                )

                # Store output vector into [B, 32, 128]
                output[b, h, :] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
