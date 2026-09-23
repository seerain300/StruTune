import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_gqa_bh_kernel(
    q_ptr,          # *bfloat16, [B, H, D]
    k_ptr,          # *float32,  [N, D], where N = num_kv_heads
    v_ptr,          # *float32,  [N, D]
    kv_indptr_ptr,  # *int32,    [B+1]
    kv_indices_ptr, # *int32,    [num_kv_indices]
    output_ptr,     # *bfloat16, [B, H, D]
    lse_ptr,        # *float32,  [B, H]
    sm_scale,       # float32 scalar
    B: tl.constexpr,        # batch size
    H: tl.constexpr,        # num query heads
    D: tl.constexpr,        # head dim
    N: tl.constexpr,        # num kv heads
    gqa_ratio: tl.constexpr # H // N (e.g., 4)
):
    # One program per (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Read token range for this batch
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    num_tokens = end - start

    # GQA mapping: each query head uses a specific KV head
    kvh = h // gqa_ratio  # 0..N-1

    # Load q vector for this (b, h): q is [B, H, D] contiguous, offset = b*(H*D) + h*D
    q_base = q_ptr + b * (H * D) + h * D
    q_vec = tl.load(q_base).to(tl.float32)  # [D] float32

    # Prepare base pointers for k and v rows for kvh
    k_row_base = k_ptr + kvh * D  # points to start of kvh row in [N, D]
    v_row_base = v_ptr + kvh * D  # points to start of kvh row in [N, D]

    # Vectorize across tokens: offs in [0, BLOCK_T)
    BLOCK_T = 128  # tuneable tile size; D=128 works well
    offs = tl.arange(0, BLOCK_T)  # [BLOCK_T]
    mask = offs < num_tokens      # validity mask

    # Load token indices for this batch
    tok_idx = tl.load(kv_indices_ptr + start + offs, mask=mask, other=0).to(tl.int32)  # [BLOCK_T], int32

    # Load K_chunk and V_chunk: shapes [BLOCK_T, D]
    # k_ptr is [N, D], so element at (kvh, tok) is k_ptr[kvh*D + tok*D + arange(D)]
    # But more cleanly: k_ptr[kvh, tok, :] == k_ptr[kvh*D + tok*D + arange(D)]
    k_offsets = kvh * D + tok_idx * D + tl.arange(0, D)  # [BLOCK_T, D]
    v_offsets = kvh * D + tok_idx * D + tl.arange(0, D)  # [BLOCK_T, D]

    # Mask invalid tokens by loading zeros for invalid positions
    k_chunk = tl.load(k_row_base + k_offsets, mask=mask[:, None], other=0.0)  # [BLOCK_T, D] float32
    v_chunk = tl.load(v_row_base + v_offsets, mask=mask[:, None], other=0.0)  # [BLOCK_T, D] float32

    # Compute logits_scaled_vec: [BLOCK_T] = (q_vec · k_chunk_row) * sm_scale
    # q_vec: [D], k_chunk_row[i, :]: [D]
    dot_row = tl.sum(q_vec[None, :] * k_chunk, axis=1)  # [BLOCK_T]
    logits_scaled_vec = dot_row * sm_scale              # [BLOCK_T] float32

    # Mask invalid positions: set logits to -inf so they don't affect max/sum
    neg_inf = -float("inf")
    logits_scaled_vec = tl.where(mask, logits_scaled_vec, neg_inf)  # [BLOCK_T]

    # Compute LSE (base-2): max over valid, sum_exp = sum(exp(logits - max))
    max_logit = tl.max(logits_scaled_vec, axis=0)  # scalar
    sum_exp = tl.sum(tl.exp(logits_scaled_vec - max_logit), axis=0)  # scalar
    lse_val = max_logit + tl.log(sum_exp) / tl.log(2.0)  # float32 scalar
    tl.store(lse_ptr + b * H + h, lse_val)

    # Compute attention weights and accumulate output
    attn_vec = tl.exp(logits_scaled_vec - max_logit) / sum_exp  # [BLOCK_T]
    attn_vec = tl.where(mask, attn_vec, 0.0)  # [BLOCK_T]

    # acc += sum_t attn_vec[t] * V_chunk[t, :]
    # V_chunk[t, :] is [D], attn_vec[t] is scalar
    acc = tl.zeros((D,), dtype=tl.float32)
    # Triton supports broadcasting on reduction: sum over rows
    # Using a loop to accumulate per dimension is fine:
    for t in range(BLOCK_T):
        # If mask[t] is False, attn_vec[t] is 0, so no contribution
        v_row = v_chunk[t, :]  # [D] float32
        acc += attn_vec[t] * v_row

    # Store output[b, h, :] as bfloat16
    out_base = output_ptr + b * (H * D) + h * D
    acc_bf16 = acc.to(tl.bfloat16)
    tl.store(out_base, acc_bf16)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        device = q.device
        # Ensure CUDA tensors
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All tensors must be on CUDA for Triton execution."

        # Shapes (assertions based on original setup)
        B, H, D = q.shape
        assert H == 32, "num_qo_heads must be 32"
        assert k_cache.shape[2] == 8, "num_kv_heads must be 8"
        assert D == 128, "head_dim must be 128"

        # Prepare k_ptr and v_ptr as [N, D] float32 by squeezing dim=1
        N = k_cache.shape[2]
        k_ptr = k_cache.squeeze(1).to(torch.float32)  # [N, D]
        v_ptr = v_cache.squeeze(1).to(torch.float32)  # [N, D]

        # Allocate outputs
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        gqa_ratio = H // N  # 4

        attention_gqa_bh_kernel[grid](
            q, k_ptr, v_ptr, kv_indptr, kv_indices, output, lse, sm_scale,
            B=B, H=H, D=D, N=N, gqa_ratio=gqa_ratio,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
