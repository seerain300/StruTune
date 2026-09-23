import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_output_kernel(
    q_ptr,            # *bfloat16, [B, H, D], contiguous
    token_ids_ptr,    # *int32,    [B, T_CONST], contiguous
    k_ptr,            # *bfloat16, [P, 1, N, D], contiguous (we assume P>=max tokens; will cast to float32 in-kernel)
    v_ptr,            # *bfloat16, [P, 1, N, D], contiguous
    output_ptr,       # *bfloat16, [B, H, D], contiguous
    lse_ptr,          # *float32,  [B, H], contiguous
    sm_scale,         # float32 scalar
    B: tl.constexpr,        # batch size
    H: tl.constexpr,        # num query heads (32)
    D: tl.constexpr,        # head dim (128)
    T_CONST: tl.constexpr,  # compile-time max tokens (>= actual max)
    gqa_ratio: tl.constexpr # H // N (e.g., 4)
):
    # Program ids for batch and head
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load q vector for (b, h) as float32
    q_base = q_ptr + b * (H * D) + h * D
    q_vec = tl.load(q_base).to(tl.float32)  # [D]

    # GQA mapping: kv head for query head h
    kvh = h // gqa_ratio  # 0..7

    # Initialize lse components
    l_max = -float("inf")
    lse_sum = 0.0
    out_vec = tl.zeros((D,), dtype=tl.float32)

    # Loop over tokens (static range)
    for t in range(T_CONST):
        tok_id = tl.load(token_ids_ptr + b * T_CONST + t).to(tl.int32)
        if tok_id >= 0:
            # Load k_row and v_row for this token: k_ptr[tok_id, 0, kvh, :]
            # Cast to float32 for stable math
            # Note: Triton supports indexing pointer by scalar, and tl.load returns vector.
            k_row = tl.load(k_ptr + tok_id * (1 * N * D) + kvh * D).to(tl.float32)  # [D]
            v_row = tl.load(v_ptr + tok_id * (1 * N * D) + kvh * D).to(tl.float32)  # [D]

            # Compute logits_scaled
            logits = tl.sum(q_vec * k_row, axis=0)  # scalar
            logits_scaled = logits * sm_scale

            # Update l_max
            l_max = tl.maximum(l_max, logits_scaled)

            # Accumulate sum of exp(logits_scaled - l_max)
            exp_term = tl.exp(logits_scaled - l_max)
            lse_sum += exp_term

            # Accumulate output: out_vec += exp_term * v_row
            out_vec += exp_term * v_row

    # Write lse (base-2): lse = l_max + log(lse_sum) / ln(2)
    inv_log2 = 1.0 / math.log(2.0)
    lse_b_h = l_max + tl.log(lse_sum) * inv_log2
    tl.store(lse_ptr + b * H + h, lse_b_h)

    # Store output vector for this (b, h)
    out_base = output_ptr + b * (H * D) + h * D
    tl.store(out_base, out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # q: [B, H, D], B,BATCH, H=32, D=128, dtype bfloat16 (assertable)
        B, H, D = q.shape
        assert H == 32 and D == 128, "This Triton implementation expects H=32 and D=128."
        N = 8
        gqa_ratio = H // N  # 4

        # Compute actual token counts per batch and prepare token_ids_all with padding (-1)
        num_tokens_per_b = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_per_b.append(end - start)

        max_tokens = max(num_tokens_per_b) if num_tokens_per_b else 0
        # Triton requires static loop bound; choose T_CONST >= max_tokens, e.g. 1024
        T_CONST = 1024

        token_ids_all = torch.empty((B, T_CONST), dtype=torch.int32, device=q.device)
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens = end - start
            # copy valid token_ids, pad with -1
            if num_tokens > 0:
                token_ids_all[b, :num_tokens] = kv_indices[start:start + num_tokens].to(torch.int32)
            token_ids_all[b, num_tokens:] = -1

        # Output and lse tensors
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel
        grid = (B, H)
        lse_and_output_kernel[grid](
            q, token_ids_all, k_cache, v_cache, output, lse, sm_scale,
            B, H, D, T_CONST, gqa_ratio,
            num_warps=4, num_stages=2
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
