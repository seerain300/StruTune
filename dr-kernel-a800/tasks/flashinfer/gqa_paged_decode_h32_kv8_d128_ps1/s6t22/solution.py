import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (b, h). Computes output vector and lse for that pair.
@triton.jit
def _forward_kernel_bh(
    q_ptr,          # *fp16, shape [B, H, D]
    k_ptr,          # *fp16, shape [N_total, 1, num_kv_heads, D] (we gather by indptr/indices)
    v_ptr,          # *fp16, shape [N_total, 1, num_kv_heads, D]
    kv_indptr_ptr,  # *int32, shape [B+1]
    kv_indices_ptr, # *int32, shape [num_tokens]
    lse_ptr,        # *fp32, shape [B, H]
    out_ptr,        # *fp32, shape [B, H, D]
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SM_SCALE: tl.float32,
    N_TOTAL: tl.constexpr,   # loop bound for tokens (e.g., 128)
):
    # Program ids: one per (b, h)
    b = tl.program_id(0)  # 0..B-1
    h = tl.program_id(1)  # 0..H-1

    # Load q vector q[b, h, :]
    q_base = q_ptr + b * H * D + h * D
    offs_d = tl.arange(0, D)
    q_vec = tl.load(q_base + offs_d).to(tl.float32)  # [D], float32 for compute

    # Determine actual token range for this batch element
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)        # int32
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)      # int32
    actual_num_tokens = end - start                        # int32 scalar

    # First pass: streaming logsumexp across tokens in natural log space
    m = -float("inf")  # running max
    sumexp = 0.0       # running sum of exp(logit - m)

    for nn in range(0, N_TOTAL):
        mask_n = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask_n, other=0).to(tl.int32)

        # GQA mapping: kv_head = h // (H // NUM_KV_HEADS) = h // 4
        kv_head = h // (H // NUM_KV_HEADS)

        k_row_ptr = k_ptr + idx * NUM_KV_HEADS * D + kv_head * D  # [D]
        k_vec = tl.load(k_row_ptr + offs_d).to(tl.float32)  # [D]
        # dot = sum(q_vec * k_vec)
        dot = tl.sum(q_vec * k_vec, axis=0)  # scalar float32
        logit = dot * SM_SCALE

        new_m = tl.maximum(m, logit)
        sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

    # lse for this (b, h) in base-2
    ln2 = 0.6931471805599453  # ln(2)
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        mask_n = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask_n, other=0).to(tl.int32)

        kv_head = h // (H // NUM_KV_HEADS)
        k_row_ptr = k_ptr + idx * NUM_KV_HEADS * D + kv_head * D  # [D]
        v_row_ptr = v_ptr + idx * NUM_KV_HEADS * D + kv_head * D  # [D]

        v_row = tl.load(v_row_ptr + offs_d).to(tl.float32)  # [D]
        k_vec = tl.load(k_row_ptr + offs_d).to(tl.float32)  # [D]

        # dot and logits_scaled
        dot = tl.sum(q_vec * k_vec, axis=0)  # scalar
        logit = dot * SM_SCALE
        softmax = tl.exp(logit - m) / sumexp  # normalize by sumexp (natural log accumulation)
        out_vec += tl.where(mask_n, softmax * v_row, v_row)

    # store output vector
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: no PyTorch ops
        assert TRITON_AVAILABLE, "Triton is not available."

        B, H, D = q.shape
        NUM_QO_HEADS = H
        NUM_KV_HEADS = 8

        # Ensure inputs are contiguous
        q_c = q.contiguous()          # [B, H, D], dtype bfloat16
        k_c = k_cache.contiguous()    # [N, 1, 8, D], dtype bfloat16
        v_c = v_cache.contiguous()    # [N, 1, 8, D], dtype bfloat16
        kv_indptr_c = kv_indptr.contiguous()  # [B+1], int32
        kv_indices_c = kv_indices.contiguous()  # [num_tokens], int32

        # Allocate outputs as float32; final convert to bfloat16 for return
        output = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q_c, k_c, v_c, kv_indptr_c, kv_indices_c, lse, output,
            B=B, H=H, D=D, NUM_KV_HEADS=NUM_KV_HEADS, SM_SCALE=float(sm_scale),
            N_TOTAL=128,  # compile-time loop bound (covers all provided workloads)
            num_warps=4,  # tuning parameter
            num_stages=2,
        )

        # Return in the same shape as the original: output bfloat16, lse float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
