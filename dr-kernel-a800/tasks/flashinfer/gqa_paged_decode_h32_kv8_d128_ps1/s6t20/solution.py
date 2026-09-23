import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _forward_kernel_bh(
    q_ptr,           # *fp16, shape [B, H, D]
    k_ptr,           # *fp16, shape [N_total, 1, num_kv_heads, D]
    v_ptr,           # *fp16, shape [N_total, 1, num_kv_heads, D]
    kv_indices_ptr,  # *int32, shape [num_kv_indices]
    kv_indptr_ptr,   # *int32, shape [B+1]
    lse_ptr,         # *fp32, shape [B, H]
    out_ptr,         # *fp32, shape [B, H, D]
    B: tl.constexpr,           # batch size
    H: tl.constexpr,           # num query heads
    D: tl.constexpr,           # head dim
    NUM_QO_HEADS: tl.constexpr,  # 32
    NUM_KV_HEADS: tl.constexpr,  # 8
    SM_SCALE: tl.constexpr,      # 1/sqrt(D)
    N_TOTAL: tl.constexpr,       # loop bound for tokens (e.g., 128)
):
    # program ids
    b = tl.program_id(0)  # 0..B-1
    h = tl.program_id(1)  # 0..H-1

    # GQA mapping
    gqa_ratio = NUM_QO_HEADS // NUM_KV_HEADS
    kv_head = h // gqa_ratio

    # load q vector q[b, h, :]
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_val = tl.load(q_base + d).to(tl.float32)
        q_vec[d] = q_val

    # gather token indices for this batch
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start

    # first pass: streaming logsumexp in base-2 across tokens
    m = -float("inf")
    sumexp = 0.0
    ln2 = 0.6931471805599453  # log(2)

    for nn in range(0, N_TOTAL):
        mask_n = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask_n, other=0).to(tl.int32)

        k_row_ptr = k_ptr + idx * NUM_KV_HEADS * D + kv_head * D
        v_row_ptr = v_ptr + idx * NUM_KV_HEADS * D + kv_head * D

        # load k row
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_row_ptr + d).to(tl.float32)
            k_vec[d] = k_val

        # dot product q · k
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]

        logit = dot * SM_SCALE
        new_m = tl.maximum(m, logit)
        sumexp = tl.where(mask_n, sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m), sumexp)
        m = new_m

    # lse for this (b, h)
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        mask_n = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask_n, other=0).to(tl.int32)

        k_row_ptr = k_ptr + idx * NUM_KV_HEADS * D + kv_head * D
        v_row_ptr = v_ptr + idx * NUM_KV_HEADS * D + kv_head * D

        v_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_val = tl.load(v_row_ptr + d).to(tl.float32)
            v_row[d] = v_val

        # recompute dot and softmax
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_row_ptr + d).to(tl.float32)
            k_vec[d] = k_val

        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]

        logit = dot * SM_SCALE
        softmax = tl.exp(logit - m) / (sumexp * ln2)
        out_vec += tl.where(mask_n, softmax * v_row, v_row)

    # store output
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: no PyTorch ops
        assert TRITON_AVAILABLE, "Triton is not available."

        B, H, D = q.shape
        # Ensure inputs are contiguous (the provided inputs are already contiguous)
        q_c = q.contiguous()
        k_c = k_cache.contiguous()  # [N, 1, 8, D]
        v_c = v_cache.contiguous()  # [N, 1, 8, D]
        kv_indptr_c = kv_indptr.contiguous()
        kv_indices_c = kv_indices.contiguous()

        # Allocate outputs as float32; we'll convert to bfloat16 before returning
        output = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q_c, k_c, v_c, kv_indices_c, kv_indptr_c, lse, output,
            B, H, D, NUM_QO_HEADS=32, NUM_KV_HEADS=8, SM_SCALE=sm_scale, N_TOTAL=128,
            num_warps=4, num_stages=2,
        )

        # Convert output to bfloat16 to match the original signature
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
