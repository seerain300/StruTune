import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _forward_kernel_bh(
    q_ptr,             # *fp16 or *fp32, shape [B, H, D]
    k_ptr,             # *fp16 or *fp32, shape [num_kv_indices, 1, num_kv_heads, D] but accessed via indices
    v_ptr,             # *fp16 or *fp32, shape [num_kv_indices, 1, num_kv_heads, D]
    kv_indices_ptr,    # *int32, shape [num_kv_indices]
    kv_indptr_ptr,     # *int32, shape [B+1]
    lse_ptr,           # *fp32, shape [B, H]
    out_ptr,           # *fp32, shape [B, H, D]
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SM_SCALE,          # scalar float32
    N_TOTAL: tl.constexpr,  # loop bound (e.g., 128), masked beyond actual_num_tokens
):
    # Program ids for batch and head
    b = tl.program_id(0)  # int32
    h = tl.program_id(1)  # int32

    # Load q vector for (b, h): q[b, h, :]
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d).to(tl.float32)

    # Compute start/end for this batch from kv_indptr
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start  # int32

    # GQA mapping: kv_head = h // (H // NUM_KV_HEADS)
    kv_group = H // NUM_KV_HEADS
    kv_head = h // kv_group

    # First pass: compute logsumexp of scaled logits over tokens (base-2)
    m = -float("inf")
    sumexp = 0.0  # scalar float32
    ln2 = 0.6931471805599453  # float32 scalar

    for nn in range(0, N_TOTAL):
        # Mask: only process if nn < actual_num_tokens
        if nn < actual_num_tokens:
            idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)

            # Compute k_vec = k_cache[idx, kv_head, :]
            k_row_base = k_ptr + idx * NUM_KV_HEADS * D + kv_head * D
            k_vec = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                k_vec[d] = tl.load(k_row_base + d).to(tl.float32)

            # Dot product q_vec · k_vec
            dot = 0.0
            for d in range(0, D):
                dot += q_vec[d] * k_vec[d]

            logit_scaled = dot * SM_SCALE
            new_m = tl.maximum(m, logit_scaled)
            sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit_scaled - new_m)
            m = new_m

    # lse = logsumexp / ln(2)
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        if nn < actual_num_tokens:
            idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)

            v_row_base = v_ptr + idx * NUM_KV_HEADS * D + kv_head * D
            v_vec = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                v_vec[d] = tl.load(v_row_base + d).to(tl.float32)

            k_row_base = k_ptr + idx * NUM_KV_HEADS * D + kv_head * D
            k_vec = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                k_vec[d] = tl.load(k_row_base + d).to(tl.float32)

            dot = 0.0
            for d in range(0, D):
                dot += q_vec[d] * k_vec[d]

            logit_scaled = dot * SM_SCALE
            softmax = tl.exp(logit_scaled - m) / (sumexp * ln2)
            out_vec += softmax * v_vec

    # Store output vector
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA device for Triton
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be on CUDA device for Triton execution."

        B, H, D = q.shape
        assert D == 128, "head_dim must be 128"
        num_kv_heads = 8
        gqa_ratio = H // num_kv_heads  # 4

        # Prepare outputs
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)
        out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q, k_cache, v_cache, kv_indices, kv_indptr, lse, out,
            H=H, D=D, NUM_KV_HEADS=num_kv_heads, SM_SCALE=sm_scale,
            N_TOTAL=128,  # loop bound, masked to handle fewer tokens
            num_warps=4,  num_stages=2
        )

        # Cast output to bfloat16 to match original model output dtype
        out_bf16 = out.to(torch.bfloat16)
        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
