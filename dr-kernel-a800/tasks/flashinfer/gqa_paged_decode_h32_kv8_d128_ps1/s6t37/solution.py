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
    q_ptr,             # *fp16, shape [B, H, D]
    k_ptr,             # *fp16, shape [N_total, 1, num_kv_heads, D]
    v_ptr,             # *fp16, shape [N_total, 1, num_kv_heads, D]
    kv_indices_ptr,    # *int32, shape [num_kv_indices]
    kv_indptr_ptr,     # *int32, shape [B+1]
    lse_ptr,           # *float32, shape [B, H]
    out_ptr,           # *float32, shape [B, H, D]
    SM_SCALE,          # scalar float32 (1 / sqrt(D))
    N_TOTAL: tl.constexpr,   # loop bound (e.g., 128)
    D: tl.constexpr,         # head dim, 128
    NUM_KV_HEADS: tl.constexpr,  # 8
):
    # Program ids for batch and head
    b = tl.program_id(0)  # int32
    h = tl.program_id(1)  # int32

    # GQA mapping: kv_head = h // (H // NUM_KV_HEADS) = h // 4
    kv_group = 8  # H // NUM_KV_HEADS == 32 // 8 == 4
    kv_head = h // kv_group

    # Load q vector q[b, h, :]
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d).to(tl.float32)

    # Load start and end from kv_indptr
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start  # Triton scalar int32

    # First pass: compute logsumexp of scaled logits over tokens (base-2)
    m = -float("inf")
    sumexp = 0.0  # scalar float32
    ln2 = 0.6931471805599453  # float32 scalar

    for nn in range(0, N_TOTAL):
        # Only process if nn < actual_num_tokens; mask guard
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

    # Store output
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are contiguous
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k_cache.to(torch.float32).contiguous()
        v_f32 = v_cache.to(torch.float32).contiguous()
        kv_indices = kv_indices.contiguous().to(torch.int32)
        kv_indptr = kv_indptr.contiguous().to(torch.int32)

        # Shapes
        B, H, D = q_f32.shape
        NUM_KV_HEADS = 8

        # Allocate outputs (float32; cast to bfloat16 at the end to match original)
        output = torch.empty((B, H, D), dtype=torch.float32, device=q_f32.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_f32.device)

        # Launch Triton: one program per (b, h)
        grid = (B, H)
        N_TOTAL = 128  # compile-time bound; mask nn < actual_num_tokens

        _forward_kernel_bh[grid](
            q_f32,
            k_f32,
            v_f32,
            kv_indices,
            kv_indptr,
            lse,
            output,
            sm_scale,
            N_TOTAL,
            D, NUM_KV_HEADS,
            num_warps=4,
            num_stages=2,
        )

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
