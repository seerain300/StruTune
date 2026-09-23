import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (b, h)
@triton.jit
def _forward_bh_kernel(
    q_ptr,            # *fp16, [B, H, D]
    k_ptr,            # *fp16, [N_total, num_kv_heads, D] (gathered via indices)
    v_ptr,            # *fp16, [N_total, num_kv_heads, D] (gathered via indices)
    kv_indptr_ptr,    # *int32, [B+1]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    lse_ptr,          # *fp32,  [B, H]
    out_ptr,          # *fp32,  [B, H, D]
    B: tl.int32,      # batch size (runtime)
    H: tl.int32,      # num query heads (runtime)
    D: tl.int32,      # head_dim (runtime)
    num_kv_heads: tl.constexpr,  # 8
    sm_scale,         # scalar fp32
    N_TOTAL: tl.constexpr,       # loop bound, e.g., 128
):
    # program ids
    b = tl.program_id(0)  # int
    h = tl.program_id(1)  # int

    # load q[b, h, :] as float32 vector
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d).to(tl.float32)

    # compute start/end for this batch element
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start  # runtime int

    # GQA mapping: kv_head = h // 4 (since num_qo_heads=32, num_kv_heads=8)
    kv_head = h // 4

    # streaming logsumexp across tokens in base-2
    m = -float("inf")
    sumexp = 0.0
    ln2 = 0.6931471805599453  # log(2)

    for nn in range(0, N_TOTAL):
        if nn < actual_num_tokens:
            idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)

            # load k_row for this token and head
            k_base = k_ptr + idx * num_kv_heads * D + kv_head * D
            k_vec = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                k_vec[d] = tl.load(k_base + d).to(tl.float32)

            dot = 0.0
            for d in range(0, D):
                dot += q_vec[d] * k_vec[d]

            logit = dot * sm_scale
            new_m = tl.maximum(m, logit)
            sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
            m = new_m

    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        if nn < actual_num_tokens:
            idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)

            k_base = k_ptr + idx * num_kv_heads * D + kv_head * D
            v_base = v_ptr + idx * num_kv_heads * D + kv_head * D

            k_vec = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                k_vec[d] = tl.load(k_base + d).to(tl.float32)

            dot = 0.0
            for d in range(0, D):
                dot += q_vec[d] * k_vec[d]
            logit = dot * sm_scale

            softmax = tl.exp(logit - lse_val) / ln2
            v_vec = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                v_vec[d] = tl.load(v_base + d).to(tl.float32)
            out_vec += softmax * v_vec

    # store output
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton requires CUDA and availability
        assert TRITON_AVAILABLE, "Triton not available."
        assert q.is_cuda, "Inputs must be on CUDA device for Triton kernel."

        B, H, D = q.shape
        num_kv_heads = 8  # fixed as per provided code

        # Ensure inputs are contiguous and correct dtype; forward should not use PyTorch ops
        q_f32 = q.to(torch.float32)
        k_f32 = k_cache.to(torch.float32)
        v_f32 = v_cache.to(torch.float32)
        kv_indptr_i32 = kv_indptr.to(torch.int32)
        kv_indices_i32 = kv_indices.to(torch.int32)

        # Output buffers
        out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_bh_kernel[grid](
            q_f32, k_f32, v_f32, kv_indptr_i32, kv_indices_i32, lse, out,
            B, H, D,
            num_kv_heads,
            float(sm_scale),  # pass as fp32 scalar
            N_TOTAL=128       # loop bound; mask with nn < actual_num_tokens
        )

        # Force completion of kernel to avoid reading partially written outputs
        torch.cuda.synchronize()

        # Return outputs as list: [output (B,H,D) bfloat16], lse (B,H) float32
        output_bf16 = out.to(torch.bfloat16)
        return [output_bf16, lse]


def run(*args):
    return ModelNew()(*args)
