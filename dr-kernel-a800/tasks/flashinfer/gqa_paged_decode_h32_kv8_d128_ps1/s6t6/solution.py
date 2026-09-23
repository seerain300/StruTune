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
    q_ptr,           # *ptr to q, shape [B, H, D], float32
    k_ptr,           # *ptr to k_cache, shape [N_total, num_kv_heads, D], but gathered by indices
    v_ptr,           # *ptr to v_cache, shape [N_total, num_kv_heads, D], similarly gathered
    kv_indices_ptr,  # *ptr to int32 indices, shape [num_kv_indices]
    kv_indptr_ptr,   # *ptr to int32 indptr, shape [B+1]
    lse_ptr,         # *ptr to lse, shape [B, H], float32
    out_ptr,         # *ptr to output, shape [B, H, D], float32
    B: tl.constexpr,         # batch size
    H: tl.constexpr,         # num_qo_heads
    D: tl.constexpr,         # head_dim
    num_kv_heads: tl.constexpr,  # number of kv heads
    sm_scale,                # scalar float32
    N_TOTAL: tl.constexpr,   # loop bound (e.g., 128), mask beyond actual_num_tokens
):
    # Program ids for batch and head
    b = tl.program_id(0)  # int
    h = tl.program_id(1)  # int

    # Compute q vector for this (b, h): q[b, h, :]
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d).to(tl.float32)

    # Determine token range for this batch from kv_indptr
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start

    # First pass: streaming logsumexp over tokens
    m = -float("inf")
    sumexp = 0.0  # scalar float32
    ln2 = 0.6931471805599453
    for nn in range(0, N_TOTAL):
        valid = start + nn < end
        idx = tl.load(kv_indices_ptr + start + nn, mask=valid, other=0).to(tl.int32)

        kv_group = idx * num_kv_heads
        kv_head_offset = kv_head * D

        k_base = k_ptr + kv_group + kv_head_offset
        v_base = v_ptr + kv_group + kv_head_offset

        # Load k_vec and compute dot with q_vec
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

    # lse for this (b, h)
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        valid = start + nn < end
        idx = tl.load(kv_indices_ptr + start + nn, mask=valid, other=0).to(tl.int32)

        kv_group = idx * num_kv_heads
        kv_head_offset = kv_head * D

        k_base = k_ptr + kv_group + kv_head_offset
        v_base = v_ptr + kv_group + kv_head_offset

        # Load v_vec
        v_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_vec[d] = tl.load(v_base + d).to(tl.float32)

        # Recompute dot and logits_scaled
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_vec[d] = tl.load(k_base + d).to(tl.float32)
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]
        logit = dot * sm_scale

        softmax = tl.exp(logit - m) / (sumexp * ln2)
        out_vec += softmax * v_vec

    # Store output vector for this (b, h)
    for d in range(0, D):
        tl.store(out_ptr + b * H * D + h * D + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # No PyTorch ops here; all compute in Triton
        assert q.dim() == 3 and k_cache.dim() == 4 and v_cache.dim() == 4
        B, H, D = q.shape
        num_kv_heads = 8  # fixed by original assertions
        gqa_ratio = H // num_kv_heads  # 4
        device = q.device

        # Ensure inputs are contiguous and float32 for compute
        q_f32 = q.contiguous().to(torch.float32)
        k_f32 = k_cache.contiguous().to(torch.float32)
        v_f32 = v_cache.contiguous().to(torch.float32)
        kv_indices_i32 = kv_indices.contiguous().to(torch.int32)
        kv_indptr_i32 = kv_indptr.contiguous().to(torch.int32)

        # Allocate outputs (float32 for compute, cast to bfloat16 after)
        output = torch.empty((B, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        triton.run(
            _forward_kernel_bh,
            grid=grid,
            num_warps=4,
            num_stages=2,
            q_ptr=q_f32,
            k_ptr=k_f32,
            v_ptr=v_f32,
            kv_indices_ptr=kv_indices_i32,
            kv_indptr_ptr=kv_indptr_i32,
            lse_ptr=lse,
            out_ptr=output,
            B=B,
            H=H,
            D=D,
            num_kv_heads=num_kv_heads,
            sm_scale=float(sm_scale),
            N_TOTAL=128,
        )

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
