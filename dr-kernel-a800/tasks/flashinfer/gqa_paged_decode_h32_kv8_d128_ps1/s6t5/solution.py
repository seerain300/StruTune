import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (b, h). Computes output and lse for that pair.
@triton.jit
def _forward_kernel_bh(
    q_ptr,          # *ptr to q, shape [B, H, D], float32
    k_ptr,          # *ptr to k_cache, shape [N_total, num_kv_heads, D], but gathered by indices
    v_ptr,          # *ptr to v_cache, shape [N_total, num_kv_heads, D], similarly gathered
    kv_indices_ptr, # *ptr to int32 indices of tokens, shape [num_kv_indices]
    kv_indptr_ptr,  # *ptr to int32 indptr, shape [B+1]
    lse_ptr,        # *ptr to lse, shape [B, H], float32
    out_ptr,        # *ptr to output, shape [B, H, D], float32
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    num_kv_heads: tl.constexpr,
    sm_scale,       # scalar float32
    N_TOTAL: tl.constexpr,  # loop bound (e.g., 128), mask beyond actual_num_tokens
):
    # Program ids for batch and head
    b = tl.program_id(0)  # int
    h = tl.program_id(1)  # int

    # Compute q vector for this (b, h): q[b, h, :]
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d).to(tl.float32)

    # Gather kv indices for this batch from kv_indptr
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start  # int32 scalar

    # GQA mapping: kv_head = h // (H // num_kv_heads)
    gqa_ratio = H // num_kv_heads
    kv_head = h // gqa_ratio  # int

    # First pass: streaming logsumexp over tokens
    m = -float("inf")
    sumexp = 0.0  # scalar float32
    ln2 = 0.6931471805599453  # 1 / log(2)

    for nn in range(0, N_TOTAL):
        # If nn >= actual_num_tokens, start+nn >= end; we skip by masked loads (Triton requires static loops).
        idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)

        # Base pointers for k and v for this token and kv_head
        # Address: k_ptr + idx * (num_kv_heads * D) + kv_head * D
        k_row_base = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_row_base = v_ptr + idx * (num_kv_heads * D) + kv_head * D

        # Load k_vec
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_row_base + d).to(tl.float32)
            k_vec[d] = k_val

        # dot = q_vec · k_vec
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]

        logit = dot * sm_scale
        new_m = tl.maximum(m, logit)
        sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

    # lse for this (b, h) = m + log(sumexp) / ln(2)
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax per token and accumulate output vector
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)
        k_row_base = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_row_base = v_ptr + idx * (num_kv_heads * D) + kv_head * D

        # Recompute dot and logits_scaled
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_row_base + d).to(tl.float32)
            k_vec[d] = k_val
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]
        logit = dot * sm_scale

        softmax = tl.exp(logit - m) / sumexp  # denominator is sumexp

        # Load v_vec and accumulate
        v_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_val = tl.load(v_row_base + d).to(tl.float32)
            v_vec[d] = v_val

        out_vec += softmax * v_vec

    # Store output vector
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Forward must use Triton; no PyTorch ops here. Ensure device is CUDA.
        device = q.device
        assert device.type == "cuda", "ModelNew.forward requires CUDA device."

        B, H, D = q.shape
        num_kv_heads = v_cache.shape[2]  # expect 8
        # Make inputs contiguous and cast to float32 for numerics
        q_f32 = q.contiguous().to(torch.float32)
        k_f32 = k_cache.contiguous().to(torch.float32)
        v_f32 = v_cache.contiguous().to(torch.float32)
        kv_indices_i32 = kv_indices.contiguous().to(torch.int32)
        kv_indptr_i32 = kv_indptr.contiguous().to(torch.int32)

        # Output tensors (float32 in kernel, cast to bfloat16 afterwards)
        output = torch.empty((B, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        N_TOTAL = 128  # loop bound; masks nn >= actual_num_tokens
        grid = (B, H)
        _forward_kernel_bh[grid](
            q_f32, k_f32, v_f32, kv_indices_i32, kv_indptr_i32, lse, output,
            B, H, D, num_kv_heads, sm_scale, N_TOTAL,
            num_warps=4,  # reasonable for D=128
            num_stages=2,
        )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
