import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_single_q_idx_h_kernel(
    q_ptr,           # *fp32, [T, H, D], contiguous
    output_ptr,      # *bf16, [T, H, D], contiguous
    lse_ptr,         # *fp32, [T, H], contiguous
    sm_scale,        # fp32
    # Fixed dimensions known to host
    T: tl.constexpr,       # total_q (for bounds safety; not used directly)
    H: tl.constexpr,       # num_qo_heads
    D: tl.constexpr,       # head_dim
    # Precomputed K/V slices for this (b, h) up to max_kv_idx (dynamic at launch)
    k_ptr,              # *fp32, [BLOCK_K, D], contiguous
    v_ptr,              # *fp32, [BLOCK_K, D], contiguous
    BLOCK_K: tl.constexpr,  # number of KV tokens to consider for this triple
    qo_start,           # int32, segment start for queries
    qo_end,             # int32, segment end for queries
    kv_start,           # int32, segment start for KV
    kv_end,             # int32, segment end for KV
):
    # Program ids: (b, q_idx, h) determined by launch grid
    # Triton doesn't expose program_id mapping here; host sets grid accordingly.
    # We rely on host to launch with grid=(num_segments, num_q_tokens, num_qo_heads)
    # and pass segment boundaries as scalar args.

    # Compute global query index and max number of KV tokens to consider (causal mask)
    global_q_idx = qo_start  # Not used; we assume grid encodes q_idx separately.
    # The host launches this kernel with grid dimension for q_idx as program_id(1).
    # Triton kernels cannot read tensors, so we do not compute global_q_idx here.
    # Instead, host provides qo_start, qo_end, kv_start, kv_end; and launches
    # with q_idx as program_id(1).

    # To make this work, we remove this kernel's reliance on q_idx and assume
    # host launches a version that computes q_idx internally. However, Triton
    # requires static code. Therefore, we instead split logic into host and
    # kernel such that host computes q_idx and passes it. Triton doesn't allow
    # dynamic q_idx in signature; so we restructure: host launches a kernel
    # that expects q_idx as a tl.constexpr argument. To keep it simple and
    # correct, we redesign forward to launch this kernel with q_idx passed as
    # tl.constexpr. Triton supports constexpr scalar args.

    # Conclusion: we pass q_idx as tl.constexpr to the kernel. The launch grid
    # will be (num_segments, num_q_tokens, num_qo_heads). Inside the kernel,
    # we can read tl.program_id(1) which corresponds to q_idx.

    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Recompute global_q_idx based on qo_start and q_idx
    global_q_idx = qo_start + q_idx

    # Compute delta for causal masking
    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    delta = num_kv_tokens - num_q_tokens
    max_kv_idx = tl.minimum(q_idx + 1 + delta, num_kv_tokens)  # can be 0; handle in math

    # Load q vector for this head: q[global_q_idx, h, :]
    q_base = global_q_idx * H * D
    q_offset = q_base + h * D
    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32

    # Compute logits for k in [0..BLOCK_K-1], masked by max_kv_idx
    logits_scaled = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(BLOCK_K):
        valid = k < max_kv_idx
        # k_ptr is [BLOCK_K, D], contiguous
        k_row = tl.load(k_ptr + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
        prod = q_vec * k_row
        logits_scaled[k] = tl.sum(prod, axis=0)  # reduce over D

    # Scale logits
    logits_scaled = logits_scaled * sm_scale

    # Compute logsumexp in natural log, then convert to base-2
    m = logits_scaled[0]
    for i in range(1, BLOCK_K):
        m = tl.maximum(m, logits_scaled[i])
    sum_exp = 0.0
    for i in range(BLOCK_K):
        sum_exp += tl.exp(logits_scaled[i] - m)
    lse_val = m + tl.log(sum_exp)  # natural log
    lse_base2 = lse_val / tl.log(2.0)
    # Atomic add lse contribution for this (global_q_idx, h)
    tl.atomic_add(lse_ptr + global_q_idx * H + h, lse_base2)

    # Compute softmax of logits_scaled (masked k >= max_kv_idx contribute 0 in scaling)
    for i in range(BLOCK_K):
        logits_scaled[i] = -float('inf') if (i >= max_kv_idx) else logits_scaled[i]
    denom = 0.0
    for i in range(BLOCK_K):
        denom += tl.exp(logits_scaled[i] - m)
    softmax_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(BLOCK_K):
        softmax_vals[i] = tl.exp(logits_scaled[i] - m) / denom

    # Compute output vector: out_vec += softmax[k] * v_rows[k, :]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for k in range(BLOCK_K):
        valid = k < max_kv_idx
        v_row = tl.load(v_ptr + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
        out_vec += softmax_vals[k] * v_row

    # Store output as bfloat16 to output_ptr[global_q_idx, h, :]
    out_base = global_q_idx * H * D
    out_offset = out_base + h * D
    tl.store(output_ptr + out_offset + tl.arange(0, D), out_vec.to(tl.bfloat16), mask=tl.arange(0, D) < D)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and contiguity
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()  # [T, H, D]
        # k_cache and v_cache are [N,1,8,128]; squeeze dim=1 => [N,8,128]
        k_cache_f32 = k_cache.to(torch.float32).contiguous()
        v_cache_f32 = v_cache.to(torch.float32).contiguous()
        qo_indptr = qo_indptr.to(torch.int32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        total_q, num_qo_heads, head_dim = q_f32.shape
        len_indptr = qo_indptr.shape[0]
        num_segments = len_indptr - 1

        # Flatten caches: [N, 8, D]
        k_cache_flat = k_cache_f32.squeeze(1)  # [N, 8, 128]
        v_cache_flat = v_cache_f32.squeeze(1)  # [N, 8, 128]

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.zeros((total_q, num_qo_heads), dtype=torch.float32, device=device)

        gqa_ratio = num_q


def run(*args):
    return ModelNew()(*args)
