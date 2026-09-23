import torch
import math
import triton
import triton.language as tl

# Kernel: one program handles one batch element b
# It:
# - Reads q, k_cache_flat, v_cache_flat based on qo_indptr[b], qo_indptr[b+1] and kv_indptr[b], kv_indptr[b+1]
# - Iterates over q tokens in that batch, per head h, computes logits, scaled, logsumexp, softmax, and dot with V,
#   writes output and updates lse per (q_idx, h).
@triton.jit
def _batch_compute_kernel(
    q_ptr, k_ptr, v_ptr,
    qo_indptr_ptr, kv_indptr_ptr, kv_indices_ptr,
    output_ptr, lse_ptr,
    sm_scale: tl.constexpr,
    total_q, num_qo_heads, head_dim, gqa_ratio,
    B,  # number of batches, we loop up to B-1 inside the kernel
    len_indptr
):
    # program id is batch index
    b = tl.program_id(0)

    # Read qo_indptr[b], qo_indptr[b+1] (scalar loads)
    q_start = tl.load(qo_indptr_ptr + b)
    q_end = tl.load(qo_indptr_ptr + b + 1)

    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Compute number of queries and KV tokens for this batch
    num_q_tokens = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    # If nothing to do, skip
    if (num_q_tokens <= 0) or (num_kv_tokens <= 0) or (b >= len_indptr - 1):
        return

    # Load kv_indices slice for this batch: [kv_start, kv_end)
    # We'll build a local vector of indices
    # Create a local indices vector (max length = num_kv_tokens)
    idx_vec = tl.zeros((num_kv_tokens,), dtype=tl.int32)
    # We need to load each element. Triton doesn't have a "load with mask vector" in a straightforward way for runtime sizes.
    # Instead, we compute offsets and load sequentially using a for loop.
    for i in range(0, num_kv_tokens):
        idx_vec[i] = tl.load(kv_indices_ptr + kv_start + i)

    # Compute delta for causal masking: delta = num_kv_tokens - num_q_tokens
    delta = num_kv_tokens - num_q_tokens

    # Prepare base offsets for q batch slice
    # q is laid out as [total_q, num_qo_heads, head_dim]; we pass offsets to q_ptr and let q_end_ptr be q_ptr + q_offset
    # We'll compute offset for q batch base: q_offset = q_start * num_qo_heads * head_dim
    q_offset = q_start * num_qo_heads * head_dim

    # k_cache_flat and v_cache_flat are [num_pages, num_kv_heads, head_dim] after squeezing time dim (asserted as 1).
    # We index k by (idx_vec[i], kv_head, :). We'll build a 2D slice view using strides. But Triton expects contiguous pointers.
    # We'll index k and v as 1D arrays by computing linear indices:
    # For a given idx and kv_head: linear = idx * (num_kv_heads * head_dim) + kv_head * head_dim
    # That is, since the (idx, kv_head, :) row is contiguous of length head_dim, and each idx row has num_kv_heads * head_dim elements.

    # For each token q_idx in this batch
    for q_idx in range(0, num_q_tokens):
        global_q_idx = q_start + q_idx

        # Compute max_kv_idx based on causal mask: max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
        # If delta >= 0, max_kv_idx = q_idx + 1; if delta < 0, we can still clamp to num_kv_tokens.
        # delta is computed above.
        max_kv_idx = q_idx + 1 + delta
        # clamp
        if max_kv_idx > num_kv_tokens:
            max_kv_idx = num_kv_tokens
        if max_kv_idx <= 0:
            # no KV for this query, skip? But with our setup, q_idx + 1 + (num_kv_tokens - num_q_tokens) >= 1 if num_kv_tokens >= num_q_tokens,
            # and >=0 otherwise. We keep skipping if <= 0, but in practice it won't be. Let's handle it anyway.
            continue

        # Per-head lse accumulator (float32)
        # We'll keep scalar per head; we'll iterate h and update a tensor with shape (num_qo_heads,) to reuse values across heads.
        # But Triton allows scalar variables; we'll use a vector across heads using a compile-time constant loop.
        # Note: Triton loop variable 'h' must be constexpr to be unrolled; we pass num_qo_heads as constexpr so loop is unrolled.

        # For each head h
        for h in range(0, num_qo_heads):
            kv_head = h // gqa_ratio

            # Load q vector for this head: q[global_q_idx, h, :]
            # q_ptr is a flat pointer to q with row stride = num_qo_heads * head_dim, but we passed offsets already.
            # We can compute address as q_ptr + q_offset + q_idx * num_qo_heads * head_dim + h * head_dim
            # Wait: we passed q_offset computed as q_start * num_qo_heads * head_dim; then per token q_idx we need to add q_idx * num_qo_heads * head_dim? No.
            # We passed q_offset for the batch base. Within batch, we iterate q_idx, but we already computed q_offset for batch base.
            # To get per-token pointer, we need to compute address: output_ptr points to output[0,0,0]; output layout is [total_q, num_qo_heads, head_dim],
            # contiguous, so output[q, h, :] address is output_ptr + (q * num_qo_heads + h) * head_dim.
            # Similarly, q[q, h, :] address is q_ptr + (q * num_qo_heads + h) * head_dim. But we passed q_ptr base offset already.

            # Build q_vec: [head_dim]
            # We'll load q vector directly from q_ptr: q_ptr is already pointing to q slice, and we need to read h-th head across head_dim.
            # That means we need to compute base for q[global_q_idx, h, :]. Since q is contiguous with [Q, H, D] layout,
            # linear index is (global_q_idx * num_qo_heads + h) * head_dim to ( + head_dim - 1).
            # However, q_ptr points to the whole tensor; to read q[global_q_idx, h, :], we need base offset as:
            # q_global_base = q_ptr + (global_q_idx * num_qo_heads + h) * head_dim
            # We can compute q_vec as a vector of length head_dim by loading with offset + arange(0, head_dim).
            # We'll compute q_vec directly: q_ptr + q_offset + q_idx * num_qo_heads * head_dim + h * head_dim -> q_ptr + q_global_base offset.
            # q_offset was computed as q_start * num_qo_heads * head_dim; but q_global base needs to be for global index q_start+q_idx.
            # We must pass q_ptr as pointing to q[q_start:q_end]; so we need to compute q_global_base offset relative to q_ptr base.

            # A simpler approach: we create q_vec by reading q[q_idx, h, :] directly using torch indexing on host, then pass to Triton as 1D vector.
            # However, Triton kernel doesn't support taking torch tensors as input arrays directly. We need to load from memory.
            # We can compute q_global_base offset as (global_q_idx * num_qo_heads + h) * head_dim, but q_ptr currently points to the start of q batch slice.
            # To avoid confusion, we'll recompute q base address using total_q layout. We'll pass q_ptr as the base of q tensor, and inside kernel,
            # compute q_global_base = q_ptr + (global_q_idx * num_qo_heads + h) * head_dim. Then we can load q_vec from q_ptr.
            # But since we passed q_offset, we need to combine. Let's fix this by passing q_ptr as base pointer and computing q_global_base.

            # Let's restructure: pass q_ptr as the base pointer to q tensor, then for q_idx we add (q_idx * num_qo_heads + h) * head_dim to read the vector.
            # We'll pass q_ptr as argument and compute q_global_base accordingly.
            # However, in the previous comment, we said q_ptr is pointing to q slice; we need consistency. To keep it simple and correct, we'll compute q_global_base using q_ptr = base q tensor pointer.
            # We'll assume q_ptr is pointing to the base of q tensor; then q[q_idx, h, :] is at offset (q_idx * num_qo_heads + h) * head_dim from q_ptr.
            # We'll do the same for k_ptr, v_ptr: k_ptr points to base of k_cache_flat; v_ptr points to base of v_cache_flat.

            # Compute base offsets:
            # q_global_base = q_ptr + (q_idx * num_qo_heads + h) * head_dim
            q_global_base = q_ptr + (q_idx * num_qo_heads + h) * head_dim
            # Load q vector: [head_dim]
            q_vec = tl.zeros((head_dim,), dtype=tl.float32)
            # We need to load from q_global_base with offset arange(0, head_dim)
            # Triton supports pointer arithmetic with tl.arange
            q_offs = tl.arange(0, head_dim)
            q_vec = tl.load(q_global_base + q_offs)

            # Compute k_vec for this head: k[idx, kv_head, :] for i in [0..max_kv_idx-1]
            k_vec = tl.zeros((max_kv_idx * head_dim,), dtype=tl.float32)
            # For each i in [0..max_kv_idx-1], linear index = idx_vec[i] * (num_kv_heads * head_dim) + kv_head * head_dim + offs
            for i in range(0, max_kv_idx):
                idx = idx_vec[i]
                linear_k = idx * (num_kv_heads * head_dim) + kv_head * head_dim
                k_offs = tl.arange(0, head_dim)
                k_vec[i * head_dim : (i + 1) * head_dim] = tl.load(k_ptr + linear_k + k_offs)

            # Compute logits: q_vec dot k_vec, but k_vec is [max_kv_idx, head_dim] flattened; we want [head_dim] dot [head_dim] across i slices.
            # Actually, we want scalar logits for each i, but we can compute sum over i: logits_total = sum_i q_vec dot k_vec[i, :]
            # We need to compute dot for each i and accumulate? No, that's not correct because we need max over all i for logsumexp and then softmax over i.
            # Better: compute per-i dot using k_vec[i*head_dim : (i+1)*head_dim] which is already loaded into k_vec.
            # But to keep it simple and avoid confusion, we can reconstruct k_vec per i by loading from k_ptr again inside the per-i loop. It's fine: Triton allows loops.
            # We'll recompute k_vec per i by loading from k_ptr with linear index. That will be slightly heavier, but manageable for head_dim=128.

            # Instead of storing k_vec, we can compute logits per i directly:
            logits = tl.zeros((), dtype=tl.float32)
            for i in range(0, max_kv_idx):
                linear_k = idx_vec[i] * (num_kv_heads * head_dim) + kv_head * head_dim
                k_offs = tl.arange(0, head_dim)
                k_row = tl.load(k_ptr + linear_k + k_offs)  # [head_dim]
                # Dot product q_vec @ k_row
                dot_val = 0.0
                # Unroll over head_dim in chunks
                for d in range(0, head_dim, 16):
                    q_chunk = q_vec[d : d + 16]
                    k_chunk = k_row[d : d + 16]
                    # Multiply and reduce
                    # For masked elements, we can pad; q_chunk and k_chunk have at most 16; compute sum
                    # Note: tl.sum over axis=0 works on vector
                    dot_val += tl.sum(q_chunk * k_chunk, axis=0)
                logits += dot_val

            # Compute logsumexp over all i: we need vector of logits per i. We'll do it by building a vector and then reducing.
            # To do so, we compute per_i logits and store in a vector.
            # However, Triton doesn't allow dynamic vector creation of runtime size easily. A better approach is to compute per_i logits in a loop,
            # maintain running max and sum, and then compute logsumexp.
            # Initialize lse accumulators
            lse_sum = 0.0
            running_max = tl.full((), -float("inf"), dtype=tl.float32)
            for i in range(0, max_kv_idx):
                linear_k = idx_vec[i] * (num_kv_heads * head_dim) + kv_head * head_dim
                k_offs = tl.arange(0, head_dim)
                k_row = tl.load(k_ptr + linear_k + k_offs)  # [head_dim]
                dot_val = 0.0
                for d in range(0, head_dim, 16):
                    q_chunk = q_vec[d : d + 16]
                    k_chunk = k_row[d : d + 16]
                    dot_val += tl.sum(q_chunk * k_chunk, axis=0)
                logits_i = dot_val * sm_scale
                # logsumexp update
                # new_max = max(running_max, logits_i)
                # sum_term = sum * exp(running_max - logits_i) + 1 * exp(new_max - logits_i)
                # Then set running_max = new_max and sum = sum_term
                new_max = tl.maximum(running_max, logits_i)
                sum_term = lse_sum * tl.exp(running_max - logits_i) + tl.exp(new_max - logits_i)
                running_max = new_max
                lse_sum = sum_term

            lse_value = (running_max + tl.log(lse_sum)) / tl.log(2.0)

            # Now compute attention and output for each i and accumulate output vector
            out_vec = tl.zeros((head_dim,), dtype=tl.float32)
            # We need to compute attn for each i and out_vec += attn * v_vec[i, :]
            for i in range(0, max_kv_idx):
                linear_k = idx_vec[i] * (num_kv_heads * head_dim) + kv_head * head_dim
                k_offs = tl.arange(0, head_dim)
                k_row = tl.load(k_ptr + linear_k + k_offs)  # [head_dim]
                dot_val = 0.0
                for d in range(0, head_dim, 16):
                    q_chunk = q_vec[d : d + 16]
                    k_chunk = k_row[d : d + 16]
                    dot_val += tl.sum(q_chunk * k_chunk, axis=0)
                logits_i = dot_val * sm_scale
                attn_i = tl.exp(logits_i - running_max)  # softmax over i (normalized by max)
                # Load v_row
                linear_v = idx_vec[i] * (num_kv_heads * head_dim) + kv_head * head_dim
                v_offs = tl.arange(0, head_dim)
                v_row = tl.load(v_ptr + linear_v + v_offs)  # [head_dim]
                out_vec += attn_i * v_row

            # Store output: output[global_q_idx, h, :] in bfloat16
            # output_ptr layout is [total_q, num_qo_heads, head_dim] contiguous, so address is:
            # output_ptr + (global_q_idx * num_qo_heads + h) * head_dim
            out_addr = output_ptr + (global_q_idx * num_qo_heads + h) * head_dim
            # Store out_vec as bfloat16: cast
            # Triton: cast to bf16
            tl.store(out_addr + tl.arange(0, head_dim), out_vec.to(tl.bfloat16))

            # Update lse[global_q_idx, h] += lse_value / ln(2)
            lse_addr = lse_ptr + (global_q_idx * num_qo_heads + h)
            # Read current lse, add, write back
            curr_lse = tl.load(lse_addr)
            new_lse = curr_lse + lse_value / tl.log(2.0)
            tl.store(lse_addr, new_lse)

# Helper to call the Triton kernel in forward
def _run_triton(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    """
    q: [total_q, num_qo_heads, head_dim], bfloat16
    k_cache: [num_pages, 1, num_kv_heads, head_dim], bfloat16
    v_cache: [num_pages, 1, num_kv_heads, head_dim], bfloat16
    qo_indptr, kv_indptr: int32
    kv_indices: int32
    sm_scale: float32
    Returns: (output, lse)
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be on CUDA for Triton."
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, _, num_kv_heads, _ = k_cache.shape
    assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Shape assertions must hold."
    gqa_ratio = num_qo_heads // num_kv_heads  # 4

    # Make inputs contiguous and flatten k_cache_flat, v_cache_flat by squeezing time dim (1)
    # We'll work with bfloat16; Triton will compute in fp32.
    q = q.contiguous()
    k_cache_flat = k_cache.squeeze(1).contiguous()
    v_cache_flat = v_cache.squeeze(1).contiguous()

    len_indptr = qo_indptr.shape[0]
    # Allocate output and lse
    output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)

    # Launch kernel: one program per batch b
    # We need B; but we don't have B explicitly. We can infer it from len_indptr - 1 valid batches.
    B = len_indptr - 1  # since each b is in [0, len_indptr-2], we use B = len_indptr - 1
    grid = (B,)
    _batch_compute_kernel[grid](
        q, k_cache_flat, v_cache_flat,
        qo_indptr, kv_indptr, kv_indices,
        output, lse,
        sm_scale,
        total_q, num_qo_heads, head_dim, gqa_ratio,
        B, len_indptr,
        num_warps=4, num_stages=2
    )

    return output, lse

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # All computation is done by Triton kernel; no torch ops inside.
        output, lse = _run_triton(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)
        return output, lse

# Example helper functions from the original snippet (not strictly needed by evaluator but provided for completeness):
def get_inputs():
    # This will create CPU tensors; move to CUDA before calling ModelNew.forward
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16)
    k_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16)
    v_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 51, [34], dtype=torch.int32)
    sm_scale = 1.0 / math.sqrt(128)
    return [q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
