import math
import triton
import triton.language as tl


@triton.jit
def attn_kernel(
    q_ptr,              # *f32, [total_q, 32, 128]
    k_ptr,              # *f32, [num_pages * 8 * 128], flattened
    v_ptr,              # *f32, [num_pages * 8 * 128], flattened
    output_ptr,         # *f32, [total_q, 32, 128]
    output_lse_ptr,     # *f32, [total_q, 32]
    q_start,            # i32
    q_end,              # i32
    kv_start,           # i32
    kv_end,             # i32
    head_dim,           # i32, 128
    num_qo_heads,       # i32, 32
    num_kv_heads,       # i32, 8
    gqa_ratio,          # i32, 4
    sm_scale,           # f32
    ln2_inv,            # f32
    MAX_Q_SEG: tl.constexpr,
    MAX_KV_SEG: tl.constexpr,
):
    b = tl.program_id(0)

    num_q_tokens_segment = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    dim = tl.arange(0, head_dim)

    # Iterate over query tokens in this segment
    for q_i in range(0, MAX_Q_SEG):
        q_valid = q_i < num_q_tokens_segment
        global_q_idx = q_start + q_i

        if not q_valid:
            continue

        # Iterate over query heads
        for h in range(0, num_qo_heads):
            kv_head = h // gqa_ratio

            # Load q[h] vector
            q_vec = tl.load(q_ptr + global_q_idx * (num_qo_heads * head_dim) + h * head_dim + dim)

            # Compute max_val and sum_exp for logsumexp over keys
            max_val = -float("inf")
            sum_exp = 0.0

            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk))
                k_vec = tl.load(k_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + dim)
                prod = tl.sum(q_vec * k_vec, axis=0)
                scaled = prod * sm_scale
                if scaled > max_val:
                    sum_exp = sum_exp * tl.exp(max_val - scaled) + 1.0
                    max_val = scaled
                else:
                    sum_exp = sum_exp + tl.exp(scaled - max_val)

            lse_val = (max_val + tl.log(sum_exp)) * ln2_inv

            # Compute output vector: sum of attn[k] * v[k], where attn = softmax(scaled) over valid keys
            out_vec = 0.0
            sum_attn = 0.0

            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                max_kv_idx = tl.minimum(q_i + 1 + (num_kv_tokens - num_q_tokens_segment), num_kv_tokens)
                valid_k = kk < max_kv_idx
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk))
                k_vec = tl.load(k_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + dim)
                prod = tl.sum(q_vec * k_vec, axis=0)
                scaled = prod * sm_scale
                attn = tl.exp(scaled - lse_val) * valid_k  # valid_k is 0 for invalid keys
                v_vec = tl.load(v_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + dim)
                # out_vec += attn * (sum of v_vec * k_vec) doesn't make sense for dot; we want attn * v_vec.
                # Since v_vec and k_vec are same length, attn is scalar; compute elementwise product and reduce.
                # Better: compute out_vec += attn * v_vec (each element), but Triton kernel is vectorized; use dot instead.
                # Compute dot v_vec · some vector? We need a scalar multiplier per element; softmax is per kk, so attn scales v_vec as a whole.
                # Instead, compute out_vec += attn * (sum of v_vec * some basis). Since v_vec is vector, we need a reduction. We can
                # interpret out_vec += attn * (sum of v_vec), i.e., attn scales the vector by a scalar. That's not correct.
                # Correction: attn is scalar; out_vec += attn * (sum of v_vec * k_vec) is also not consistent.
                # To keep things simple and correct, we compute out_vec as the weighted sum of v_vec components using a dummy dot.
                # However, Triton requires consistent vector operations. We'll instead compute out_vec elementwise: out_vec += attn * v_vec elementwise.
                # Triton does not support per-lane separate store here; we'll compute out_vec as a scalar accumulation, which is incorrect.
                # Therefore, we compute out_vec using the same logic as the original: out_vec = sum of attn * v_vec. For vector addition,
                # Triton will require a proper vector accumulator, which we can emulate by maintaining a vector and summing.
                # Since Triton lacks per-iteration vector accumulation, we can compute out_vec as a reduction across kk using a vector tmp.
                # We'll initialize out_vec as a vector and add attn * v_vec to it; but Triton does not support variable-length vector storage.
                # Thus, we revert to computing out_vec as scalar accumulation over kk and use softmax. To keep correctness, we compute
                # attn * v_vec as a scalar per kk (dot with a ones vector). This is acceptable because the original code computes
                # attn per key and then matvec. For simplicity and correctness, we compute out_vec as sum of attn * (sum of v_vec * k_vec)
                # which is not ideal, but Triton lacks proper vector accumulation here.

            # The above out_vec accumulation is oversimplified; to maintain correctness, we compute output via an alternative method:
            # We know output is the softmax-weighted sum of v vectors. We can recompute attn vector and then compute out_vec by doing
            # a vectorized softmax and matvec. Triton supports vector operations. So we compute:
            # 1) lse_val (done above).
            # 2) Compute attn_vec[k] = exp(scaled - lse_val) for valid k, else 0.
            # 3) Compute out_vec = sum_k attn_vec[k] * v_selected[k].
            # Implement vectorized attn_vec and matvec below:

            # Recompute scaled scores and build attn_vec
            attn_vec = tl.zeros([MAX_KV_SEG], dtype=tl.float32)
            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                max_kv_idx = tl.minimum(q_i + 1 + (num_kv_tokens - num_q_tokens_segment), num_kv_tokens)
                valid_k = kk < max_kv_idx
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk))
                k_vec = tl.load(k_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + dim)
                prod = tl.sum(q_vec * k_vec, axis=0)
                scaled = prod * sm_scale
                attn_vec[kk] = tl.exp(scaled - lse_val) * valid_k

            # Compute output vector using attn_vec
            out_vec = tl.zeros([head_dim], dtype=tl.float32)
            for kk in range(0, MAX_KV_SEG):
                if kk >= num_kv_tokens:
                    break
                k_idx = tl.load(kv_indices_ptr + (kv_start + kk))
                v_vec = tl.load(v_ptr + k_idx * (num_kv_heads * head_dim) + kv_head * head_dim + dim)
                out_vec += attn_vec[kk] * v_vec

            # Store results
            tl.store(output_ptr + global_q_idx * (num_qo_heads * head_dim) + h * head_dim + dim, out_vec)
            tl.store(output_lse_ptr + global_q_idx * num_qo_heads + h, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on the same device
        assert q.device == k_cache.device == v_cache.device == qo_indptr.device == kv_indptr.device == kv_indices.device
        device = q.device

        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and head_dim == 128 and num_kv_heads == 8

        # Flatten k_cache and v_cache (remove size-1 dim)
        k_cache_flat = k_cache.squeeze(1).contiguous()  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).contiguous()  # [num_pages, 8, 128]

        # Cast to fp32 for compute
        q_f32 = q.to(torch.float32)
        k_ptr_f32 = k_cache_flat.to(torch.float32).view(-1)  # 1D flattened
        v_ptr_f32 = v_cache_flat.to(torch.float32).view(-1)

        # Allocate outputs (fp32)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.numel()
        grid = (len_indptr - 1,)

        ln2_inv = 1.0 / math.log(2.0)

        # Launch Triton kernel
        attn_kernel[grid](
            q_f32, k_ptr_f32, v_ptr_f32, output, lse,
            qo_indptr[0].item(), qo_indptr[-1].item(), kv_indptr[0].item(), kv_indptr[-1].item(),
            head_dim, num_qo_heads, num_kv_heads, 4,
            sm_scale, ln2_inv,
            MAX_Q_SEG=100,  # upper bound, masked
            MAX_KV_SEG=100, # upper bound, masked
            num_warps=4,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
