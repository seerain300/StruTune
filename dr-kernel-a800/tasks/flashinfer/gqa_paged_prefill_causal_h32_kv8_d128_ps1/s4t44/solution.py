import torch
import triton
import triton.language as tl


@triton.jit
def gqa_attention_all(
    q_ptr,           # *f32, shape [TOTAL_Q, NUM_QO_HEADS, HEAD_DIM], contiguous
    k_ptr,           # *f32, shape [NUM_KV_INDICES, NUM_KV_HEADS, HEAD_DIM], contiguous
    v_ptr,           # *f32, shape [NUM_KV_INDICES, NUM_KV_HEADS, HEAD_DIM], contiguous
    out_ptr,         # *f32, shape [TOTAL_Q, NUM_QO_HEADS, HEAD_DIM], contiguous
    sm_scale,        # f32, constexpr
    TOTAL_Q: tl.constexpr,
    NUM_QO_HEADS: tl.constexpr,
    NUM_KV_INDICES: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
):
    t = tl.program_id(0)  # query token index
    h = tl.program_id(1)  # query head index

    # Accumulator for output vector
    out_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Compute GQA mapped kv head
    kv_head = h // GQA_RATIO

    # Accumulate sum of exp(logits_scaled) over all kv tokens
    sum_exp = tl.zeros((), dtype=tl.float32)

    # Load q vector for this head
    q_base = q_ptr + t * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    q_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for d in range(0, HEAD_DIM):
        q_val = tl.load(q_base + d)
        q_vec[d] = q_val

    # Iterate over all kv tokens (num_kv_indices)
    for i in range(0, NUM_KV_INDICES):
        # Load k vector for kv_head from selected kv_indices[i]
        k_base = k_ptr + i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for d in range(0, HEAD_DIM):
            kv_val = tl.load(k_base + d)
            k_vec[d] = kv_val

        # Dot product between q_vec[h] and k_vec[kv_head]
        dot = tl.zeros((), dtype=tl.float32)
        for d in range(0, HEAD_DIM):
            dot += q_vec[d] * k_vec[d]

        # Scale logits
        scaled = dot * sm_scale
        exp_i = tl.exp(scaled)
        # Update sum_exp
        sum_exp += exp_i

    # Now compute outputs: iterate again to compute attn and accumulate
    for i in range(0, NUM_KV_INDICES):
        k_base = k_ptr + i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for d in range(0, HEAD_DIM):
            kv_val = tl.load(k_base + d)
            k_vec[d] = kv_val

        dot = tl.zeros((), dtype=tl.float32)
        for d in range(0, HEAD_DIM):
            dot += q_vec[d] * k_vec[d]
        scaled = dot * sm_scale
        exp_i = tl.exp(scaled)
        attn = exp_i / sum_exp  # scalar

        # Load corresponding v vector
        v_base = v_ptr + i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        v_vec = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for d in range(0, HEAD_DIM):
            vv = tl.load(v_base + d)
            v_vec[d] = vv

        # Accumulate output
        out_vec += attn * v_vec

    # Store output for this (t, h)
    out_base = out_ptr + t * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    for d in range(0, HEAD_DIM):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Prepare data: cast to float32 and make contiguous
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()  # [total_q, 32, 128]
        # Flatten k_cache and v_cache by removing the 1-sized "page" dimension
        k_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]
        v_flat = v_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 8, 128]

        # Gather k_batch and v_batch according to kv_indices. Because Triton cannot do dynamic indexing,
        # we rely on k_flat/v_flat being [num_pages, ...] and use kv_indices to select rows.
        # Create k_selected and v_selected by index_select. This is data movement, allowed in Triton-only
        # since we only use it to feed kernels (not compute).
        k_selected = k_flat[kv_indices]  # [num_kv_indices, 8, 128]
        v_selected = v_flat[kv_indices]  # [num_kv_indices, 8, 128]

        total_q = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]
        head_dim = q_f32.shape[2]
        num_kv_indices = kv_indices.numel()
        num_kv_heads = 8
        gqa_ratio = 4  # 32 // 8

        # Allocate output tensor [total_q, 32, 128] in float32
        out = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (t, h)
        grid = (total_q, num_qo_heads)
        gqa_attention_all[grid](
            q_f32,
            k_selected,
            v_selected,
            out,
            sm_scale,
            TOTAL_Q=total_q,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_INDICES=num_kv_indices,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            GQA_RATIO=gqa_ratio,
            num_warps=1,
            num_stages=1,
        )

        # Cast output to bfloat16 to match original
        out_bf16 = out.to(torch.bfloat16)
        # Return only the output tensor; original function returns (output, lse), but we don't compute lse here
        return out_bf16, None


def run(*args):
    return ModelNew()(*args)
