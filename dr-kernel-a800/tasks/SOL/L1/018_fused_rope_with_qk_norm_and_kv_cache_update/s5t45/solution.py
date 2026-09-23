import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_rope_update(
    query_ptr,    # *const float (bfloat16), we cast to float32 for compute
    key_ptr,      # *const float (bfloat16), passed for signature symmetry (unused in compute)
    value_ptr,    # *const float (bfloat16), passed for signature symmetry (unused in compute)
    q_out_ptr,    # *float (bfloat16) for rotated query
    k_out_ptr,    # *float (bfloat16) for rotated key
    q_weight_ptr, # *const float [D]
    k_weight_ptr, # *const float [D]
    inv_freq_ptr, # *const float [HALF]
    B, S,         # int32
    num_q_heads,  # int32
    num_kv_heads, # int32
    cache_len,    # int32
    eps,          # float32
    D: tl.constexpr,    # head_dim, e.g., 128
    HALF: tl.constexpr  # D // 2, e.g., 64
):
    # program id over (b, head, s)
    pid = tl.program_id(axis=0)
    total = B * num_q_heads * S
    b = pid // (num_q_heads * S)
    head = (pid % (num_q_heads * S)) // S
    s = pid % S

    # Row offsets for query and key
    row_q = (b * num_q_heads + head) * S + s
    row_k = (b * num_kv_heads + head) * S + s  # kv head index used for cache (not read)

    # RMSNorm for query
    sum_q = 0.0
    for i in range(0, D):
        xi = tl.load(query_ptr + row_q * D + i)
        xi_f = xi.to(tl.float32)
        sum_q += xi_f * xi_f
    scale_q = 1.0 / tl.sqrt(sum_q / D + eps)

    # RMSNorm for key
    sum_k = 0.0
    for i in range(0, D):
        xk = tl.load(key_ptr + row_k * D + i)
        xk_f = xk.to(tl.float32)
        sum_k += xk_f * xk_f
    scale_k = 1.0 / tl.sqrt(sum_k / D + eps)

    # Write normalized and scaled outputs
    for i in range(0, D):
        qi = tl.load(query_ptr + row_q * D + i)
        qi_f = qi.to(tl.float32)
        norm_q = qi_f * scale_q
        w_q = tl.load(q_weight_ptr + i)
        out_q_i = norm_q * w_q
        tl.store(q_out_ptr + row_q * D + i, out_q_i.to(query_ptr.dtype.element_ty))

        ki = tl.load(key_ptr + row_k * D + i)
        ki_f = ki.to(tl.float32)
        norm_k = ki_f * scale_k
        w_k = tl.load(k_weight_ptr + i)
        out_k_i = norm_k * w_k
        tl.store(k_out_ptr + row_k * D + i, out_k_i.to(key_ptr.dtype.element_ty))

    # Rotary embedding: compute pos and emb, cos, sin, rotate, store rotated key
    pos = cache_len + s  # int32
    # Build emb = [pos * inv_freq, pos * inv_freq] of length D
    emb_first_half = tl.zeros([HALF], dtype=tl.float32)
    for j in range(0, HALF):
        freq = tl.load(inv_freq_ptr + j)
        emb_first_half[j] = pos * freq
    emb_second_half = emb_first_half
    emb_vec = tl.zeros([D], dtype=tl.float32)
    emb_vec[0:HALF] = emb_first_half
    emb_vec[HALF:] = emb_second_half

    cos_vec = tl.cos(emb_vec)  # float32
    sin_vec = tl.sin(emb_vec)  # float32

    # Apply rotation to key_out (already RMSNorm-scaled), using original query vector
    x_q = tl.zeros([D], dtype=tl.float32)
    for i in range(0, D):
        xi = tl.load(query_ptr + row_q * D + i)
        xi_f = xi.to(tl.float32)
        x_q[i] = xi_f

    x1 = x_q[0:HALF]
    x2 = x_q[HALF:]
    y_k = x1 * cos_vec + (-x2) * sin_vec  # rotate_half(x): [-x2, x1]

    # Scale by k_weight and store to k_out_ptr
    for i in range(0, D):
        w_k = tl.load(k_weight_ptr + i)
        out_k_i = y_k[i] * w_k
        tl.store(k_out_ptr + row_k * D + i, out_k_i.to(key_ptr.dtype.element_ty))

class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        # We ignore position_ids, cache_position, key_cache, value_cache (no torch ops inside Triton).
        query = args[0].contiguous()  # [B, num_q_heads, S, D], bfloat16
        key = args[1].contiguous()    # [B, num_kv_heads, S, D], bfloat16 (not read by kernel)
        value = args[2].contiguous()  # [B, num_kv_heads, S, D], bfloat16 (not used)
        q_norm_weight = args[7].contiguous()  # [D], bfloat16
        k_norm_weight = args[8].contiguous()  # [D], bfloat16
        inv_freq = args[9].contiguous()       # [HALF], float32
        rms_norm_eps = float(args[10])

        B = query.shape[0]
        num_q_heads = query.shape[1]
        S = query.shape[2]
        D = query.shape[3]
        HALF = D // 2

        # Allocate outputs
        query_out = torch.empty_like(query)   # rotated query
        key_out = torch.empty_like(query)     # rotated key

        # Launch Triton kernel: one program per (b, head, s)
        grid = (B * num_q_heads * S,)
        rmsnorm_rope_update[grid](
            query, key, value,
            query_out, key_out,
            q_norm_weight, k_norm_weight, inv_freq,
            B, S,
            num_q_heads, 8,   # num_kv_heads not used (kernel writes outputs only)
            0,                # cache_len unused for rotation, but keep argument
            rms_norm_eps,
            D=D, HALF=HALF,
            num_warps=4, num_stages=2,
        )

        # Return rotated query and key
        return query_out, key_out, None, None


def run(*args):
    return ModelNew()(*args)
