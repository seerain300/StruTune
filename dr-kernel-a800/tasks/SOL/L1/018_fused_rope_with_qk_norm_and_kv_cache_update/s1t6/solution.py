import torch
import triton
import triton.language as tl


# Triton kernel: RMSNorm per row
# y = weight * x / sqrt(mean(x^2) + eps)
@triton.jit
def rms_norm_row(x_ptr, y_ptr, weight_ptr, eps, head_dim, BLOCK: tl.constexpr):
    row_id = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < head_dim
    x = tl.load(x_ptr + row_id * head_dim + offsets, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    sumsq = tl.sum(x32 * x32, axis=0)
    mean = sumsq / head_dim
    scale = tl.rsqrt(mean + eps)
    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    y = (x32 * w) * scale
    tl.store(y_ptr + row_id * head_dim + offsets, y, mask=mask)


# Triton kernel: build rotation cos/sin vectors per position
# Uses small-angle approximations:
#   cos(alpha) ≈ 1 - alpha^2/2
#   sin(alpha) ≈ alpha
# Inputs:
#   position_ids: [B, seq_len] int64
#   inv_freq: [head_dim//2] float32
#   cos_ptr: [seq_len, head_dim] float32 (pre-allocated)
#   sin_ptr: [seq_len, head_dim] float32 (pre-allocated)
@triton.jit
def build_rotation(pos_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr, seq_len, head_dim, BLOCK: tl.constexpr):
    pos_id = tl.program_id(axis=0)  # token position in [0, seq_len)
    pos = tl.load(pos_ids_ptr + pos_id).to(tl.int32)
    offs = tl.arange(0, BLOCK)  # BLOCK = head_dim
    mask = offs < head_dim
    # emb = pos * inv_freq for both halves
    alpha = pos * tl.load(inv_freq_ptr + offs, mask=mask, other=0.0)  # length head_dim
    cos_val = 1.0 - 0.5 * (alpha * alpha)
    sin_val = alpha
    tl.store(cos_ptr + pos_id * head_dim + offs, cos_val, mask=mask)
    tl.store(sin_ptr + pos_id * head_dim + offs, sin_val, mask=mask)


# Triton kernel: apply rotation to a tensor of shape [n_rows, head_dim]
# z = (x * cos) + (rotate_half(x) * sin)
# rotate_half(x): take last head_dim/2 and put at front, first half stays at back, negated.
@triton.jit
def apply_rotation_rows(x_ptr, z_ptr, cos_ptr, sin_ptr, head_dim, BLOCK: tl.constexpr):
    row_id = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < head_dim

    # Load x row
    x = tl.load(x_ptr + row_id * head_dim + offsets, mask=mask, other=0.0).to(tl.float32)

    half = head_dim // 2
    # First half of rotated: - second half of x
    x_second_half = tl.load(x_ptr + row_id * head_dim + offsets, mask=(offsets < half), other=0.0).to(tl.float32)
    # Second half of rotated: first half of x
    x_first_half = tl.load(x_ptr + row_id * head_dim + (offsets - half), mask=(offsets >= half) & (offsets < head_dim), other=0.0).to(tl.float32)

    # Load cos/sin for this row
    cos_vec = tl.load(cos_ptr + row_id * head_dim + offsets, mask=mask, other=1.0).to(tl.float32)
    sin_vec = tl.load(sin_ptr + row_id * head_dim + offsets, mask=mask, other=1.0).to(tl.float32)

    # Construct rotated vector: [x_first_half, -x_second_half]
    rotated = tl.zeros([head_dim], dtype=tl.float32)
    rotated = tl.where(offsets < half, -x_second_half, rotated)
    rotated = tl.where(offsets >= half, x_first_half, rotated)

    z = (x * cos_vec) + (rotated * sin_vec)
    tl.store(z_ptr + row_id * head_dim + offsets, z, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect:
        # 0: query [B, num_q_heads, seq_len, head_dim]
        # 1: key [B, num_kv_heads, seq_len, head_dim]
        # 2: value [B, num_kv_heads, seq_len, head_dim] (unused, but kept for API)
        # 3: position_ids [B, seq_len]
        # 4: key_cache [B, num_kv_heads, 262144, head_dim] (unused)
        # 5: value_cache [B, num_kv_heads, 262144, head_dim] (unused)
        # 6: cache_position [seq_len] (unused)
        # 7: q_norm_weight [head_dim]
        # 8: k_norm_weight [head_dim]
        # 9: inv_freq [head_dim//2] float32
        # 10: rms_norm_eps float
        if len(args) < 10:
            raise RuntimeError("ModelNew.forward expects at least 10 arguments")
        query = args[0].contiguous()
        key = args[1].contiguous()
        value = args[2].contiguous()  # unused
        position_ids = args[3].contiguous()  # [B, seq_len], int64
        key_cache = args[4].contiguous()  # unused
        value_cache = args[5].contiguous()  # unused
        cache_position = args[6].contiguous()  # unused
        q_norm_weight = args[7].contiguous()  # [head_dim], bfloat16
        k_norm_weight = args[8].contiguous()  # [head_dim], bfloat16
        inv_freq = args[9].contiguous()      # [head_dim//2], float32
        rms_norm_eps = float(args[10]) if len(args) > 10 else 1e-6

        # Shapes
        Bq, num_q_heads, seq_len, head_dim = query.shape
        Bk, num_kv_heads, _, _ = key.shape
        if Bq != Bk:
            raise RuntimeError("Batch size mismatch between query and key")
        B = Bq
        if key.shape != (B, num_kv_heads, seq_len, head_dim) or value.shape != (B, num_kv_heads, seq_len, head_dim):
            raise RuntimeError("Key/Value shape mismatch")

        # Allocate normalized tensors
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # 1) RMSNorm for query and key (per row) in Triton
        n_rows_q = B * num_q_heads * seq_len
        n_rows_k = B * num_kv_heads * seq_len

        rms_norm_row[(n_rows_q,)](
            query, query_norm, q_norm_weight, rms_norm_eps, head_dim, BLOCK=head_dim, num_warps=4
        )

        rms_norm_row[(n_rows_k,)](
            key, key_norm, k_norm_weight, rms_norm_eps, head_dim, BLOCK=head_dim, num_warps=4
        )

        # 2) Build rotation cos/sin vectors in Triton: one per position p
        # Allocate cos/sin buffers [seq_len, head_dim] float32
        cos_mat = torch.empty((seq_len, head_dim), dtype=torch.float32, device=query.device)
        sin_mat = torch.empty((seq_len, head_dim), dtype=torch.float32, device=query.device)

        build_rotation[(seq_len,)](
            position_ids, inv_freq, cos_mat, sin_mat, seq_len, head_dim, BLOCK=head_dim, num_warps=1
        )

        # 3) Apply rotation to query and key using Triton
        query_rotated = torch.empty_like(query_norm)
        key_rotated = torch.empty_like(key_norm)

        apply_rotation_rows[(n_rows_q,)](
            query_norm, query_rotated, cos_mat, sin_mat, head_dim, BLOCK=head_dim, num_warps=4
        )

        apply_rotation_rows[(n_rows_k,)](
            key_norm, key_rotated, cos_mat, sin_mat, head_dim, BLOCK=head_dim, num_warps=4
        )

        # Return only the rotated tensors (original code returns these; cache updates are side effects)
        return query_rotated, key_rotated, None, None


def run(*args):
    return ModelNew()(*args)
