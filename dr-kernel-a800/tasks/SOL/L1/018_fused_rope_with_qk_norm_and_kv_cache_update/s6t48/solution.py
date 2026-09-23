import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    x_stride0, x_stride1, x_stride2,
                    sum_stride0,
                    BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    sum_val = tl.zeros((), dtype=tl.float32)
    base = b * x_stride0 + h * x_stride1 + s * x_stride2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        vals = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(vals * vals, axis=0)
    tl.store(sum_sums_ptr + pid, sum_val)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    x_stride0, x_stride1, x_stride2,
                    out_stride0, out_stride1, out_stride2,
                    sum_stride0,
                    eps: tl.constexpr,
                    BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    sum_val = tl.load(sum_sums_ptr + pid)
    inv_rms = 1.0 / tl.sqrt(sum_val / H + eps)

    base_x = b * x_stride0 + h * x_stride1 + s * x_stride2
    base_out = b * out_stride0 + h * out_stride1 + s * out_stride2

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_vals = tl.load(x_ptr + base_x + idx, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y = x_vals * inv_rms * w_vals
        tl.store(out_ptr + base_out + idx, y.to(tl.bfloat16), mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, HALF: tl.constexpr,
                              pos_stride0, cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    b = tl.program_id(0)
    s = tl.program_id(1)
    if b >= B or s >= S:
        return

    pos = tl.load(position_ids_ptr + b * pos_stride0 + s).to(tl.float32)

    half = HALF  # H // 2
    for off in range(0, half, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < half
        # inv_freq is length half: [0..half-1]
        vals = tl.load(inv_freq_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        emb = pos * vals  # first half
        # second half duplicates first half
        # store first half
        base = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base + idx * cos_stride2, emb, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, tl.zeros([BLOCK_H], dtype=tl.float32), mask=mask)
        # store second half
        base2 = b * cos_stride0 + s * cos_stride1 + half * cos_stride2
        tl.store(cos_ptr + base2 + idx * cos_stride2, emb, mask=mask)
        tl.store(sin_ptr + base2 + idx * sin_stride2, tl.zeros([BLOCK_H], dtype=tl.float32), mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr, HALF: tl.constexpr,
                           x_stride0, x_stride1, x_stride2,
                           out_stride0, out_stride1, out_stride2,
                           cos_stride0, cos_stride1, cos_stride2,
                           sin_stride0, sin_stride1, sin_stride2,
                           BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    h = (pid % (num_heads * S)) // S
    s = pid % S

    base_x = b * x_stride0 + h * x_stride1 + s * x_stride2
    base_out = b * out_stride0 + h * out_stride1 + s * out_stride2
    base_cos = b * cos_stride0 + s * cos_stride1
    base_sin = b * sin_stride0 + s * sin_stride1

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # Load x
        x_vals = tl.load(x_ptr + base_x + idx, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin
        cos_vals = tl.load(cos_ptr + base_cos + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base_sin + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        # Build rotated = rotate_half(x)
        rotated = tl.zeros([BLOCK_H], dtype=tl.float32)
        # For idx < half: rotated[idx] = -x[half + idx]
        # For idx >= half: rotated[idx] = x[idx - half]
        half = HALF
        mask_lt = (idx < half)
        mask_ge = (idx >= half)
        rotated = tl.where(mask_lt, -tl.load(x_ptr + base_x + (half + idx), mask=mask_lt, other=0.0).to(tl.float32),
                           tl.where(mask_ge, tl.load(x_ptr + base_x + (idx - half), mask=mask_ge, other=0.0).to(tl.float32),
                                    rotated))
        # y = x * cos + rotated * sin
        y = x_vals * cos_vals + rotated * sin_vals

        tl.store(out_ptr + base_out + idx, y.to(tl.bfloat16), mask=mask)


@triton.jit
def update_cache_kernel(key_ptr, value_ptr, key_cache_ptr, value_cache_ptr, cache_pos_ptr,
                         B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         key_stride0, key_stride1, key_stride2,
                         value_stride0, value_stride1, value_stride2,
                         key_cache_stride0, key_cache_stride1, key_cache_stride2,
                         value_cache_stride0, value_cache_stride1, value_cache_stride2,
                         cache_stride0):
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    h = (pid % (num_kv_heads * S)) // S
    s = pid % S

    # read rotated key at (b, h, s, :)
    base_k = b * key_stride0 + h * key_stride1 + s * key_stride2
    rotated_k = tl.load(key_ptr + base_k + tl.arange(0, H)).to(tl.float32)  # key is already normalized in forward

    # destination cache position
    dest = tl.load(cache_pos_ptr + s).to(tl.int32)

    base_kc = b * key_cache_stride0 + h * key_cache_stride1 + dest * key_cache_stride2
    base_vc = b * value_cache_stride0 + h * value_cache_stride1 + dest * value_cache_stride2

    # write to key_cache
    tl.store(key_cache_ptr + base_kc + tl.arange(0, H), rotated_k.to(tl.bfloat16), mask=(tl.arange(0, H) < H))
    # write to value_cache: value[b, h, s, :]
    base_v = b * value_stride0 + h * value_stride1 + s * value_stride2
    v = tl.load(value_ptr + base_v + tl.arange(0, H)).to(tl.float32)
    tl.store(value_cache_ptr + base_vc + tl.arange(0, H), v.to(tl.bfloat16), mask=(tl.arange(0, H) < H))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                position_ids: torch.Tensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor,
                k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor,
                rms_norm_eps: float):
        """
        Returns:
          query_rotated: [B, num_q_heads, S, H], bfloat16
          key_rotated: [B, num_kv_heads, S, H], bfloat16
          key_cache: updated [B, num_kv_heads, max_pos, H], bfloat16
          value_cache: updated [B, num_kv_heads, max_pos, H], bfloat16
        """
        assert query.is_cuda and key.is_cuda and value.is_cuda and position_ids.is_cuda \
               and key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda \
               and q_norm_weight.is_cuda and k_norm_weight.is_cuda and inv_freq.is_cuda, "All tensors must be on CUDA."

        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        device = query.device
        half = H // 2

        # 1) RMSNorm for query -> query_norm
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=device)
        grid_q = (B * num_q_heads * S,)
        rms_sum_kernel[grid_q](query, sum_sums_q,
                               B, num_q_heads, S, H,
                               query.stride(0), query.stride(1), query.stride(2),
                               sum_sums_q.stride(0),
                               BLOCK_H=128, num_warps=4)

        query_norm = torch.empty_like(query, dtype=torch.float16)
        rms_norm_kernel[grid_q](query, q_norm_weight, query_norm, sum_sums_q,
                                B, num_q_heads, S, H,
                                query.stride(0), query.stride(1), query.stride(2),
                                query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                sum_sums_q.stride(0),
                                eps=rms_norm_eps,
                                BLOCK_H=128, num_warps=4)

        # 2) Build rotation cos/sin per (b, s) and apply to query_norm
        cos = torch.empty((B, S, H), dtype=torch.float32, device=device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_rc = (B, S)
        rotate_sin_cos_kernel_b_s[grid_rc](position_ids, inv_freq, cos, sin,
                                           B, S, H, half,
                                           position_ids.stride(0),
                                           cos.stride(0), cos.stride(1), cos.stride(2),
                                           sin.stride(0), sin.stride(1), sin.stride(2),
                                           BLOCK_H=128, num_warps=4)

        # 3) Apply rotation to query_norm -> query_rotated
        query_rotated = torch.empty_like(query_norm, dtype=torch.float16)
        grid_rot_q = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot_q](query_norm, cos, sin, query_rotated,
                                          B, num_q_heads, S, H, half,
                                          query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                          query_rotated.stride(0), query_rotated.stride(1), query_rotated.stride(2),
                                          cos.stride(0), cos.stride(1), cos.stride(2),
                                          sin.stride(0), sin.stride(1), sin.stride(2),
                                          BLOCK_H=128, num_warps=4)

        # 4) RMSNorm for key -> key_norm
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=device)
        grid_k = (B * num_kv_heads * S,)
        rms_sum_kernel[grid_k](key, sum_sums_k,
                               B, num_kv_heads, S, H,
                               key.stride(0), key.stride(1), key.stride(2),
                               sum_sums_k.stride(0),
                               BLOCK_H=128, num_warps=4)

        key_norm = torch.empty_like(key, dtype=torch.float16)
        rms_norm_kernel[grid_k](key, k_norm_weight, key_norm, sum_sums_k,
                                B, num_kv_heads, S, H,
                                key.stride(0), key.stride(1), key.stride(2),
                                key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                sum_sums_k.stride(0),
                                eps=rms_norm_eps,
                                BLOCK_H=128, num_warps=4)

        # 5) Build rotation cos/sin again (for key) and apply to key_norm
        # We can reuse the same cos/sin computed for (B, S) since they depend only on position_ids
        # But for clarity, compute it again (cheap).
        # Note: if H changes across workloads, this is fine; here H is consistent with original model.
        # If H differs, previous kernels use HALF for idx; that’s fine as long as HALF = H//2.
        # However, to be robust, recompute cos/sin for key using the same position_ids tensor shape logic.
        # Recompute cos/sin for key using same B,S,H
        rotate_sin_cos_kernel_b_s[grid_rc](position_ids, inv_freq, cos, sin,
                                           B, S, H, half,
                                           position_ids.stride(0),
                                           cos.stride(0), cos.stride(1), cos.stride(2),
                                           sin.stride(0), sin.stride(1), sin.stride(2),
                                           BLOCK_H=128, num_warps=4)

        # Apply rotation to key_norm -> key_rotated
        key_rotated = torch.empty_like(key_norm, dtype=torch.float16)
        grid_rot_k = (B * num_kv_heads * S,)
        apply_rotation_kernel[grid_rot_k](key_norm, cos, sin, key_rotated,
                                          B, num_kv_heads, S, H, half,
                                          key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                          key_rotated.stride(0), key_rotated.stride(1), key_rotated.stride(2),
                                          cos.stride(0), cos.stride(1), cos.stride(2),
                                          sin.stride(0), sin.stride(1), sin.stride(2),
                                          BLOCK_H=128, num_warps=4)

        # 6) Update caches: in-place updates at positions cache_position
        # Ensure cache_position is int32 for Triton indexing
        cache_position_i32 = cache_position.to(torch.int32)
        # We will update key_cache and value_cache in-place using Triton.
        # For each (b, kv_head, s), copy rotated key/value into cache row at cache_position[s].
        grid_cache = (B * num_kv_heads * S,)
        update_cache_kernel[grid_cache](key_rotated, value, key_cache, value_cache, cache_position_i32,
                                        B, num_kv_heads, S, H,
                                        key_rotated.stride(0), key_rotated.stride(1), key_rotated.stride(2),
                                        value.stride(0), value.stride(1), value.stride(2),
                                        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                                        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                                        cache_position_i32.stride(0),
                                        num_warps=4)

        # 7) Return outputs
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
