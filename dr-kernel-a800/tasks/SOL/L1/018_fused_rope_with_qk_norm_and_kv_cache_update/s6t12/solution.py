import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride, h_stride, s_stride,
                    BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_sq = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs = b * batch_stride + head * h_stride + s * s_stride + idx
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sum_sums_ptr + pid, sum_sq)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_sums_ptr,
                     B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                     batch_stride_x, h_stride_x, s_stride_x,
                     batch_stride_out, h_stride_out, s_stride_out,
                     eps: tl.constexpr, BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_sq = tl.load(sum_sums_ptr + pid)
    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = tl.load(weight_ptr + tl.arange(0, BLOCK_H), mask=tl.arange(0, BLOCK_H) < H, other=1.0).to(tl.float32)
    scale = inv_rms * scale

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        y_vals = x_vals * scale
        offs_out = b * batch_stride_out + head * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + offs_out, y_vals, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, H2: tl.constexpr,
                              pos_stride,
                              cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # Each program handles one (b, s)
    pid = tl.program_id(0)
    if pid >= B * S:
        return
    b = pid // S
    s = pid % S

    pos = tl.load(position_ids_ptr + b * pos_stride + s).to(tl.float32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        first_half = idx < H2
        inv_freq_idx = idx // 2  # only valid for first half
        inv_freq_vals = tl.load(inv_freq_ptr + inv_freq_idx, mask=first_half, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals
        # Duplicate for second half
        emb = tl.where(first_half, emb_first, emb_first)

        c = tl.cos(emb)
        s_ = tl.sin(emb)

        base = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                           batch_stride_x, h_stride_x, s_stride_x,
                           batch_stride_out, h_stride_out, s_stride_out,
                           cos_stride0, cos_stride1, cos_stride2,
                           sin_stride0, sin_stride1, sin_stride2,
                           BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    base = b * cos_stride0 + s * cos_stride1
    half = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # Load x
        offs_x = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for this (b, s)
        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        # Build rotated vector: for i < half, rotated[i] = -x[half + i]; for i >= half, rotated[i] = x[i - half]
        rotated = tl.zeros((BLOCK_H,), dtype=tl.float32)
        first_mask = idx < half
        second_mask = idx >= half
        # For first half
        x_second = tl.load(x_ptr + offs_x, mask=second_mask, other=0.0).to(tl.float32)
        x_first = tl.load(x_ptr + offs_x, mask=first_mask, other=0.0).to(tl.float32)
        rotated = tl.where(first_mask, -x_second, x_first)

        y = x_vals * cos_vals + rotated * sin_vals

        offs_out = b * batch_stride_out + head * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + offs_out, y, mask=mask)


@triton.jit
def update_cache_kernel(x_ptr, value_ptr, key_cache_ptr, value_cache_ptr, cache_pos_ptr,
                        B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                        batch_stride_x, h_stride_x, s_stride_x,
                        batch_stride_kc, h_stride_kc, dest_stride, d_stride_kc,
                        batch_stride_vc, h_stride_vc, dest_stride_vc, d_stride_vc,
                        BLOCK_H: tl.constexpr):
    # Each program handles one (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    dest = tl.load(cache_pos_ptr + s).to(tl.int32)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        # key_cache write
        src_offs = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        src_vals = tl.load(x_ptr + src_offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(key_cache_ptr + b * batch_stride_kc + head * h_stride_kc + dest * dest_stride + idx * d_stride_kc, src_vals, mask=mask)
        # value_cache write (original value, not rotated)
        src_offs_val = b * batch_stride_vc + head * h_stride_vc + s * s_stride_x + idx  # note: s used for src, dest uses cache_pos
        src_vals_val = tl.load(value_ptr + src_offs_val, mask=mask, other=0.0).to(tl.float32)
        tl.store(value_cache_ptr + b * batch_stride_vc + head * h_stride_vc + dest * dest_stride + idx * d_stride_vc, src_vals_val, mask=mask)


# Example use in ModelNew.forward (you would replace torch code with these kernels):
# get_inputs: same as original helper
# def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
#     # ... returns dict with tensors including cache_len, but we won't use cache_position there in host
#     pass
# run: same signature as original, but ModelNew.forward uses Triton
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; all computation in Triton

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
        # Ensure dtypes: use float32 for trig, bfloat16 for outputs
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # 1) RMSNorm query -> query_norm
        query_norm = torch.empty_like(query, dtype=torch.float32)  # normalized in float32, then cast
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        grid_sum_q = (B * num_q_heads * S,)
        BLOCK_H = 64  # works for H up to 128; mask handles tails
        rms_sum_kernel[grid_sum_q](query, sum_sums_q,
                                   B, num_q_heads, S, H,
                                   query.stride(0), query.stride(1), query.stride(2),
                                   BLOCK_H=BLOCK_H, num_warps=4)
        rms_norm_kernel[grid_sum_q](query, q_norm_weight.to(torch.float32), query_norm, sum_sums_q,
                                    B, num_q_heads, S, H,
                                    query.stride(0), query.stride(1), query.stride(2),
                                    query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                    rms_norm_eps, BLOCK_H=BLOCK_H, num_warps=4)

        # 2) Compute cos/sin for rotation per (b, s)
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        H2 = H // 2
        grid_rc = (B, S)
        rotate_sin_cos_kernel_b_s[grid_rc](position_ids, inv_freq, cos, sin,
                                           B, S, H, H2,
                                           position_ids.stride(0),
                                           cos.stride(0), cos.stride(1), cos.stride(2),
                                           sin.stride(0), sin.stride(1), sin.stride(2),
                                           BLOCK_H=BLOCK_H, num_warps=4)

        # 3) Apply rotation to query_norm -> query_rot
        query_rot = torch.empty_like(query, dtype=torch.float32)
        grid_rot = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot](query_norm, cos, sin, query_rot,
                                        B, num_q_heads, S, H,
                                        query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
                                        query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
                                        cos.stride(0), cos.stride(1), cos.stride(2),
                                        sin.stride(0), sin.stride(1), sin.stride(2),
                                        BLOCK_H=BLOCK_H, num_warps=4)

        # 4) RMSNorm key -> key_norm
        key_norm = torch.empty_like(key, dtype=torch.float32)
        sum_sums_k = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=key.device)
        grid_sum_k = (B * num_kv_heads * S,)
        rms_sum_kernel[grid_sum_k](key, sum_sums_k,
                                   B, num_kv_heads, S, H,
                                   key.stride(0), key.stride(1), key.stride(2),
                                   BLOCK_H=BLOCK_H, num_warps=4)
        rms_norm_kernel[grid_sum_k](key, k_norm_weight.to(torch.float32), key_norm, sum_sums_k,
                                    B, num_kv_heads, S, H,
                                    key.stride(0), key.stride(1), key.stride(2),
                                    key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                    rms_norm_eps, BLOCK_H=BLOCK_H, num_warps=4)

        # 5) Apply rotation to key_norm -> key_rot
        key_rot = torch.empty_like(key, dtype=torch.float32)
        grid_rot_k = (B * num_kv_heads * S,)
        apply_rotation_kernel[grid_rot_k](key_norm, cos, sin, key_rot,
                                          B, num_kv_heads, S, H,
                                          key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
                                          key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
                                          cos.stride(0), cos.stride(1), cos.stride(2),
                                          sin.stride(0), sin.stride(1), sin.stride(2),
                                          BLOCK_H=BLOCK_H, num_warps=4)

        # 6) Update caches with query_rot and key_rot at cache_position
        # Cast to bfloat16 for cache writes (original caches are bfloat16)
        query_rot_cast = query_rot.to(torch.bfloat16)
        key_rot_cast = key_rot.to(torch.bfloat16)
        value_cast = value.to(torch.bfloat16)

        # Ensure cache_position is int32 on device
        cache_pos_i32 = cache_position.to(torch.int32)

        grid_cache = (B * num_kv_heads * S,)
        update_cache_kernel[grid_cache](query_rot_cast, value_cast, key_cache, value_cache, cache_pos_i32,
                                        B, num_kv_heads, S, H,
                                        query_rot_cast.stride(0), query_rot_cast.stride(1), query_rot_cast.stride(2),
                                        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
                                        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
                                        BLOCK_H=BLOCK_H, num_warps=4)

        # Return as original run did: rotated query and key, and updated caches
        return query_rot_cast, key_rot_cast, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
