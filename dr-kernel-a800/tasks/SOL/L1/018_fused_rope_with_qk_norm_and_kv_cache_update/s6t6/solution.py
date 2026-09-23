import torch
import triton
import triton.language as tl

@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                    H: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    BLOCK_H: tl.constexpr):
    # One program per (b, h, s)
    pid = tl.program_id(0)
    total = B * num_q_heads * S
    if pid >= total:
        return
    b = pid // (num_q_heads * S)
    hs = pid % (num_q_heads * S)
    h = hs // S
    s = hs % S

    acc = 0.0
    # Loop over head_dim in chunks of BLOCK_H
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offsets = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    # Accumulate into per-token sum (we have only one s program per (b,h), so we write once)
    tl.store(sum_sums_ptr + pid, acc)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                    H: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    batch_stride_out, h_stride_out, s_stride_out,
                    eps: tl.constexpr,
                    num_warps=4):
    # One program per (b, h, s)
    pid = tl.program_id(0)
    total = B * num_q_heads * S
    if pid >= total:
        return
    b = pid // (num_q_heads * S)
    hs = pid % (num_q_heads * S)
    h = hs // S
    s = hs % S

    # Load sum of squares for this token
    sum_val = tl.load(sum_sums_ptr + pid)
    mean = sum_val / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Load x and weight, compute normalized and scaled
    for off in range(0, H, 128):
        idx = off + tl.arange(0, 128)
        mask = idx < H
        offsets_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        x = tl.load(x_ptr + offsets_x, mask=mask, other=0.0).to(tl.float32)
        offsets_w = idx  # weight is 1D
        w = tl.load(weight_ptr + offsets_w, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * w
        # Store as bfloat16
        offsets_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + offsets_out, y.to(tl.bfloat16), mask=mask)


@triton.jit
def precompute_sin_cos_per_token(pos_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                                 H: tl.constexpr):
    # One program per token (we expect host to launch grid=(B,S))
    b = tl.program_id(0)  # actually batch index
    s = tl.program_id(1)  # token index
    pos = tl.load(pos_ptr + b * S + s)
    pos = pos.to(tl.float32)
    arange = tl.arange(0, H)
    mask = arange < H
    # Build emb = cat([pos * inv_freq[:H//2], pos * inv_freq[:H//2]])
    half = H // 2
    first = arange < half
    second = arange >= half
    emb = tl.zeros((H,), dtype=tl.float32)
    # inv_freq is 1D length H//2; load corresponding entries
    inv_first = tl.load(inv_freq_ptr + tl.where(first, arange, 0), mask=first, other=0.0)
    inv_second = tl.load(inv_freq_ptr + tl.where(second, arange - half, 0), mask=second, other=0.0)
    # Note: inv_freq_ptr length is H//2; we need to index by column index
    # Since inv_freq has length half, we compute indices via idx//2 for first half and (idx - half)//2 for second half
    idx1 = arange
    idx2 = arange - half
    inv_first = tl.load(inv_freq_ptr + (idx1 // 2), mask=first, other=0.0)
    inv_second = tl.load(inv_freq_ptr + (idx2 // 2), mask=second, other=0.0)
    emb = tl.where(first, pos * inv_first, pos * inv_second)
    c = tl.cos(emb)
    s = tl.sin(emb)
    base = b * (S * H) + s * H
    tl.store(cos_ptr + base + arange, c, mask=mask)
    tl.store(sin_ptr + base + arange, s, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                           H: tl.constexpr,
                           batch_stride_x, h_stride_x, s_stride_x,
                           batch_stride_cos, h_stride_cos, s_stride_cos,
                           batch_stride_sin, h_stride_sin, s_stride_sin,
                           batch_stride_out, h_stride_out, s_stride_out,
                           num_warps=4):
    # One program per (b, h, s)
    pid = tl.program_id(0)
    total = B * num_q_heads * S
    if pid >= total:
        return
    b = pid // (num_q_heads * S)
    hs = pid % (num_q_heads * S)
    h = hs // S
    s = hs % S

    half = H // 2
    arange = tl.arange(0, H)
    mask = arange < H

    offsets_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x + arange
    offsets_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out + arange

    x = tl.load(x_ptr + offsets_x, mask=mask, other=0.0).to(tl.float32)

    base = b * batch_stride_cos + h * h_stride_cos + s * s_stride_cos
    cos_vals = tl.load(cos_ptr + base + arange, mask=mask, other=0.0).to(tl.float32)
    base_sin = b * batch_stride_sin + h * h_stride_sin + s * s_stride_sin
    sin_vals = tl.load(sin_ptr + base_sin + arange, mask=mask, other=0.0).to(tl.float32)

    first = arange < half
    second = arange >= half

    x_first = tl.load(x_ptr + offsets_x, mask=first, other=0.0).to(tl.float32)
    x_second = tl.load(x_ptr + offsets_x, mask=second, other=0.0).to(tl.float32)

    rotated = tl.zeros((H,), dtype=tl.float32)
    rotated = tl.where(first, x_first, -x_second)
    y = x * cos_vals + rotated * sin_vals

    tl.store(out_ptr + offsets_out, y.to(tl.bfloat16), mask=mask)


@triton.jit
def update_cache_kernel(key_rot_ptr, value_ptr,
                        key_cache_ptr, value_cache_ptr,
                        dest_ptr,
                        B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr,
                        H: tl.constexpr,
                        batch_stride_kr, h_stride_kr, s_stride_kr, d_stride_kr,
                        batch_stride_v, h_stride_v, s_stride_v, d_stride_v,
                        batch_stride_kc, h_stride_kc, dest_stride, d_stride_kc,
                        batch_stride_vc, h_stride_vc, dest_stride_vc, d_stride_vc,
                        num_warps=4):
    # 2D grid: (B*S, num_kv_heads)
    pid0 = tl.program_id(0)  # over B*S
    pid1 = tl.program_id(1)  # over num_kv_heads
    b = pid0 // S
    s = pid0 % S
    k = pid1

    dest = tl.load(dest_ptr + s).to(tl.int64)

    arange = tl.arange(0, H)
    mask = arange < H

    # Load key_rot at (b, k, s, :)
    offsets_kr = b * batch_stride_kr + k * h_stride_kr + s * s_stride_kr + arange * d_stride_kr
    key_vals = tl.load(key_rot_ptr + offsets_kr, mask=mask, other=0.0).to(tl.float32)

    # Store into key_cache at position dest
    offsets_kc = b * batch_stride_kc + k * h_stride_kc + dest * dest_stride + arange * d_stride_kc
    tl.store(key_cache_ptr + offsets_kc, key_vals.to(tl.bfloat16), mask=mask)

    # Load value at (b, k, s, :)
    offsets_v = b * batch_stride_v + k * h_stride_v + s * s_stride_v + arange * d_stride_v
    vals = tl.load(value_ptr + offsets_v, mask=mask, other=0.0).to(tl.float32)

    # Store into value_cache at position dest
    offsets_vc = b * batch_stride_vc + k * h_stride_vc + dest * dest_stride_vc + arange * d_stride_vc
    tl.store(value_cache_ptr + offsets_vc, vals.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value,
                position_ids,  # int64 [B, S]
                key_cache, value_cache,  # [B, num_kv_heads, max_pos, H] bfloat16
                cache_position,  # int64 [S]
                q_norm_weight, k_norm_weight,  # [H] bfloat16
                inv_freq,  # [H//2] float32
                rms_norm_eps):
        # Shapes
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # Ensure contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        position_ids = position_ids.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        cache_position = cache_position.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()

        # 1) RMSNorm: compute sum of squares
        sum_sums_q = torch.empty(B * num_q_heads * S, dtype=torch.float32, device=query.device)
        grid_rms_sum = (B * num_q_heads * S,)
        rms_sum_kernel[grid_rms_sum](
            query, sum_sums_q,
            B, num_q_heads, S,
            H,
            query.stride(0), query.stride(1), query.stride(2),
            BLOCK_H=128,
            num_warps=4
        )

        # 2) RMSNorm normalization for query
        query_norm = torch.empty_like(query, dtype=torch.bfloat16, device=query.device)
        rms_norm_kernel[grid_rms_sum](
            query, q_norm_weight, query_norm, sum_sums_q,
            B, num_q_heads, S,
            H,
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            rms_norm_eps,
            num_warps=4
        )

        # 3) RMSNorm for key
        sum_sums_k = torch.empty(B * num_q_heads * S, dtype=torch.float32, device=query.device)
        rms_sum_kernel[grid_rms_sum](
            key, sum_sums_k,
            B, num_q_heads, S,
            H,
            key.stride(0), key.stride(1), key.stride(2),
            BLOCK_H=128,
            num_warps=4
        )

        key_norm = torch.empty_like(key, dtype=torch.bfloat16, device=query.device)
        rms_norm_kernel[grid_rms_sum](
            key, k_norm_weight, key_norm, sum_sums_k,
            B, num_q_heads, S,
            H,
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            rms_norm_eps,
            num_warps=4
        )

        # 4) Precompute sin/cos per token: grid over (B, S)
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        grid_rc = (B, S)
        precompute_sin_cos_per_token[grid_rc](
            position_ids, inv_freq, cos, sin,
            H=H,
            num_warps=4
        )

        # 5) Apply rotation to query and key: one program per (b, h, s)
        query_rot = torch.empty_like(query_norm, dtype=torch.bfloat16, device=query.device)
        key_rot = torch.empty_like(key_norm, dtype=torch.bfloat16, device=query.device)

        grid_rot = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            cos.stride(0), 0, 0,
            sin.stride(0), 0, 0,
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            num_warps=4
        )

        apply_rotation_kernel[grid_rot](
            key_norm, cos, sin, key_rot,
            B, num_q_heads, S, H,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            cos.stride(0), 0, 0,
            sin.stride(0), 0, 0,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            num_warps=4
        )

        # 6) Update caches: 2D grid (B*S, num_kv_heads)
        grid_up = (B * S, num_kv_heads)
        update_cache_kernel[grid_up](
            key_rot, value,
            key_cache, value_cache,
            cache_position,
            B, num_kv_heads, S, H,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), 1,
            value.stride(0), value.stride(1), value.stride(2), 1,
            key_cache.stride(0), key_cache.stride(1), cache_position.stride(0), 1,
            value_cache.stride(0), value_cache.stride(1), cache_position.stride(0), 1,
            num_warps=4
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
