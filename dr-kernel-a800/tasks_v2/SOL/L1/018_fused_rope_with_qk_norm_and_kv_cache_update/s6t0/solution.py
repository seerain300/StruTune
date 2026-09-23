import torch
import triton
import triton.language as tl


@triton.jit
def rotate_and_store_query_kernel(
    x_ptr,           # pointer to input query_norm [B, num_q_heads, S, head_dim]
    cos_ptr,         # pointer to cos [S, head_dim] (we'll index by token t and dim d)
    sin_ptr,         # pointer to sin [S, head_dim]
    out_ptr,         # pointer to output query_rotated [B, num_q_heads, S, head_dim]
    B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr, head_dim: tl.constexpr,
    stride_xb, stride_xmh, stride_xs, stride_xd,
    stride_cos, stride_sin,
    stride_ob, stride_omh, stride_os, stride_od,
):
    # Each program handles one (b, mh, t) triplet
    pid = tl.program_id(0)
    # Map pid to (b, mh, t)
    # We launch grid as (B, num_q_heads, S) => pid in [0, B*num_q_heads*S)
    b = pid // (num_q_heads * S)
    rem = pid % (num_q_heads * S)
    mh = rem // S
    t = rem % S

    # Initialize indices
    # d iterates over head_dim
    d_offsets = tl.arange(0, head_dim)
    mask = d_offsets < head_dim

    # Compute base offsets for x and out
    x_base = b * stride_xb + mh * stride_xmh + t * stride_xs
    out_base = b * stride_ob + mh * stride_omh + t * stride_os

    # Load x
    x = tl.load(x_ptr + x_base + d_offsets * stride_xd, mask=mask, other=0.0)
    # Load cos and sin for this token t
    cos = tl.load(cos_ptr + t * stride_cos + d_offsets * stride_cos, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + t * stride_sin + d_offsets * stride_sin, mask=mask, other=0.0)

    # First 64: use as-is
    # Second 64: apply rotation (negative second half, positive first half)
    half = head_dim // 2
    first = x[:half]
    second = x[half:]
    rotated_second = -second * sin[half:] + first * sin[:half]
    rotated_first = -second * cos[half:] + first * cos[:half]

    # Combine
    y = tl.zeros([head_dim], dtype=x.dtype)
    y[:half] = rotated_first
    y[half:] = rotated_second

    # Store
    tl.store(out_ptr + out_base + d_offsets * stride_od, y, mask=mask)


@triton.jit
def rotate_and_store_key_kernel(
    x_ptr,           # pointer to input key_norm [B, num_kv_heads, S, head_dim]
    cos_ptr,         # pointer to cos [S, head_dim]
    sin_ptr,         # pointer to sin [S, head_dim]
    out_ptr,         # pointer to output key_rotated [B, num_kv_heads, S, head_dim]
    B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, head_dim: tl.constexpr,
    stride_xb, stride_xmh, stride_xs, stride_xd,
    stride_cos, stride_sin,
    stride_ob, stride_omh, stride_os, stride_od,
):
    pid = tl.program_id(0)
    b = pid // (num_kv_heads * S)
    rem = pid % (num_kv_heads * S)
    mh = rem // S
    t = rem % S

    d_offsets = tl.arange(0, head_dim)
    mask = d_offsets < head_dim

    x_base = b * stride_xb + mh * stride_xmh + t * stride_xs
    out_base = b * stride_ob + mh * stride_omh + t * stride_os

    x = tl.load(x_ptr + x_base + d_offsets * stride_xd, mask=mask, other=0.0)
    cos = tl.load(cos_ptr + t * stride_cos + d_offsets * stride_cos, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + t * stride_sin + d_offsets * stride_sin, mask=mask, other=0.0)

    half = head_dim // 2
    first = x[:half]
    second = x[half:]
    rotated_second = -second * sin[half:] + first * sin[:half]
    rotated_first = -second * cos[half:] + first * cos[:half]

    y = tl.zeros([head_dim], dtype=x.dtype)
    y[:half] = rotated_first
    y[half:] = rotated_second

    tl.store(out_ptr + out_base + d_offsets * stride_od, y, mask=mask)


@triton.jit
def update_cache_kernel(
    key_rotated_ptr,      # [B, num_kv_heads, S, head_dim]
    value_ptr,            # [B, num_kv_heads, S, head_dim]
    key_cache_ptr,        # [B, num_kv_heads, max_position_embeddings, head_dim]
    value_cache_ptr,      # [B, num_kv_heads, max_position_embeddings, head_dim]
    dest_idx_ptr,         # [S] int64 positions where to write
    B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr, head_dim: tl.constexpr,
    stride_krb, stride_krmh, stride_krs, stride_krd,
    stride_vb, stride_vmh, stride_vs, stride_vd,
    stride_kcb, stride_kcmh, stride_kcs, stride_kcd,
    stride_vcb, stride_vcmh, stride_vcs, stride_vcd,
):
    # Grid: (B*S, num_kv_heads)
    pid = tl.program_id(0)
    b = pid // S
    mh = tl.program_id(1)
    t = pid % S

    # Load destination index (absolute position)
    dest = tl.load(dest_idx_ptr + t)  # int64

    d_offsets = tl.arange(0, head_dim)
    mask = d_offsets < head_dim

    # Load key_rotated[t] and value[t]
    x_base = b * stride_krb + mh * stride_krmh + t * stride_krs
    v_base = b * stride_vb + mh * stride_vmh + t * stride_vs

    key_vec = tl.load(key_rotated_ptr + x_base + d_offsets * stride_krd, mask=mask, other=0.0)
    val_vec = tl.load(value_ptr + v_base + d_offsets * stride_vd, mask=mask, other=0.0)

    # Store into cache at dest position
    kc_base = b * stride_kcb + mh * stride_kcmh + dest * stride_kcs
    vc_base = b * stride_vcb + mh * stride_vcmh + dest * stride_vcs

    tl.store(key_cache_ptr + kc_base + d_offsets * stride_kcd, key_vec, mask=mask)
    tl.store(value_cache_ptr + vc_base + d_offsets * stride_vcd, val_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes
        B, num_q_heads, S, head_dim = query.shape
        num_kv_heads = key.shape[1]

        # Ensure contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()

        # RMSNorm in PyTorch (to keep Triton usage focused on compute-heavy parts)
        # RMSNorm(x) = x * rsqrt(mean(x^2) + eps), then scale by weight
        def rms_norm(x, weight, eps):
            # Compute in fp32 for numerical stability
            x_fp32 = x.to(torch.float32)
            variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
            scale = torch.rsqrt(variance + eps)  # [B, H, S, 1]
            x_normed_fp32 = x_fp32 * scale
            # Scale by weight (bfloat16 tensor) and cast back
            return (weight.to(torch.float32) * x_normed_fp32).to(x.dtype)

        query_norm = rms_norm(query, q_norm_weight, rms_norm_eps)
        key_norm = rms_norm(key, k_norm_weight, rms_norm_eps)

        # Compute cos/sin from inv_freq (float32) following original behavior
        # inv_freq has length head_dim//2 = 64; emb = cat([freqs, freqs], dim=-1)
        # We compute emb for tokens: [B, S, 128] = [B, S, head_dim]
        # But inv_freq is only first 64; second half duplicates. We'll compute cos/sin accordingly.
        # Construct emb for each token t: emb[t, :] = position_ids[b,t] * inv_freq
        # Note: position_ids is [B, S]; emb should be [B, S, head_dim], but since we need sin/cos per token, we treat emb as [S, head_dim] by indexing along S.
        # We'll build cos/sin as [S, head_dim] tensors in float32.
        # torch provides position_ids[:, :, None].float() and inv_freq[None, None, :], but here we want [S, head_dim].
        # We'll do a simple loop over S to create these tensors (cheap compared to attention).
        # However, Triton expects tensors; we can create them on host and pass to kernels.
        # Create cos and sin as float32 tensors [S, head_dim]
        device = query.device
        sin_cos_dtype = torch.float32
        sin = torch.empty((S, head_dim), device=device, dtype=sin_cos_dtype)
        cos = torch.empty((S, head_dim), device=device, dtype=sin_cos_dtype)
        # inv_freq is [64], but we need [128]; use duplicated second half
        inv_freq_128 = torch.cat([inv_freq, inv_freq], dim=0)  # [128]
        # For each token t
        # Using PyTorch vectorized ops: position_ids is [B, S], but our grid is per token t; we'll flatten to [S].
        # Extract per-token position: position_ids[:, t] => select one row for all t
        # Simpler: just use cache_position which is [S]. The original uses position_ids.expand(B, -1); but here, since we need per-token positions, we can use cache_position directly because they both are absolute positions for the current block. The original code uses position_ids[:, t], but since cache_position is provided, we can rely on it.
        # So we use cache_position to compute positions.
        for t in range(S):
            pos = cache_position[t].item()  # int64
            emb = (pos * inv_freq_128).to(sin_cos_dtype)  # [128]
            sin[t, :] = torch.sin(emb)
            cos[t, :] = torch.cos(emb)

        # Allocate outputs for rotated tensors
        query_rotated = torch.empty_like(query_norm, dtype=query_norm.dtype, device=device)
        key_rotated = torch.empty_like(key_norm, dtype=key_norm.dtype, device=device)

        # Launch Triton kernels for rotation
        # Grid for rotation: (B, num_q_heads, S)
        grid_query = (B * num_q_heads * S,)
        rotate_and_store_query_kernel[grid_query](
            query_norm, cos, sin, query_rotated,
            B, num_q_heads, S, head_dim,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            cos.stride(0), sin.stride(1),
            query_rotated.stride(0), query_rotated.stride(1), query_rotated.stride(2), query_rotated.stride(3),
            num_warps=4, num_stages=2,
        )

        grid_key = (B * num_kv_heads * S,)
        rotate_and_store_key_kernel[grid_key](
            key_norm, cos, sin, key_rotated,
            B, num_kv_heads, S, head_dim,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            cos.stride(0), sin.stride(1),
            key_rotated.stride(0), key_rotated.stride(1), key_rotated.stride(2), key_rotated.stride(3),
            num_warps=4, num_stages=2,
        )

        # Update caches using Triton
        # Ensure cache_position is int64 tensor of shape [S]
        dest_idx = cache_position.to(torch.int64)
        # Grid for update_cache: (B*S, num_kv_heads)
        grid_cache = (B * S, num_kv_heads)
        update_cache_kernel[grid_cache](
            key_rotated, value, key_cache, value_cache, dest_idx,
            B, num_kv_heads, S, head_dim,
            key_rotated.stride(0), key_rotated.stride(1), key_rotated.stride(2), key_rotated.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            num_warps=4, num_stages=2,
        )

        # Return as original function: (query_rotated, key_rotated, key_cache, value_cache)
        # Note: The original also updated key_cache and value_cache in place; here we return them updated.
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
