import torch
import triton
import triton.language as tl

# -------------------------------
# Triton kernels
# -------------------------------

@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                    H: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    sum_stride,
                    num_warps=4):
    # Each program handles one (b, h, s) token
    pid = tl.program_id(0)
    total = B * num_q_heads * S
    if pid >= total:
        return
    b = pid // (num_q_heads * S)
    hs = pid % (num_q_heads * S)
    h = hs // S
    s = hs % S

    arange = tl.arange(0, 128)  # we will mask for H
    mask = arange < H
    offsets = b * batch_stride_x + h * h_stride_x + s * s_stride_x + arange * 1  # d stride assumed 1 for simplicity
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    sq = x_fp32 * x_fp32
    # sum across H dimension
    sum_val = tl.sum(sq, axis=0)
    tl.store(sum_sums_ptr + pid, sum_val)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_sums_ptr,
                     B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                     H: tl.constexpr,
                     batch_stride_x, h_stride_x, s_stride_x,
                     batch_stride_out, h_stride_out, s_stride_out,
                     eps,
                     num_warps=4):
    pid = tl.program_id(0)
    total = B * num_q_heads * S
    if pid >= total:
        return
    b = pid // (num_q_heads * S)
    hs = pid % (num_q_heads * S)
    h = hs // S
    s = hs % S

    arange = tl.arange(0, 128)
    mask = arange < H
    offsets_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x + arange * 1
    offsets_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out + arange * 1

    x = tl.load(x_ptr + offsets_x, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.load(sum_sums_ptr + pid).to(tl.float32)
    # mean = sum_val / H
    mean = sum_val / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    w = tl.load(weight_ptr + arange, mask=mask, other=1.0).to(tl.float32)
    y = x * inv_rms * w
    # store
    tl.store(out_ptr + offsets_out, y, mask=mask)


@triton.jit
def rotate_sin_cos_kernel(pos_ids_ptr, inv_freq_ptr,
                           cos_ptr, sin_ptr,
                           B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                           pos_stride,
                           cos_stride, sin_stride,
                           num_warps=4):
    # One program per (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)
    if b >= B or s >= S:
        return

    pos = tl.load(pos_ids_ptr + b * pos_stride + s).to(tl.float32)

    # emb vector of length H, emulate cat([freqs, freqs], -1) where
    # first half emb[i] = pos * inv_freq[i//2], second half emb[i] = pos * inv_freq[(i-H//2)//2]
    half = H // 2
    arange = tl.arange(0, H)
    # first half
    idx1 = arange < half
    idx2 = arange >= half
    emb_first = pos * tl.load(inv_freq_ptr + arange // 2, mask=idx1, other=0.0).to(tl.float32)
    # second half values come from inv_freq with index (i - half) // 2
    idx2_part = arange - half
    emb_second = pos * tl.load(inv_freq_ptr + idx2_part // 2, mask=idx2, other=0.0).to(tl.float32)
    emb = tl.where(idx1, emb_first, emb_second)

    c = tl.cos(emb)
    s1 = tl.sin(emb)

    # Store cos and sin as [B, S, H]
    base = b * cos_stride + s * sin_stride
    tl.store(cos_ptr + base + arange, c)
    tl.store(sin_ptr + base + arange, s1)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr,
                           out_ptr,
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

    arange = tl.arange(0, 128)
    mask = arange < H

    offsets_x = b * batch_stride_x + h * h_stride_x + s * s_stride_x + arange * 1
    offsets_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out + arange * 1

    x = tl.load(x_ptr + offsets_x, mask=mask, other=0.0).to(tl.float32)

    # Load cos/sin for this (b, s)
    base = b * batch_stride_cos + h * h_stride_cos + s * s_stride_cos
    cos_vals = tl.load(cos_ptr + base + arange, mask=mask, other=0.0).to(tl.float32)
    base_sin = b * batch_stride_sin + h * h_stride_sin + s * s_stride_sin
    sin_vals = tl.load(sin_ptr + base_sin + arange, mask=mask, other=0.0).to(tl.float32)

    # rotate_half(x) = [-x[..., H//2:], x[..., :H//2]]
    half = H // 2
    idx_first = arange < half
    x_first = tl.load(x_ptr + offsets_x, mask=idx_first, other=0.0).to(tl.float32)
    x_second = tl.load(x_ptr + offsets_x, mask=(~idx_first), other=0.0).to(tl.float32)
    rotated = tl.zeros((128,), dtype=tl.float32)
    rotated = tl.where(idx_first, x_first, -x_second)

    y = x * cos_vals + rotated * sin_vals
    tl.store(out_ptr + offsets_out, y, mask=mask)


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
    # grid: (B*S, num_kv_heads)
    pid0 = tl.program_id(0)  # over B*S
    pid1 = tl.program_id(1)  # over num_kv_heads
    total = B * S
    if pid0 >= total or pid1 >= num_kv_heads:
        return
    b = pid0 // S
    s = pid0 % S
    k = pid1

    dest = tl.load(dest_ptr + s).to(tl.int64)

    arange = tl.arange(0, 128)
    mask = arange < H

    # key_cache write
    offsets_kr = b * batch_stride_kr + k * h_stride_kr + s * s_stride_kr + arange * d_stride_kr
    key_vals = tl.load(key_rot_ptr + offsets_kr, mask=mask, other=0.0).to(tl.float32)
    offsets_kc = b * batch_stride_kc + k * h_stride_kc + dest * dest_stride + arange * d_stride_kc
    tl.store(key_cache_ptr + offsets_kc, key_vals, mask=mask)

    # value_cache write (original value)
    offsets_v = b * batch_stride_v + k * h_stride_v + s * s_stride_v + arange * d_stride_v
    val_vals = tl.load(value_ptr + offsets_v, mask=mask, other=0.0).to(tl.float32)
    offsets_vc = b * batch_stride_vc + k * h_stride_vc + dest * dest_stride_vc + arange * d_stride_vc
    tl.store(value_cache_ptr + offsets_vc, val_vals, mask=mask)


# -------------------------------
# ModelNew.forward
# -------------------------------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                position_ids: torch.Tensor,
                key_cache: torch.Tensor, value_cache: torch.Tensor,
                cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor,
                rms_norm_eps: float):
        # Extract shapes
        B, num_q_heads, S, H = query.shape
        _, num_kv_heads, _, _ = key.shape  # num_kv_heads = 8 in original, but we will not assume; pass as param
        # We restrict head_dim to 64 or 128 for simplicity; mask handles other cases but we'll enforce 64/128 by host.
        assert H in (64, 128), "head_dim must be 64 or 128"

        device = query.device

        # 1) RMSNorm sum for query and key
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=device)
        grid_sum = (B * num_q_heads * S,)
        # Strides: we assume contiguous last dim, so d_stride=1; we pass stride(0), stride(1), stride(2) for batch, head, seq.
        rms_sum_kernel[grid_sum](query, sum_sums_q,
                                 B=B, num_q_heads=num_q_heads, S=S,
                                 H=H,
                                 batch_stride_x=query.stride(0), h_stride_x=query.stride(1),
                                 s_stride_x=query.stride(2),
                                 sum_stride=sum_sums_q.stride(0),
                                 num_warps=4)

        # 2) RMSNorm normalization for query
        query_norm = torch.empty_like(query, dtype=torch.float32, device=device)
        rms_norm_kernel[grid_sum](query, q_norm_weight, query_norm, sum_sums_q,
                                  B=B, num_q_heads=num_q_heads, S=S,
                                  H=H,
                                  batch_stride_x=query.stride(0), h_stride_x=query.stride(1),
                                  s_stride_x=query.stride(2),
                                  batch_stride_out=query_norm.stride(0), h_stride_out=query_norm.stride(1),
                                  s_stride_out=query_norm.stride(2),
                                  eps=rms_norm_eps,
                                  num_warps=4)

        # 3) RMSNorm sum for key
        sum_sums_k = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=device)
        rms_sum_kernel[grid_sum](key, sum_sums_k,
                                 B=B, num_q_heads=num_q_heads, S=S,
                                 H=H,
                                 batch_stride_x=key.stride(0), h_stride_x=key.stride(1),
                                 s_stride_x=key.stride(2),
                                 sum_stride=sum_sums_k.stride(0),
                                 num_warps=4)

        # 4) RMSNorm normalization for key
        key_norm = torch.empty_like(key, dtype=torch.float32, device=device)
        rms_norm_kernel[grid_sum](key, k_norm_weight, key_norm, sum_sums_k,
                                  B=B, num_q_heads=num_q_heads, S=S,
                                  H=H,
                                  batch_stride_x=key.stride(0), h_stride_x=key.stride(1),
                                  s_stride_x=key.stride(2),
                                  batch_stride_out=key_norm.stride(0), h_stride_out=key_norm.stride(1),
                                  s_stride_out=key_norm.stride(2),
                                  eps=rms_norm_eps,
                                  num_warps=4)

        # 5) Precompute cos/sin for rotation using Triton kernel over (B, S)
        cos = torch.empty((B, S, H), dtype=torch.float32, device=device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_rc = (B, S)
        rotate_sin_cos_kernel[grid_rc](position_ids, inv_freq, cos, sin,
                                       B=B, S=S, H=H,
                                       pos_stride=position_ids.stride(0),
                                       cos_stride=cos.stride(0), sin_stride=sin.stride(0),
                                       num_warps=4)

        # 6) Apply rotation to query and key
        query_rot = torch.empty((B, num_q_heads, S, H), dtype=torch.float32, device=device)
        key_rot = torch.empty((B, num_q_heads, S, H), dtype=torch.float32, device=device)

        # For query
        apply_rotation_kernel[(B * num_q_heads * S,)](query_norm, cos, sin, query_rot,
                                                      B=B, num_q_heads=num_q_heads, S=S, H=H,
                                                      batch_stride_x=query_norm.stride(0), h_stride_x=query_norm.stride(1),
                                                      s_stride_x=query_norm.stride(2),
                                                      batch_stride_cos=cos.stride(0), h_stride_cos=cos.stride(1), s_stride_cos=cos.stride(2),
                                                      batch_stride_sin=sin.stride(0), h_stride_sin=sin.stride(1), s_stride_sin=sin.stride(2),
                                                      batch_stride_out=query_rot.stride(0), h_stride_out=query_rot.stride(1), s_stride_out=query_rot.stride(2),
                                                      num_warps=4)

        # For key
        apply_rotation_kernel[(B * num_q_heads * S,)](key_norm, cos, sin, key_rot,
                                                      B=B, num_q_heads=num_q_heads, S=S, H=H,
                                                      batch_stride_x=key_norm.stride(0), h_stride_x=key_norm.stride(1),
                                                      s_stride_x=key_norm.stride(2),
                                                      batch_stride_cos=cos.stride(0), h_stride_cos=cos.stride(1), s_stride_cos=cos.stride(2),
                                                      batch_stride_sin=sin.stride(0), h_stride_sin=sin.stride(1), s_stride_sin=sin.stride(2),
                                                      batch_stride_out=key_rot.stride(0), h_stride_out=key_rot.stride(1), s_stride_out=key_rot.stride(2),
                                                      num_warps=4)

        # 7) Update caches: key_cache[:, :, cache_position] = key_rot, value_cache[:, :, cache_position] = value (original)
        # We need to write per (b, kv_head, s) to dest position cache_position[s]
        grid_update = (B * num_kv_heads * S,)
        # Cast cache_position to int64 for Triton
        dest_positions = cache_position.to(torch.int64)
        # For key_cache
        key_rot_contig = key_rot.contiguous()
        value_contig = value.contiguous()
        key_cache_contig = key_cache.contiguous()
        value_cache_contig = value_cache.contiguous()

        # Strides for loads/stores
        d_stride_kr = key_rot_contig.stride(3)  # typically 1
        d_stride_v = value_contig.stride(3)     # typically 1
        d_stride_kc = key_cache_contig.stride(3)  # typically 1
        dest_stride = dest_positions.stride(0)    # 1
        d_stride_vc = value_cache_contig.stride(3)  # typically 1

        # Launch update kernel
        update_cache_kernel[grid_update](key_rot_contig, value_contig,
                                         key_cache_contig, value_cache_contig,
                                         dest_positions,
                                         B=B, num_kv_heads=num_kv_heads, S=S,
                                         H=H,
                                         batch_stride_kr=key_rot_contig.stride(0),
                                         h_stride_kr=key_rot_contig.stride(1),
                                         s_stride_kr=key_rot_contig.stride(2),
                                         d_stride_kr=key_rot_contig.stride(3),
                                         batch_stride_v=value_contig.stride(0),
                                         h_stride_v=value_contig.stride(1),
                                         s_stride_v=value_contig.stride(2),
                                         d_stride_v=value_contig.stride(3),
                                         batch_stride_kc=key_cache_contig.stride(0),
                                         h_stride_kc=key_cache_contig.stride(1),
                                         dest_stride=dest_stride,
                                         d_stride_kc=key_cache_contig.stride(3),
                                         batch_stride_vc=value_cache_contig.stride(0),
                                         h_stride_vc=value_cache_contig.stride(1),
                                         dest_stride_vc=dest_positions.stride(0),
                                         d_stride_vc=value_cache_contig.stride(3),
                                         num_warps=4)

        # Return rotated tensors and updated caches
        # Note: original run returns (query_rotated, key_rotated, key_cache, value_cache).
        # Our computation uses query_norm and key_norm for rotation; we need to rotate original query and key after RMSNorm, which we have.
        # So we return query_rot, key_rot, updated key_cache, updated value_cache.
        return query_rot, key_rot, key_cache_contig, value_cache_contig


# -------------------------------
# Example usage if needed
# -------------------------------
# ModelNew must be callable like the original Model.
# ModelNew.forward expects the same inputs as the original run.


def run(*args):
    return ModelNew()(*args)
