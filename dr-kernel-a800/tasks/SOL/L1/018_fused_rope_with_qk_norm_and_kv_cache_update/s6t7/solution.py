import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                    H: tl.constexpr,
                    batch_stride: tl.constexpr, h_stride: tl.constexpr, s_stride: tl.constexpr,
                    BLOCK_H: tl.constexpr):
    total = B * num_q_heads * S
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (num_q_heads * S)
    hs = pid % (num_q_heads * S)
    h = hs // S
    s = hs % S

    sum_val = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * batch_stride + h * h_stride + s * s_stride + idx
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x * x, axis=0)
    tl.store(sum_sums_ptr + pid, sum_val)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                    H: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    batch_stride_w,
                    batch_stride_out, h_stride_out, s_stride_out,
                    eps: tl.constexpr,
                    BLOCK_H: tl.constexpr):
    total = B * num_q_heads * S
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (num_q_heads * S)
    hs = pid % (num_q_heads * S)
    h = hs // S
    s = hs % S

    sum_val = tl.load(sum_sums_ptr + pid).to(tl.float32)
    mean = sum_val / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx
        w_offs = batch_stride_w  # weight is 1D, element per dim
        out_offs = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + w_offs).to(tl.float32)
        y = x * inv_rms * weight
        tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr,
                               cos_ptr, sin_ptr,
                               B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                               batch_stride_pos, s_stride_pos,
                               cos_batch_stride, cos_h_stride, cos_s_stride,
                               sin_batch_stride, sin_h_stride, sin_s_stride,
                               BLOCK_H: tl.constexpr):
    # One program per (b, s)
    pid = tl.program_id(0)
    if pid >= B * S:
        return
    b = pid // S
    s = pid % S
    pos = tl.load(position_ids_ptr + b * batch_stride_pos + s * s_stride_pos).to(tl.float32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        half = H // 2
        # Load inv_freq for first half: inv_freq[i//2] for i < half
        freq_idx = tl.minimum(idx, half - 1) // 2  # safe for i < half
        inv_vals = tl.load(inv_freq_ptr + freq_idx, mask=idx < half, other=0.0).to(tl.float32)
        base = pos * inv_vals
        emb = base
        # Duplicate second half: emb[i] = emb[i - half] for i >= half
        emb_second = tl.where(idx >= half, emb[idx - half], 0.0)
        emb = tl.where(idx >= half, emb_second, emb)
        # Compute sin/cos
        c = tl.cos(emb)
        s = tl.sin(emb)
        # Store to cos/sin tensors at [b, s, :]
        base_cos = b * cos_batch_stride + 0 * cos_h_stride + s * cos_s_stride
        base_sin = b * sin_batch_stride + 0 * sin_h_stride + s * sin_s_stride
        tl.store(cos_ptr + base_cos + idx, c, mask=mask)
        tl.store(sin_ptr + base_sin + idx, s, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr,
                           out_ptr,
                           B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                           H: tl.constexpr,
                           batch_stride_x, h_stride_x, s_stride_x,
                           cos_batch_stride, cos_h_stride, cos_s_stride,
                           sin_batch_stride, sin_h_stride, sin_s_stride,
                           batch_stride_out, h_stride_out, s_stride_out,
                           BLOCK_H: tl.constexpr):
    total = B * num_q_heads * S
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (num_q_heads * S)
    hs = pid % (num_q_heads * S)
    h = hs // S
    s = hs % S

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * batch_stride_x + h * batch_stride_x + s * s_stride_x + idx  # h contributes via out_ptr's h_stride, but x is [B,H,S,], so use strides from shape; corrected below
        # Correction: x_ptr has strides for [B,H,S,H], i.e., batch_stride_x, h_stride_x, s_stride_x, d_stride_x.
        # Since we don't pass d_stride_x, we need to pass x as [B,H,S,H] and assume h_stride_x is the dim stride for H.
        # We previously omitted d_stride_x; define properly here:
        # We need to pass x_ptr with 4D, but to keep simple, we restructure:
        # The following code assumes x_ptr layout [B,H,S,H] with strides [batch,h,s,dim]. For clarity, we relaunch with proper 4D.

        # Relaunch with correct 4D apply rotation kernel:
        # (We keep code here minimal; in practice, Triton kernels should be passed 4D pointers.)
        # To avoid confusion, we provide a corrected kernel below and rerun this forward with it.


# Corrected, complete 4D apply rotation kernel with masks and proper strides:
@triton.jit
def apply_rotation_kernel_4d(x_ptr, cos_ptr, sin_ptr, out_ptr,
                             B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                             batch_stride_x, h_stride_x, s_stride_x, d_stride_x,
                             cos_batch_stride, cos_h_stride, cos_s_stride, cos_d_stride,
                             sin_batch_stride, sin_h_stride, sin_s_stride, sin_d_stride,
                             batch_stride_out, h_stride_out, s_stride_out, d_stride_out,
                             BLOCK_H: tl.constexpr):
    total = B * num_q_heads * S
    pid = tl.program_id(0)
    if pid >= total:
        return
    b = pid // (num_q_heads * S)
    hs = pid % (num_q_heads * S)
    h = hs // S
    s = hs % S

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x_offs = b * batch_stride_x + h * h_stride_x + s * s_stride_x + idx * d_stride_x
        # cos/sin for (b, s) applies to all h in rotation, but we only need per-token cos/sin vectors. The previous kernel mixed h incorrectly; here we fix:
        # We don't need cos/sin for each h; cos/sin are per token (b, s). So we load cos/sin for (b, s) and apply across h. But out_ptr is per h. We need to apply rotation per (h).
        # To avoid this confusion, we will instead compute rotation by using out_ptr = x_ptr rotated; we'll keep cos/sin loaded for (b, s).

        # Load x[b,h,s,:]
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for token (b, s)
        base_cos = b * cos_batch_stride + 0 * cos_h_stride + s * cos_s_stride + 0 * cos_d_stride
        base_sin = b * sin_batch_stride + 0 * sin_h_stride + s * sin_s_stride + 0 * sin_d_stride
        cos_vals = tl.load(cos_ptr + base_cos + idx, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base_sin + idx, mask=mask, other=0.0).to(tl.float32)

        half = H // 2
        # Rotate: y = x * cos + rotated * sin, where rotated = [-x_second, x_first]
        x_first = x[:half]
        x_second = x[half:]
        rotated = -x_second + x_first  # incorrect sign; correct below
        # Correct rotation: rotated = [-x_second, x_first] -> for first half use -x_second, for second half use x_first
        rotated = tl.where(idx < half, -x_second, x_first)
        y = x * cos_vals + rotated * sin_vals

        out_offs = b * batch_stride_out + h * h_stride_out + s * s_stride_out + idx * d_stride_out
        tl.store(out_ptr + out_offs, y, mask=mask)


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
                        BLOCK_H: tl.constexpr):
    # Grid: (B*S, num_kv_heads)
    pid = tl.program_id(0)
    kv = tl.program_id(1)
    if pid >= B * S or kv >= num_kv_heads:
        return
    b = pid // S
    s = pid % S

    dest = tl.load(dest_ptr + s).to(tl.int32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        src_offs = b * batch_stride_kr + kv * h_stride_kr + s * s_stride_kr + idx * d_stride_kr
        src_vals = tl.load(key_rot_ptr + src_offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(key_cache_ptr + b * batch_stride_kc + kv * h_stride_kc + dest * dest_stride + idx * d_stride_kc, src_vals, mask=mask)

        src_offs_val = b * batch_stride_v + kv * h_stride_v + s * s_stride_v + idx * d_stride_v
        src_vals_val = tl.load(value_ptr + src_offs_val, mask=mask, other=0.0).to(tl.float32)
        tl.store(value_cache_ptr + b * batch_stride_vc + kv * h_stride_vc + dest * dest_stride_vc + idx * d_stride_vc, src_vals_val, mask=mask)


# -----------------------
# ModelNew.forward
# -----------------------
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # BLOCK_H should be a power of two up to 128; here we use 64, which covers head_dim=64 or 128.
        self.BLOCK_H = 64

    def forward(self, query, key, value,
                position_ids,  # [B, S] int64
                key_cache, value_cache,  # [B, num_kv_heads, max_len, H]
                cache_position,  # [S] int64
                q_norm_weight, k_norm_weight,  # [H], bfloat16
                inv_freq,  # [H//2], float32
                rms_norm_eps):
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # 1) Compute per-token sum of squares for RMSNorm (query)
        sum_sums_q = torch.empty(B * num_q_heads * S, dtype=torch.float32, device=query.device)
        grid_rms = (B * num_q_heads * S,)
        rms_sum_kernel[grid_rms](query, sum_sums_q,
                                 B=B, num_q_heads=num_q_heads, S=S,
                                 H=H,
                                 batch_stride=query.stride(0), h_stride=query.stride(1), s_stride=query.stride(2),
                                 BLOCK_H=self.BLOCK_H)

        # 2) Normalize query using q_norm_weight
        query_norm = torch.empty_like(query, dtype=torch.float32)  # compute in fp32
        rms_norm_kernel[grid_rms](query, q_norm_weight, query_norm,
                                  sum_sums_q,
                                  B=B, num_q_heads=num_q_heads, S=S,
                                  H=H,
                                  batch_stride_x=query.stride(0), h_stride_x=query.stride(1), s_stride_x=query.stride(2),
                                  batch_stride_w=q_norm_weight.stride(0),
                                  batch_stride_out=query_norm.stride(0), h_stride_out=query_norm.stride(1), s_stride_out=query_norm.stride(2),
                                  eps=rms_norm_eps, BLOCK_H=self.BLOCK_H)

        # 3) Precompute cos/sin for rotation per (b, s)
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        grid_rc = (B, S)
        rotate_sin_cos_kernel_b_s[grid_rc](position_ids, inv_freq, cos, sin,
                                           B=B, S=S, H=H,
                                           batch_stride_pos=position_ids.stride(0), s_stride_pos=position_ids.stride(1),
                                           cos_batch_stride=cos.stride(0), cos_h_stride=cos.stride(1), cos_s_stride=cos.stride(2),
                                           sin_batch_stride=sin.stride(0), sin_h_stride=sin.stride(1), sin_s_stride=sin.stride(2),
                                           BLOCK_H=self.BLOCK_H)

        # 4) Apply rotation to query
        query_rot = torch.empty_like(query_norm, dtype=torch.float32)  # output rotated query
        apply_rotation_kernel_4d[grid_rms](query_norm, cos, sin, query_rot,
                                           B=B, num_q_heads=num_q_heads, S=S, H=H,
                                           batch_stride_x=query_norm.stride(0), h_stride_x=query_norm.stride(1), s_stride_x=query_norm.stride(2), d_stride_x=H,  # dummy; kernel handles slicing
                                           cos_batch_stride=cos.stride(0), cos_h_stride=cos.stride(1), cos_s_stride=cos.stride(2), cos_d_stride=cos.stride(2),
                                           sin_batch_stride=sin.stride(0), sin_h_stride=sin.stride(1), sin_s_stride=sin.stride(2), sin_d_stride=sin.stride(2),
                                           batch_stride_out=query_rot.stride(0), h_stride_out=query_rot.stride(1), s_stride_out=query_rot.stride(2), d_stride_out=H,
                                           BLOCK_H=self.BLOCK_H)

        # 5) Compute per-token sum of squares for RMSNorm (key)
        sum_sums_k = torch.empty(B * num_q_heads * S, dtype=torch.float32, device=key.device)
        rms_sum_kernel[grid_rms](key, sum_sums_k,
                                 B=B, num_q_heads=num_q_heads, S=S,
                                 H=H,
                                 batch_stride=key.stride(0), h_stride=key.stride(1), s_stride=key.stride(2),
                                 BLOCK_H=self.BLOCK_H)

        # 6) Normalize key using k_norm_weight
        key_norm = torch.empty_like(key, dtype=torch.float32)  # compute in fp32
        rms_norm_kernel[grid_rms](key, k_norm_weight, key_norm,
                                  sum_sums_k,
                                  B=B, num_q_heads=num_q_heads, S=S,
                                  H=H,
                                  batch_stride_x=key.stride(0), h_stride_x=key.stride(1), s_stride_x=key.stride(2),
                                  batch_stride_w=k_norm_weight.stride(0),
                                  batch_stride_out=key_norm.stride(0), h_stride_out=key_norm.stride(1), s_stride_out=key_norm.stride(2),
                                  eps=rms_norm_eps, BLOCK_H=self.BLOCK_H)

        # 7) Apply rotation to key
        key_rot = torch.empty_like(key_norm, dtype=torch.float32)  # output rotated key
        # Apply rotation kernel over (B*num_q_heads*S). Note: We need per-(b,h,s) program. We already used grid_rms; redefine grid as (B*S, num_q_heads) and inside use hs.
        total = B * num_q_heads * S
        grid_apply = (total,)
        apply_rotation_kernel_4d[grid_apply](key_norm, cos, sin, key_rot,
                                             B=B, num_q_heads=num_q_heads, S=S, H=H,
                                             batch_stride_x=key_norm.stride(0), h_stride_x=key_norm.stride(1), s_stride_x=key_norm.stride(2), d_stride_x=H,
                                             cos_batch_stride=cos.stride(0), cos_h_stride=cos.stride(1), cos_s_stride=cos.stride(2), cos_d_stride=cos.stride(2),
                                             sin_batch_stride=sin.stride(0), sin_h_stride=sin.stride(1), sin_s_stride=sin.stride(2), sin_d_stride=sin.stride(2),
                                             batch_stride_out=key_rot.stride(0), h_stride_out=key_rot.stride(1), s_stride_out=key_rot.stride(2), d_stride_out=H,
                                             BLOCK_H=self.BLOCK_H)

        # 8) Update caches: write rotated keys at cache_position and values at same positions
        grid_update = (B * S, num_kv_heads)
        update_cache_kernel[grid_update](key_rot, value,
                                         key_cache, value_cache,
                                         cache_position,
                                         B=B, num_kv_heads=num_kv_heads, S=S,
                                         H=H,
                                         batch_stride_kr=key_rot.stride(0), h_stride_kr=key_rot.stride(1), s_stride_kr=key_rot.stride(2), d_stride_kr=H,
                                         batch_stride_v=value.stride(0), h_stride_v=value.stride(1), s_stride_v=value.stride(2), d_stride_v=H,
                                         batch_stride_kc=key_cache.stride(0), h_stride_kc=key_cache.stride(1), dest_stride=1, d_stride_kc=key_cache.stride(3),
                                         batch_stride_vc=value_cache.stride(0), h_stride_vc=value_cache.stride(1), dest_stride_vc=1, d_stride_vc=value_cache.stride(3),
                                         BLOCK_H=self.BLOCK_H)

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
