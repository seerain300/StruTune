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
    head = (pid % (num_heads * S)) // S
    s = pid % S
    sum_val = 0.0
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs = b * x_stride0 + head * x_stride1 + s * x_stride2 + idx
        x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sum_sums_ptr + pid * sum_stride0, sum_val)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, y_ptr, sum_sums_ptr,
                     B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                     x_stride0, x_stride1, x_stride2,
                     y_stride0, y_stride1, y_stride2,
                     w_stride0, sum_stride0,
                     eps: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_val = tl.load(sum_sums_ptr + pid * sum_stride0).to(tl.float32)
    H_f = H.to(tl.float32)
    inv_rms = 1.0 / tl.sqrt(sum_val / H_f + eps)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_in = b * x_stride0 + head * x_stride1 + s * x_stride2 + idx
        x_vals = tl.load(x_ptr + offs_in, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y_vals = x_vals * inv_rms * w_vals
        offs_out = b * y_stride0 + head * y_stride1 + s * y_stride2 + idx
        tl.store(y_ptr + offs_out, y_vals, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                              B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                              position_ids_stride0,
                              cos_stride0, cos_stride1, cos_stride2,
                              sin_stride0, sin_stride1, sin_stride2,
                              BLOCK_H: tl.constexpr):
    # One program per (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)
    pos = tl.load(position_ids_ptr + b * position_ids_stride0 + s).to(tl.float32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        half = H // 2
        # first half uses inv_freq directly
        first = idx < half
        inv_freq_vals = tl.load(inv_freq_ptr + idx, mask=first, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals
        # second half duplicates first half
        emb = tl.where(first, emb_first, emb_first)

        c = tl.cos(emb)
        s_ = tl.sin(emb)

        base = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr,
                           B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                           x_stride0, x_stride1, x_stride2,
                           y_stride0, y_stride1, y_stride2,
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

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        x_offs = b * x_stride0 + head * x_stride1 + s * x_stride2 + idx
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        cos_offs = b * cos_stride0 + s * cos_stride1 + idx * cos_stride2
        sin_offs = b * sin_stride0 + s * sin_stride1 + idx * sin_stride2
        cos_vals = tl.load(cos_ptr + cos_offs, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + sin_offs, mask=mask, other=0.0).to(tl.float32)

        half = H // 2
        # Build rotated according to idx ranges
        rot_first = idx < half
        rot_second = idx >= half

        # rotated_first = -x[half + idx] for idx < half
        rotated_first = tl.load(x_ptr + (b * x_stride0 + head * x_stride1 + s * x_stride2 + (half + idx)), mask=rot_first, other=0.0).to(tl.float32) * (-1.0)
        # rotated_second = x[idx - half] for idx >= half
        rotated_second = tl.load(x_ptr + (b * x_stride0 + head * x_stride1 + s * x_stride2 + (idx - half)), mask=rot_second, other=0.0).to(tl.float32)

        rotated = tl.where(rot_first, rotated_first, rotated_second)

        y_vals = x_vals * cos_vals + rotated * sin_vals
        y_offs = b * y_stride0 + head * y_stride1 + s * y_stride2 + idx
        tl.store(y_ptr + y_offs, y_vals, mask=mask)


@triton.jit
def update_cache_kernel(query_rot_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
                         B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr,
                         query_rot_stride0, query_rot_stride1, query_rot_stride2,
                         value_stride0, value_stride1, value_stride2,
                         key_cache_stride0, key_cache_stride1, key_cache_stride2,
                         value_cache_stride0, value_cache_stride1, value_cache_stride2,
                         cache_pos_ptr,
                         max_len_stride,  # stride along max_position_embeddings dim
                         BLOCK_H: tl.constexpr):
    # One program per (b, kv_head, s)
    pid = tl.program_id(0)
    if pid >= B * num_kv_heads * S:
        return
    b = pid // (num_kv_heads * S)
    kv = (pid % (num_kv_heads * S)) // S
    s = pid % S

    dest_pos = tl.load(cache_pos_ptr + s).to(tl.int32)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # write query_rot into key_cache at position dest_pos
        q_offs_in = b * query_rot_stride0 + kv * query_rot_stride1 + s * query_rot_stride2 + idx
        q_vals = tl.load(query_rot_ptr + q_offs_in, mask=mask, other=0.0).to(tl.float32)

        k_offs = b * key_cache_stride0 + kv * key_cache_stride1 + dest_pos * max_len_stride + idx
        tl.store(key_cache_ptr + k_offs, q_vals, mask=mask)

        # write value into value_cache at position dest_pos
        v_offs_in = b * value_stride0 + kv * value_stride1 + s * value_stride2 + idx
        v_vals = tl.load(value_ptr + v_offs_in, mask=mask, other=0.0).to(tl.float32)

        vc_offs = b * value_cache_stride0 + kv * value_cache_stride1 + dest_pos * max_len_stride + idx
        tl.store(value_cache_ptr + vc_offs, v_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size: int, seq_len: int, cache_len: int,
                 num_attention_heads: int, num_key_value_heads: int,
                 head_dim: int, max_position_embeddings: int, rope_theta: float, rms_norm_eps: float):
        super().__init__()
        self.num_q_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_len = max_position_embeddings
        self.rope_theta = rope_theta
        self.eps = rms_norm_eps
        # We'll compute inv_freq on-the-fly; BLOCK_H chosen for typical head_dim=128
        self.BLOCK_H = 128

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                position_ids: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor, inv_freq: torch.Tensor):
        B = query.shape[0]
        num_q_heads = query.shape[1]
        S = query.shape[2]
        H = query.shape[3]
        num_kv_heads = key.shape[1]

        # 1) RMSNorm for query -> query_norm
        sum_query = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        rms_sum_kernel[(B * num_q_heads * S,)](
            query, sum_query,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            sum_query.stride(0),
            BLOCK_H=self.BLOCK_H, num_warps=4
        )
        query_norm = torch.empty_like(query, dtype=torch.float32)
        rms_norm_kernel[(B * num_q_heads * S,)](
            query, q_norm_weight, query_norm, sum_query,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            k_norm_weight.stride(0), sum_query.stride(0),
            self.eps, BLOCK_H=self.BLOCK_H, num_warps=4
        )
        query_norm = query_norm.to(torch.bfloat16)

        # 2) Build cos/sin for rotation (float32), shape [B, S, H]
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        rotate_sin_cos_kernel_b_s[(B, S)](
            position_ids, inv_freq, cos, sin,
            B, S, H,
            position_ids.stride(0),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=self.BLOCK_H, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rot (bfloat16 output)
        query_rot = torch.empty((B, num_q_heads, S, H), dtype=torch.bfloat16, device=query.device)
        apply_rotation_kernel[(B * num_q_heads * S,)](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=self.BLOCK_H, num_warps=4
        )

        # 4) RMSNorm for key -> key_norm (compute in float32, output bfloat16)
        sum_key = torch.empty((B * num_kv_heads * S,), dtype=torch.float32, device=key.device)
        rms_sum_kernel[(B * num_kv_heads * S,)](
            key, sum_key,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            sum_key.stride(0),
            BLOCK_H=self.BLOCK_H, num_warps=4
        )
        key_norm = torch.empty((B, num_kv_heads, S, H), dtype=torch.float32, device=key.device)
        rms_norm_kernel[(B * num_kv_heads * S,)](
            key, k_norm_weight, key_norm, sum_key,
            B, num_kv_heads, S, H,
            key.stride(0), key.stride(1), key.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            k_norm_weight.stride(0), sum_key.stride(0),
            self.eps, BLOCK_H=self.BLOCK_H, num_warps=4
        )
        key_norm = key_norm.to(torch.bfloat16)

        # 5) Update caches using cache_position (int64 vector length S)
        # Note: original code updates key_cache with rotated_query; value_cache with value.
        # We launch the Triton kernel to perform this.
        update_cache_kernel[(B * num_kv_heads * S,)](
            query_rot, value, key_cache, value_cache,
            B, num_kv_heads, S,
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            value.stride(0), value.stride(1), value.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            cache_position,
            key_cache.stride(2),  # max_len_stride along last dim
            BLOCK_H=self.BLOCK_H, num_warps=4
        )

        return query_rot, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
