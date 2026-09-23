import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,         # *const T (input), shape [B, H, S, D]
    w_ptr,         # *const T (weight), shape [D]
    y_ptr,         # *T (output), shape [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    y_stride0, y_stride1, y_stride2, y_stride3,
    w_stride0,                          # w is 1D, stride0
    BLOCK_SIZE: tl.constexpr,
):
    # program ids: (b, h, s)
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # base pointers for this (b, h, s) row
    x_row_ptr = x_ptr + b * x_stride0 + h * x_stride1 + s * x_stride2
    y_row_ptr = y_ptr + b * y_stride0 + h * y_stride1 + s * y_stride2

    # compute sum of squares across D in fp32
    sumsq = 0.0
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * x_stride3, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)
        sumsq += tl.sum(x_vals * x_vals, axis=0)
    mean = sumsq / D
    scale = 1.0 / tl.sqrt(mean + 1e-6)

    # apply weight and store
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * x_stride3, mask=mask, other=0.0)
        x_vals = x_vals.to(tl.float32)
        w_vals = tl.load(w_ptr + idx * w_stride0, mask=mask, other=1.0).to(tl.float32)
        y_vals = x_vals * scale * w_vals
        tl.store(y_row_ptr + idx * y_stride3, y_vals, mask=mask)


@triton.jit
def rotate_query_kernel(
    x_ptr,         # *const float32 (normalized query), shape [B, H, S, D]
    cos_ptr,       # *const float32, shape [B, S, D] (cos_all)
    sin_ptr,       # *const float32, shape [B, S, D] (sin_all)
    y_ptr,         # *bf16 (output rotated query), shape [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    y_stride0, y_stride1, y_stride2, y_stride3,
    cos_stride0, cos_stride1, cos_stride2,  # cos/sin are [B, S, D]
    sin_stride0, sin_stride1, sin_stride2,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    x_row_ptr = x_ptr + b * x_stride0 + h * x_stride1 + s * x_stride2
    y_row_ptr = y_ptr + b * y_stride0 + h * y_stride1 + s * y_stride2

    # Load cos_all and sin_all for this (b, s)
    cos_base = cos_ptr + b * cos_stride0 + s * cos_stride1
    sin_base = sin_ptr + b * sin_stride0 + s * sin_stride1
    d = tl.arange(0, D)
    cos_all = tl.load(cos_base + d * cos_stride2)  # [D] float32
    sin_all = tl.load(sin_base + d * sin_stride2)  # [D] float32

    # Compute rotated output across D in blocks
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * x_stride3, mask=mask, other=0.0).to(tl.float32)
        half = D // 2
        x_first = x_vals[:half]
        x_second = x_vals[half:]
        rotate = -x_second + x_first  # rotate_half(x)
        y_vals = x_vals * cos_all - rotate * sin_all
        # store as bf16
        tl.store(y_row_ptr + idx * y_stride3, y_vals, mask=mask)  # Triton will convert to y dtype (bf16)


@triton.jit
def rotate_key_kernel(
    x_ptr,         # *const float32 (normalized key), shape [B, H, S, D]
    sin_ptr,       # *const float32, shape [B, S, D] (sin_all)
    cos_ptr,       # *const float32, shape [B, S, D] (cos_all)
    y_ptr,         # *bf16 (output rotated key), shape [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    y_stride0, y_stride1, y_stride2, y_stride3,
    sin_stride0, sin_stride1, sin_stride2,  # sin is [B, S, D]
    cos_stride0, cos_stride1, cos_stride2,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    x_row_ptr = x_ptr + b * x_stride0 + h * x_stride1 + s * x_stride2
    y_row_ptr = y_ptr + b * y_stride0 + h * y_stride1 + s * y_stride2

    # Load sin_all and cos_all for this (b, s)
    sin_base = sin_ptr + b * sin_stride0 + s * sin_stride1
    cos_base = cos_ptr + b * cos_stride0 + s * cos_stride1
    d = tl.arange(0, D)
    sin_all = tl.load(sin_base + d * sin_stride2)  # [D] float32
    cos_all = tl.load(cos_base + d * cos_stride2)  # [D] float32

    # Compute rotated output across D in blocks
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * x_stride3, mask=mask, other=0.0).to(tl.float32)
        half = D // 2
        x_first = x_vals[:half]
        x_second = x_vals[half:]
        rotate = -x_second + x_first  # rotate_half(x)
        y_vals = x_vals * sin_all - rotate * cos_all
        tl.store(y_row_ptr + idx * y_stride3, y_vals, mask=mask)  # Triton will convert to y dtype (bf16)


@triton.jit
def scatter_update_kernel(
    src_ptr,       # *const bf16 (rotated key or original value), shape [B, H, S, D]
    dst_ptr,       # *bf16 (key_cache or value_cache), shape [B, H, L, D]
    cache_pos_ptr, # *const int32, shape [S]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
    src_stride0, src_stride1, src_stride2, src_stride3,
    dst_stride0, dst_stride1, dst_stride2, dst_stride3,
    BLOCK_SIZE: tl.constexpr,
):
    # We launch with grid = (B, H, S). Each program copies one src row to dst at cache_pos[s].
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(cache_pos_ptr + s).to(tl.int32)
    # Guard: if pos >= L, skip (not used in given workloads, but safe)
    if pos >= L:
        return

    src_row_ptr = src_ptr + b * src_stride0 + h * src_stride1 + s * src_stride2
    dst_row_ptr = dst_ptr + b * dst_stride0 + h * dst_stride1 + pos * dst_stride2

    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        src_vals = tl.load(src_row_ptr + idx * src_stride3, mask=mask, other=0.0)
        # src_vals are bf16; store as bf16 into dst
        tl.store(dst_row_ptr + idx * dst_stride3, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_q_heads: int = 96, num_kv_heads: int = 8, head_dim: int = 128):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(self, *args):
        # Args from get_inputs: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        assert len(args) == 11, "Expected 11 inputs"
        query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps = args

        B = query.shape[0]
        H_q = query.shape[1]
        S = query.shape[2]
        D = query.shape[3]
        assert H_q == self.num_q_heads, f"num_q_heads mismatch: expected {self.num_q_heads}, got {H_q}"
        H_kv = key.shape[1]
        assert H_kv == self.num_kv_heads, f"num_kv_heads mismatch: expected {self.num_kv_heads}, got {H_kv}"
        assert D == self.head_dim, f"head_dim mismatch: expected {self.head_dim}, got {D}"

        # Ensure tensors are on same device
        device = query.device

        # RMSNorm: output normalized and scaled tensors
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)
        rmsnorm_kernel[(B, H_q, S)](
            query, q_norm_weight.to(torch.bfloat16), query_norm,
            B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            q_norm_weight.stride(0),
            BLOCK_SIZE=128,
        )
        rmsnorm_kernel[(B, H_kv, S)](
            key, k_norm_weight.to(torch.bfloat16), key_norm,
            B, H_kv, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            k_norm_weight.stride(0),
            BLOCK_SIZE=128,
        )

        # Build cos_all and sin_all for each (b, s): angle = pos * inv_freq, D//2 = 64, concatenate cos/sin to D
        # Move position_ids to float32
        pos_ids = position_ids.to(torch.float32)  # [B, S]
        # inv_freq: [D//2] float32
        inv_freq = inv_freq.to(device=device, dtype=torch.float32)
        # angle per (b, s): [D//2]
        angle = (pos_ids[:, :, None] * inv_freq[None, None, :]).to(torch.float32)  # [B, S, D//2]
        cos_half = torch.cos(angle)  # [B, S, D//2]
        sin_half = torch.sin(angle)  # [B, S, D//2]
        # Concatenate with itself to form cos_all and sin_all: [B, S, D]
        cos_all = torch.cat([cos_half, cos_half], dim=-1)  # [B, S, D]
        sin_all = torch.cat([sin_half, sin_half], dim=-1)  # [B, S, D]

        # Rotate query and key (store as bf16 like inputs)
        query_rot = torch.empty_like(query)  # bf16 output
        rotate_query_kernel[(B, H_q, S)](
            query_norm.to(torch.float32), cos_all, sin_all, query_rot,
            B, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            cos_all.stride(0), cos_all.stride(1), cos_all.stride(2),
            sin_all.stride(0), sin_all.stride(1), sin_all.stride(2),
            BLOCK_SIZE=128,
        )
        key_rot = torch.empty_like(key)  # bf16 output
        rotate_key_kernel[(B, H_kv, S)](
            key_norm.to(torch.float32), sin_all, cos_all, key_rot,
            B, H_kv, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            sin_all.stride(0), sin_all.stride(1), sin_all.stride(2),
            cos_all.stride(0), cos_all.stride(1), cos_all.stride(2),
            BLOCK_SIZE=128,
        )

        # Scatter update caches: write rotated keys and original values at cache_position
        # Convert cache_position to int32 for Triton
        cache_pos = cache_position.to(torch.int32)  # [S]
        L = key_cache.shape[2]
        scatter_update_kernel[(B, H_kv, S)](
            key_rot.to(torch.bfloat16), key_cache, cache_pos,
            B, H_kv, S, D, L,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            BLOCK_SIZE=128,
        )
        scatter_update_kernel[(B, H_kv, S)](
            value.to(torch.bfloat16), value_cache, cache_pos,
            B, H_kv, S, D, L,
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            BLOCK_SIZE=128,
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
