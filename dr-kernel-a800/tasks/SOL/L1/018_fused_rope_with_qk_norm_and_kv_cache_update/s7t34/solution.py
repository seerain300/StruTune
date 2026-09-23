import torch
import triton
import triton.language as tl

# Triton RMSNorm: per (b, h, s) row across D
@triton.jit
def rmsnorm_kernel(
    x_ptr, out_ptr, weight_ptr,
    B, H, S, D,
    x_stride0, x_stride1, x_stride2, x_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = b * x_stride0 + h * x_stride1 + s * x_stride2
    base_out = b * out_stride0 + h * out_stride1 + s * out_stride2

    sum_sq = 0.0
    for offs in range(0, D, BLOCK_SIZE):
        d_idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = d_idx < D
        x = tl.load(x_ptr + base_x + d_idx * x_stride3, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sum_sq += tl.sum(x32 * x32)

    mean_sq = sum_sq / D
    inv_rms = tl.rsqrt(mean_sq + eps)

    for offs in range(0, D, BLOCK_SIZE):
        d_idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = d_idx < D
        x = tl.load(x_ptr + base_x + d_idx * x_stride3, mask=mask, other=0.0)
        w = tl.load(weight_ptr + d_idx, mask=mask, other=1.0).to(tl.float32)
        y = (x.to(tl.float32) * inv_rms) * w
        tl.store(out_ptr + base_out + d_idx * out_stride3, y.to(x.dtype), mask=mask)


# Triton query rotation: y = x * cos - rotate_half(x) * sin
@triton.jit
def rotate_query_kernel(
    x_ptr, out_ptr, cos_ptr, sin_ptr,
    B, H_q, S, D,
    x_stride0, x_stride1, x_stride2, x_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = b * x_stride0 + h * x_stride1 + s * x_stride2
    base_out = b * out_stride0 + h * out_stride1 + s * out_stride2

    # Load cos_all and sin_all vectors (length D)
    cos_all = tl.load(cos_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)
    sin_all = tl.load(sin_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)

    for offs in range(0, D, BLOCK_SIZE):
        d_idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = d_idx < D
        x = tl.load(x_ptr + base_x + d_idx * x_stride3, mask=mask, other=0.0).to(tl.float32)

        half = D // 2
        x1 = x[:half]
        x2 = x[half:]
        xr = tl.concatenate([-x2, x1], axis=0)  # shape [D]

        y = x * cos_all - xr * sin_all
        tl.store(out_ptr + base_out + d_idx * out_stride3, y.to(x_ptr.dtype.element_ty), mask=mask)


# Triton key rotation: y = x * sin - rotate_half(x) * cos
@triton.jit
def rotate_key_kernel(
    x_ptr, out_ptr, sin_ptr, cos_ptr,
    B, H_kv, S, D,
    x_stride0, x_stride1, x_stride2, x_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = b * x_stride0 + h * x_stride1 + s * x_stride2
    base_out = b * out_stride0 + h * out_stride1 + s * out_stride2

    sin_all = tl.load(sin_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)
    cos_all = tl.load(cos_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)

    for offs in range(0, D, BLOCK_SIZE):
        d_idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = d_idx < D
        x = tl.load(x_ptr + base_x + d_idx * x_stride3, mask=mask, other=0.0).to(tl.float32)

        half = D // 2
        x1 = x[:half]
        x2 = x[half:]
        xr = tl.concatenate([-x2, x1], axis=0)  # shape [D]

        y = x * sin_all - xr * cos_all
        tl.store(out_ptr + base_out + d_idx * out_stride3, y.to(x_ptr.dtype.element_ty), mask=mask)


# Triton scatter update: write src row to dst[b, h, pos, :]
@triton.jit
def scatter_update_kernel(
    src_ptr, dst_ptr,
    B, H, S, D, L,
    src_stride0, src_stride1, src_stride2, src_stride3,
    dst_stride0, dst_stride1, dst_stride2, dst_stride3,
    cache_pos_ptr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    pos = tl.load(cache_pos_ptr + s).to(tl.int32)

    base_src = b * src_stride0 + h * src_stride1 + s * src_stride2
    base_dst = b * dst_stride0 + h * dst_stride1 + pos * dst_stride2

    for d in range(0, D):
        val = tl.load(src_ptr + base_src + d * src_stride3)
        tl.store(dst_ptr + base_dst + d * dst_stride3, val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_q_heads: int = 96, num_kv_heads: int = 8, head_dim: int = 128, cache_len: int = 0):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.cache_len = cache_len
        self.eps = 1e-6

    def forward(self, *args):
        # Expect: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        assert len(args) == 11, "Expected 11 inputs"
        query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps = args

        # Shapes and asserts
        B = query.shape[0]
        H_q = query.shape[1]
        S = query.shape[2]
        D = query.shape[3]
        assert H_q == self.num_q_heads, f"num_q_heads mismatch: expected {self.num_q_heads}, got {H_q}"
        H_kv = key.shape[1]
        assert H_kv == self.num_kv_heads, f"num_kv_heads mismatch: expected {self.num_kv_heads}, got {H_kv}"
        assert D == self.head_dim, f"head_dim mismatch: expected {self.head_dim}, got {D}"

        # Move inv_freq to device (float32)
        inv_freq = inv_freq.to(query.device).to(torch.float32)

        # Precompute cos_all and sin_all using torch on device: [B, S, D]
        # angle = pos * inv_freq[:D//2]
        pos_ids = position_ids.to(torch.float32)


def run(*args):
    return ModelNew()(*args)
