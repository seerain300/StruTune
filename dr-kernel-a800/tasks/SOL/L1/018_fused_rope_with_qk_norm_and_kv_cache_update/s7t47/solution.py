import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,         # *bf16, [B, H, S, D]
    w_ptr,         # *bf16, [D]
    y_ptr,         # *bf16, [B, H, S, D]
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Compute sum of squares across D
    sumsq = 0.0
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        sumsq += tl.sum(x_fp32 * x_fp32)

    mean = sumsq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Normalize and scale by weight, store
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y = (x * inv_rms) * w
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + cols * y_stride_d, y, mask=mask)


@triton.jit
def rotate_q_kernel(
    x_ptr,          # *bf16, RMSNormed query, [B, H_q, S, D]
    y_ptr,          # *bf16, output query_rot, [B, H_q, S, D]
    pos_ptr,        # *int64, position_ids, [B, S]
    inv_ptr,        # *float32, inv_freq, [D//2]
    B, H_q, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Load position id for this (b, s)
    pos = tl.load(pos_ptr + b * 0 + s * 0).to(tl.float32)

    # Build cos_all and sin_all of length D:
    # Compute angles for the first half (D//2), then concatenate.
    half = D // 2
    angles = tl.arange(0, half) * 0.0
    for i in tl.static_range(0, half):
        angles[i] = pos * tl.load(inv_ptr + i).to(tl.float32)
    cos_angle = tl.cos(angles)
    sin_angle = tl.sin(angles)
    cos_all = tl.concatenate([cos_angle, cos_angle], axis=0)
    sin_all = tl.concatenate([sin_angle, sin_angle], axis=0)

    # Process columns in chunks
    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        half = D // 2
        x1 = x[:half]
        x2 = x[half:]
        rotate_half = tl.concatenate([-x2, x2, -x1, x1], axis=0)

        y = x * sin_all - rotate_half * cos_all  # query uses sin_all (consistent with original code)
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + cols * y_stride_d, y, mask=mask)


@triton.jit
def rotate_k_kernel(
    x_ptr,          # *bf16, RMSNormed key, [B, H_kv, S, D]
    y_ptr,          # *bf16, output key_rot, [B, H_kv, S, D]
    pos_ptr,        # *int64, position_ids, [B, S]
    inv_ptr,        # *float32, inv_freq, [D//2]
    B, H_kv, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    y_stride_b, y_stride_h, y_stride_s, y_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(pos_ptr + b * 0 + s * 0).to(tl.float32)

    half = D // 2
    angles = tl.arange(0, half) * 0.0
    for i in tl.static_range(0, half):
        angles[i] = pos * tl.load(inv_ptr + i).to(tl.float32)
    # For keys, original code uses sin(angle) in the rotation. We use sin_angle as "cos_all" and cos_angle as "sin_all".
    sin_angle = tl.sin(angles)  # used as "cos_all"
    cos_angle = tl.cos(angles)  # used as "sin_all"

    for off in tl.static_range(0, D, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s + cols * x_stride_d, mask=mask, other=0.0).to(tl.float32)

        half = D // 2
        x1 = x[:half]
        x2 = x[half:]
        rotate_half = tl.concatenate([-x2, x2, -x1, x1], axis=0)

        # y = x * sin(angle) - rotate_half(x) * cos(angle)
        y = x * sin_angle - rotate_half * cos_angle
        tl.store(y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s + cols * y_stride_d, y, mask=mask)


@triton.jit
def scatter_update_cache_kernel(
    src_ptr,        # *bf16, [B, H, S, D]
    dst_ptr,        # *bf16, [B, H, L, D]
    B, H, S, D, L,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_h, dst_stride_l, dst_stride_d,
    cache_pos_ptr,  # *int32, [S]
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    pos = tl.load(cache_pos_ptr + s)

    # Copy src[b, h, s, :] to dst[b, h, pos, :]
    for off in tl.static_range(0, D, 128):
        cols = off + tl.arange(0, 128)
        mask = cols < D
        src_vals = tl.load(src_ptr + b * src_stride_b + h * src_stride_h + s * src_stride_s + cols * src_stride_d, mask=mask, other=0.0)
        tl.store(dst_ptr + b * dst_stride_b + h * dst_stride_h + pos * dst_stride_l + cols * dst_stride_d, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants aligned with the original code
        self.head_dim = 128
        self.num_q_heads = 96
        self.num_kv_heads = 8
        self.max_position_embeddings = 262144
        self.rope_theta = 10000000.0
        self.rms_norm_eps = 1e-6

    def forward(self, *args):
        # We expect the same 11 arguments as the original run:
        # 0=query, 1=key, 2=value, 3=position_ids, 4=key_cache, 5=value_cache,
        # 6=cache_position, 7=q_norm_weight, 8=k_norm_weight, 9=inv_freq, 10=rms_norm_eps
        assert len(args) == 11, "ModelNew.forward expects 11 arguments"
        query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps = args

        B_q = query.shape[0]
        H_q = query.shape[1]
        S = query.shape[2]
        D = query.shape[3]
        assert key.shape[0] == B_q and key.shape[3] == D
        assert value.shape[0] == B_q and value.shape[3] == D
        assert position_ids.shape[0] == B_q and position_ids.shape[1] == S
        assert key_cache.shape[0] == B_q and key_cache.shape[2] == self.max_position_embeddings and key_cache.shape[3] == D
        assert value_cache.shape[0] == B_q and value_cache.shape[2] == self.max_position_embeddings and value_cache.shape[3] == D
        assert cache_position.shape[0] == S
        assert q_norm_weight.shape[0] == D and k_norm_weight.shape[0] == D
        assert inv_freq.shape[0] == D // 2

        # Ensure contiguous memory
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

        # 1) RMSNorm on query and key (fp32 compute, bf16 store)
        query_norm = torch.empty


def run(*args):
    return ModelNew()(*args)
