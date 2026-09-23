import torch
import triton
import triton.language as tl

# Triton: RMSNorm over last dim D for each [b, h, s] row
@triton.jit
def rmsnorm_kernel(
    X_ptr, W_ptr, Out_ptr,
    B, H, S, D,
    stride_b, stride_h, stride_s, stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_per_batch = H * S
    b = pid // rows_per_batch
    rem = pid % rows_per_batch
    h = rem // S
    s = rem % S

    base = b * stride_b + h * stride_h + s * stride_s

    sumsq = 0.0
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + base + offs * stride_d, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)

    mean = sumsq / D
    inv_rms = tl.math.rsqrt(mean + eps)

    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + base + offs * stride_d, mask=mask, other=0.0)
        x = x.to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(Out_ptr + (b * out_stride_b + h * out_stride_h + s * out_stride_s + offs * out_stride_d),
                 y.to(tl.bfloat16), mask=mask)


# Triton: generate cos_all and sin_all vectors for query rotation
# Inputs:
#   PositionIds: [B, S] int64
#   InvFreq: [F=64] float32
#   OutCos: [B, S, D] float32
#   OutSin: [B, S, D] float32
# Each program computes one token (b, s), then fills D=128 entries:
#   angles = pos * inv_freq[:D//2], cos_all = [cos(angles), cos(angles)], sin_all = [sin(angles), sin(angles)]
@triton.jit
def gen_cos_sin_kernel(
    PositionIds_ptr, InvFreq_ptr, OutCos_ptr, OutSin_ptr,
    B, S,
    stride_b_p, stride_s_p,
    stride_b_c, stride_s_c, stride_d_c,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    pos = tl.load(PositionIds_ptr + b * stride_b_p + s * stride_s_p).to(tl.int32)

    # angles for first half: [0..63]
    for i in range(0, 64):
        angle = pos * tl.load(InvFreq_ptr + i).to(tl.float32)
        cos_val = tl.math.cos(angle).to(tl.float32)
        sin_val = tl.math.sin(angle).to(tl.float32)
        # store into cos_all and sin_all for both halves
        tl.store(OutCos_ptr + b * stride_b_c + s * stride_s_c + i * 2, cos_val)
        tl.store(OutCos_ptr + b * stride_b_c + s * stride_s_c + i * 2 + 1, cos_val)
        tl.store(OutSin_ptr + b * stride_b_c + s * stride_s_c + i * 2, sin_val)
        tl.store(OutSin_ptr + b * stride_b_c + s * stride_s_c + i * 2 + 1, sin_val)


# Triton: rotate query
# y = x * cos - rotate_half(x) * sin
@triton.jit
def rotate_q_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
    B, H_q, S, D,
    stride_b_x, stride_h_x, stride_s_x, stride_d_x,
    stride_b_c, stride_s_c, stride_d_c,
    stride_b_y, stride_h_y, stride_s_y, stride_d_y,
):
    pid = tl.program_id(0)
    rows_per_batch = H_q * S
    b = pid // rows_per_batch
    rem = pid % rows_per_batch
    h = rem // S
    s = rem % S

    base_x = b * stride_b_x + h * stride_h_x + s * stride_s_x
    base_y = b * stride_b_y + h * stride_h_y + s * stride_s_y
    base_c = b * stride_b_c + s * stride_s_c

    for d in range(0, D, 128):
        offs = d + tl.arange(0, 128)
        mask = offs < D
        x = tl.load(X_ptr + base_x + offs * stride_d_x, mask=mask, other=0.0).to(tl.float32)
        cos_vec = tl.load(Cos_ptr + base_c + offs * stride_d_c, mask=mask, other=1.0).to(tl.float32)
        sin_vec = tl.load(Sin_ptr + base_c + offs * stride_d_c, mask=mask, other=1.0).to(tl.float32)

        x1 = x[:128 // 2]
        x2 = x[128 // 2:]
        rot = tl.concatenate([-x2, x1], axis=0)

        y = x * cos_vec - rot * sin_vec
        tl.store(Y_ptr + base_y + offs * stride_d_y, y.to(tl.bfloat16), mask=mask)


# Triton: rotate key
# y = x * sin - rotate_half(x) * cos
@triton.jit
def rotate_k_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
    B, H_kv, S, D,
    stride_b_x, stride_h_x, stride_s_x, stride_d_x,
    stride_b_c, stride_s_c, stride_d_c,
    stride_b_y, stride_h_y, stride_s_y, stride_d_y,
):
    pid = tl.program_id(0)
    rows_per_batch = H_kv * S
    b = pid // rows_per_batch
    rem = pid % rows_per_batch
    h = rem // S
    s = rem % S

    base_x = b * stride_b_x + h * stride_h_x + s * stride_s_x
    base_y = b * stride_b_y + h * stride_h_y + s * stride_s_y
    base_c = b * stride_b_c + s * stride_s_c

    for d in range(0, D, 128):
        offs = d + tl.arange(0, 128)
        mask = offs < D
        x = tl.load(X_ptr + base_x + offs * stride_d_x, mask=mask, other=0.0).to(tl.float32)
        sin_vec = tl.load(Cos_ptr + base_c + offs * stride_d_c, mask=mask, other=1.0).to(tl.float32)  # sin_all
        cos_vec = tl.load(Sin_ptr + base_c + offs * stride_d_c, mask=mask, other=1.0).to(tl.float32)  # cos_all

        x1 = x[:128 // 2]
        x2 = x[128 // 2:]
        rot = tl.concatenate([-x2, x1], axis=0)

        y = x * sin_vec - rot * cos_vec
        tl.store(Y_ptr + base_y + offs * stride_d_y, y.to(tl.bfloat16), mask=mask)


# Triton: scatter update caches
# Y: rotated key [B, H_kv, S, D]
# Value: original value [B, H_kv, S, D]
# Cache: [B, H_kv, L, D]
# CachePos: [B, S] int32
@triton.jit
def scatter_cache_kernel(
    Y_ptr, Value_ptr, Cache_ptr, CachePos_ptr,
    B, H, S, D, L,
    stride_b_y, stride_h_y, stride_s_y, stride_d_y,
    stride_b_val, stride_h_val, stride_s_val, stride_d_val,
    stride_b_c, stride_h_c, stride_l_c, stride_d_c,
    stride_b_p, stride_s_p,
):
    pid = tl.program_id(0)
    rows_per_batch = H * S
    b = pid // rows_per_batch
    rem = pid % rows_per_batch
    h = rem // S
    s = rem % S

    pos = tl.load(CachePos_ptr + b * stride_b_p + s * stride_s_p)

    base_y = b * stride_b_y + h * stride_h_y + s * stride_s_y
    base_val = b * stride_b_val + h * stride_h_val + s * stride_s_val
    base_c = b * stride_b_c + h * stride_h_c + pos * stride_l_c

    for d in range(0, D, 128):
        offs = d + tl.arange(0, 128)
        mask = offs < D
        y_vals = tl.load(Y_ptr + base_y + offs * stride_d_y, mask=mask, other=0.0).to(tl.bfloat16)
        val_vals = tl.load(Value_ptr + base_val + offs * stride_d_val, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(Cache_ptr + base_c + offs * stride_d_c, y_vals, mask=mask)
        # value does not require rotation; original value is stored
        tl.store(Cache_ptr + base_c + offs * stride_d_c, val_vals, mask=mask)


# For safety, make sure tensors are contiguous before launching kernels
def _ensure_contiguous(t: torch.Tensor):
    if t.is_contiguous():
        return t
    return t.contiguous()


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        inv_freq: torch.Tensor,
        rms_norm_eps: float,
    ):
        assert query.is_cuda and key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda, "All tensors must be on CUDA device"
        assert position_ids.is_cuda and cache_position.is_cuda, "position_ids and cache_position must be on CUDA device"

        B = query.shape[0]
        H_q = query.shape[1]
        S = query.shape[2]
        D = query.shape[3]
        assert key.shape == (B, 8, S, D)
        assert value.shape == (B, 8, S, D)
        assert key_cache.shape == (B, 8, 262144, D)
        assert value_cache.shape == (B, 8, 262144, D)
        assert q_norm_weight.shape == (D,) and k_norm_weight.shape == (D,)

        device = query.device

        # 1) RMSNorm for query and key
        # Prepare output buffers
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # Launch RMSNorm for query
        grid_q = (B * H_q * S,)
        rmsnorm_kernel[grid_q](
            query, q_norm_weight.to(torch.float32), query_norm,
            B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            rms_norm_eps,
            BLOCK_SIZE=128,
            num_warps=4,
        )

        # Launch RMSNorm for key
        grid_k = (B * 8 * S,)
        rmsnorm_kernel[grid_k](
            key, k_norm_weight.to(torch.float32), key_norm,
            B, 8, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            rms_norm_eps,
            BLOCK_SIZE=128,
            num_warps=4,
        )

        # 2) Generate cos_all and sin_all for query rotation
        # cos_all: [B, S, D] float32, sin_all: [B, S, D] float32
        # We only need angles from inv_freq[:D//2], and then two copies
        cos_all = torch.empty((B, S, D), device=device, dtype=torch.float32)
        sin_all = torch.empty((B, S, D), device=device, dtype=torch.float32)

        # position_ids is [B, S] int64; cast to int32 for kernel
        pos_ids = _ensure_contiguous(position_ids.to(torch.int32))
        grid_gen = (B * S,)
        gen_cos_sin_kernel[grid_gen](
            pos_ids,
            inv_freq.to(torch.float32),
            cos_all,
            sin_all,
            B, S,
            pos_ids.stride(0), pos_ids.stride(1),
            cos_all.stride(0), cos_all.stride(1), cos_all.stride(2),
            num_warps=1,
        )

        # 3) Query rotation: y = x * cos - rotate_half(x) * sin
        query_rot = torch.empty_like(query_norm)  # bf16
        grid_qrot = (B * H_q * S,)
        rotate_q_kernel[grid_qrot](
            query_norm, cos_all, sin_all, query_rot,
            B, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            cos_all.stride(0), cos_all.stride(1), cos_all.stride(2),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            num_warps=4,
        )

        # 4) Key rotation: y = x * sin - rotate_half(x) * cos
        key_rot = torch.empty_like(key_norm)  # bf16
        grid_krot = (B * 8 * S,)
        rotate_k_kernel[grid_krot](
            key_norm, sin_all, cos_all, key_rot,
            B, 8, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            sin_all.stride(0), sin_all.stride(1), sin_all.stride(2),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            num_warps=4,
        )

        # 5) Scatter update caches at cache_position[s]
        # Ensure cache_position is [B, S] int32
        cache_pos = _ensure_contiguous(cache_position.to(torch.int32).unsqueeze(1))  # [B, S]

        # Launch scatter for key rotation and original value
        grid_scatter = (B * 8 * S,)
        scatter_cache_kernel[grid_scatter](
            key_rot, value, key_cache, cache_pos,
            B, 8, S, D, 262144,
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            cache_pos.stride(0), cache_pos.stride(1),
            num_warps=4,
        )

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
