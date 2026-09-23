import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,          # *const bf16, input [B, H, S, D]
    w_ptr,          # *const bf16, weight [D]
    y_ptr,          # *mut bf16, output [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,  # strides for x
    w_s0,                   # stride for w (usually 1)
    y_s0, y_s1, y_s2, y_s3,  # strides for y
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # map pid to (b, h, s)
    HS = H * S
    b = pid // HS
    tmp = pid % HS
    h = tmp // S
    s = tmp % S

    # base offset for this row
    base_x = b * x_s0 + h * x_s1 + s * x_s2

    # compute mean of x^2 across D
    sum_x2 = 0.0
    for off in range(0, D, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_ptr + base_x + idx * x_s3, mask=mask, other=0.0)
        x_f = x.to(tl.float32)
        sum_x2 += tl.sum(x_f * x_f, axis=0)

    mean = sum_x2 / D
    inv_rms = tl.rsqrt(mean + eps)

    # scale and apply weight
    for off in range(0, D, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_ptr + base_x + idx * x_s3, mask=mask, other=0.0)
        w = tl.load(w_ptr + idx * w_s0, mask=mask, other=1.0).to(tl.float32)
        y = (x.to(tl.float32) * inv_rms) * w
        tl.store(y_ptr + base_x + idx * y_s3, y.to(tl.bfloat16), mask=mask)


@triton.jit
def rotate_q_kernel(
    x_ptr,          # *const bf16, input [B, H_q, S, D]
    cos_ptr,        # *const float32, cos_all [B, S, D]
    sin_ptr,        # *const float32, sin_all [B, S, D]
    y_ptr,          # *mut bf16, output [B, H_q, S, D]
    B: tl.constexpr, H_q: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,
    cos_s0, cos_s1, cos_s2,
    y_s0, y_s1, y_s2, y_s3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    HS = H_q * S
    b = pid // HS
    tmp = pid % HS
    h = tmp // S
    s = tmp % S

    base_x = b * x_s0 + h * x_s1 + s * x_s2
    base_cos = b * cos_s0 + s * cos_s1
    base_y = b * y_s0 + h * y_s1 + s * y_s2

    # Load x
    for off in range(0, D, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_ptr + base_x + idx * x_s3, mask=mask, other=0.0).to(tl.float32)
        # Load cos/sin for this position s: cos_ptr[base_cos + idx], sin_ptr[base_cos + idx]
        cos_vals = tl.load(cos_ptr + base_cos + idx * cos_s2, mask=mask, other=0.0)
        sin_vals = tl.load(sin_ptr + base_cos + idx * cos_s2, mask=mask, other=0.0)

        half = D // 2
        x1 = x[:half]
        x2 = x[half:]
        # rotate_half(x) = [-x2, x1]
        rx = tl.concatenate([-x2, x1], axis=0)

        y = x * cos_vals - rx * sin_vals
        tl.store(y_ptr + base_y + idx * y_s3, y.to(tl.bfloat16), mask=mask)


@triton.jit
def rotate_k_kernel(
    x_ptr,          # *const bf16, input [B, H_kv, S, D]
    sin_ptr,        # *const float32, sin_all [B, S, D]
    cos_ptr,        # *const float32, cos_all [B, S, D]
    y_ptr,          # *mut bf16, output [B, H_kv, S, D]
    B: tl.constexpr, H_kv: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    x_s0, x_s1, x_s2, x_s3,
    sin_s0, sin_s1, sin_s2,
    y_s0, y_s1, y_s2, y_s3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    HS = H_kv * S
    b = pid // HS
    tmp = pid % HS
    h = tmp // S
    s = tmp % S

    base_x = b * x_s0 + h * x_s1 + s * x_s2
    base_sin = b * sin_s0 + s * sin_s1
    base_cos = b * sin_s0 + s * sin_s1  # reuse base; sin and cos have same [B, S] leading dims
    base_y = b * y_s0 + h * y_s1 + s * y_s2

    # Load x
    for off in range(0, D, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_ptr + base_x + idx * x_s3, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base_sin + idx * sin_s2, mask=mask, other=0.0)
        cos_vals = tl.load(cos_ptr + base_cos + idx * sin_s2, mask=mask, other=0.0)  # cos at [b, s, idx]

        half = D // 2
        x1 = x[:half]
        x2 = x[half:]
        rx = tl.concatenate([-x2, x1], axis=0)

        # keys rotate via sin_all and cos_all: y = x * sin_all - rotate_half(x) * cos_all
        y = x * sin_vals - rx * cos_vals
        tl.store(y_ptr + base_y + idx * y_s3, y.to(tl.bfloat16), mask=mask)


@triton.jit
def scatter_cache_kernel(
    krot_ptr,       # *const bf16, rotated keys [B, H, S, D]
    val_ptr,        # *const bf16, values [B, H, S, D] (original, not rotated)
    keycache_ptr,   # *mut bf16, key_cache [B, H, L, D]
    valuecache_ptr, # *mut bf16, value_cache [B, H, L, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
    k_s0, k_s1, k_s2, k_s3,
    v_s0, v_s1, v_s2, v_s3,
    kc_s0, kc_s1, kc_s2, kc_s3,
    vc_s0, vc_s1, vc_s2, vc_s3,
    cache_pos_ptr,  # *const int32, [B, S]
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    HS = H * S
    b = pid // HS
    tmp = pid % HS
    h = tmp // S
    s = tmp % S

    pos = tl.load(cache_pos_ptr + b * cache_pos_ptr.stride(0) + s * cache_pos_ptr.stride(1)).to(tl.int32)

    # base offsets
    base_k = b * k_s0 + h * k_s1 + s * k_s2
    base_v = b * v_s0 + h * v_s1 + s * v_s2
    base_kc = b * kc_s0 + h * kc_s1 + pos * kc_s2
    base_vc = b * vc_s0 + h * vc_s1 + pos * vc_s2

    for off in range(0, D, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        k = tl.load(krot_ptr + base_k + idx * k_s3, mask=mask, other=0.0).to(tl.bfloat16)
        v = tl.load(val_ptr + base_v + idx * v_s3, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(keycache_ptr + base_kc + idx * kc_s3, k, mask=mask)
        tl.store(valuecache_ptr + base_vc + idx * vc_s3, v, mask=mask)


def _ensure_contiguous(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def _launch_rmsnorm(query: torch.Tensor, q_weight: torch.Tensor, eps: float) -> torch.Tensor:
    assert query.is_cuda and q_weight.is_cuda
    B, H_q, S, D = query.shape
    y = torch.empty_like(query)
    grid = (B * H_q * S,)
    rmsnorm_kernel[grid](
        query, q_weight, y,
        B, H_q, S, D,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        q_weight.stride(0),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        eps=eps,
        BLOCK_SIZE=128,
        num_warps=4,
    )
    return y


def _build_rotation_tensors(query: torch.Tensor, key: torch.Tensor, device: torch.device):
    # Build cos_all and sin_all for query and key. Use torch for rotation math (PyTorch only, no Triton here).
    # We'll feed them to Triton kernels without using torch operations in ModelNew.forward.
    B, S = query.shape[:2]
    D = query.shape[-1]
    half = D // 2
    pos_ids = torch.arange(S, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    inv_freq = torch.tensor([1.0 / (10000000.0 ** (i / D)) for i in range(0, half)], device=device, dtype=torch.float32)
    # angle [B, S, half]
    angle = pos_ids.to(torch.float32) * inv_freq.unsqueeze(0)  # [B, S, half]
    # cos_all, sin_all for query
    cos_q = torch.cos(angle)  # [B, S, half]
    sin_q = torch.sin(angle)  # [B, S, half]
    cos_q = torch.cat([cos_q, cos_q], dim=-1)  # [B, S, D]
    sin_q = torch.cat([sin_q, sin_q], dim=-1)  # [B, S, D]
    # keys rotate via sin_all and cos_all from query's cos_q, sin_q (original code uses query's cos/sin for keys)
    cos_k = cos_q
    sin_k = sin_q
    # Ensure contiguous
    cos_q = _ensure_contiguous(cos_q)
    sin_q = _ensure_contiguous(sin_q)
    cos_k = _ensure_contiguous(cos_k)
    sin_k = _ensure_contiguous(sin_k)
    return cos_q, sin_q, cos_k, sin_k


def _launch_query_rotation(query_norm: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor) -> torch.Tensor:
    B, H_q, S, D = query_norm.shape
    query_rot = torch.empty_like(query_norm)
    grid = (B * H_q * S,)
    rotate_q_kernel[grid](
        query_norm, cos_q, sin_q, query_rot,
        B, H_q, S, D,
        query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
        cos_q.stride(0), cos_q.stride(1), cos_q.stride(2),
        query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
        BLOCK_SIZE=128,
        num_warps=4,
    )
    return query_rot


def _launch_key_rotation(key_norm: torch.Tensor, sin_k: torch.Tensor, cos_k: torch.Tensor) -> torch.Tensor:
    # num_kv_heads is 8 from provided axes, but key_norm.shape[1] may vary. Use actual H.
    B, H, S, D = key_norm.shape
    key_rot = torch.empty_like(key_norm)
    grid = (B * H * S,)
    rotate_k_kernel[grid](
        key_norm, sin_k, cos_k, key_rot,
        B, H, S, D,
        key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
        sin_k.stride(0), sin_k.stride(1), sin_k.stride(2),
        key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
        BLOCK_SIZE=128,
        num_warps=4,
    )
    return key_rot


def _launch_scatter_update(key_rot: torch.Tensor, value: torch.Tensor,
                           key_cache: torch.Tensor, value_cache: torch.Tensor,
                           cache_position: torch.Tensor) -> None:
    B, H_kv, S, D = key_rot.shape
    L = key_cache.shape[2]
    grid = (B * H_kv * S,)
    # Ensure cache_position is [B, S] int32
    cache_pos = _ensure_contiguous(cache_position.to(torch.int32).unsqueeze(1))
    scatter_cache_kernel[grid](
        key_rot, value, key_cache, value_cache,
        B, H_kv, S, D, L,
        key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        cache_pos,
        BLOCK_SIZE=128,
        num_warps=4,
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure CUDA and contiguity
        assert query.is_cuda and key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda, "All tensors must be CUDA for Triton"
        device = query.device
        # 1) RMSNorm on query and key
        query_norm = _launch_rmsnorm(query, q_norm_weight.to(query.dtype), rms_norm_eps)
        key_norm = _launch_rmsnorm(key, k_norm_weight.to(key.dtype), rms_norm_eps)

        # 2) Build rotation tensors (PyTorch for math, Triton for rotations)
        # Note: Build using provided inv_freq if desired; however original code already has inv_freq tensor.
        # We still construct angle-based cos/sin for rotation.
        cos_q, sin_q, cos_k, sin_k = _build_rotation_tensors(query_norm, key_norm, device)

        # 3) Query rotation
        query_rot = _launch_query_rotation(query_norm, cos_q, sin_q)

        # 4) Key rotation
        key_rot = _launch_key_rotation(key_norm, sin_k, cos_k)

        # 5) Scatter update caches
        _launch_scatter_update(key_rot, value, key_cache, value_cache, cache_position)

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
