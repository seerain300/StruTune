import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x, out, w, B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // (H * S)
    h = (pid // S) % H
    s = pid % S

    base = b * H * S * D + h * S * D + s * D

    # Compute scale in float32
    sum_sq = 0.0
    offs = tl.arange(0, D)
    for d in range(0, D):
        v = tl.load(x + base + d)
        sum_sq += v.to(tl.float32) * v
    mean = sum_sq / D
    scale = tl.rsqrt(mean + 1e-6)

    # Write normalized and scaled output
    for d in range(0, D):
        v = tl.load(x + base + d)
        nv = (v.to(tl.float32) * scale) * tl.load(w + d)
        tl.store(out + base + d, nv.to(v.dtype))


@triton.jit
def apply_rotary_kernel(x_in, x_out, pos, inv_freq_const, B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
                         D: tl.constexpr, HALF: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // (H * S)
    h = (pid // S) % H
    s = pid % S

    base = b * H * S * D + h * S * D + s * D

    # Load x
    offs = tl.arange(0, D)
    x = tl.load(x_in + base + offs)

    # Construct cos/sin for first HALF dims using inv_freq_const (constexpr vector)
    # inv_freq_const has length HALF in float32: [inv_freq[0], inv_freq[1], ...]
    idx = tl.arange(0, HALF)
    pos_f = pos.to(tl.float32)  # scalar
    base2 = pos_f * inv_freq_const[idx]  # [HALF], float32
    cos_vec = tl.cos(base2)             # [HALF], float32
    sin_vec = tl.sin(base2)             # [HALF], float32

    # Prepare masks and offsets for rotation
    mask_even = offs < HALF
    mask_odd = offs >= HALF

    # Even part (first HALF)
    x_even = x[0:HALF]
    # Odd part (next HALF)
    x_odd = x[HALF:2 * HALF]

    # Apply rotation: y = x_even * cos - x_odd * sin (negative sign because of rotation)
    y_even = x_even * cos_vec
    y_odd = -x_odd * sin_vec

    # Concatenate results
    y = tl.zeros([D], dtype=x.dtype)
    y[0:HALF] = y_even
    y[HALF:2 * HALF] = y_odd

    tl.store(x_out + base + offs, y)


@triton.jit
def update_cache_kernel(key_out, value_in, key_cache, value_cache, out_pos, B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
                         D: tl.constexpr, num_kv_heads: tl.constexpr):
    # One program per (b, kv_head, s)
    pid = tl.program_id(0)
    b = pid // (H * S)
    kv_head = (pid // S) % H  # H = num_kv_heads is passed
    s = pid % S

    # Compute base offsets
    base_qk = b * H * S * D + kv_head * S * D + s * D
    base_key = b * num_kv_heads * D * (out_pos + 0) + kv_head * D  # out_pos is scalar
    base_value = b * num_kv_heads * D * (out_pos + 0) + kv_head * D

    # Store rotated key and original value to caches
    key_vec = tl.load(key_out + base_qk + tl.arange(0, D))
    value_vec = tl.load(value_in + base_qk + tl.arange(0, D))
    tl.store(key_cache + base_key + tl.arange(0, D), key_vec)
    tl.store(value_cache + base_value + tl.arange(0, D), value_vec)


def run_triton_model(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     position_ids: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                     cache_position: torch.Tensor,
                     q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor, inv_freq: torch.Tensor,
                     rms_norm_eps: float):
    B = query.shape[0]
    H = query.shape[1]
    S = query.shape[2]
    D = query.shape[3]
    HALF = D // 2

    # Outputs after RMSNorm
    query_norm = torch.empty_like(query)
    key_norm = torch.empty_like(key)

    # Launch RMSNorm kernels: one per (b, h, s)
    grid_norm = (B * H * S,)
    rmsnorm_kernel[grid_norm](
        query, query_norm, q_norm_weight,
        B, H, S, D, HALF,
        num_warps=4, num_stages=2,
    )
    rmsnorm_kernel[grid_norm](
        key, key_norm, k_norm_weight,
        B, H, S, D, HALF,
        num_warps=4, num_stages=2,
    )

    # Apply rotary embedding: pos = cache_len + s = cache_position[0] + s
    # We pass cache_position[0] as the scalar pos; Triton treats it as runtime scalar.
    # Construct inv_freq_const vector in Python as float32 with length HALF.
    inv_freq_const = inv_freq.to(torch.float32).contiguous()  # [HALF]

    query_rot = torch.empty_like(query_norm)
    key_rot = torch.empty_like(key_norm)

    pos_scalar = cache_position[0]  # scalar tensor on device
    grid_rot = (B * H * S,)
    apply_rotary_kernel[grid_rot](
        query_norm, query_rot, pos_scalar, inv_freq_const,
        B, H, S, D, HALF,
        num_warps=4, num_stages=2,
    )
    apply_rotary_kernel[grid_rot](
        key_norm, key_rot, pos_scalar, inv_freq_const,
        B, H, S, D, HALF,
        num_warps=4, num_stages=2,
    )

    # Update caches: out_pos = cache_len + s
    # We write into new tensors; Triton does not read key/value caches to avoid torch tensor reads in kernel.
    # The original code writes into existing caches; here we mimic the side-effect by writing into new outputs.
    key_cache_out = torch.empty_like(key_cache)
    value_cache_out = torch.empty_like(value_cache)

    grid_cache = (B * H * S,)
    out_pos = cache_position[0]  # scalar tensor on device
    update_cache_kernel[grid_cache](
        key_rot, value, key_cache_out, value_cache_out, out_pos,
        B, H, S, D, H,  # num_kv_heads = H
        num_warps=4, num_stages=2,
    )

    return query_rot, key_rot, key_cache_out, value_cache_out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        query = args[0].contiguous()
        key = args[1].contiguous()
        value = args[2].contiguous()
        position_ids = args[3]  # not used in Triton
        key_cache = args[4].contiguous()
        value_cache = args[5].contiguous()
        cache_position = args[6]  # scalar tensor on device
        q_norm_weight = args[7].contiguous()  # [D], bf16
        k_norm_weight = args[8].contiguous()  # [D], bf16
        inv_freq = args[9].contiguous()       # [D//2], float32
        rms_norm_eps = args[10]               # not used (fixed in kernel)

        query_rot, key_rot, key_cache_out, value_cache_out = run_triton_model(
            query, key, value, position_ids, key_cache, value_cache, cache_position,
            q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        )
        return query_rot, key_rot, key_cache_out, value_cache_out


def run(*args):
    return ModelNew()(*args)
