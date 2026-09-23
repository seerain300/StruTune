import torch
import triton
import triton.language as tl

# Build cos/sin for rotation: emb = cat([pos * inv_freq, pos * inv_freq]), pos=0 (emb is position-agnostic in original)
@triton.jit
def build_cos_sin_kernel(inv_freq_ptr, cos_ptr, sin_ptr, head_dim, BLOCK: tl.constexpr):
    half = head_dim // 2
    offs = tl.arange(0, half)
    # pos = 0; emb_half = inv_freq
    inv = tl.load(inv_freq_ptr + offs)
    # Create emb_vec of length head_dim: first half is inv, second half is inv
    emb_vec = tl.zeros((head_dim,), dtype=tl.float32)
    emb_vec[:half] = inv
    emb_vec[half:] = inv

    cos_vals = tl.cos(emb_vec).to(tl.float32)
    sin_vals = tl.sin(emb_vec).to(tl.float32)

    # Store cos/sin into arrays
    tl.store(cos_ptr + offs, cos_vals, mask=offs < half)
    tl.store(cos_ptr + half + offs, cos_vals, mask=offs < half)
    tl.store(sin_ptr + offs, sin_vals, mask=offs < half)
    tl.store(sin_ptr + half + offs, sin_vals, mask=offs < half)

# Fused RMSNorm + Rotate: per-row over last dim
@triton.jit
def fuse_norm_rotate_kernel(x_ptr, y_ptr, weight_ptr, eps, cos_ptr, sin_ptr, n_rows, head_dim, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < head_dim
    base = row_id * head_dim

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # RMSNorm: variance over last dim
    sum_sq = tl.sum(x32 * x32)
    mean = sum_sq / head_dim
    scale = tl.rsqrt(mean + eps)  # scalar
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    x_norm = x32 * (w * scale)

    # Rotation: split into halves
    half = head_dim // 2
    x1 = x_norm[:half]
    x2 = x_norm[half:]

    # Load cos/sin halves
    cos1 = tl.load(cos_ptr + offs, mask=offs < half, other=0.0).to(tl.float32)
    sin1 = tl.load(sin_ptr + offs, mask=offs < half, other=0.0).to(tl.float32)
    cos2 = tl.load(cos_ptr + half + offs, mask=offs < half, other=0.0).to(tl.float32)
    sin2 = tl.load(sin_ptr + half + offs, mask=offs < half, other=0.0).to(tl.float32)

    # Compute rotated output
    out1 = x1 * cos1 + x2 * sin1  # first half
    out2 = -x2 * cos2 + x1 * sin2  # second half

    y32 = tl.zeros((head_dim,), dtype=tl.float32)
    y32[:half] = out1
    y32[half:] = out2

    # Store back (cast handled by Triton on store)
    tl.store(y_ptr + base + offs, y32, mask=mask)

# Generic copy of one row from src to dst at base_offset (cache_len + t)
@triton.jit
def copy_slice_kernel(src_ptr, dst_ptr, B, num_heads, seq_len, head_dim, base_offset, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    head = tl.program_id(1)
    t = tl.program_id(2)

    src_base = ((b * num_heads + head) * seq_len + t) * head_dim
    dst_base = ((b * num_heads + head) * (base_offset + t)) * head_dim

    offs = tl.arange(0, BLOCK)
    mask = offs < head_dim
    src_row_ptr = src_ptr + src_base + offs
    dst_row_ptr = dst_ptr + dst_base + offs

    val = tl.load(src_row_ptr, mask=mask, other=0.0)
    tl.store(dst_row_ptr, val, mask=mask)

@torch.no_grad()
def run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position_ids: torch.Tensor,  # kept for API compatibility, not used
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    rms_norm_eps: float,
):
    # Ensure CUDA and contiguous
    assert query.is_cuda and key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda, "All tensors must be on CUDA"
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()
    q_norm_weight = q_norm_weight.contiguous()
    k_norm_weight = k_norm_weight.contiguous()
    inv_freq = inv_freq.contiguous()

    B, num_q_heads, seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    num_kv_heads_value = value.shape[1]
    assert key.shape == (B, num_kv_heads, seq_len, head_dim)
    assert value.shape == (B, num_kv_heads_value, seq_len, head_dim)
    assert key_cache.shape == (B, num_kv_heads, 262144, head_dim)
    assert value_cache.shape == (B, num_kv_heads, 262144, head_dim)

    # Allocate outputs (rotated)
    query_rotated = torch.empty_like(query)
    key_rotated = torch.empty_like(key)

    # Build cos/sin (single program)
    cos_vec = torch.empty(head_dim, dtype=torch.float32, device=query.device)
    sin_vec = torch.empty(head_dim, dtype=torch.float32, device=query.device)
    # Choose BLOCK = head_dim for convenience
    BLOCK = head_dim
    build_cos_sin_kernel[(1,)](inv_freq, cos_vec, sin_vec, head_dim, BLOCK=BLOCK)

    # Fused RMSNorm + Rotate for query and key
    n_rows_q = B * num_q_heads * seq_len
    n_rows_k = B * num_kv_heads * seq_len

    fuse_norm_rotate_kernel[(n_rows_q,)](
        query, query_rotated, q_norm_weight, rms_norm_eps, cos_vec, sin_vec,
        n_rows_q, head_dim, BLOCK=BLOCK, num_warps=4
    )

    fuse_norm_rotate_kernel[(n_rows_k,)](
        key, key_rotated, k_norm_weight, rms_norm_eps, cos_vec, sin_vec,
        n_rows_k, head_dim, BLOCK=BLOCK, num_warps=4
    )

    # Update caches: copy rows into key_cache[:, :, cache_len + t] and value_cache[:, :, cache_len + t]
    base_offset = int(cache_position[0].item())  # cache_len from cache_position[0]
    grid_k = (B, num_kv_heads, seq_len)
    grid_v = (B, num_kv_heads, seq_len)

    copy_slice_kernel[grid_k](
        key_rotated, key_cache, B, num_kv_heads, seq_len, head_dim, base_offset, BLOCK=BLOCK, num_warps=4
    )

    copy_slice_kernel[grid_v](
        value, value_cache, B, num_kv_heads, seq_len, head_dim, base_offset, BLOCK=BLOCK, num_warps=4
    )

    return query_rotated, key_rotated, key_cache, value_cache

class ModelNew(torch.nn.Module):
    def forward(self, *args):
        if len(args) < 11:
            raise RuntimeError("ModelNew.forward expects at least 11 arguments")
        query = args[0]
        key = args[1]
        value = args[2]
        position_ids = args[3]
        key_cache = args[4]
        value_cache = args[5]
        cache_position = args[6]
        q_norm_weight = args[7]
        k_norm_weight = args[8]
        inv_freq = args[9]
        rms_norm_eps = float(args[10]) if len(args) > 10 else 1e-6
        return run(query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps)


def run(*args):
    return ModelNew()(*args)
