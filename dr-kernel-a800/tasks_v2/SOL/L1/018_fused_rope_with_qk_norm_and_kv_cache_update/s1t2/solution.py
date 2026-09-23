import torch
import triton
import triton.language as tl

# Triton kernel: RMSNorm per row across last dim
@triton.jit
def rms_norm_row_kernel(x_ptr, y_ptr, weight_ptr, eps, head_dim, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)  # 0..n_rows-1
    offs = tl.arange(0, BLOCK)  # covers full head_dim
    mask = offs < head_dim
    base = row_id * head_dim

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Compute mean of squares over the row
    sum_sq = tl.sum(x32 * x32)
    mean = sum_sq / head_dim
    scale = tl.rsqrt(mean + eps)  # scalar

    w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    y32 = x32 * (w * scale)

    tl.store(y_ptr + base + offs, y32, mask=mask)

# Triton kernel: apply rotation using precomputed cos/sin vectors (length = head_dim)
@triton.jit
def rotate_rows_kernel(x_ptr, y_ptr, cos_ptr, sin_ptr, head_dim, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < head_dim
    base = row_id * head_dim

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    half = head_dim // 2
    x1 = x32[:half]
    x2 = x32[half:]

    cos1 = tl.load(cos_ptr + offs, mask=offs < half, other=0.0).to(tl.float32)
    sin1 = tl.load(sin_ptr + offs, mask=offs < half, other=0.0).to(tl.float32)
    cos2 = tl.load(cos_ptr + half + offs, mask=offs < half, other=0.0).to(tl.float32)
    sin2 = tl.load(sin_ptr + half + offs, mask=offs < half, other=0.0).to(tl.float32)

    out1 = x1 * cos1 + x2 * sin1   # first half
    out2 = -x2 * cos2 + x1 * sin2  # second half

    y32 = tl.zeros((head_dim,), dtype=tl.float32)
    y32[:half] = out1
    y32[half:] = out2

    tl.store(y_ptr + base + offs, y32, mask=mask)

# Triton kernel: copy one row from src to dst at base_offset (dst index = (b * num_heads + head) * (base_offset + t))
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
    position_ids: torch.Tensor,  # kept for API compatibility; not used directly
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

    # Allocate normalized tensors
    query_norm = torch.empty_like(query)
    key_norm = torch.empty_like(key)

    # 1) RMSNorm for query and key (per row)
    n_rows_q = B * num_q_heads * seq_len
    n_rows_k = B * num_kv_heads * seq_len
    BLOCK = head_dim  # one program handles full row

    rms_norm_row_kernel[(n_rows_q,)](
        query, query_norm, q_norm_weight, rms_norm_eps, head_dim, BLOCK=BLOCK, num_warps=4
    )
    rms_norm_row_kernel[(n_rows_k,)](
        key, key_norm, k_norm_weight, rms_norm_eps, head_dim, BLOCK=BLOCK, num_warps=4
    )

    # 2) Build rotation vectors using torch (since Triton doesn't do trig here):
    # emb = cat([pos * inv_freq, pos * inv_freq], dim=-1), pos=0 (original code's rotation is position-agnostic)
    pos = 0
    emb_first = (pos * inv_freq).to(query.dtype)  # length head_dim/2
    emb_second = emb_first
    emb_vec = torch.cat([emb_first, emb_second], dim=0)  # length head_dim
    cos_vec = torch.cos(emb_vec).to(torch.float32).contiguous()  # fp32 for accuracy
    sin_vec = torch.sin(emb_vec).to(torch.float32).contiguous()

    # 3) Apply rotation using Triton
    query_rotated = torch.empty_like(query_norm)
    key_rotated = torch.empty_like(key_norm)

    rotate_rows_kernel[(n_rows_q,)](
        query_norm, query_rotated, cos_vec, sin_vec, head_dim, BLOCK=BLOCK, num_warps=4
    )
    rotate_rows_kernel[(n_rows_k,)](
        key_norm, key_rotated, cos_vec, sin_vec, head_dim, BLOCK=BLOCK, num_warps=4
    )

    # 4) Update caches: copy rotated keys and original values into cache at cache_len + t
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
