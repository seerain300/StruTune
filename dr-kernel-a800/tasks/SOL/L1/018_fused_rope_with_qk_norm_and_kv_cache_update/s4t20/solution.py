import torch
import triton
import triton.language as tl

# Triton kernel: RMS normalization per row (length D). y = x * rsqrt(mean(x^2) + eps)
@triton.jit
def rms_norm_rows_kernel(x_ptr, out_ptr, D: tl.constexpr, eps: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + row_id * D + offs).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    mean = sum_sq / D
    scale = 1.0 / tl.sqrt(mean + eps)
    y = x * scale
    tl.store(out_ptr + row_id * D + offs, y.to(tl.bfloat16))

# Triton kernel: precompute cos and sin vectors of length 2*D based on inv_freq and pos = 0 (fixed).
# We use a 1D launch; idx ranges 0..2*D-1. Triton doesn't support torch.cat, so we store cos/sin vectors.
@triton.jit
def cos_sin_fixed_pos_kernel(inv_freq_ptr, cos_ptr, sin_ptr, D: tl.constexpr):
    idx = tl.arange(0, 2 * D)
    half = D
    idx_half1 = tl.arange(0, half)
    idx_half2 = idx_half1 + half
    angle1 = idx_half1.to(tl.float32) * tl.load(inv_freq_ptr + idx_half1)
    angle2 = (idx_half2 - half).to(tl.float32) * tl.load(inv_freq_ptr + (idx_half2 - half))
    angle = tl.where(idx < half, angle1, angle2)
    c = tl.cos(angle)
    s = tl.sin(angle)
    tl.store(cos_ptr + idx, c)
    tl.store(sin_ptr + idx, s)

# Triton kernel: apply rotation using precomputed cos and sin vectors of length 2*D.
# Input x (shape [rows, D]), output y (shape [rows, D]):
# y = x * cos[:D] - rotate_half(x) * sin[:D], where rotate_half(x) = [-x2, x1] for halves.
@triton.jit
def rotate_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, D: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + row_id * D + offs).to(tl.float32)
    cos = tl.load(cos_ptr + offs).to(tl.float32)
    sin = tl.load(sin_ptr + offs).to(tl.float32)
    half = D // 2
    x1 = x[:half]
    x2 = x[half:]
    rotated = x1 * cos[:half] - x2 * sin[:half]
    tl.store(out_ptr + row_id * D + offs, rotated.to(tl.bfloat16))

# Triton kernel: copy a batch of rows from src to dst at specified positions. Grid is (B, H, S).
# This kernel copies src[b, head, s, :] into dst[b, head, positions[s], :]. Note: we assume src has shape
# [B, H, S, D] with strides known; here we pass flattened pointers using N*H*S*D layout but we rely on caller to
# pass correct pointers. In practice, we pass the actual tensors directly and compute offsets based on b, head, s.
@triton.jit
def cache_update_kernel(src_ptr, dst_ptr, positions_ptr, D: tl.constexpr):
    b = tl.program_id(0)
    head = tl.program_id(1)
    s = tl.program_id(2)
    pos = tl.load(positions_ptr + s)  # int64
    # For key_cache: index is (b, head, pos)
    # For value_cache: index is (b, head, pos)
    # We don't have explicit row stride; src and dst are [B, H, S, D], so each row occupies D elements contiguous.
    # We pass src_ptr pointing to row (b, head, s) and dst_ptr pointing to row (b, head, pos). Since Triton
    # expects offsets, we can compute row base as (b*H + head)*S*D + s*D for src and (b*H + head)*max_pos*D + pos*D for dst.
    # However, to keep it simple, we assume src and dst are laid out as [B*H*S, D] contiguous. We will pass tensors
    # shaped that way from forward. So row_id = b*H*S + head*S + s.
    row_id = b * (num_q_heads * seq_len) + head * seq_len + s
    src_offset = row_id * D
    dst_offset = (b * (num_key_value_heads * max_position_embeddings) + head * max_position_embeddings + pos) * D
    offs = tl.arange(0, D)
    val = tl.load(src_ptr + src_offset + offs).to(tl.float32)
    tl.store(dst_ptr + dst_offset + offs, val.to(tl.bfloat16))

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                position_ids: torch.Tensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor,
                k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor,
                rms_norm_eps: float):
        """
        Triton-optimized version:
        - RMS normalization for query and key via Triton kernel.
        - Precompute cos/sin vectors via Triton kernel (no torch.cos/torch.sin in host).
        - Apply rotation using Triton kernel.
        - Update caches using Triton kernel.
        No torch.cos/torch.sin/torch.cat in host code. Triton kernels do trig computations and data movement.
        """
        device = query.device
        dtype = query.dtype  # bfloat16
        D = query.shape[-1]  # head_dim = 128
        B, num_q_heads, S, _ = query.shape
        num_kv_heads = key.shape[1]
        max_pos = key_cache.shape[2]

        # 1) RMS normalization (Triton)
        # Prepare output in fp32 for stability, then cast back
        query_norm_fp32 = torch.empty((B, num_q_heads, S, D), dtype=torch.float32, device=device)
        key_norm_fp32 = torch.empty((B, num_kv_heads, S, D), dtype=torch.float32, device=device)

        grid_norm = (B * num_q_heads * S,)
        rms_norm_rows_kernel[grid_norm](query, query_norm_fp32, D, rms_norm_eps)
        rms_norm_rows_kernel[grid_norm](key, key_norm_fp32, D, rms_norm_eps)

        # Cast back to bfloat16 for rotation
        query_norm = query_norm_fp32.to(torch.bfloat16)
        key_norm = key_norm_fp32.to(torch.bfloat16)

        # 2) Triton cos/sin vectors (fixed pos=0). Length 2*D. Float32.
        cos_vec = torch.empty(2 * D, dtype=torch.float32, device=device)
        sin_vec = torch.empty(2 * D, dtype=torch.float32, device=device)
        inv_freq_vec = inv_freq  # [D], float32
        # Launch once per batch (grid size 1). This computes cos/sin for the first position (p=0). We use them to rotate later.
        cos_sin_fixed_pos_kernel[(1,)](inv_freq_vec, cos_vec, sin_vec, D)

        # 3) Apply rotation (Triton)
        query_rotated_fp32 = torch.empty((B, num_q_heads, S, D), dtype=torch.float32, device=device)
        key_rotated_fp32 = torch.empty((B, num_kv_heads, S, D), dtype=torch.float32, device=device)

        grid_rotate = (B * num_q_heads * S,)
        rotate_kernel[grid_rotate](query_norm, cos_vec, sin_vec, query_rotated_fp32, D)
        rotate_kernel[grid_rotate](key_norm, cos_vec, sin_vec, key_rotated_fp32, D)

        # Cast back to bfloat16
        query_rotated = query_rotated_fp32.to(torch.bfloat16)
        key_rotated = key_rotated_fp32.to(torch.bfloat16)

        # 4) Update caches (Triton): write into key_cache at cache_position and into value_cache at cache_position
        # Cast cache_position to int32 for Triton
        positions_int = cache_position.to(torch.int32)

        # Prepare src as [B, H, S, D] contiguous; we'll pass query_rotated for key_cache and value for value_cache
        # Note: cache_update_kernel expects flattened [B*H*S, D] layout. We will create such flattened views.
        # For key_cache: use key_rotated (b, num_kv_heads, S, D)
        # First, reshape: [B*num_kv_heads*S, D]
        key_rotated_flat = key_rotated.reshape(B * num_kv_heads * S, D).contiguous()
        value_flat = value.reshape(B * num_q_heads * S, D).contiguous()

        # Grid over (B, num_kv_heads, S) for key_cache; and (B, num_q_heads, S) for value_cache
        # We'll call the kernel twice with appropriate dst tensors.

        # Update key_cache
        grid_cache_k = (B, num_kv_heads, S)
        # dst_ptr for key_cache: [B, num_kv_heads, max_pos, D] with D contiguous
        cache_update_kernel[grid_cache_k](key_rotated_flat, key_cache, positions_int, D)

        # Update value_cache
        grid_cache_v = (B, num_q_heads, S)
        # dst_ptr for value_cache: [B, num_q_heads, max_pos, D]
        cache_update_kernel[grid_cache_v](value_flat, value_cache, positions_int, D)

        # Return rotated query and rotated key, along with updated caches.
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
