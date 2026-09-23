import torch
import triton
import triton.language as tl

# Kernel: RMSNorm on X -> Y
# Inputs:
#   X: [B, num_heads, S, D] (bf16), Weight: [D] (bf16), eps: fp32
# Output:
#   Y: [B, num_heads, S, D] (same dtype as X)
@triton.jit
def rmsnorm_kernel(X, Weight, Y, eps, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    B, H, S, _ = tl.shape(X)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    # Row index in X
    x = X[pid_b, pid_h, pid_s, :]
    # Reduction over D in chunks of BLOCK_SIZE
    sumsq = 0.0
    for offs in range(0, D, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_chunk = tl.load(x + cols, mask=mask, other=0.0)
        x_chunk_f = x_chunk.to(tl.float32)
        sumsq += tl.sum(x_chunk_f * x_chunk_f, axis=0)
    mean = sumsq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    # Apply normalization and weight
    for offs in range(0, D, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_chunk = tl.load(x + cols, mask=mask, other=0.0)
        w = tl.load(Weight + cols, mask=mask, other=0.0).to(tl.float32)
        y_chunk = x_chunk.to(tl.float32) * (w * inv_rms)
        # Cast back to original dtype
        y_chunk = y_chunk.to(x_chunk.dtype)
        tl.store(Y[pid_b, pid_h, pid_s, cols], y_chunk, mask=mask)

# Kernel: compute cos and sin per (b, s) for rotation using inv_freq
# Inputs:
#   inv_freq: [D_half] (fp32), pos_ids: [B, S] (int32), cos_out: [B, S, D_half] (fp32), sin_out: [B, S, D_half] (fp32)
@triton.jit
def cosine_sin_kernel(inv_freq, pos_ids, cos_out, sin_out, D_half: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pos = pos_ids[pid_b, pid_s].to(tl.float32)
    for i in range(0, D_half):
        freq = inv_freq[i]
        angle = pos * freq
        c = tl.cos(angle)
        s = tl.sin(angle)
        tl.store(cos_out[pid_b, pid_s, i], c)  # fp32
        tl.store(sin_out[pid_b, pid_s, i], s)  # fp32

# Kernel: apply rotation to X using cos/sin -> Y
# Inputs:
#   X: [B, H, S, D] (bf16), cos: [B, S, D_half] (fp32), sin: [B, S, D_half] (fp32), Weight: [D] (bf16), Output Y
@triton.jit
def apply_rope_kernel(X, cos, sin, Weight, Y, D: tl.constexpr, D_half: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    x = X[pid_b, pid_h, pid_s, :]
    # Load x and weight
    for offs in range(0, D, BLOCK_SIZE):
        cols = offs + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x_chunk = tl.load(x + cols, mask=mask, other=0.0)
        w = tl.load(Weight + cols, mask=mask, other=0.0).to(tl.float32)
        x_chunk_f = x_chunk.to(tl.float32)
        # Split into two halves: x1 and x2
        x1 = x_chunk_f[..., :D_half]
        x2 = x_chunk_f[..., D_half:]
        # Load cos/sin for this (b, s)
        # Broadcast to [D] using repeating: cos[:D_half] twice; sin[:D_half] twice
        cos_vals = tl.load(cos[pid_b, pid_s, :]).to(tl.float32)  # [D_half]
        sin_vals = tl.load(sin[pid_b, pid_s, :]).to(tl.float32)  # [D_half]
        # Create cos_all and sin_all of shape [D]
        cos_all = tl.zeros([D], dtype=tl.float32)
        sin_all = tl.zeros([D], dtype=tl.float32)
        # Fill first half
        cos_all[:D_half] = cos_vals
        sin_all[:D_half] = sin_vals
        # Fill second half with same cos/sin (as original applies sin twice for keys; for query applies cos twice)
        cos_all[D_half:] = cos_vals
        sin_all[D_half:] = sin_vals
        # Compute rotate_half(x) = [-x2, x1]
        rot = tl.cat([-x2, x1], axis=0)  # shape [D]
        y_chunk = x_chunk_f * cos_all - rot * sin_all
        # Cast back to original dtype and store
        y_chunk = y_chunk.to(x_chunk.dtype)
        tl.store(Y[pid_b, pid_h, pid_s, cols], y_chunk, mask=mask)

# Kernel: scatter update caches at positions cache_position
# Inputs:
#   Src: [B, H, S, D] (bf16), Cache: [B, H, L, D] (bf16), cache_position: [S] (int32), Output Cache
@triton.jit
def scatter_update_cache_kernel(Src, Cache, cache_position, B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    for s in range(0, S):
        idx = cache_position[s]  # int32
        row_src = Src[pid_b, pid_h, s, :]
        row_cache = Cache[pid_b, pid_h, idx, :]
        # Copy row_src into row_cache
        for d in range(0, D):
            val = tl.load(row_src + d)
            tl.store(row_cache + d, val)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure tensors are on CUDA and contiguous
        device = query.device
        assert device.type == "cuda", "ModelNew requires CUDA device"
        B, H_q, S, D = query.shape
        _, H_kv, _, _ = key.shape
        inv_half = D // 2
        # Allocate outputs
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)
        # Launch RMSNorm for query and key
        grid = (B, H_q, S)
        # Choose BLOCK_SIZE as 128 (since D=128 in workload). Keep it constexpr for Triton.
        BLOCK_SIZE = 128
        rmsnorm_kernel[grid](
            query, q_norm_weight, query_norm, rms_norm_eps,
            D=D, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4, num_stages=2
        )
        rmsnorm_kernel[grid](
            key, k_norm_weight, key_norm, rms_norm_eps,
            D=D, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4, num_stages=2
        )
        # Compute cos/sin using Triton
        # position_ids: [B, S], cast to int32
        pos_ids_int = position_ids.to(torch.int32)
        cos_out = torch.empty((B, S, inv_half), dtype=torch.float32, device=device)
        sin_out = torch.empty((B, S, inv_half), dtype=torch.float32, device=device)
        grid_cs = (B, S)
        cosine_sin_kernel[grid_cs](
            inv_freq, pos_ids_int, cos_out, sin_out,
            D_half=inv_half, num_warps=2, num_stages=2
        )
        # Apply rotation: query uses cos, key uses sin (inferred logical rotation)
        # Launch apply_rope for query with cos_out
        query_rotated = torch.empty_like(query)
        grid_rope = (B, H_q, S)
        apply_rope_kernel[grid_rope](
            query_norm, cos_out, sin_out, q_norm_weight, query_rotated,
            D=D, D_half=inv_half,
            num_warps=4, num_stages=2
        )
        # Launch apply_rope for key with sin_out
        key_rotated = torch.empty_like(key)
        apply_rope_kernel[grid_rope](
            key_norm, cos_out, sin_out, k_norm_weight, key_rotated,
            D=D, D_half=inv_half,
            num_warps=4, num_stages=2
        )
        # Update caches: create new tensors (not in-place) to match original return semantics
        new_key_cache = torch.empty_like(key_cache)
        new_value_cache = torch.empty_like(value_cache)
        grid_sc = (B, H_kv)
        # cache_position: [S], cast to int32
        cache_pos_int = cache_position.to(torch.int32)
        # Copy src rows into cache at cache_position
        scatter_update_cache_kernel[grid_sc](
            key_rotated, new_key_cache, cache_pos_int,
            B=B, H=H_kv, S=S, D=D,
            num_warps=4, num_stages=2
        )
        scatter_update_cache_kernel[grid_sc](
            value, new_value_cache, cache_pos_int,
            B=B, H=H_kv, S=S, D=D,
            num_warps=4, num_stages=2
        )
        return query_rotated, key_rotated, new_key_cache, new_value_cache


def run(*args):
    return ModelNew()(*args)
