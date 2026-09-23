import torch
import triton
import triton.language as tl

# Triton kernel: RMS normalization per row (length D).
# For each row, compute scale = 1/sqrt(mean(x^2) + eps), then y = x * scale.
# Assumes input/output tensors are laid out as [rows, D], where rows = N * S and D is constexpr.
@triton.jit
def rms_norm_rows_kernel(x_ptr, out_ptr, D: tl.constexpr, eps):
    row_id = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + row_id * D + offs).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    mean = sum_sq / D
    scale = 1.0 / tl.sqrt(mean + eps)
    y = x * scale
    tl.store(out_ptr + row_id * D + offs, y.to(tl.bfloat16))

# Triton kernel: copy rows from src (flattened [rows, D]) into dst (flattened [rows_out, D]) at positions given by positions_ptr (int64).
# Grid: (B, num_heads, S). Each program handles one (b, head, s) and copies row (b, head, s) -> (b, head, positions[s]).
@triton.jit
def copy_rows_to_positions_kernel(src_ptr, dst_ptr, positions_ptr, D: tl.constexpr):
    b = tl.program_id(0)
    head = tl.program_id(1)
    s = tl.program_id(2)
    # Load target position for this (b, head, s)
    pos = tl.load(positions_ptr + s).to(tl.int64)
    # Compute row start offsets
    src_row_start = (b * head + head * s) * D
    dst_row_start = (b * head + head * pos) * D
    offs = tl.arange(0, D)
    vals = tl.load(src_ptr + src_row_start + offs).to(tl.bfloat16)
    tl.store(dst_ptr + dst_row_start + offs, vals)

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
        Triton:
        - Perform RMS normalization for query and key using Triton kernels.
        - Update key/value caches using Triton copy kernels.
        PyTorch:
        - Apply rotation position embedding (apply_rope) using original logic (requires torch.cos/torch.sin/torch.cat).
        Inputs:
          - query: [B, num_q_heads, S, D]
          - key: [B, num_kv_heads, S, D] (note: original run uses batch_size and num_kv_heads; we match that here)
          - value: [B, num_kv_heads, S, D]
          - key_cache: [B, num_kv_heads, max_len, D]
          - value_cache: [B, num_kv_heads, max_len, D]
          - cache_position: [len], int64, positions to write
        Returns:
          - query_rotated, key_rotated, updated key_cache, updated value_cache
        """
        device = query.device
        dtype = query.dtype

        # 1) Triton RMS normalization for query and key
        # Ensure contiguous
        query_contig = query.contiguous()
        key_contig = key.contiguous()
        value_contig = value.contiguous()

        B, num_q_heads, S, D = query_contig.shape
        Bk, num_kv_heads, Sk, Dk = key_contig.shape
        assert D == Dk, "Dimension mismatch: query and key last dim must be equal"
        assert S == Sk, "Dimension mismatch: query seq_len and key seq_len must be equal"

        # Normalize query
        query_norm = torch.empty_like(query_contig, dtype=dtype)
        N_rows_q = B * num_q_heads * S
        rms_norm_rows_kernel[(N_rows_q,)](query_contig.view(-1, D), query_norm.view(-1, D), D, float(rms_norm_eps))

        # Normalize key
        key_norm = torch.empty_like(key_contig, dtype=dtype)
        N_rows_k = Bk * num_kv_heads * Sk
        rms_norm_rows_kernel[(N_rows_k,)](key_contig.view(-1, D), key_norm.view(-1, D), D, float(rms_norm_eps))

        # 2) PyTorch: apply rotation position embedding (original logic)
        # Prepare rotation components: position_ids [B, S], inv_freq [D//2], emb [B, S, D], cos, sin
        # Note: original code uses position_ids shape [batch_size, seq_len] -> [B, S]
        # inv_freq is provided; reshape and expand appropriately.
        # emb = [pos * inv_freq, pos * inv_freq] over last dim
        # cos = emb.cos, sin = emb.sin
        # rotate_half(x) = [-x[..., D//2:], x[..., :D//2]]
        # y = x * cos + rotate_half(x) * sin

        # Make inv_freq shape usable: inv_freq is [D//2], expand to [B, S, D]
        # We can reuse inv_freq directly since it's already in fp32.
        # Build emb: position_ids float
        position_ids = position_ids.to(torch.float32)  # [B, S]
        # inv_freq: [D//2], we need [B, S, D]
        # Create emb by repeating: emb[:, :, :D//2] = position_ids[:, :, None] * inv_freq[:D//2], emb[:, :, D//2:] = same
        # But emb must be [B, S, D]; let's do it via broadcasting:
        B_pos, S_pos = position_ids.shape
        emb = torch.empty((B_pos, S_pos, D), device=device, dtype=torch.float32)
        emb[..., :D // 2] = position_ids.unsqueeze(-1) * inv_freq[:D // 2]  # [B, S, D//2]
        emb[..., D // 2:] = position_ids.unsqueeze(-1) * inv_freq[:D // 2]  # [B, S, D//2]
        cos = torch.cos(emb)  # [B, S, D]
        sin = torch.sin(emb)  # [B, S, D]

        # Apply rotation to normalized query and key
        def apply_rope(x, cos, sin):
            # x: [B, heads, S, D]
            x_fp32 = x.to(torch.float32)
            x1 = x_fp32[..., :D // 2]
            x2 = x_fp32[..., D // 2:]
            # Broadcast cos/sin: [B, S, D] -> [B, 1, S, D] by adding singleton dims
            cos_4d = cos.unsqueeze(1)  # [B, 1, S, D]
            sin_4d = sin.unsqueeze(1)  # [B, 1, S, D]
            rotated = (x_fp32 * cos_4d) + (torch.cat([-x2, x1], dim=-1) * sin_4d)
            return rotated.to(dtype)

        query_rotated = apply_rope(query_norm, cos, sin)
        key_rotated = apply_rope(key_norm, cos, sin)

        # 3) Triton cache update: write rotated rows into key_cache/value_cache at cache_position
        # Grid over (B, num_kv_heads, S)
        grid = (Bk, num_kv_heads, Sk)
        # For key_cache: write key_rotated into key_cache at cache_position[s] for each (b, head, s)
        # We need to flatten src for Triton: key_rotated viewed as [rows, D] where rows = Bk * num_kv_heads * Sk
        key_rot_flat = key_rotated.view(-1, D)
        key_cache_flat = key_cache.view(Bk, num_kv_heads, -1, D).reshape(-1, D)
        copy_rows_to_positions_kernel[grid](key_rot_flat, key_cache_flat, cache_position, D)

        # For value_cache: write value (original value) into value_cache at same positions; rotation doesn't modify value's structure.
        # However, the original code updates value_cache with 'value' not 'value_rotated'. We need to ensure we write value into cache.
        # Create a flattened version of value and copy.
        value_cache_flat = value_cache.view(Bk, num_kv_heads, -1, D).reshape(-1, D)
        # Note: We don't have 'value_rotated' since rotation applies to query/key only. The original code uses 'value' (not rotated) in run().
        # To match original behavior, we copy original value rows into cache at cache_position.
        value_src_flat = value_contig.view(-1, D)
        copy_rows_to_positions_kernel[grid](value_src_flat, value_cache_flat, cache_position, D)

        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
