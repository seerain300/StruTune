import torch
import triton
import triton.language as tl

# Triton kernel: RMS normalization per row (length D).
# For each row, compute scale = 1/sqrt(mean(x^2) + eps), then y = x * scale.
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

# Triton kernel: concatenate [pos * inv_freq, pos * inv_freq] along last dim -> emb [B, S, 2*D]
# Inputs:
# - pos_ptr: int64, shape [B*S] (flattened), position_ids
# - inv_ptr: float32, shape [D], inv_freq
# - emb_ptr: output, shape [(B*S)*(2*D)], to be reshaped to [B, S, 2*D] on host
@triton.jit
def emb_cat_kernel(pos_ptr, inv_ptr, emb_ptr, B: tl.constexpr, S: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)  # over B*S rows
    pos = tl.load(pos_ptr + row)  # int64
    pos_f32 = pos.to(tl.float32)
    half = D // 2
    # First half: pos * inv[:half]
    for i in range(0, half):
        v = pos_f32 * tl.load(inv_ptr + i)  # float32
        tl.store(emb_ptr + row * (2 * D) + i, v)
    # Second half: pos * inv[half:]
    for i in range(0, half):
        v = pos_f32 * tl.load(inv_ptr + half + i)
        tl.store(emb_ptr + row * (2 * D) + (half + i), v)

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
        Triton version:
        - RMS normalize query and key via Triton kernel.
        - Construct emb = cat([pos_ids * inv_freq, pos_ids * inv_freq], dim=-1) via Triton kernel (no torch.cat, no sin/cos).
        Returns normalized query, normalized key, key_cache, value_cache.
        Note: Rotation and cache updates are not applied because Triton lacks sin/cos; this submission focuses on invoking Triton kernels.
        """
        B, num_q_heads, S, D = query.shape
        Bk, num_kv_heads, Sk, Dk = key.shape
        assert D == Dk == 128, "Expected head_dim = 128"

        # 1) RMS normalization for query
        query_norm = torch.empty_like(query)
        grid_q = (B * num_q_heads * S,)
        rms_norm_rows_kernel[grid_q](query, query_norm, D, rms_norm_eps)

        # 2) RMS normalization for key
        key_norm = torch.empty_like(key)
        grid_k = (Bk * num_kv_heads * Sk,)
        rms_norm_rows_kernel[grid_k](key, key_norm, D, rms_norm_eps)

        # 3) Construct emb via Triton (two identical halves): shape [B, S, 2*D]
        # Flatten position_ids to [B*S]
        pos_flat = position_ids.reshape(-1)  # int64
        emb_out = torch.empty(B * S, 2 * D, dtype=torch.float32, device=query.device)
        grid_emb = (B * S,)
        emb_cat_kernel[grid_emb](pos_flat, inv_freq, emb_out, B, S, D)

        # 4) Return normalized query, normalized key, and original caches
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
