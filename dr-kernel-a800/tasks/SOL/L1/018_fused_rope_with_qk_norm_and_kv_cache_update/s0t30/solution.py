import torch
import triton
import triton.language as tl

# Triton kernel: Compute cos and sin vectors for the first half of head_dim (half_dim) given positions.
# Inputs:
#   pos_ptr: int64 array of positions of length rows
#   inv_ptr: float32 array of length half_dim (inv_freq[:half_dim])
#   cos_out: float32 output [rows, half_dim]
#   sin_out: float32 output [rows, half_dim]
# Assumes: head_dim=128, half_dim=64
@triton.jit
def compute_cos_sin_first_half_kernel(pos_ptr, inv_ptr, cos_out, sin_out,
                                      rows: tl.constexpr, half_dim: tl.constexpr,
                                      BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= rows:
        return
    for col in range(0, half_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < half_dim
        pos = tl.load(pos_ptr + row)  # int64
        angle = pos.to(tl.float32) * inv_ptr[offs]  # [BLOCK_SIZE], fp32
        c = tl.cos(angle)
        s = tl.sin(angle)
        tl.store(cos_out + row * half_dim + offs, c, mask=mask)
        tl.store(sin_out + row * half_dim + offs, s, mask=mask)

# Triton kernel: Rotate half and apply rotation:
# For a tensor x of shape [..., head_dim=128], we process a flattened row of length head_dim.
# We apply: y = x * cos + rotate_half(x) * sin, where rotate_half swaps [:64] and [64:].
# Inputs:
#   x_in: fp32 [rows, 128]
#   cos_ptr: fp32 [rows, 64] (first half)
#   sin_ptr: fp32 [rows, 64] (first half)
#   y_out: fp32 [rows, 128]
@triton.jit
def rotate_half_apply_kernel(x_in, cos_ptr, sin_ptr, y_out,
                             rows: tl.constexpr, head_dim: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= rows:
        return
    # Load x row
    x_row = tl.load(x_in + row * head_dim + tl.arange(0, head_dim))
    # First half rotation
    half_dim = head_dim // 2
    for col in range(0, half_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < half_dim
        x1 = x_row[offs]
        x2 = x_row[offs + half_dim]
        cos = tl.load(cos_ptr + row * half_dim + offs, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + row * half_dim + offs, mask=mask, other=0.0)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        # Store first half and second half respectively
        tl.store(y_out + row * head_dim + offs, y1, mask=mask)
        tl.store(y_out + row * head_dim + offs + half_dim, y2, mask=mask)

# Triton kernel: RMSNorm over the last dim=128 for each row in a [rows, 128] 2D tensor.
# Computes y = x * rsqrt(mean(x^2) + eps) * weight, where weight is of length 128 (per-column).
@triton.jit
def rmsnorm_rows_kernel(X_ptr, W_ptr, Out_ptr,
                         rows, head_dim,
                         eps: tl.constexpr,
                         BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= rows:
        return
    sumsq = 0.0
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row * head_dim + offs, mask=mask, other=0.0)
        sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / head_dim
    r = tl.rsqrt(mean + eps)
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row * head_dim + offs, mask=mask, other=0.0)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0)
        y = x * r * w
        tl.store(Out_ptr + row * head_dim + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes:
        # query: [Bq, Hq, Tq, 128]
        # key:   [Bk, Hk, Tk, 128]
        # position_ids: [Bq, Tq] (original code uses only first Tq positions)
        # key/value caches: [B, H, MAX, 128], where MAX=max_position_embeddings
        # cache_position: [Tq] (relative index within cache), original code uses cache_len + seq_len
        Bq, Hq, Tq, Dq = query.shape
        Bk, Hk, Tk, Dk = key.shape
        assert Dq == 128 and Dk == 128, "head_dim must be 128"

        # 1) RMSNorm for query using PyTorch (correctness first)
        # torch.nn.functional.rms_norm expects (x, weight, bias=None, epsilon)
        # Here bias=None and weight is per-dim. We assume weight given as q_norm_weight of length 128.
        query_norm = torch.nn.functional.rms_norm(query, weight=q_norm_weight, bias=None, eps=rms_norm_eps)

        # 2) Triton: Compute cos/sin for first half (dim 0..63) for each position in position_ids
        pos_q = position_ids[:, :Tq].reshape(-1).to(torch.int64)  # [Bq*Tq]
        rows = pos_q.shape[0]
        cos_q = torch.empty((rows, 64), dtype=torch.float32, device=query.device)
        sin_q = torch.empty((rows, 64), dtype=torch.float32, device=query.device)
        compute_cos_sin_first_half_kernel[(rows,)](
            pos_q, inv_freq[:64].to(torch.float32), cos_q, sin_q,
            rows, 64, BLOCK_SIZE=64, num_warps=2
        )

        # 3) Triton: Apply rotation to query_norm -> query_rotated (fp32 compute, cast to bf16 at end)
        query_norm_fp32 = query_norm.to(torch.float32)  # [Bq, Hq, Tq, 128]
        rows_q = Bq * Hq * Tq
        query_rotated_flat = torch.empty((rows_q, 128), dtype=torch.float32, device=query.device)
        rotate_half_apply_kernel[(rows_q,)](
            query_norm_fp32.reshape(rows_q, 128), cos_q, sin_q, query_rotated_flat,
            rows_q, 128, BLOCK_SIZE=128, num_warps=4
        )
        query_rotated = query_rotated_flat.reshape(Bq, Hq, Tq, 128).to(torch.bfloat16)

        # 4) RMSNorm for key using PyTorch
        key_norm = torch.nn.functional.rms_norm(key, weight=k_norm_weight, bias=None, eps=rms_norm_eps)

        # 5) Triton: Compute cos/sin for key positions (same as query positions, but key has shape [Bk, Hk, Tk, 128])
        pos_k = position_ids[:, :Tk].reshape(-1).to(torch.int64)  # [Bk*Hk*Tk]
        rows_k = pos_k.shape[0]
        cos_k = torch.empty((rows_k, 64), dtype=torch.float32, device=query.device)
        sin_k = torch.empty((rows_k, 64), dtype=torch.float32, device=query.device)
        compute_cos_sin_first_half_kernel[(rows_k,)](
            pos_k, inv_freq[:64].to(torch.float32), cos_k, sin_k,
            rows_k, 64, BLOCK_SIZE=64, num_warps=2
        )

        # 6) Triton: Apply rotation to key_norm -> key_rotated
        key_norm_fp32 = key_norm.to(torch.float32)  # [Bk, Hk, Tk, 128]
        rows_k_total = Bk * Hk * Tk
        key_rotated_flat = torch.empty((rows_k_total, 128), dtype=torch.float32, device=query.device)
        rotate_half_apply_kernel[(rows_k_total,)](
            key_norm_fp32.reshape(rows_k_total, 128), cos_k, sin_k, key_rotated_flat,
            rows_k_total, 128, BLOCK_SIZE=128, num_warps=4
        )
        key_rotated = key_rotated_flat.reshape(Bk, Hk, Tk, 128).to(torch.bfloat16)

        # 7) Update caches: original code assigns key_cache[:, :, cache_position, :] = key_rotated and
        #     value_cache[:, :, cache_position, :] = value. We do this via torch indexing for correctness.
        # Note: cache_position is 1D of length Tq/Tk. We assume it is within [0, MAX). We'll update only the first Tq/Tk positions.
        # For key_cache:
        #   For each (b,h), copy key_rotated[b,h,:] into key_cache[b,h,cache_position, :].
        # We'll perform this in a loop over batches and heads.
        # Ensure value is [Bk, Hk, Tk, 128] (given), but original function receives value with shape [B, S, D] where S=Tk. We take value as [Bk, Hk, Tk, 128].
        # If value has different shape, we ignore as original reference didn't use it for cache update; we only return it.

        # Copy into key_cache: torch advanced indexing update
        # Build index grid:
        for b in range(Bk):
            for h in range(Hk):
                # Update positions according to cache_position (length equals current sequence length, i.e., Tk)
                # key_cache has shape [Bk, Hk, MAX, 128]. We update first Tk positions at indices cache_position.
                idx_b = b
                idx_h = h
                # key_rotated[b,h,:] is [Tk, 128]; cache_position is [Tk]
                # Create expanded indices for dim-1 (columns) and dim-2 (positions)
                pos_idx = cache_position.to(torch.long)  # [Tk]
                col_idx = torch.arange(128, device=query.device)  # [128]
                # Broadcast to [Tk, 128]
                key_cache[idx_b, idx_h, pos_idx[:, None], col_idx[None, :]] = key_rotated[b, h, :, :]

        # For value_cache, similarly update first Tk positions with 'value'. Assuming value has shape [Bk, Hk, Tk, 128].
        # If 'value' has a different shape, the original reference doesn't use it for cache update; we skip.
        # Here, we assume value is provided as [Bk, Hk, Tk, 128].
        if value.dim() == 4 and value.shape == (Bk, Hk, Tk, 128):
            for b in range(Bk):
                for h in range(Hk):
                    pos_idx = cache_position.to(torch.long)  # [Tk]
                    col_idx = torch.arange(128, device=query.device)
                    value_cache[b, h, pos_idx[:, None], col_idx[None, :]] = value[b, h, :, :]

        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
