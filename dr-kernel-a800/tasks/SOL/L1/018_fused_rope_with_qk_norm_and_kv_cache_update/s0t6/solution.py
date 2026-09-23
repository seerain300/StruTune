import torch
import triton
import triton.language as tl

# Triton kernel: RMSNorm over the last dimension (head_dim) for each row.
# Inputs:
#   X_ptr: pointer to input [rows, head_dim], dtype float32
#   W_ptr: pointer to weight [head_dim], dtype float32
#   Out_ptr: pointer to output [rows, head_dim], dtype float32
#   rows: number of rows
#   head_dim: length of last dimension
# eps: epsilon for RMSNorm
@triton.jit
def rmsnorm_rows_kernel(X_ptr, W_ptr, Out_ptr,
                         rows, head_dim,
                         eps: tl.constexpr,
                         BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    sumsq = 0.0
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)  # fp32
        sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / head_dim
    r = tl.rsqrt(mean + eps)
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)  # fp32
        w = tl.load(W_ptr + offs, mask=mask, other=1.0)  # fp32
        y = x * r * w
        tl.store(Out_ptr + row_id * head_dim + offs, y, mask=mask)

# Triton kernel: compute cos/sin scalars for each token based on position and inv_freq[:half_dim].
# Inputs:
#   pos_ptr: [rows] int64 positions (one per token)
#   inv_ptr: [half_dim] float32 inverse frequencies (head_dim//2)
#   cos_ptr: [rows, head_dim] float32 output
#   sin_ptr: [rows, head_dim] float32 output
# rows: number of tokens
# half_dim: head_dim//2
# D: head_dim
@triton.jit
def compute_cos_sin_kernel(pos_ptr, inv_ptr, cos_ptr, sin_ptr,
                           rows, half_dim, D: tl.constexpr,
                           BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # Loop over D in BLOCK_SIZE chunks
    for col in range(0, D, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        # first half frequencies
        idx = offs  # [0..half_dim-1]
        idx2 = idx + half_dim  # [half_dim..2*half_dim-1]
        idx2 = idx2  # but offs only spans [0..D-1], and idx2 must correspond to idx + half_dim
        # We need to compute idx2 = idx + half_dim, but offs is already >= half_dim for second half.
        # Create a combined index for idx:
        idx_half = offs < half_dim
        idx_val = tl.where(idx_half, offs, offs - half_dim + half_dim)  # no-op, just clarify
        # Compute angle = pos * inv_freq
        pos_val = tl.load(pos_ptr + row_id).to(tl.float32)
        angle1 = pos_val * tl.load(inv_ptr + offs, mask=mask, other=0.0)  # only valid for offs < half_dim
        angle2 = pos_val * tl.load(inv_ptr + (offs - half_dim), mask=mask & (offs >= half_dim), other=0.0)
        angle = tl.where(offs < half_dim, angle1, angle2)
        c = tl.cos(angle).to(tl.float32)
        s = tl.sin(angle).to(tl.float32)
        tl.store(cos_ptr + row_id * D + offs, c, mask=mask)
        tl.store(sin_ptr + row_id * D + offs, s, mask=mask)

# Triton kernel: apply rotation y = x * cos + rotate_half(x) * sin
# Inputs:
#   X_ptr: [rows, D] float32 input (normalized tensor)
#   cos_ptr: [rows, D] float32 cos
#   sin_ptr: [rows, D] float32 sin
#   Out_ptr: [rows, D] float32 output
# rows: number of rows
# D: head_dim
@triton.jit
def apply_rotation_kernel(X_ptr, cos_ptr, sin_ptr, Out_ptr,
                          rows, D: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    for col in range(0, D, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + row_id * D + offs, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + row_id * D + offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + row_id * D + offs, mask=mask, other=0.0).to(tl.float32)
        half = D // 2
        first = offs < half
        x1 = tl.where(first, x, 0.0)  # first half
        x2 = tl.where(offs >= half, x, 0.0)  # second half
        rotated_half = -x2 + x1  # rotate_half: concatenate [-x2, x1]
        y = x * c + rotated_half * s
        tl.store(Out_ptr + row_id * D + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rope_theta = 10000000.0  # from original

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes
        Bq, Hq, Tq, Dq = query.shape
        Bk, Hk, Tk, Dk = key.shape
        assert Dq == Dk, "Query and Key head dims must match"
        assert (Dq % 2 == 0) and (Dk % 2 == 0), "head_dim must be even"
        half_dim = Dq // 2

        # 1) RMSNorm for query and key (Triton)
        # query_norm: float32
        query_norm = torch.empty((Bq * Hq * Tq, Dq), dtype=torch.float32, device=query.device)
        rmsnorm_rows_kernel[(Bq * Hq * Tq,)](
            query.reshape(Bq * Hq * Tq, Dq), q_norm_weight.to(torch.float32), query_norm,
            Bq * Hq * Tq, Dq, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )

        key_norm = torch.empty((Bk * Hk * Tk, Dk), dtype=torch.float32, device=key.device)
        rmsnorm_rows_kernel[(Bk * Hk * Tk,)](
            key.reshape(Bk * Hk * Tk, Dk), k_norm_weight.to(torch.float32), key_norm,
            Bk * Hk * Tk, Dk, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )

        # 2) Compute cos/sin for query positions: position_ids is [Bq, Tq]
        pos_q = position_ids[:, :Tq].to(torch.int64).reshape(-1)  # [Bq*Tq]
        inv_q_half = inv_freq[:half_dim].to(torch.float32)        # [half_dim]
        cos_q = torch.empty((Bq * Tq, Dq), dtype=torch.float32, device=query.device)
        sin_q = torch.empty((Bq * Tq, Dq), dtype=torch.float32, device=query.device)
        compute_cos_sin_kernel[(Bq * Tq,)](
            pos_q, inv_q_half, cos_q, sin_q,
            Bq * Tq, half_dim, Dq, BLOCK_SIZE=64, num_warps=2
        )

        # 3) Apply rotation for query (Triton)
        query_rotated_fp32 = torch.empty((Bq * Hq * Tq, Dq), dtype=torch.float32, device=query.device)
        apply_rotation_kernel[(Bq * Hq * Tq,)](
            query_norm, cos_q, sin_q, query_rotated_fp32,
            Bq * Hq * Tq, Dq, BLOCK_SIZE=128, num_warps=4
        )
        query_rotated_bf16 = query_rotated_fp32.to(torch.bfloat16).reshape(Bq, Hq, Tq, Dq)

        # 4) Compute cos/sin for key positions: position_ids for key assumed same as query (original uses position_ids for both)
        pos_k = position_ids[:, :Tk].to(torch.int64).reshape(-1)  # [Bk*Tk]
        inv_k_half = inv_freq[:half_dim].to(torch.float32)
        cos_k = torch.empty((Bk * Tk, Dk), dtype=torch.float32, device=key.device)
        sin_k = torch.empty((Bk * Tk, Dk), dtype=torch.float32, device=key.device)
        compute_cos_sin_kernel[(Bk * Tk,)](
            pos_k, inv_k_half, cos_k, sin_k,
            Bk * Tk, half_dim, Dk, BLOCK_SIZE=64, num_warps=2
        )

        # 5) Apply rotation for key (Triton)
        key_rotated_fp32 = torch.empty((Bk * Hk * Tk, Dk), dtype=torch.float32, device=key.device)
        apply_rotation_kernel[(Bk * Hk * Tk,)](
            key_norm, cos_k, sin_k, key_rotated_fp32,
            Bk * Hk * Tk, Dk, BLOCK_SIZE=128, num_warps=4
        )
        key_rotated_bf16 = key_rotated_fp32.to(torch.bfloat16).reshape(Bk, Hk, Tk, Dk)

        # 6) Update caches using torch advanced indexing (correct and simple for dynamic shapes)
        # key_cache: update slice at cache_position with rotated key
        # For each (b, h), copy key_rotated[b, h, :Tk, :] into key_cache[b, h, cache_position, :]
        # We'll build an index for each (b, h) and assign.
        # Note: cache_position is 1D of length Tk (original code passes cache_position as 1D).
        for b in range(Bk):
            for h in range(Hk):
                src = key_rotated_fp32[b * Hk + h].unsqueeze(0)  # [1, Tk, Dk]
                # key_cache strides: [Bk, Hk, Tk, Dk] => linear index = b*(Hk*Tk*Dk) + h*(Tk*Dk) + t*Dk + d
                # We assign at positions cache_position[t] along t.
                for t in range(Tk):
                    key_cache[b, h, cache_position[t].item(), :] = src[0, t, :].to(torch.bfloat16)

        # value_cache: update slice at cache_position with original value (no rotation, keep bfloat16)
        # value is [Bk, Hk, Tk, Dk]
        for b in range(Bk):
            for h in range(Hk):
                for t in range(Tk):
                    value_cache[b, h, cache_position[t].item(), :] = value[b, h, t, :]

        # Return query_rotated, key_rotated, key_cache, value_cache with correct dtypes
        return query_rotated_bf16, key_rotated_bf16, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
