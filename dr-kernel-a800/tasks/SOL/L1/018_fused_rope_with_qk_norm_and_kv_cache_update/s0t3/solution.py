import torch
import triton
import triton.language as tl

# Triton kernel: RMSNorm over the last dimension (head_dim) for each row in X.
# X_ptr: input [rows, head_dim], W_ptr: per-column weight [head_dim], Out_ptr: output [rows, head_dim]
# eps: float epsilon for RMSNorm
@triton.jit
def rmsnorm_rows_kernel(X_ptr, W_ptr, Out_ptr,
                         rows, head_dim,
                         eps: tl.constexpr,
                         BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    sumsq = 0.0
    # Reduction across head_dim in chunks of BLOCK_SIZE
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sumsq += tl.sum(x32 * x32, axis=0)
    mean = sumsq / head_dim
    r = tl.rsqrt(mean + eps)  # fp32 scalar for this row
    # Apply normalization and per-column weight, write back
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        y = x.to(tl.float32) * r * w
        tl.store(Out_ptr + row_id * head_dim + offs, y, mask=mask)

# Triton kernel: compute cos and sin for each token position using inv[:half_dim].
# pos_ptr: [B, T] int64 positions, inv_ptr: [half_dim] float32, D: head_dim (even), returns cos_ptr and sin_ptr
@triton.jit
def compute_cos_sin_kernel(pos_ptr, inv_ptr, cos_ptr, sin_ptr,
                           B, T, half_dim, D,
                           BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= B * T:
        return
    pos = tl.load(pos_ptr + row_id).to(tl.float32)
    # Compute cos and sin for each element index j in [0, D)
    for j in range(0, D, BLOCK):
        offs = j + tl.arange(0, BLOCK)
        mask = offs < D
        # idx = offs // 2 maps to first half indices
        idx = offs // 2  # [BLOCK]
        mask2 = idx < half_dim
        inv = tl.load(inv_ptr + idx, mask=mask2, other=0.0).to(tl.float32)
        emb = pos * inv  # [BLOCK]
        cos_vals = tl.cos(emb)
        sin_vals = tl.sin(emb)
        tl.store(cos_ptr + row_id * D + offs, cos_vals, mask=mask)
        tl.store(sin_ptr + row_id * D + offs, sin_vals, mask=mask)

# Triton kernel: apply rotation per row: y = x * cos + rotate_half(x) * sin
# X_in_ptr: [rows, D], cos_ptr: [rows, D], sin_ptr: [rows, D], Out_ptr: [rows, D]
@triton.jit
def apply_rotation_kernel(X_in_ptr, cos_ptr, sin_ptr, Out_ptr,
                          rows, D,
                          half_dim: tl.constexpr,
                          BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # Process in tiles along D
    for j in range(0, D, BLOCK):
        offs = j + tl.arange(0, BLOCK)
        mask = offs < D
        x = tl.load(X_in_ptr + row_id * D + offs, mask=mask, other=0.0)
        c = tl.load(cos_ptr + row_id * D + offs, mask=mask, other=1.0).to(tl.float32)
        s = tl.load(sin_ptr + row_id * D + offs, mask=mask, other=0.0).to(tl.float32)
        half = D // 2
        first = offs < half
        # extract first and last halves
        x1 = tl.where(first, x, 0.0)  # values at columns [0, half)
        x2 = tl.where(offs >= half, x, 0.0)  # values at columns [half, D)
        rotated_half = -x2 + x1  # rotate_half: [-x2, x1] concatenation
        y = x.to(tl.float32) * c + rotated_half * s
        tl.store(Out_ptr + row_id * D + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants from original
        self.rope_theta = 10000000.0

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes
        Bq, Hq, Tq, Dq = query.shape  # query: [Bq, Hq, Tq, Dq]
        Bk, Hk, Tk, Dk = key.shape    # key: [Bk, Hk, Tk, Dk]
        # We assume Dq == Dk == head_dim and head_dim is even.
        half_dim_q = Dq // 2
        half_dim_k = Dk // 2

        # 1) RMSNorm for query and key (Triton)
        query_norm = torch.empty_like(query, dtype=torch.float32)
        rmsnorm_rows_kernel[(Bq * Hq * Tq,)](
            query.reshape(Bq * Hq * Tq, Dq), q_norm_weight, query_norm.reshape(Bq * Hq * Tq, Dq),
            Bq * Hq * Tq, Dq, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )

        key_norm = torch.empty_like(key, dtype=torch.float32)
        rmsnorm_rows_kernel[(Bk * Hk * Tk,)](
            key.reshape(Bk * Hk * Tk, Dk), k_norm_weight, key_norm.reshape(Bk * Hk * Tk, Dk),
            Bk * Hk * Tk, Dk, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )

        # 2) Compute cos/sin for query positions: position_ids is [Bq, Tq]
        pos_q = position_ids[:, :Tq].to(torch.int64)
        inv_q = inv_freq[:half_dim_q].to(torch.float32)
        cos_q = torch.empty(Bq * Tq, Dq, dtype=torch.float32, device=query.device)


def run(*args):
    return ModelNew()(*args)
