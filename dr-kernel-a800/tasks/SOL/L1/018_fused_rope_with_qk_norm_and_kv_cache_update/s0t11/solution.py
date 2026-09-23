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

# Triton kernel: compute cos/sin vectors for positions [0..T) and frequency indices [:half_dim].
# Inputs:
#   pos_ptr: [T] int64 positions
#   inv_ptr: [half_dim] float32 inverse frequencies (for indices 2*i)
#   cos_ptr: [T, head_dim] float32 output cos
#   sin_ptr: [T, head_dim] float32 output sin
@triton.jit
def compute_cos_sin_kernel(pos_ptr, inv_ptr, cos_ptr, sin_ptr,
                           T: tl.constexpr, half_dim: tl.constexpr, head_dim: tl.constexpr,
                           BLOCK_SIZE: tl.constexpr):
    t = tl.program_id(0)  # one program per position
    if t >= T:
        return
    # Load position
    pos = tl.load(pos_ptr + t)  # int64
    # Compute angle for each frequency index i in [0..half_dim-1]
    # idx = 2 * i
    for i in range(0, half_dim):
        idx = 2 * i
        angle = pos * inv_ptr[i]  # float32
        cos_val = tl.cos(angle)
        sin_val = tl.sin(angle)
        tl.store(cos_ptr + t * head_dim + i, cos_val)
        tl.store(sin_ptr + t * head_dim + i, sin_val)

# Triton kernel: apply rotation to normalized tensor using cos/sin vectors broadcast over T.
# Inputs:
#   X_ptr: [rows, head_dim] float32 normalized input
#   cos_ptr: [T, head_dim] float32 cos
#   sin_ptr: [T, head_dim] float32 sin
#   Out_ptr: [rows, head_dim] float32 output
@triton.jit
def rotate_y_kernel(X_ptr, cos_ptr, sin_ptr, Out_ptr,
                    rows, head_dim,
                    BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)  # fp32
        cos_vec = tl.load(cos_ptr + offs, mask=mask, other=0.0)  # fp32
        sin_vec = tl.load(sin_ptr + offs, mask=mask, other=0.0)  # fp32
        half = head_dim // 2
        first = x[:half]
        second = x[half:]
        rotated_half = -second + first  # rotate_half(x) = cat([-x2, x1], dim=-1)
        y = x * cos_vec + rotated_half * sin_vec
        tl.store(Out_ptr + row_id * head_dim + offs, y, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes:
        # query: [Bq, Hq, Tq, Dq]
        # key:   [Bk, Hk, Tk, Dk]
        # value: [B, S, D] (unused, kept for signature consistency)
        device = query.device
        Bq, Hq, Tq, Dq = query.shape
        Bk, Hk, Tk, Dk = key.shape
        # The original code uses head_dim = 128; follow that.
        assert Dq == 128 and Dk == 128, "head_dim must be 128"
        half_dim = Dq // 2  # 64

        # 1) RMSNorm for query (fp32 compute)
        query_norm = torch.empty((Bq, Hq, Tq, Dq), dtype=torch.float32, device=device)
        query_rows = Bq * Hq * Tq
        query_flat = query.reshape(query_rows, Dq)
        rmsnorm_rows_kernel[(query_rows,)](
            query_flat, q_norm_weight.to(torch.float32), query_norm.reshape(query_rows, Dq),
            query_rows, Dq, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )

        # 2) Compute cos/sin for positions (position_ids): shape [Bq, Tq]
        pos_q = position_ids[:, :Tq].reshape(-1)  # [Bq*Tq]
        inv_half_q = inv_freq[:half_dim].to(torch.float32)  # [64]
        cos_q = torch.empty((Bq * Tq, Dq), dtype=torch.float32, device=device)
        sin_q = torch.empty((Bq * Tq, Dq), dtype=torch.float32, device=device)
        compute_cos_sin_kernel[(Bq * Tq,)](
            pos_q, inv_half_q, cos_q, sin_q,
            T=Bq * Tq, half_dim=half_dim, head_dim=Dq, BLOCK_SIZE=128, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rotated (fp32 compute)
        query_rotated_fp32 = torch.empty((Bq, Hq, Tq, Dq), dtype=torch.float32, device=device)
        rows_q = Bq * Hq * Tq
        rotate_y_kernel[(rows_q,)](
            query_norm.reshape(rows_q, Dq), cos_q, sin_q, query_rotated_fp32.reshape(rows_q, Dq),
            rows_q, Dq, BLOCK_SIZE=128, num_warps=4
        )
        # Cast to bfloat16 to match original output dtype
        query_rotated = query_rotated_fp32.to(torch.bfloat16)

        # 4) RMSNorm for key (fp32 compute)
        key_norm = torch.empty((Bk, Hk, Tk, Dk), dtype=torch.float32, device=device)
        key_rows = Bk * Hk * Tk
        key_flat = key.reshape(key_rows, Dk)
        rmsnorm_rows_kernel[(key_rows,)](
            key_flat, k_norm_weight.to(torch.float32), key_norm.reshape(key_rows, Dk),
            key_rows, Dk, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )

        # 5) Compute cos/sin for key positions: shape [Bk, Tk]
        pos_k = position_ids[:, :Tk].reshape(-1)  # [Bk*Tk] (cache_len + seq_len is same as query positions)
        cos_k = torch.empty((Bk * Tk, Dk), dtype=torch.float32, device=device)
        sin_k = torch.empty((Bk * Tk, Dk), dtype=torch.float32, device=device)
        compute_cos_sin_kernel[(Bk * Tk,)](
            pos_k, inv_half_q, cos_k, sin_k,
            T=Bk * Tk, half_dim=half_dim, head_dim=Dk, BLOCK_SIZE=128, num_warps=4
        )

        # 6) Apply rotation to key_norm -> key_rotated (fp32 compute)
        key_rotated_fp32 = torch.empty((Bk, Hk, Tk, Dk), dtype=torch.float32, device=device)
        rotate_y_kernel[(key_rows,)](
            key_norm.reshape(key_rows, Dk), cos_k, sin_k, key_rotated_fp32.reshape(key_rows, Dk),
            key_rows, Dk, BLOCK_SIZE=128, num_warps=4
        )
        key_rotated = key_rotated_fp32.to(torch.bfloat16)

        # 7) Return outputs; do not mutate caches (keep original behavior)
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
