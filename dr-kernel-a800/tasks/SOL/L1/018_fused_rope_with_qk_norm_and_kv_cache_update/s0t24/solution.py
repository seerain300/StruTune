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
# eps: epsilon for RMSNorm (compile-time constant)
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

# Triton kernel: compute cos and sin for each position using inv_freq[:half_dim].
# Inputs:
#   pos_ptr: int64 positions [rows]
#   inv_ptr: float32 inv_freq[:half_dim] [half_dim]
#   cos_ptr: float32 cos[rows, half_dim]
#   sin_ptr: float32 sin[rows, half_dim]
#   rows: number of positions
#   half_dim: length of cosine/sine vector (e.g., 64)
@triton.jit
def compute_cos_sin_rows_kernel(pos_ptr, inv_ptr, cos_ptr, sin_ptr,
                                rows, half_dim,
                                BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    pos = tl.load(pos_ptr + row_id).to(tl.float32)
    for j in range(0, half_dim, BLOCK_SIZE):
        offs = j + tl.arange(0, BLOCK_SIZE)
        mask = offs < half_dim
        inv = tl.load(inv_ptr + offs, mask=mask, other=0.0)  # fp32
        f = pos * inv  # [BLOCK_SIZE], fp32
        c = tl.cos(f)  # [BLOCK_SIZE], fp32
        s = tl.sin(f)  # [BLOCK_SIZE], fp32
        base = row_id * half_dim + j
        tl.store(cos_ptr + base, c, mask=mask)
        tl.store(sin_ptr + base, s, mask=mask)

# Triton kernel: apply rotation to x using cos/sin per row.
# x: input [rows, head_dim], fp32
# cos: [rows, half_dim], fp32
# sin: [rows, half_dim], fp32
# y: output [rows, head_dim], fp32
@triton.jit
def apply_rotation_rows_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr,
                               rows, head_dim,
                               half_dim: tl.constexpr,
                               BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # Process the first half (0..half_dim-1) and second half (half_dim..head_dim-1)
    for k in range(0, half_dim, BLOCK_SIZE):
        offs = k + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(x_ptr + row_id * head_dim + offs, mask=mask, other=0.0)  # fp32
        first_mask = offs < half_dim
        x_first = tl.where(first_mask, x, 0.0)  # [BLOCK_SIZE]
        x_second = tl.where((~first_mask) & mask, x, 0.0)  # [BLOCK_SIZE]

        # Load cos/sin for this row at frequency indices k..k+BLOCK_SIZE-1
        cos_vals = tl.load(cos_ptr + row_id * half_dim + offs, mask=(offs < half_dim), other=0.0)  # [BLOCK_SIZE]
        sin_vals = tl.load(sin_ptr + row_id * half_dim + offs, mask=(offs < half_dim), other=0.0)  # [BLOCK_SIZE]

        # Compute rotated second half: [-x_second * cos + x_first * sin]
        rotated_second = -x_second * cos_vals + x_first * sin_vals
        # Store into y's second half positions
        y_offs = half_dim + offs
        tl.store(y_ptr + row_id * head_dim + y_offs, rotated_second, mask=mask)

# ModelNew forward: launch Triton kernels for heavy compute, return results
class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes:
        # query: [Bq, Hq, Tq, Dq] (Dq=128)
        # key:   [Bk, Hk, Tk, Dk] (Dk=128)
        # position_ids: [Bq, Tq] (int64)
        # inv_freq: [Dq//2] (float32), here Dq//2 = 64
        Bq, Hq, Tq, Dq = query.shape
        Bk, Hk, Tk, Dk = key.shape
        assert Dq == 128 and Dk == 128, "head_dim must be 128"
        half_dim = Dq // 2  # 64

        # 1) RMSNorm for query (fp32 compute), output fp32
        query_norm = torch.empty((Bq, Hq, Tq, Dq), dtype=torch.float32, device=query.device)
        query_rows = Bq * Hq * Tq
        rmsnorm_rows_kernel[(query_rows,)](
            query.reshape(query_rows, Dq), q_norm_weight.to(torch.float32), query_norm.reshape(query_rows, Dq),
            query_rows, Dq, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )

        # 2) Compute cos/sin for query positions: position_ids is [Bq, Tq]
        pos_q = position_ids[:, :Tq].reshape(-1).to(torch.int64)  # [Bq*Tq]
        inv_half_q = inv_freq[:half_dim].to(torch.float32)       # [64]
        cos_q = torch.empty((Bq * Tq, Dq), dtype=torch.float32, device=query.device)
        sin_q = torch.empty((Bq * Tq, Dq), dtype=torch.float32, device=query.device)
        compute_cos_sin_rows_kernel[(Bq * Tq,)](
            pos_q, inv_half_q, cos_q, sin_q,
            Bq * Tq, half_dim, BLOCK_SIZE=128, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rotated (fp32 compute)
        query_rotated_fp32 = torch.empty((Bq, Hq, Tq, Dq), dtype=torch.float32, device=query.device)
        apply_rotation_rows_kernel[(query_rows,)](
            query_norm.reshape(query_rows, Dq), cos_q, sin_q, query_rotated_fp32.reshape(query_rows, Dq),
            query_rows, Dq, half_dim=half_dim, BLOCK_SIZE=128, num_warps=4
        )
        # Cast to bf16 to match original return type
        query_rotated = query_rotated_fp32.to(torch.bfloat16)

        # Repeat for key: RMSNorm, cos/sin, rotation
        key_norm = torch.empty((Bk, Hk, Tk, Dk), dtype=torch.float32, device=key.device)
        key_rows = Bk * Hk * Tk
        rmsnorm_rows_kernel[(key_rows,)](
            key.reshape(key_rows, Dk), k_norm_weight.to(torch.float32), key_norm.reshape(key_rows, Dk),
            key_rows, Dk, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )

        pos_k = position_ids[:, :Tk].reshape(-1).to(torch.int64)  # [Bk*Tk]
        inv_half_k = inv_freq[:half_dim].to(torch.float32)       # [64]
        cos_k = torch.empty((Bk * Tk, Dk), dtype=torch.float32, device=key.device)
        sin_k = torch.empty((Bk * Tk, Dk), dtype=torch.float32, device=key.device)
        compute_cos_sin_rows_kernel[(Bk * Tk,)](
            pos_k, inv_half_k, cos_k, sin_k,
            Bk * Tk, half_dim, BLOCK_SIZE=128, num_warps=4
        )

        key_rotated_fp32 = torch.empty((Bk, Hk, Tk, Dk), dtype=torch.float32, device=key.device)
        apply_rotation_rows_kernel[(key_rows,)](
            key_norm.reshape(key_rows, Dk), cos_k, sin_k, key_rotated_fp32.reshape(key_rows, Dk),
            key_rows, Dk, half_dim=half_dim, BLOCK_SIZE=128, num_warps=4
        )
        key_rotated = key_rotated_fp32.to(torch.bfloat16)

        # Return results; key_cache and value_cache are returned unchanged
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
