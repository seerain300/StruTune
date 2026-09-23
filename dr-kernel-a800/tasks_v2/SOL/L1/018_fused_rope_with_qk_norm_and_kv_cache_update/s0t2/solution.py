import torch
import triton
import triton.language as tl

# Triton kernel: RMSNorm over the last dimension (head_dim) for each row of a 3D tensor X[B, H, T, D].
# We flatten X into [rows, D] where rows = B*H*T. The kernel operates on per-row vectors of length D.
@triton.jit
def rmsnorm_rows_kernel(X_ptr, W_ptr, Out_ptr,
                         rows, head_dim,
                         eps: tl.constexpr,
                         BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # Accumulate sum of squares in fp32
    sumsq = 0.0
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / head_dim
    r = tl.rsqrt(mean + eps)  # fp32
    # Write normalized output: y = W * x * r
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        y = x.to(tl.float32) * r * w
        tl.store(Out_ptr + row_id * head_dim + offs, y, mask=mask)

# Triton kernel: compute cos/sin scalars for each token based on position and inv_freq[:half_dim].
# Inputs:
#   pos_ptr: [B, T] int64 positions
#   inv_ptr: [half_dim] float32 inverse frequencies (length head_dim//2)
#   cos_ptr: [B, T, head_dim] float32
#   sin_ptr: [B, T, head_dim] float32
# We write cos and sin as length head_dim by duplicating half_dim entries: [a0, a1, ..., a(h-1), b0, b1, ..., b(h-1)].
@triton.jit
def compute_cos_sin_kernel(pos_ptr, inv_ptr, cos_ptr, sin_ptr,
                           B, T, half_dim, head_dim):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return
    pos = tl.load(pos_ptr + b * T + t).to(tl.float32)  # pos as float
    for i in range(0, half_dim):
        inv_i = tl.load(inv_ptr + i)  # float32
        angle = pos * inv_i  # float32
        cos_val = tl.cos(angle)
        sin_val = tl.sin(angle)
        # Write cos and sin repeated across head_dim
        base = b * (T * head_dim) + t * head_dim
        # First half: cos
        tl.store(cos_ptr + base + i, cos_val)
        tl.store(sin_ptr + base + i, sin_val)
        # Second half: cos repeated
        tl.store(cos_ptr + base + i + half_dim, cos_val)
        tl.store(sin_ptr + base + i + half_dim, sin_val)

# Triton kernel: apply rotation y = x * cos + rotate_half(x) * sin
# Input X: [rows, head_dim], cos_ptr/sin_ptr: [B, T, head_dim] laid out contiguously in rows order.
# We decode (b, head, t) from row_id to fetch correct cos/sin for that token.
# Output Out: [rows, head_dim]
@triton.jit
def apply_rotation_kernel(X_ptr, Cos_ptr, Sin_ptr, Out_ptr,
                          rows, head_dim, half_dim,
                          B, T,
                          cache_start: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # Decode (b, head, t) from row_id: rows = B * head_dim * T
    t = row_id % T
    tmp = row_id // T
    head = tmp % head_dim
    b = tmp // head_dim
    # Compute base offset for cos/sin vectors of length head_dim for this (b, t)
    base = b * (T * head_dim) + t * head_dim
    # Load cos and sin scalars for this row
    for i in range(0, half_dim):
        cos_val = tl.load(Cos_ptr + base + i)
        sin_val = tl.load(Sin_ptr + base + i)
    # Apply rotation across head_dim
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        rotate = tl.cat([-x2, x1], axis=0)  # rotate_half as [head_dim]
        y = x.to(tl.float32) * cos_val + rotate.to(tl.float32) * sin_val
        tl.store(Out_ptr + row_id * head_dim + offs, y, mask=mask)

# Triton kernel: copy from source [rows_in, D] to destination [B_out, H_out, T_out, D] using provided strides,
# writing into positions cache_position (length T_out). We receive cache_position as int64 and map per row.
@triton.jit
def copy_cache_rows_kernel(From_ptr, To_ptr, pos_ptr, rows_out, B_out, H_out, T_out, D,
                           stride_b_out, stride_h_out, stride_t_out, stride_d_out):
    row_id = tl.program_id(0)
    if row_id >= rows_out:
        return
    # Decode (b_out, h_out, t_out) from row_id: rows_out = B_out * H_out * T_out
    t_out = row_id % T_out
    tmp = row_id // T_out
    h_out = tmp % H_out
    b_out = tmp // H_out
    # Load cache position for this token
    pos = tl.load(pos_ptr + t_out).to(tl.int32)
    # Compute source row corresponding to (b_out, h_out, t_out) in From_ptr which is laid out as [rows_in, D]
    # We don't have mapping here; assume From_ptr is contiguous per (b,h,t). Triton cannot decode without metadata.
    # Therefore, we will pass From_ptr as [rows_out, D] where rows_out = B_out * H_out * T_out and each row maps to itself.
    # In practice, From_ptr is flattened; so row_id is the source row. If From_ptr has more rows, we need mapping.
    # Given we launch with rows_out equal to the number of rows to copy, we can proceed:
    for d in range(0, D):
        src = tl.load(From_ptr + row_id * D + d)
        to_offset = b_out * stride_b_out + h_out * stride_h_out + pos * stride_t_out + d * stride_d_out
        tl.store(To_ptr + to_offset, src)

class ModelNew(torch.nn.Module):
    def forward(self, query: torch.Tensor,
                key: torch.Tensor, value: torch.Tensor,
                position_ids: torch.Tensor,
                key_cache: torch.Tensor, value_cache: torch.Tensor,
                cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor,
                rms_norm_eps: float):
        # Ensure all tensors are CUDA
        assert query.is_cuda and key.is_cuda and value.is_cuda and position_ids.is_cuda \
               and key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda \
               and q_norm_weight.is_cuda and k_norm_weight.is_cuda and inv_freq.is_cuda, "All tensors must be on CUDA."
        Bq, Hq, Tq, D = query.shape
        Bk, Hk, Tk, Dk = key.shape
        assert D == Dk and D == 128, "head_dim must be 128."
        half_dim = D // 2

        # Prepare weight vectors (fp32)
        q_w = q_norm_weight.to(torch.float32).contiguous()  # [D], fp32
        k_w = k_norm_weight.to(torch.float32).contiguous()  # [D], fp32

        # Launch RMSNorm for query: rows = Bq * Hq * Tq
        query_norm = torch.empty(Bq * Hq * Tq, D, dtype=torch.float32, device=query.device)
        rmsnorm_rows_kernel[(Bq * Hq * Tq,)](
            query.reshape(Bq * Hq * Tq, D), q_w, query_norm,
            Bq * Hq * Tq, D,
            eps=rms_norm_eps,
            BLOCK_SIZE=128, num_warps=4
        )
        query_norm = query_norm.reshape(Bq, Hq, Tq, D)

        # Launch RMSNorm for key: rows = Bk * Hk * Tk
        key_norm = torch.empty(Bk * Hk * Tk, D, dtype=torch.float32, device=key.device)
        rmsnorm_rows_kernel[(Bk * Hk * Tk,)](
            key.reshape(Bk * Hk * Tk, D), k_w, key_norm,
            Bk * Hk * Tk, D,
            eps=rms_norm_eps,
            BLOCK_SIZE=128, num_warps=4
        )
        key_norm = key_norm.reshape(Bk, Hk, Tk, D)

        # Compute cos and sin for query tokens
        # position_ids: [Bq, Tq], int64
        pos_q = position_ids  # use position_ids for query (original code uses it)
        cos_q = torch.empty(Bq * Tq, D, dtype=torch.float32, device=query.device)
        sin_q = torch.empty(Bq * Tq, D, dtype=torch.float32, device=query.device)
        compute_cos_sin_kernel[(Bq, Tq)](
            pos_q, inv_freq, cos_q, sin_q,
            Bq, Tq, half_dim, D
        )

        # Compute cos and sin for key tokens
        pos_k = pos_q  # original code uses position_ids for query; for key, it uses the same in many cases
        cos_k = torch.empty(Bk * Tk, D, dtype=torch.float32, device=key.device)
        sin_k = torch.empty(Bk * Tk, D, dtype=torch.float32, device=key.device)
        compute_cos_sin_kernel[(Bk, Tk)](
            pos_k, inv_freq, cos_k, sin_k,
            Bk, Tk, half_dim, D
        )

        # Apply rotation for query
        query_rotated = torch.empty(Bq, Hq, Tq, D, dtype=torch.float32, device=query.device)
        apply_rotation_kernel[(Bq * Hq * Tq,)](
            query_norm.reshape(Bq * Hq * Tq, D), cos_q, sin_q, query_rotated.reshape(Bq * Hq * Tq, D),
            Bq * Hq * Tq, D, half_dim,
            Bq, Tq,
            cache_start=0,
            BLOCK_SIZE=128, num_warps=4
        )

        # Apply rotation for key
        key_rotated = torch.empty(Bk, Hk, Tk, D, dtype=torch.float32, device=key.device)
        apply_rotation_kernel[(Bk * Hk * Tk,)](
            key_norm.reshape(Bk * Hk * Tk, D), cos_k, sin_k, key_rotated.reshape(Bk * Hk * Tk, D),
            Bk * Hk * Tk, D, half_dim,
            Bk, Tk,
            cache_start=0,
            BLOCK_SIZE=128, num_warps=4
        )

        # Update caches using Triton copy kernel:
        # We need to write into key_cache[:, :, cache_position, :] = key_rotated
        # And value_cache[:, :, cache_position, :] = value (no rotation).
        # For key_cache: rows_out = Bk * Hk * Tq (assuming we update with key_rotated of T=Tq, as typical).
        # Flatten key_rotated to [rows_out, D] where rows_out = Bk * Hk * Tq, and cache_position length Tq.
        rows_k = Bk * Hk * Tq
        from_k = key_rotated.reshape(rows_k, D)  # fp32
        # Launch Triton copy to key_cache
        # We need strides for key_cache: [Bk, Hk, Tq, D]
        stride_b_k = Hk * Tq * D
        stride_h_k = Tq * D
        stride_t_k = D
        stride_d_k = 1
        copy_cache_rows_kernel[(rows_k,)](
            from_k, key_cache, cache_position.to(torch.int64), rows_k, Bk, Hk, Tq, D,
            stride_b_k, stride_h_k, stride_t_k, stride_d_k
        )

        # For value_cache: update rows_out = Bk * Hk * Tq as well


def run(*args):
    return ModelNew()(*args)
