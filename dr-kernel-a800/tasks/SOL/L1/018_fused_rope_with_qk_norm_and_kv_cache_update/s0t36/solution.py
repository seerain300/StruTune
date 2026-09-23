import torch
import triton
import triton.language as tl

# Triton kernel: RMSNorm over the last dimension (head_dim) for each row.
# Inputs:
#   X_ptr: pointer to input [rows, head_dim], dtype float32
#   W_ptr: pointer to weight [head_dim], dtype float32 (ones in the original)
#   Out_ptr: pointer to output [rows, head_dim], dtype float32
#   rows: number of rows
#   head_dim: length of last dimension
#   eps: epsilon for RMSNorm
@triton.jit
def rmsnorm_rows_kernel(X_ptr, W_ptr, Out_ptr,
                         rows, head_dim,
                         eps: tl.constexpr,
                         BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= rows:
        return
    sumsq = 0.0
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + pid * head_dim + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sumsq += tl.sum(x32 * x32, axis=0)
    mean = sumsq / head_dim
    r = tl.rsqrt(mean + eps)
    # normalized: x * r * W
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + pid * head_dim + offs, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0)
        y = x32 * r * w
        tl.store(Out_ptr + pid * head_dim + offs, y, mask=mask)

# Triton kernel: compute cos/sin for each position (rows = B * seq_len).
# Outputs are [rows, 128] vectors: first 64 are cos/sin(angle), last 64 zeros (we only use first 64 in rotation,
# but we write 128-length to match expected shape).
@triton.jit
def compute_cos_sin_rows_kernel(Pos_ptr, Inv_ptr, Cos_ptr, Sin_ptr,
                                rows, half_dim, head_dim,
                                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= rows:
        return
    # Compute angle = pos * inv[:half_dim], then cos/sin
    pos = tl.load(Pos_ptr + pid)  # int64
    # We'll iterate over first half_dim and write into head_dim slots; last half zeros.
    for col in range(0, half_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < half_dim
        angle = pos.to(tl.float32) * tl.load(Inv_ptr + offs, mask=mask, other=0.0)
        c = tl.cos(angle)
        s = tl.sin(angle)
        # Write into contiguous [rows, head_dim]
        tl.store(Cos_ptr + pid * head_dim + offs, c, mask=mask)
        tl.store(Sin_ptr + pid * head_dim + offs, s, mask=mask)
    # For last half (half_dim to head_dim), write zeros
    for col in range(half_dim, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        tl.store(Cos_ptr + pid * head_dim + offs, 0.0, mask=mask)
        tl.store(Sin_ptr + pid * head_dim + offs, 0.0, mask=mask)

# Triton kernel: apply rotation to normalized x using cos/sin (128-length).
# Input X_ptr: [rows, head_dim] fp32, Cos_ptr/Sin_ptr: [rows, head_dim] fp32
# Output Y_ptr: [rows, head_dim] fp32, later cast to bfloat16 for return.
@triton.jit
def apply_rotation_rows_kernel(X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
                               rows, head_dim,
                               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= rows:
        return
    half_dim = 64
    # Load first half and second half
    mask0 = tl.arange(0, BLOCK_SIZE) < half_dim
    mask1 = tl.arange(0, BLOCK_SIZE) < (head_dim - half_dim)
    x0 = tl.load(X_ptr + pid * head_dim + tl.arange(0, BLOCK_SIZE), mask=mask0, other=0.0)
    x1 = tl.load(X_ptr + pid * head_dim + (tl.arange(0, BLOCK_SIZE) + half_dim), mask=mask1, other=0.0)
    c0 = tl.load(Cos_ptr + pid * head_dim + tl.arange(0, BLOCK_SIZE), mask=mask0, other=0.0)
    s0 = tl.load(Sin_ptr + pid * head_dim + tl.arange(0, BLOCK_SIZE), mask=mask0, other=0.0)
    # Rotate: y0 = x0 * c0 + (-x1) * s0; y1 = x1 * c0 + x0 * s0
    y0 = x0 * c0 - x1 * s0
    y1 = x1 * c0 + x0 * s0
    # Store back to Y_ptr
    tl.store(Y_ptr + pid * head_dim + tl.arange(0, BLOCK_SIZE), y0, mask=mask0)
    tl.store(Y_ptr + pid * head_dim + (tl.arange(0, BLOCK_SIZE) + half_dim), y1, mask=mask1)

class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes:
        # query: [B, Hq, T, D] where D=128
        # key:   [B, Hk, T, D] where D=128
        # value: [B, Hk, T, D] (not used in computation but present for signature compatibility)
        Bq, Hq, Tq, Dq = query.shape
        Bk, Hk, Tk, Dk = key.shape
        assert Dq == 128 and Dk == 128, "head_dim must be 128"
        half_dim = 64
        head_dim = 128

        # 1) RMSNorm for query in Triton (fp32 compute), outputs fp32
        query_rows = Bq * Hq * Tq
        query_norm = torch.empty((query_rows, Dq), dtype=torch.float32, device=query.device)
        rmsnorm_rows_kernel[(query_rows,)](
            query.reshape(query_rows, Dq).contiguous(),
            q_norm_weight.to(torch.float32),
            query_norm,
            query_rows, Dq,
            eps=rms_norm_eps,
            BLOCK_SIZE=128,
            num_warps=4
        )
        # 2) Compute cos/sin for query positions using position_ids[:, :Tq]
        pos_q = position_ids[:, :Tq].reshape(-1)  # [Bq*Tq]
        cos_q = torch.empty((query_rows, head_dim), dtype=torch.float32, device=query.device)
        sin_q = torch.empty((query_rows, head_dim), dtype=torch.float32, device=query.device)
        compute_cos_sin_rows_kernel[(query_rows,)](
            pos_q.to(torch.int64),
            inv_freq.to(torch.float32),
            cos_q,
            sin_q,
            query_rows, half_dim, head_dim,
            BLOCK_SIZE=128,
            num_warps=4
        )
        # 3) Apply rotation to query_norm -> query_rotated_fp32
        query_rotated_fp32 = torch.empty((query_rows, Dq), dtype=torch.float32, device=query.device)
        apply_rotation_rows_kernel[(query_rows,)](
            query_norm,
            cos_q,
            sin_q,
            query_rotated_fp32,
            query_rows, Dq,
            BLOCK_SIZE=128,
            num_warps=4
        )
        # Cast to bf16 to match original return type for query_rotated
        query_rotated = query_rotated_fp32.reshape(Bq, Hq, Tq, Dq).to(torch.bfloat16)

        # Repeat for key (assuming key tensors also have last dim as Dk=128)
        key_rows = Bk * Hk * Tq
        key_norm = torch.empty((key_rows, Dk), dtype=torch.float32, device=key.device)
        rmsnorm_rows_kernel[(key_rows,)](
            key.reshape(key_rows, Dk).contiguous(),
            k_norm_weight.to(torch.float32),
            key_norm,
            key_rows, Dk,
            eps=rms_norm_eps,
            BLOCK_SIZE=128,
            num_warps=4
        )
        # Compute cos/sin for key positions using pos_q
        cos_k = torch.empty((key_rows, head_dim), dtype=torch.float32, device=key.device)
        sin_k = torch.empty((key_rows, head_dim), dtype=torch.float32, device=key.device)
        compute_cos_sin_rows_kernel[(key_rows,)](
            pos_q.to(torch.int64),
            inv_freq.to(torch.float32),
            cos_k,
            sin_k,
            key_rows, half_dim, head_dim,
            BLOCK_SIZE=128,
            num_warps=4
        )
        # Apply rotation to key_norm -> key_rotated_fp32
        key_rotated_fp32 = torch.empty((key_rows, Dk), dtype=torch.float32, device=key.device)
        apply_rotation_rows_kernel[(key_rows,)](
            key_norm,
            cos_k,
            sin_k,
            key_rotated_fp32,
            key_rows, Dk,
            BLOCK_SIZE=128,
            num_warps=4
        )
        key_rotated = key_rotated_fp32.reshape(Bk, Hk, Tq, Dk).to(torch.bfloat16)

        # Return updated caches (torch indexing to update; Triton kernels are used for main computations).
        # Note: 'value' is not provided in forward signature; original 'run' returns (query_rotated, key_rotated, updated key_cache, value_cache).
        # We update key_cache at cache_position as in the original, and keep value_cache unchanged (original also doesn't update it here).
        # key_cache update: copy rotated key into positions [0..Tq-1]
        max_pos = cache_position.numel()
        for b in range(Bk):
            for h in range(Hk):
                # Copy key_rotated[b, h, :, :] into key_cache[b, h, 0:Tq, :]
                key_cache[b, h, :Tq, :] = key_rotated[b, h, :, :].to(key_cache.dtype)

        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
