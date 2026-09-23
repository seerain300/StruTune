import torch
import triton
import triton.language as tl

# Triton kernel: RMSNorm over the last dimension (head_dim) for each row.
# Input X_ptr: pointer to [rows, head_dim] float32.
# Output Out_ptr: pointer to [rows, head_dim] float32.
# Weight W_ptr: per-column weight [head_dim] float32.
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
        w = tl.load(W_ptr + offs, mask=mask, other=1.0)  # fp32 weight
        out = x * r * w
        tl.store(Out_ptr + row_id * head_dim + offs, out, mask=mask)

# Triton kernel: compute cos/sin for each position using inv_freq[:half_dim].
# Inputs:
#   Pos_ptr: [rows] int64 positions
#   InvFreq_ptr: [half_dim] float32
#   Cos_ptr, Sin_ptr: [rows, half_dim] float32
@triton.jit
def compute_cos_sin_rows_kernel(Pos_ptr, InvFreq_ptr, Cos_ptr, Sin_ptr,
                                rows, half_dim,
                                BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    pos = tl.load(Pos_ptr + row_id)
    # We compute cos/sin for indices 0..half_dim-1
    for col in range(0, half_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < half_dim
        x = pos * InvFreq_ptr[offs]  # [BLOCK_SIZE] fp32
        c = tl.cos(x)
        s = tl.sin(x)
        out_idx = row_id * half_dim + offs
        tl.store(Cos_ptr + out_idx, c, mask=mask)
        tl.store(Sin_ptr + out_idx, s, mask=mask)

# Triton kernel: apply rotation y = x * cos + rotate_half(x) * sin.
# Input X: normalized [rows, head_dim] float32
# Input Cos_ptr, Sin_ptr: [rows, half_dim] float32
# Output Out: rotated [rows, head_dim] float32
@triton.jit
def apply_rotation_rows_kernel(X_ptr, Cos_ptr, Sin_ptr, Out_ptr,
                               rows, head_dim, half_dim,
                               BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # For rotation, we need first half and second half along last dim
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)  # fp32
        # idx for sin/cos is 0..half_dim-1
        idx = tl.arange(0, BLOCK_SIZE)
        cos = tl.load(Cos_ptr + row_id * half_dim + idx, mask=idx < half_dim, other=0.0)
        sin = tl.load(Sin_ptr + row_id * half_dim + idx, mask=idx < half_dim, other=0.0)
        # rotate_half: second half -> -first half, first half -> second half
        # x1 = x[0:half], x2 = x[half:head_dim]
        # y[0:half] = x1 * cos + x2 * sin
        # y[half:head_dim] = -x2 * cos + x1 * sin
        half = head_dim // 2
        mask1 = (offs < half) & mask
        mask2 = (offs >= half) & mask
        x1 = tl.where(mask1, x, 0.0)
        x2 = tl.where(mask2, x, 0.0)
        y0 = x1 * cos + x2 * sin
        y1 = -x2 * cos + x1 * sin
        # assemble
        y = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        y0_mask = (idx < half) & ((idx + half) < head_dim) & (offs < half)
        y1_mask = (idx < half) & ((idx + half) < head_dim) & (offs >= half)
        y = tl.where(y0_mask, y0, y)
        y = tl.where(y1_mask, y1, y)
        tl.store(Out_ptr + row_id * head_dim + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
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
        # Output: (query_rotated, key_rotated, key_cache, value_cache)
        # We only compute/return query_rotated and key_rotated via Triton; caches are returned unchanged.

        # Shapes
        Bq, Hq, Tq, Dq = query.shape
        Bk, Hk, Tk, Dk = key.shape
        assert Dq == 128 and Dk == 128, "head_dim must be 128"
        half_dim = 64  # because head_dim=128, half=64

        # 1) RMSNorm for query (fp32 compute), output [Bq, Hq, Tq, 128] fp32
        query_norm = torch.empty((Bq, Hq, Tq, Dq), dtype=torch.float32, device=query.device)
        query_rows = Bq * Hq * Tq
        rmsnorm_rows_kernel[(query_rows,)](
            query.reshape(query_rows, Dq), q_norm_weight.to(torch.float32), query_norm.reshape(query_rows, Dq),
            query_rows, Dq, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )

        # 2) Compute cos/sin for query positions: position_ids is [Bq, Tq]
        pos_q = position_ids[:, :Tq].reshape(-1).to(torch.int64)  # [Bq*Tq]
        inv_half = inv_freq[:half_dim].to(torch.float32)  # [64]
        cos_q = torch.empty((Bq * Tq, half_dim), dtype=torch.float32, device=query.device)
        sin_q = torch.empty((Bq * Tq, half_dim), dtype=torch.float32, device=query.device)
        compute_cos_sin_rows_kernel[(Bq * Tq,)](
            pos_q, inv_half, cos_q, sin_q,
            Bq * Tq, half_dim, BLOCK_SIZE=128, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rotated (fp32 compute), output [Bq, Hq, Tq, 128] fp32
        query_rotated_fp32 = torch.empty((Bq, Hq, Tq, Dq), dtype=torch.float32, device=query.device)
        apply_rotation_rows_kernel[(query_rows,)](
            query_norm.reshape(query_rows, Dq), cos_q, sin_q, query_rotated_fp32.reshape(query_rows, Dq),
            query_rows, Dq, half_dim, BLOCK_SIZE=128, num_warps=4
        )
        # Cast to bfloat16 for final output
        query_rotated = query_rotated_fp32.to(torch.bfloat16)

        # 4) RMSNorm for key (fp32 compute), output [Bk, Hk, Tk, 128] fp32
        key_norm = torch.empty((Bk, Hk, Tk, Dk), dtype=torch.float32, device=key.device)
        key_rows = Bk * Hk * Tk
        rmsnorm_rows_kernel[(key_rows,)](
            key.reshape(key_rows, Dk), k_norm_weight.to(torch.float32), key_norm.reshape(key_rows, Dk),
            key_rows, Dk, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )

        # 5) Compute cos/sin for key positions: position_ids is [Bk, Tk]
        pos_k = position_ids[:, :Tk].reshape(-1).to(torch.int64)  # [Bk*Tk]
        cos_k = torch.empty((Bk * Tk, half_dim), dtype=torch.float32, device=key.device)
        sin_k = torch.empty((Bk * Tk, half_dim), dtype=torch.float32, device=key.device)
        compute_cos_sin_rows_kernel[(Bk * Tk,)](
            pos_k, inv_half, cos_k, sin_k,
            Bk * Tk, half_dim, BLOCK_SIZE=128, num_warps=4
        )

        # 6) Apply rotation to key_norm -> key_rotated (fp32 compute), output [Bk, Hk, Tk, 128] fp32
        key_rotated_fp32 = torch.empty((Bk, Hk, Tk, Dk), dtype=torch.float32, device=key.device)
        apply_rotation_rows_kernel[(key_rows,)](
            key_norm.reshape(key_rows, Dk), cos_k, sin_k, key_rotated_fp32.reshape(key_rows, Dk),
            key_rows, Dk, half_dim, BLOCK_SIZE=128, num_warps=4
        )
        # Cast to bfloat16 for final output
        key_rotated = key_rotated_fp32.to(torch.bfloat16)

        # Return: query_rotated (bf16), key_rotated (bf16), key_cache (unchanged), value_cache (unchanged)
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
