import torch
import triton
import triton.language as tl

# Triton kernel: RMSNorm over the last dimension (head_dim) for each row.
# X: [rows, head_dim], fp32; W: [head_dim], fp32; Out: [rows, head_dim], fp32
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
        w = tl.load(W_ptr + offs, mask=mask, other=0.0)  # fp32
        y = x * r
        y = y * w  # apply per-column weight (ones)
        tl.store(Out_ptr + row_id * head_dim + offs, y, mask=mask)


# Triton kernel: compute cos/sin per row for even indices [0,2,..,126] (half_dim=64)
# Inputs:
#   pos_ptr: [rows] int64 positions
#   inv_ptr: [half_dim] fp32 (64 elements)
# Outputs:
#   cos_out: [rows, 128] fp32, only even indices written
#   sin_out: [rows, 128] fp32, only even indices written
@triton.jit
def compute_cos_sin_rows_kernel(pos_ptr, inv_ptr, cos_out_ptr, sin_out_ptr,
                                rows, half_dim,
                                BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    pos = tl.load(pos_ptr + row_id)
    for col in range(0, half_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < half_dim
        inv = tl.load(inv_ptr + offs, mask=mask, other=0.0)  # fp32
        idx = (2.0 * pos.to(tl.float32)) * inv  # fp32
        c = tl.cos(idx)
        s = tl.sin(idx)
        base_col = 2 * offs  # even indices in [0..126] step 2
        tl.store(cos_out_ptr + row_id * 128 + base_col, c, mask=mask)
        tl.store(sin_out_ptr + row_id * 128 + base_col, s, mask=mask)


# Triton kernel: apply rotation y = x * cos + rotate_half(x) * sin
# Inputs:
#   x: [rows, 128] fp32
#   cos: [rows, 128] fp32 (only even indices have valid values, odd zeros)
#   sin: [rows, 128] fp32 (only even indices have valid values, odd zeros)
# Outputs:
#   y: [rows, 128] fp32
@triton.jit
def apply_rotation_rows_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr,
                                rows, head_dim,
                                BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(x_ptr + row_id * head_dim + offs, mask=mask, other=0.0)  # fp32
        # Load cos/sin only for even positions; fill others with 1/0 (neutral for multiply)
        even_mask = (offs % 2) == 0
        c = tl.load(cos_ptr + row_id * 128 + offs, mask=mask & even_mask, other=1.0)
        s = tl.load(sin_ptr + row_id * 128 + offs, mask=mask & even_mask, other=0.0)
        # First half: original columns [0..63]
        half = head_dim // 2
        first = x[:half]
        second = x[half:]
        rotated_half = tl.cat([-second, first], axis=0)  # [64]
        y = x * c + rotated_half * s  # apply rotation only to even positions (half part), others unaffected
        tl.store(y_ptr + row_id * head_dim + offs, y, mask=mask)


# ModelNew entry point: performs all computational work via Triton and returns outputs.
class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes:
        # query: [Bq, Hq, Tq, Dq] with Dq=128
        # key:   [Bk, Hk, Tk, Dk] with Dk=128
        # value: not used for output, kept for signature compatibility
        Bq, Hq, Tq, Dq = query.shape
        Bk, Hk, Tk, Dk = key.shape
        assert Dq == 128 and Dk == 128, "head_dim must be 128"
        half_dim = 64  # since head_dim=128
        device = query.device

        # 1) RMSNorm for query (fp32 compute)
        query_rows = Bq * Hq * Tq
        query_norm_fp32 = torch.empty((query_rows, 128), dtype=torch.float32, device=device)
        rmsnorm_rows_kernel[(query_rows,)](
            query.reshape(query_rows, 128), q_norm_weight.to(torch.float32), query_norm_fp32,
            query_rows, 128, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )

        # 2) Compute cos/sin for query positions: position_ids is [Bq, Tq]
        pos_q = position_ids[:, :Tq].reshape(-1).to(torch.int64)  # [Bq*Tq]
        inv_half = inv_freq[:half_dim].to(torch.float32)  # [64]
        cos_q = torch.empty((Bq * Tq, 128), dtype=torch.float32, device=device)
        sin_q = torch.empty((Bq * Tq, 128), dtype=torch.float32, device=device)
        compute_cos_sin_rows_kernel[(Bq * Tq,)](
            pos_q, inv_half, cos_q, sin_q, Bq * Tq, half_dim, BLOCK_SIZE=128, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rotated (fp32 compute, return bf16)
        query_rotated_fp32 = torch.empty((query_rows, 128), dtype=torch.float32, device=device)
        apply_rotation_rows_kernel[(query_rows,)](
            query_norm_fp32, cos_q, sin_q, query_rotated_fp32, query_rows, 128, BLOCK_SIZE=128, num_warps=4
        )
        query_rotated = query_rotated_fp32.view(Bq, Hq, Tq, 128).to(torch.bfloat16)

        # 4) RMSNorm for key (fp32 compute)
        key_rows = Bk * Hk * Tk
        key_norm_fp32 = torch.empty((key_rows, 128), dtype=torch.float32, device=device)
        rmsnorm_rows_kernel[(key_rows,)](
            key.reshape(key_rows, 128), k_norm_weight.to(torch.float32), key_norm_fp32,
            key_rows, 128, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )

        # 5) Compute cos/sin for key positions: position_ids shape is [Bk, Tk]
        pos_k = position_ids[:Bk, :Tk].reshape(-1).to(torch.int64)  # [Bk*Tk]
        cos_k = torch.empty((key_rows, 128), dtype=torch.float32, device=device)
        sin_k = torch.empty((key_rows, 128), dtype=torch.float32, device=device)
        compute_cos_sin_rows_kernel[(key_rows,)](
            pos_k, inv_half, cos_k, sin_k, key_rows, half_dim, BLOCK_SIZE=128, num_warps=4
        )

        # 6) Apply rotation to key_norm -> key_rotated (fp32 compute, return bf16)
        key_rotated_fp32 = torch.empty((key_rows, 128), dtype=torch.float32, device=device)
        apply_rotation_rows_kernel[(key_rows,)](
            key_norm_fp32, cos_k, sin_k, key_rotated_fp32, key_rows, 128, BLOCK_SIZE=128, num_warps=4
        )
        key_rotated = key_rotated_fp32.view(Bk, Hk, Tk, 128).to(torch.bfloat16)

        # 7) Return outputs. We keep key_cache and value_cache as provided (not modified), since the original forward
        #    also does not return modified caches for this benchmark. We ensure Triton kernels are invoked for computation.
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
