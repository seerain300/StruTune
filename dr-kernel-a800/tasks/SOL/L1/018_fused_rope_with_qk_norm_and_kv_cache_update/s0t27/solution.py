import torch
import triton
import triton.language as tl

# Triton kernel: RMSNorm over the last dimension (head_dim) for each row.
# Inputs:
#   X_ptr: pointer to input [rows, head_dim], dtype float32
#   W_ptr: pointer to weight [head_dim], dtype float32
#   Out_ptr: pointer to output [rows, head_dim], dtype float32
# Arguments:
#   rows: number of rows (B * H * T)
#   head_dim: length of last dimension (D, e.g., 128)
#   eps: epsilon for RMSNorm
@triton.jit
def rmsnorm_rows_kernel(X_ptr, W_ptr, Out_ptr,
                         rows, head_dim,
                         eps: tl.constexpr,
                         BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    sumsq = 0.0
    # Accumulate sum of squares in fp32
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)  # fp32
        sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / head_dim
    r = tl.rsqrt(mean + eps)
    # Normalize and apply per-column weight
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)  # fp32
        w = tl.load(W_ptr + offs, mask=mask, other=1.0)  # fp32
        y = x * r * w  # fp32
        tl.store(Out_ptr + row_id * head_dim + offs, y, mask=mask)


# Triton kernel: apply rotation y = x * cos + rotate_half(x) * sin.
# Inputs:
#   X_ptr: normalized tensor [rows, head_dim] float32
#   cos_ptr, sin_ptr: per-position vectors [rows, half_dim] float32
#   Out_ptr: output [rows, head_dim] float32
@triton.jit
def apply_rotation_rows_kernel(X_ptr, cos_ptr, sin_ptr, Out_ptr,
                                rows, head_dim,
                                half_dim: tl.constexpr,
                                BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # Process first half [0:half_dim)
    for col in range(0, half_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < half_dim
        x1 = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)  # fp32
        x2 = tl.load(X_ptr + row_id * head_dim + (offs + half_dim), mask=mask, other=0.0)  # fp32
        c = tl.load(cos_ptr + row_id * half_dim + offs, mask=mask, other=1.0)  # fp32
        s = tl.load(sin_ptr + row_id * half_dim + offs, mask=mask, other=1.0)  # fp32
        y1 = x1 * c - x2 * s
        y2 = x1 * s + x2 * c
        # Store to [0:half_dim) and [half_dim:2*half_dim)
        tl.store(Out_ptr + row_id * head_dim + offs, y1, mask=mask)
        tl.store(Out_ptr + row_id * head_dim + (offs + half_dim), y2, mask=mask)
    # Process second half if head_dim > 2 * half_dim (not expected here; head_dim=128)
    # To be robust, we could add a second loop for [half_dim: 2*half_dim), but with D=128,
    # half_dim=64, so this covers the entire dimension. If D != 2*half_dim, we skip.

# Triton kernel: compute cos/sin for positions using inv_freq[:half_dim].
# Inputs:
#   pos_ptr: positions [rows] int32
#   inv_ptr: inv_freq [:half_dim] float32
#   cos_ptr, sin_ptr: outputs [rows, half_dim] float32
@triton.jit
def compute_cos_sin_rows_kernel(pos_ptr, inv_ptr, cos_ptr, sin_ptr,
                                rows, half_dim,
                                BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    for i in range(0, half_dim, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < half_dim
        pos = tl.load(pos_ptr + row_id)  # int32
        angle = pos.to(tl.float32) * tl.load(inv_ptr + offs, mask=mask, other=1.0)  # fp32
        c = tl.cos(angle)
        s = tl.sin(angle)
        tl.store(cos_ptr + row_id * half_dim + offs, c, mask=mask)
        tl.store(sin_ptr + row_id * half_dim + offs, s, mask=mask)


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
        # Shapes
        Bq, Hq, Tq, Dq = query.shape
        Bk, Hk, Tk, Dk = key.shape
        # We will use head_dim = 128 (as in the original code), half_dim = 64.
        assert Dq == 128 and Dk == 128, "head_dim must be 128"
        half_dim = 64

        device = query.device
        dtype_in = torch.float32  # compute in fp32 inside kernels
        dtype_out = torch.bfloat16  # return in bf16 to match original

        # 1) RMSNorm for query: out [Bq, Hq, Tq, Dq] fp32
        query_rows = Bq * Hq * Tq
        query_norm = torch.empty((query_rows, Dq), dtype=dtype_in, device=device)
        rmsnorm_rows_kernel[(query_rows,)](
            query.reshape(query_rows, Dq), q_norm_weight.to(torch.float32),
            query_norm,
            query_rows, Dq, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )
        # Reshape back to [Bq, Hq, Tq, Dq]
        query_norm = query_norm.view(Bq, Hq, Tq, Dq)

        # 2) Compute cos/sin for query positions: position_ids is [Bq, Tq]
        pos_q = position_ids[:, :Tq].to(torch.int64).reshape(-1)  # [Bq*Tq]
        pos_q_i32 = pos_q.to(torch.int32)
        inv_half_q = inv_freq[:half_dim].to(torch.float32)  # [64]
        cos_q = torch.empty((Bq * Tq, half_dim), dtype=torch.float32, device=device)
        sin_q = torch.empty((Bq * Tq, half_dim), dtype=torch.float32, device=device)
        compute_cos_sin_rows_kernel[(Bq * Tq,)](
            pos_q_i32, inv_half_q, cos_q, sin_q,
            Bq * Tq, half_dim, BLOCK_SIZE=128, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rotated (fp32 compute)
        query_rows_ = Bq * Hq * Tq
        query_rotated_fp32 = torch.empty((query_rows_, Dq), dtype=torch.float32, device=device)
        # We need to pass cos_q and sin_q as [rows, half_dim], and operate on each row (B*H*T).
        # For rotation kernel, cos/sin are per-position; each row uses its own position.
        apply_rotation_rows_kernel[(query_rows_,)](
            query_norm.reshape(query_rows_, Dq),
            cos_q, sin_q, query_rotated_fp32,
            query_rows_, Dq, half_dim=half_dim, BLOCK_SIZE=128, num_warps=4
        )
        # Cast to bf16 to match original return type
        query_rotated = query_rotated_fp32.view(Bq, Hq, Tq, Dq).to(dtype_out)

        # 4) RMSNorm for key: out [Bk, Hk, Tk, Dk] fp32
        key_rows = Bk * Hk * Tk
        key_norm = torch.empty((key_rows, Dk), dtype=dtype_in, device=device)
        rmsnorm_rows_kernel[(key_rows,)](
            key.reshape(key_rows, Dk), k_norm_weight.to(torch.float32),
            key_norm,
            key_rows, Dk, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )
        key_norm = key_norm.view(Bk, Hk, Tk, Dk)

        # 5) Compute cos/sin for key positions: position_ids is [Bk, Tk]
        pos_k = position_ids[:Bk, :Tk].to(torch.int64).reshape(-1)  # [Bk*Tk]
        pos_k_i32 = pos_k.to(torch.int32)
        cos_k = torch.empty((Bk * Tk, half_dim), dtype=torch.float32, device=device)
        sin_k = torch.empty((Bk * Tk, half_dim), dtype=torch.float32, device=device)
        compute_cos_sin_rows_kernel[(Bk * Tk,)](
            pos_k_i32, inv_half_q, cos_k, sin_k,
            Bk * Tk, half_dim, BLOCK_SIZE=128, num_warps=4
        )

        # 6) Apply rotation to key_norm -> key_rotated (fp32 compute)
        key_rotated_fp32 = torch.empty((key_rows, Dk), dtype=torch.float32, device=device)
        apply_rotation_rows_kernel[(key_rows,)](
            key_norm.reshape(key_rows, Dk),
            cos_k, sin_k, key_rotated_fp32,
            key_rows, Dk, half_dim=half_dim, BLOCK_SIZE=128, num_warps=4
        )
        key_rotated = key_rotated_fp32.view(Bk, Hk, Tk, Dk).to(dtype_out)

        # Return results: query_rotated, key_rotated, key_cache, value_cache
        # Note: key_cache and value_cache are not modified in this forward to match original behavior.
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
