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
        base_col = 2 * offs  # even indices 0,2,...,126
        tl.store(cos_out_ptr + row_id * 128 + base_col, c, mask=mask)
        tl.store(sin_out_ptr + row_id * 128 + base_col, s, mask=mask)


# Triton kernel: apply rotation y = x * cos + rotate_half(x) * sin, x is [rows, 128], cos/sin are [rows, 128]
# For the query rotation, rows = Bq * Hq * Tq
@triton.jit
def apply_rotation_rows_kernel(X_ptr, cos_ptr, sin_ptr, Out_ptr,
                               rows, half_dim,
                               BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    for col in range(0, 128, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < 128
        x = tl.load(X_ptr + row_id * 128 + offs, mask=mask, other=0.0)  # fp32
        # first half and second half along last dim
        first = x[:64]
        second = x[64:]
        rotated_half = torch.cat([-second, first], dim=0)  # [64]
        cos = tl.load(cos_ptr + row_id * 128 + offs, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + row_id * 128 + offs, mask=mask, other=0.0)
        y = x * cos + rotated_half * sin
        tl.store(Out_ptr + row_id * 128 + offs, y, mask=mask)


# Entry point ModelNew.forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self,
                query: torch.Tensor,
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
        # Shapes:
        # query: [Bq, Hq, Tq, 128]
        # key:   [Bk, Hk, Tk, 128]
        # value: [B, S, D] (unused for outputs; kept for signature)
        Bq, Hq, Tq, Dq = query.shape
        Bk, Hk, Tk, Dk = key.shape
        assert Dq == 128 and Dk == 128, "head_dim must be 128"
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
        inv_half = inv_freq[:64].to(torch.float32)  # [64]
        cos_q = torch.empty((pos_q.numel(), 128), dtype=torch.float32, device=device)
        sin_q = torch.empty((pos_q.numel(), 128), dtype=torch.float32, device=device)
        compute_cos_sin_rows_kernel[(pos_q.numel(),)](
            pos_q, inv_half, cos_q, sin_q, pos_q.numel(), 64, BLOCK_SIZE=128, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rotated (fp32 compute)
        query_rotated_fp32 = torch.empty((query_rows, 128), dtype=torch.float32, device=device)
        apply_rotation_rows_kernel[(query_rows,)](
            query_norm_fp32, cos_q, sin_q, query_rotated_fp32, query_rows, 64, BLOCK_SIZE=128, num_warps=4
        )
        # Cast to bf16 to match original output dtype
        query_rotated = query_rotated_fp32.view(Bq, Hq, Tq, 128).to(torch.bfloat16)

        # 4) RMSNorm for key (fp32 compute)
        key_rows = Bk * Hk * Tk
        key_norm_fp32 = torch.empty((key_rows, 128), dtype=torch.float32, device=device)
        rmsnorm_rows_kernel[(key_rows,)](
            key.reshape(key_rows, 128), k_norm_weight.to(torch.float32), key_norm_fp32,
            key_rows, 128, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )

        # 5) Compute cos/sin for key positions: cache_position is [N] (N=Tq or Tk)
        # We use cache_position directly as positions (int64)
        pos_k = cache_position.to(torch.int64)  # [N]
        cos_k = torch.empty((pos_k.numel(), 128), dtype=torch.float32, device=device)
        sin_k = torch.empty((pos_k.numel(), 128), dtype=torch.float32, device=device)
        # For keys, we need positions per (b,h) token. If cache_position length is Tq, we assume updating with that.
        # But here we need the positions for keys. Since cache_position is provided, use it as key positions.
        compute_cos_sin_rows_kernel[(pos_k.numel(),)](
            pos_k, inv_half, cos_k, sin_k, pos_k.numel(), 64, BLOCK_SIZE=128, num_warps=4
        )

        # 6) Apply rotation to key_norm -> key_rotated (fp32 compute)
        key_rotated_fp32 = torch.empty((key_rows, 128), dtype=torch.float32, device=device)
        apply_rotation_rows_kernel[(key_rows,)](
            key_norm_fp32, cos_k, sin_k, key_rotated_fp32, key_rows, 64, BLOCK_SIZE=128, num_warps=4
        )
        key_rotated = key_rotated_fp32.view(Bk, Hk, Tk, 128).to(torch.bfloat16)

        # Return: query_rotated and key_rotated (bf16), and return key_cache, value_cache unchanged.
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
