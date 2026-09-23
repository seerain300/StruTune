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
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0)
        y = x * r * w
        tl.store(Out_ptr + row_id * head_dim + offs, y, mask=mask)

# Triton kernel: compute cos and sin for each position and dimension (D=128, half_dim=64).
# Inputs:
#   pos_ptr: [rows], int64 position ids
#   inv_ptr: [half_dim], float32 inv_freq[:half_dim]
#   cos_ptr: [rows, D], float32 output cos
#   sin_ptr: [rows, D], float32 output sin
#   rows: number of positions
#   D: head_dim (128)
#   half_dim: 64 (used to compute inv indices)
@triton.jit
def compute_cos_sin_kernel(pos_ptr, inv_ptr, cos_ptr, sin_ptr,
                           rows, D: tl.constexpr, half_dim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # For each row, compute cos and sin across D dims by pairing with inv_ptr[:half_dim]
    for col in range(0, D, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        # emb index for cosine/sine: first half uses inv_ptr[0:half_dim], second half repeats
        # Construct idx: [0..half_dim-1] for first half, [half_dim..2*half_dim-1] for second half
        # But since D == 2 * half_dim, we can simply index inv_ptr by offs // half_dim
        idx = offs // half_dim  # 0 for first half, 1 for second half
        # Gather inv values per element (broadcast over row)
        # We need to compute per-element idx from offs; since D = 2 * half_dim, idx = offs // half_dim
        # Note: offs ranges [0..D-1]; offs < half_dim -> idx=0; offs in [half_dim..D-1] -> idx=1
        inv = tl.load(inv_ptr + idx, mask=mask, other=0.0)  # shape [BLOCK_SIZE]
        pos = tl.load(pos_ptr + row_id)
        angle = pos * inv
        c = tl.cos(angle)
        s = tl.sin(angle)
        # Store into cos/sin at [row_id, offs]
        tl.store(cos_ptr + row_id * D + offs, c, mask=mask)
        tl.store(sin_ptr + row_id * D + offs, s, mask=mask)

# Triton kernel: apply rotation y = x * cos + rotate_half(x) * sin
# Inputs:
#   X_ptr: [rows, D], float32 normalized input
#   cos_ptr: [rows, D], float32 cos
#   sin_ptr: [rows, D], float32 sin
#   Out_ptr: [rows, D], float32 output
#   rows: number of rows
#   D: head_dim (128)
@triton.jit
def rotate_rows_kernel(X_ptr, cos_ptr, sin_ptr, Out_ptr,
                        rows, D: tl.constexpr, half_dim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    for col in range(0, D, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + row_id * D + offs, mask=mask, other=0.0)
        cos = tl.load(cos_ptr + row_id * D + offs, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + row_id * D + offs, mask=mask, other=0.0)
        # Split into first half and second half
        first_mask = offs < half_dim
        second_mask = ~first_mask
        first = tl.where(first_mask, x, 0.0)
        second = tl.where(second_mask, x, 0.0)
        # rotate_half: swap and negate second half -> [-second, first]
        rotated = tl.where(first_mask, x, -second)
        y = x * cos + rotated * sin
        tl.store(Out_ptr + row_id * D + offs, y, mask=mask)

def _launch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float, device: torch.device):
    rows = x.numel()
    head_dim = x.shape[-1]
    # Flatten to [rows, head_dim]
    x_flat = x.reshape(rows, head_dim).contiguous()
    out_flat = out.reshape(rows, head_dim).contiguous()
    weight_fp32 = weight.to(torch.float32).contiguous()
    grid = (rows,)
    rmsnorm_rows_kernel[grid](
        x_flat, weight_fp32, out_flat,
        rows, head_dim,
        eps=eps, BLOCK_SIZE=128, num_warps=4
    )

def _launch_compute_cos_sin(pos: torch.Tensor, inv: torch.Tensor, cos_out: torch.Tensor, sin_out: torch.Tensor, device: torch.device):
    rows = pos.numel()
    D = inv.shape[0] * 2  # head_dim, 128
    half_dim = inv.shape[0]  # 64
    # pos: [rows], int64
    pos_flat = pos.reshape(rows).contiguous()
    cos_out_flat = cos_out.reshape(rows, D).contiguous()
    sin_out_flat = sin_out.reshape(rows, D).contiguous()
    inv_fp32 = inv.to(torch.float32).contiguous()
    grid = (rows,)
    compute_cos_sin_kernel[grid](
        pos_flat, inv_fp32, cos_out_flat, sin_out_flat,
        rows, D, half_dim, BLOCK_SIZE=128, num_warps=4
    )

def _launch_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor, device: torch.device):
    rows = x.numel()
    D = x.shape[-1]
    half_dim = D // 2
    x_flat = x.reshape(rows, D).contiguous()
    cos_flat = cos.reshape(rows, D).contiguous()
    sin_flat = sin.reshape(rows, D).contiguous()
    out_flat = out.reshape(rows, D).contiguous()
    grid = (rows,)
    rotate_rows_kernel[grid](
        x_flat, cos_flat, sin_flat, out_flat,
        rows, D, half_dim, BLOCK_SIZE=128, num_warps=4
    )

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes:
        # query: [Bq, Hq, Tq, Dq] with Dq == 128
        # key:   [Bk, Hk, Tk, Dk] with Dk == 128
        # value: unused for this computation
        Bq, Hq, Tq, Dq = query.shape
        Bk, Hk, Tk, Dk = key.shape
        assert Dq == 128 and Dk == 128, "head_dim must be 128 for this Triton implementation"
        assert query.device.type == "cuda" and key.device.type == "cuda", "Triton kernels require CUDA tensors"

        # 1) RMSNorm for query
        query_norm = torch.empty((Bq, Hq, Tq, Dq), dtype=torch.float32, device=query.device)
        _launch_rmsnorm(query.reshape(Bq * Hq * Tq, Dq), q_norm_weight, query_norm, rms_norm_eps, query.device)

        # 2) Compute cos/sin for query positions: position_ids is [Bq, Tq]
        pos_q = position_ids[:, :Tq].reshape(-1).to(torch.int64)  # [Bq*Tq]
        inv_half_q = inv_freq[:64].to(torch.float32)              # [64]
        cos_q = torch.empty((Bq * Tq, Dq), dtype=torch.float32, device=query.device)
        sin_q = torch.empty((Bq * Tq, Dq), dtype=torch.float32, device=query.device)
        _launch_compute_cos_sin(pos_q, inv_half_q, cos_q, sin_q, query.device)

        # 3) Apply rotation to query_norm -> query_rotated (fp32 compute, cast to bf16)
        query_rotated_fp32 = torch.empty((Bq, Hq, Tq, Dq), dtype=torch.float32, device=query.device)
        _launch_rotate(query_norm.reshape(Bq * Hq * Tq, Dq), cos_q, sin_q, query_rotated_fp32, query.device)
        query_rotated = query_rotated_fp32.to(torch.bfloat16)

        # 4) RMSNorm for key
        key_norm = torch.empty((Bk, Hk, Tk, Dk), dtype=torch.float32, device=key.device)
        _launch_rmsnorm(key.reshape(Bk * Hk * Tk, Dk), k_norm_weight, key_norm, rms_norm_eps, key.device)

        # 5) Compute cos/sin for key positions: position_ids is [Bk, Tk] (same as input, but cache_len+seq_len not used here)
        # We can reuse inv_freq for key as well since it's per head
        pos_k = position_ids[:Bk, :Tk].reshape(-1).to(torch.int64)  # [Bk*Tk]
        cos_k = torch.empty((Bk * Tk, Dk), dtype=torch.float32, device=key.device)
        sin_k = torch.empty((Bk * Tk, Dk), dtype=torch.float32, device=key.device)
        _launch_compute_cos_sin(pos_k, inv_freq[:64].to(torch.float32), cos_k, sin_k, key.device)

        # 6) Apply rotation to key_norm -> key_rotated (fp32 compute, cast to bf16)
        key_rotated_fp32 = torch.empty((Bk, Hk, Tk, Dk), dtype=torch.float32, device=key.device)
        _launch_rotate(key_norm.reshape(Bk * Hk * Tk, Dk), cos_k, sin_k, key_rotated_fp32, key.device)
        key_rotated = key_rotated_fp32.to(torch.bfloat16)

        # Return query_rotated, key_rotated (bf16), and leave caches as-is
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
