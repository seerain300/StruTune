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
    col = 0
    while col < head_dim:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)  # fp32
        sumsq += tl.sum(x * x, axis=0)
        col += BLOCK_SIZE
    mean = sumsq / head_dim
    r = tl.rsqrt(mean + eps)
    col = 0
    while col < head_dim:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0)
        y = x * r * w
        tl.store(Out_ptr + row_id * head_dim + offs, y, mask=mask)
        col += BLOCK_SIZE

# Triton kernel: compute cos and sin for each position and dimension (half_dim=64 -> D=128).
# Inputs:
#   pos_ptr: [rows], int64 position ids
#   inv_ptr: [half_dim], float32 inv_freq[:half_dim]
#   cos_ptr: [rows, D], float32 output cos
#   sin_ptr: [rows, D], float32 output sin
#   rows: number of positions
#   D: head_dim (128)
#   half_dim: 64
@triton.jit
def compute_cos_sin_kernel(pos_ptr, inv_ptr, cos_ptr, sin_ptr,
                           rows, D: tl.constexpr, half_dim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # Compute emb using inv[:half_dim]: emb = [pos * inv[:half_dim], pos * inv[:half_dim]]
    for col in range(0, D, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        # Map offs: first half uses offs, second half uses offs - half_dim
        offs1 = offs
        offs2 = offs - half_dim
        # Gather inv indices (offs2 is out-of-range for second half; we'll handle via masks)
        # We only need inv[:half_dim]; offs1 covers first half. For second half, reuse inv[:half_dim].
        inv1 = tl.load(inv_ptr + offs1, mask=offs1 < half_dim, other=0.0)
        # Broadcast pos to vector
        pos_val = tl.load(pos_ptr + row_id)
        # Compute emb for first half: pos_val * inv1
        emb1 = pos_val * inv1  # [BLOCK_SIZE]
        # For second half, reuse inv1 (same values), but offset logically by half_dim
        # Since emb2 equals emb1, we can compute sin/cos for both halves by doubling emb1.
        emb = emb1 * 2.0  # 64 elements correspond to emb1, second half repeats emb1
        # Load sin and cos of emb; since BLOCK_SIZE may exceed 64, we use masks to limit to first 64
        # But D=128, BLOCK_SIZE=128, half_dim=64: mask by offs < half_dim yields zeros beyond.
        # Compute cos/sin for first 64; second half repeats values.
        cos_val = tl.cos(emb)  # tl.cos expects float32
        sin_val = tl.sin(emb)
        # Store for first 64 columns
        tl.store(cos_ptr + row_id * D + offs, cos_val, mask=mask & (offs < half_dim))
        tl.store(sin_ptr + row_id * D + offs, sin_val, mask=mask & (offs < half_dim))
        # For second half, store same cos/sin under columns [half_dim + offs - half_dim]
        tl.store(cos_ptr + row_id * D + (offs + half_dim), cos_val, mask=mask & (offs < half_dim))
        tl.store(sin_ptr + row_id * D + (offs + half_dim), sin_val, mask=mask & (offs < half_dim))

# Triton kernel: apply rotation y = x * cos + rotate_half(x) * sin on [rows, 128]
# Inputs:
#   x_ptr: [rows, 128], fp32 normalized input
#   cos_ptr: [rows, 128], fp32
#   sin_ptr: [rows, 128], fp32
#   out_ptr: [rows, 128], fp32 output
@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                          rows, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    for col in range(0, D, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(x_ptr + row_id * D + offs, mask=mask, other=0.0)
        cos = tl.load(cos_ptr + row_id * D + offs, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + row_id * D + offs, mask=mask, other=0.0)
        # Rotate half: swap and negate second half
        half = D // 2  # 64
        first = x[..., :half]
        second = x[..., half:]
        rotated_half = -second + first  # Actually need -second, +first, here we compute it for this vector slice
        # Correct rotation:
        # Define first_half = x[..., :half], second_half = x[..., half:]
        # y_first = x_first * cos + (-x_second) * sin
        # y_second = x_second * cos + x_first * sin
        # We can compute y by splitting x into first/second and using above formulas.
        # However, Triton doesn't support multi-d slicing here; compute per element using masks:
        # For each offs in [0..half): y = x[offs] * cos[offs] + (-x[offs+half]) * sin[offs+half]
        # For offs in [half..D): y = x[offs] * cos[offs] + x[offs-half] * sin[offs-half]
        # Implement per-element via masks using two small loops over half.
        # Easier: load first and second halves as separate vectors and compute y as concat.
        # We'll implement it elementwise using double for-loops over half (since D=128 and half=64, this is fine).
        # For simplicity and correctness, we compute y using the standard formula by reusing x:
        # Let x1=x[:half], x2=x[half:], cos1=cos[:half], sin1=cos[half:], cos2=cos[half:], sin2=sin[:half]
        # Then y[:half] = x1 * cos1 - x2 * sin1
        # y[half:] = x2 * cos2 + x1 * sin2
        # But Triton slicing is limited; instead, we compute y directly from x and cos/sin:
        # y = x * cos; and separately compute rotated contribution using second half values with appropriate sin/cos.
        # Here, due to vectorized nature, we compute y per element using the above formulas by reusing the loaded x vector
        # and selecting appropriate cos/sin segments via indexing. Since Triton doesn't support Python slicing on tl.tensor,
        # we compute it as:
        # For offs in [0..half): use cos[offs], sin[offs] and x_second = tl.load(x_ptr + row_id*D + (offs+half), mask=mask_first, other=0.0)
        # For offs in [half..D): use cos[offs], sin[offs] and x_first = tl.load(x_ptr + row_id*D + (offs-half), mask=mask_second, other=0.0)
        # Implementation note: Triton requires static loops; we use while to iterate half elements.
        j = 0
        while j < half:
            idx_first = row_id * D + j
            idx_second = row_id * D + (j + half)
            xj = tl.load(x_ptr + idx_first)
            cosj = tl.load(cos_ptr + idx_first)
            sinj = tl.load(sin_ptr + idx_first)
            xj2 = tl.load(x_ptr + idx_second)
            cosj2 = tl.load(cos_ptr + idx_second)
            sinj2 = tl.load(sin_ptr + idx_second)
            # y_first = xj * cosj - xj2 * sinj
            # y_second = xj2 * cosj2 + xj * sinj2
            y_first = xj * cosj - xj2 * sinj
            y_second = xj2 * cosj2 + xj * sinj2
            # Store y_first
            tl.store(out_ptr + idx_first, y_first)
            # Store y_second
            tl.store(out_ptr + idx_second, y_second)
            j += 1

# Host code for ModelNew.forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes:
        # query: [Bq, Hq, Tq, 128]
        # key:   [Bk, Hk, Tk, 128]
        Bq, Hq, Tq, Dq = query.shape
        Bk, Hk, Tk, Dk = key.shape
        assert Dq == 128 and Dk == 128, "head_dim must be 128"

        # 1) RMSNorm for query (fp32 compute)
        rows_q = Bq * Hq * Tq
        query_norm = torch.empty((rows_q, Dq), dtype=torch.float32, device=query.device)
        query_flat = query.reshape(rows_q, Dq).to(torch.float32)
        qw = q_norm_weight.to(torch.float32)
        rmsnorm_rows_kernel[(rows_q,)](
            query_flat, qw, query_norm,
            rows_q, Dq, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )
        # Reshape back
        query_norm = query_norm.view(Bq, Hq, Tq, Dq)

        # 2) Compute cos/sin for query positions: position_ids is [Bq, Tq]
        pos_q = position_ids[:, :Tq].reshape(-1).to(torch.int64)
        inv_q = inv_freq[:64].to(torch.float32)
        cos_q = torch.empty((pos_q.shape[0], 128), dtype=torch.float32, device=query.device)
        sin_q = torch.empty((pos_q.shape[0], 128), dtype=torch.float32, device=query.device)
        compute_cos_sin_kernel[(pos_q.shape[0],)](
            pos_q, inv_q, cos_q, sin_q,
            pos_q.shape[0], 128, 64, BLOCK_SIZE=128, num_warps=4
        )

        # 3) Apply rotation to query_norm -> query_rotated (fp32 compute)
        rows_rot_q = Bq * Hq * Tq
        query_rotated_fp32 = torch.empty((rows_rot_q, Dq), dtype=torch.float32, device=query.device)
        apply_rotation_kernel[(rows_rot_q,)](
            query_norm.reshape(rows_rot_q, Dq), cos_q, sin_q, query_rotated_fp32,
            rows_rot_q, 128, BLOCK_SIZE=128, num_warps=4
        )
        query_rotated = query_rotated_fp32.view(Bq, Hq, Tq, Dq).to(torch.bfloat16)

        # 4) RMSNorm for key (fp32 compute)
        rows_k = Bk * Hk * Tk
        key_norm = torch.empty((rows_k, Dk), dtype=torch.float32, device=key.device)
        key_flat = key.reshape(rows_k, Dk).to(torch.float32)
        kw = k_norm_weight.to(torch.float32)
        rmsnorm_rows_kernel[(rows_k,)](
            key_flat, kw, key_norm,
            rows_k, Dk, rms_norm_eps, BLOCK_SIZE=128, num_warps=4
        )
        key_norm = key_norm.view(Bk, Hk, Tk, Dk)

        # 5) Compute cos/sin for key positions: cache_len..cache_len+Tk
        pos_k = (cache_position + torch.arange(Bk * Tk, dtype=torch.int64, device=key.device)).view(-1)
        inv_k = inv_freq[:64].to(torch.float32)
        cos_k = torch.empty((pos_k.shape[0], 128), dtype=torch.float32, device=key.device)
        sin_k = torch.empty((pos_k.shape[0], 128), dtype=torch.float32, device=key.device)
        compute_cos_sin_kernel[(pos_k.shape[0],)](
            pos_k, inv_k, cos_k, sin_k,
            pos_k.shape[0], 128, 64, BLOCK_SIZE=128, num_warps=4
        )

        # 6) Apply rotation to key_norm -> key_rotated (fp32 compute)
        rows_rot_k = Bk * Hk * Tk
        key_rotated_fp32 = torch.empty((rows_rot_k, Dk), dtype=torch.float32, device=key.device)
        apply_rotation_kernel[(rows_rot_k,)](
            key_norm.reshape(rows_rot_k, Dk), cos_k, sin_k, key_rotated_fp32,
            rows_rot_k, 128, BLOCK_SIZE=128, num_warps=4
        )
        key_rotated = key_rotated_fp32.view(Bk, Hk, Tk, Dk).to(torch.bfloat16)

        # 7) Return results (note: value is unused)
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
