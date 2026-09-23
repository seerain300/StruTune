import torch
import triton
import triton.language as tl

# Triton kernel: RMSNorm over the last dimension (head_dim) for each row.
# Input tensor X is viewed as [rows, head_dim], output Out is [rows, head_dim].
# W is per-column weight of length head_dim, provided as fp32.
@triton.jit
def rmsnorm_rows_kernel(X_ptr, W_ptr, Out_ptr,
                         rows, head_dim,
                         eps: tl.constexpr,
                         BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # Compute sum of squares in fp32
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
        # Store back to Out (dtype of Out_ptr determines cast)
        tl.store(Out_ptr + row_id * head_dim + offs, y, mask=mask)

# Triton kernel: compute cos/sin scalars for each token based on position and inv_freq[:half_dim].
# Inputs:
#   pos_ptr: [B, T] int64 positions
#   inv_ptr: [half_dim] float32 inverse frequencies (length head_dim//2)
#   cos_ptr: [B, T, half_dim] float32
#   sin_ptr: [B, T, half_dim] float32
# We write into cos_ptr and sin_ptr as [B, T, half_dim].
@triton.jit
def compute_cos_sin_kernel(pos_ptr, inv_ptr, cos_ptr, sin_ptr,
                           B, T, half_dim):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return
    pos = tl.load(pos_ptr + b * T + t).to(tl.float32)  # pos as float
    # Compute angles for half_dim components
    for i in range(0, half_dim):
        inv_i = tl.load(inv_ptr + i)  # float32
        angle = pos * inv_i  # float32
        cos_val = tl.cos(angle)
        sin_val = tl.sin(angle)
        # Store into cos_ptr and sin_ptr at [b, t, i]
        tl.store(cos_ptr + b * (T * half_dim) + t * half_dim + i, cos_val)
        tl.store(sin_ptr + b * (T * half_dim) + t * half_dim + i, sin_val)

# Triton kernel: apply rotation y = x * cos + rotate_half(x) * sin
# Input x viewed as [rows, head_dim], cos and sin scalars for that row as length half_dim vectors.
# Output y viewed as [rows, head_dim]. cos_ptr/sin_ptr are [B, T, half_dim].
@triton.jit
def apply_rotation_kernel(X_ptr, CosSin_ptr, Out_ptr,
                          rows, head_dim, half_dim,
                          cache_start: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    # Map row_id to (b, head, t) for reading cos/sin scalars. Since cos/sin do not depend on head,
    # we just read per token t.
    # We need t and b. We'll read pos from Cache position at row_id - cache_start and use it to compute cos/sin.
    # However, we do not have pos here. To keep things simple and correct, we assume cos_sin is precomputed
    # outside the kernel based on token t, and we pass cache_start. Triton cannot read pos here, so we
    # will compute cos/sin per (b,t) outside and pass to kernel as CosSin_ptr for the corresponding token.
    # The host will ensure Out_ptr is laid out as [rows, D] and Out_ptr for each row corresponds to the same t
    # as row_id - cache_start. Thus, we set up Out_ptr such that each row maps to its token position.
    # Here, we load cos/sin scalars for this row_id from CosSin_ptr assuming it is precomputed at host.
    # For y = x * cos + rotate_half(x) * sin, we need to read cos and sin for token (b, t) corresponding to row_id.
    # Since we cannot access pos here, we rely on host to pass correct cos/sin.
    # Let's assume CosSin_ptr is actually [B, T, half_dim] and we can index by (b,t).
    # We need to decode b and t from row_id: b = row_id // T, t = row_id % T.
    # NOTE: This is only valid if rows == B * T. In our forward, we set rows = B * H * T for query and key separately.
    # Given this, we cannot decode (b, t) reliably across different inputs. Therefore, we instead pass cos/sin
    # computed per (b,t) into Out_ptr layout matching rows, but Triton cannot fetch pos. Hence, we precompute
    # cos/sin outside and feed them as a contiguous vector per row.
    # Simplify: assume CosSin_ptr is [rows, half_dim] and we load by row_id.
    # Load cos/sin for this row:
    # cos_vec: [half_dim], sin_vec: [half_dim]
    for i in range(0, half_dim):
        cos_val = tl.load(CosSin_ptr + row_id * half_dim + i)
        sin_val = tl.load(CosSin_ptr + row_id * half_dim + i + half_dim)  # store sin right after cos
    # Now apply rotation across head_dim
    for col in range(0, head_dim, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < head_dim
        x = tl.load(X_ptr + row_id * head_dim + offs, mask=mask, other=0.0)
        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]
        rotate = tl.cat([-x2, x1], axis=0)  # rotate_half
        # Compute y = x * cos + rotate * sin
        # We need to gather cos and sin for each column. Since cos/sin are per token,
        # we apply them to all columns uniformly (as scalar). That's fine for head_dim=128.
        # For each column col, we use cos_val and sin_val scalars.
        # Implement elementwise:
        y = x.to(tl.float32) * cos_val + rotate.to(tl.float32) * sin_val
        tl.store(Out_ptr + row_id * head_dim + offs, y, mask=mask)

# Triton kernel: copy from one pointer to another, useful for value cache update (no math).
@triton.jit
def copy_rows_kernel(From_ptr, To_ptr, rows, D):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return
    for col in range(0, D, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        val = tl.load(From_ptr + row_id * D + offs, mask=mask, other=0.0)
        tl.store(To_ptr + row_id * D + offs, val, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                position_ids: torch.Tensor,
                key_cache: torch.Tensor, value_cache: torch.Tensor,
                cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor,
                rms_norm_eps: float):
        # Shapes
        Bq, Hq, Tq, D = query.shape
        Bk, Hk, Tk, Dk = key.shape
        # Ensure device and dtype
        assert query.is_cuda and key.is_cuda and value.is_cuda and position_ids.is_cuda \
               and key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda \
               and q_norm_weight.is_cuda and k_norm_weight.is_cuda and inv_freq.is_cuda, "All tensors must be on CUDA device."
        assert D == Dk and D == 128, "head_dim must be 128."
        half_dim = D // 2
        # RMSNorm for query and key
        # Prepare weight vectors (fp32)
        q_w = q_norm_weight.to(torch.float32).contiguous()  # [D], fp32
        k_w = k_norm_weight.to(torch.float32).contiguous()  # [D], fp32

        # View inputs as [rows, D] for Triton kernels
        Xq = query.contiguous()
        Xk = key.contiguous()
        Vl = value.contiguous()  # [B, Hk, T, D]

        # Compute number of rows for each (we will run two RMSNorm kernels)
        rows_q = Bq * Hq * Tq
        rows_k = Bk * Hk * Tk

        # Allocate outputs for normalization
        query_norm = torch.empty_like(query, dtype=query.dtype)
        key_norm = torch.empty_like(key, dtype=key.dtype)

        # Launch RMSNorm for query
        grid_q = (rows_q,)
        rmsnorm_rows_kernel[grid_q](
            Xq, q_w, query_norm,
            rows_q, D,
            rms_norm_eps,
            BLOCK_SIZE=128, num_warps=4
        )

        # Launch RMSNorm for key
        grid_k = (rows_k,)
        rmsnorm_rows_kernel[grid_k](
            Xk, k_w, key_norm,
            rows_k, D,
            rms_norm_eps,
            BLOCK_SIZE=128, num_warps=4
        )

        # Compute cos and sin per token for rotation using Triton
        # pos is [B, T] where T is the seq_len of each input; for query it's Tq, for key it's Tk.
        # We need to compute separately for query and key.
        Bpos_q = Bq
        Tpos_q = Tq
        Bpos_k = Bk
        Tpos_k = Tk

        # Prepare pos tensors
        pos_q = cache_position[:Bpos_q * Tpos_q].reshape(Bpos_q, Tpos_q).to(torch.float32)  # [Bq, Tq]
        pos_k = cache_position[:Bpos_k * Tpos_k].reshape(Bpos_k, Tpos_k).to(torch.float32)  # [Bk, Tk]

        # inv_freq is [half_dim] float32 (from original: 10000000.0 based arange over head_dim//2)
        inv = inv_freq[:half_dim].contiguous()  # [half_dim] fp32

        # Allocate cos/sin for query and key
        cos_q = torch.empty((Bpos_q, Tpos_q, half_dim), dtype=torch.float32, device=query.device)
        sin_q = torch.empty((Bpos_q, Tpos_q, half_dim), dtype=torch.float32, device=query.device)
        grid_cos_q = (Bpos_q, Tpos_q)
        compute_cos_sin_kernel[grid_cos_q](
            pos_q, inv, cos_q, sin_q,
            Bpos_q, Tpos_q, half_dim,
            num_warps=2
        )

        cos_k = torch.empty((Bpos_k, Tpos_k, half_dim), dtype=torch.float32, device=key.device)
        sin_k = torch.empty((Bpos_k, Tpos_k, half_dim), dtype=torch.float32, device=key.device)
        grid_cos_k = (Bpos_k, Tpos_k)
        compute_cos_sin_kernel[grid_cos_k](
            pos_k, inv, cos_k, sin_k,
            Bpos_k, Tpos_k, half_dim,
            num_warps=2
        )

        # Apply rotation: y = x * cos + rotate_half(x) * sin
        # We need to apply rotation to query_norm and key_norm. We will flatten [B,H,T] to [rows, D] and launch kernel.
        # For query:
        rows_q_rot = Bq * Hq * Tq
        query_rotated = torch.empty_like(query_norm, dtype=query_norm.dtype)
        # For key:
        rows_k_rot = Bk * Hk * Tk
        key_rotated = torch.empty_like(key_norm, dtype=key_norm.dtype)

        # We need a way to feed cos/sin scalars per row to kernel. Triton kernel expects cos_sin_ptr with shape [rows, 2*half_dim].
        # Build cos_sin_ptr for query: [rows_q_rot, 2*half_dim], fill with cos and sin interleaved.
        # Decode b and t per row to select correct cos/sin. This is tricky in Triton without host-provided mapping,
        # so we precompute per (b,t) as above and pass pointers via contiguous layout, then inside kernel load by row_id.
        # In practice, we can pass cos/sin per token t for each batch via cos_q/sin_q and rely on Out_ptr layout that matches rows_q_rot.
        # To simplify, we relabel: cos/sin for each token t applies to all rows with that t (because rotation depends only on token position).
        # However, rows_q_rot != Bq*Tq, because Hq is folded in. Triton kernel above assumed rows == B*T; to make it general,
        # we modify kernel to accept cos/sin as [rows, half_dim] and load by row_id. That's acceptable because cos/sin are scalars
        # per token and reused across rows. So we construct cos_sin_vec_q of length rows_q_rot: for each row_id, its token t is row_id % Tq,
        # batch b is row_id // Tq // Hq (but better: since we folded Hq into rows, t is row_id % Tq, b is row_id // (Hq*Tq)). Simplify:
        # b = row_id // Tq, head = (row_id // Tq) % Hq, t = row_id % Tq. We'll pass b,t to kernel as constexpr using Python mapping per launch
        # is not possible, so we use the [rows, 2*half_dim] trick by mapping row_id to (b,t) via host code. To do that without extra kernels,
        # we set up cos_sin_vec_q and cos_sin_vec_k on host: alternating cos and sin.

        # Build cos_sin_vec for query: [rows_q_rot, 2*half_dim]
        # For each row_id in [0, rows_q_rot), let b = row_id // Tq, head = (row_id // Tq) % Hq, t = row_id % Tq.
        # cos_sin_vec[row_id, 2*i] = cos_q[b, t, i], sin at 2*i+1. Note: cos_q/sin_q are [Bq, Tq, half_dim].
        cos_sin_q = torch.empty((rows_q_rot, 2 * half_dim), dtype=torch.float32, device=query.device)
        # Populate using a small loop (PyTorch indexing)
        for row in range(rows_q_rot):
            b = row // Tq
            t = row % Tq
            for i in range(half_dim):
                cos_val = cos_q[b, t, i]
                sin_val = sin_q[b, t, i]
                cos_sin_q[row, 2 * i] = cos_val
                cos_sin_q[row, 2 * i + 1] = sin_val

        # Similarly for key:
        cos_sin_k = torch.empty((rows_k_rot, 2 * half_dim), dtype=torch.float32, device=key.device)
        for row in range(rows_k_rot):
            b = row // Tk
            t = row % Tk
            for i in range(half_dim):
                cos_val = cos_k[b, t, i]
                sin_val = sin_k[b, t, i]
                cos_sin_k[row, 2 * i] = cos_val
                cos_sin_k[row, 2 * i + 1] = sin_val

        # Launch rotation kernel for query
        # We need to decode (b, head, t) inside kernel from row_id. The kernel expects a contiguous cos_sin pointer with 2*half_dim per row.
        # To keep things simple, we relaunch with grid (rows_q_rot,) and let kernel load cos_sin_vec[row_id, :].
        apply_rotation_kernel[(rows_q_rot,)](
            query_norm, cos_sin_q, query_rotated,
            rows_q_rot, D, half_dim,
            cache_start=0,
            BLOCK_SIZE=128, num_warps=4
        )

        # Launch rotation kernel for key
        apply_rotation_kernel[(rows_k_rot,)](
            key_norm, cos_sin_k, key_rotated,
            rows_k_rot, D, half_dim,
            cache_start=0,
            BLOCK_SIZE=128, num_warps=4
        )

        # Update caches using torch assignment (overwrite slice)
        # key_cache: [B, Hk, max_pos, D]
        # value_cache: [B, Hk, max_pos, D]
        # We need to place key_rotated and value along cache positions: cache_start = cache_len, length = T (query or key)
        # Note: cache_len is not passed; in the original, cache_position is of length seq_len. We assume cache_start=0 for simplicity.
        # However, original code assigns to key_cache[:, :, cache_position] where cache_position is [T]. So we need to assign
        # key_rotated[:, :, cache_position] into key_cache. Triton can do this, but simple torch assignment is fine here.
        for b in range(Bk):
            for head in range(Hk):
                # key_rotated shape: [Bk, Hk, T, D]
                # key_cache shape: [Bk, Hk, max_pos, D]
                # Assign slice at positions cache_position (length T): key_cache[b, head, cache_position, :] = key_rotated[b, head, :, :]
                # Use cache_position as provided (from original input): it's [T] int64
                # Expand to [1, 1, T] for index_put
                idx = cache_position  # [T]
                # key_rotated[b, head, :, :] is the slice we want to assign
                # torch.index_put is not available here; use advanced indexing:
                # Build a list of indices: [cache_position] along dim=2
                key_cache[b, head, idx, :] = key_rotated[b, head, :, :]

                # For value, assign value[b, head, :, :] to the same positions
                value_cache[b, head, idx, :] = value[b, head, :, :]

        # Return query_rotated, key_rotated, key_cache, value_cache
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
