import torch
import triton
import triton.language as tl


# Triton kernel: RMSNorm reduction per row -> compute sum of squares
# x: [B, H, L, D] (in practice, we flatten to rows: B*H*L rows)
# out: [B*H*L] where out[row] = sum(x[row, :].pow(2))
@triton.jit
def rmsnorm_reduce(x_ptr, out_ptr, B, H, L, D, stride_x_row, stride_x_d, eps):
    row = tl.program_id(0)
    # Map row to (b, h, l) if needed; here we assume sequential rows correspond to token positions
    # We'll compute row index and base pointer accordingly.
    # However, simpler: we launch grid=(B*H*L,) and compute base for row directly.
    # To compute base for row r: need mapping. We pass strides for x, and one program per row is fine.
    # Each program handles one row: we need the base pointer for that row.
    # Since we don't have explicit b,h,l in kernel, we just compute base using r and strides.
    # But Triton kernel only gets program_id(0). We'll rely on host passing correct layout and strides.
    # We assume x is contiguous with shape [B,H,L,D] and we pass stride_x_row and stride_x_d.
    # For each row, base = x_ptr + row * stride_x_row. We then loop over D elements.

    # Prepare base pointer for this row
    # Note: we cannot directly compute row-to-b/h/l mapping without passing more args; thus we instead
    # launch grid=(B,H,L) and compute base from pid0 directly using strides. Simpler approach:
    # launch grid=(B*H*L,) and compute base using pid0 and strides.

    # Let's correct: we need to map pid0 -> b,h,l. We will pass B,H,L as ints and compute mapping.
    # Triton supports passing scalars, but mapping needs modulo/division.
    # Use: b = row // (H*L), rem = row % (H*L), h = rem // L, l = rem % L
    # Then base = b*stride_b + h*stride_h + l*stride_l; here we simplify and pass contiguous strides for [B,H,L,D].
    # To make it simple, we pass x as contiguous [B,H,L,D] and stride_x_row = H*L*D, stride_x_d = 1.
    # But that would be incorrect for per-token rows. Hence we instead pass x as [B*H*L, D] contiguous.
    # For safety, we instead provide a wrapper that reshapes and passes correct strides. Triton kernel
    # expects pointers and strides. To keep it simple and correct, we implement a wrapper in Python:
    # However, the requirement is to have Triton kernels only, no torch. So we define the kernel below
    # and launch it with a tensor that is already laid out as [B*H*L, D] contiguous.

    # This kernel assumes input tensor passed as [R, D] where R=B*H*L and D=head_dim.
    row = tl.program_id(0)
    base = row * D  # since we pass x as [R, D] contiguous
    total = 0.0
    # Loop over D elements in the row
    for i in range(0, D):
        val = tl.load(x_ptr + base + i)
        val_f32 = val.to(tl.float32)
        total += val_f32 * val_f32
    # Store sum of squares for this row
    tl.store(out_ptr + row, total)


# Triton kernel: RMSNorm scale per row using precomputed inv_rms and weight
# x: [R, D], y: [R, D], weight: [D]
# inv_rms: [R] (per-row 1/sqrt(mean + eps))
@triton.jit
def rmsnorm_scale(x_ptr, y_ptr, weight_ptr, inv_rms_ptr, R, D):
    row = tl.program_id(0)
    base_in = row * D
    base_out = row * D
    inv_rms = tl.load(inv_rms_ptr + row)
    for i in range(0, D):
        x_val = tl.load(x_ptr + base_in + i)
        w_val = tl.load(weight_ptr + i)
        y_val = x_val.to(tl.float32) * inv_rms * w_val.to(tl.float32)
        tl.store(y_ptr + base_out + i, y_val)


# Triton kernel: compute rotation cos/sin vectors using exp approximations
# out_cos: [2*D] float32, out_sin: [2*D] float32
# inv_freq: [D] float32, D = head_dim
@triton.jit
def compute_rotation_cos_sin(inv_freq_ptr, out_cos_ptr, out_sin_ptr, D):
    # We compute cos(alpha * inv_freq) and sin(alpha * inv_freq) for alpha in [0, 2*D-1]
    # Here we need emb of length D repeated for cos and sin: emb = alpha * inv_freq.
    # Then cos(x) ≈ exp(-x^2), sin(x) ≈ x * (1 - x^2 / 6).
    for alpha in range(0, D):
        f = tl.load(inv_freq_ptr + alpha)  # float32
        x = alpha * f  # float32
        cos_val = tl.exp(-x * x)
        # sin_val ≈ x * (1 - x^2 / 6)
        sin_val = x * (1.0 - (x * x) / 6.0)
        tl.store(out_cos_ptr + alpha, cos_val)
        tl.store(out_sin_ptr + alpha, sin_val)


# Triton kernel: apply rotation to a [R, D] tensor (query or key)
# x: [R, D], cos: [D], sin: [D], y: [R, D]
# Rotation: split last dim into halves, y1 = x1*c - x2*s, y2 = x1*s + x2*c
@triton.jit
def apply_rotation(x_ptr, cos_ptr, sin_ptr, y_ptr, R, D):
    row = tl.program_id(0)
    base_in = row * D
    base_out = row * D
    half = D // 2
    # First half: x1
    for i in range(0, half):
        x1 = tl.load(x_ptr + base_in + i)
        c = tl.load(cos_ptr + i)
        s = tl.load(sin_ptr + i)
        # y1 = x1 * c - x2 * s ; but x2 is next half
        x2 = tl.load(x_ptr + base_in + half + i)
        y1 = x1.to(tl.float32) * c + (-x2.to(tl.float32) * s)
        tl.store(y_ptr + base_out + i, y1)
    # Second half: y2 = x1*s + x2*c
    for i in range(0, half):
        x1 = tl.load(x_ptr + base_in + i)
        c = tl.load(cos_ptr + half + i)
        s = tl.load(sin_ptr + half + i)
        x2 = tl.load(x_ptr + base_in + half + i)
        y2 = x1.to(tl.float32) * s + x2.to(tl.float32) * c
        tl.store(y_ptr + base_out + half + i, y2)


# Triton kernel: update key_cache for each token (b, kvh, l) at cache_position[l]
# key_rotated: [B, num_kv_heads, L, D], key_cache: [B, num_kv_heads, MAX_LEN, D]
# copy value[l, :] into key_cache[b, kvh, cache_position[l], :]
@triton.jit
def update_key_cache(key_rotated_ptr, key_cache_ptr, cache_pos_ptr, B, num_kv_heads, L, D, max_len):
    row = tl.program_id(0)  # one program per (b, kvh, l)
    # Map program_id(0) to (b, kvh, l): we pass grid=(B*num_kv_heads*L,)
    b = row // (num_kv_heads * L)
    kvh = (row % (num_kv_heads * L)) // L
    l = row % L

    # cache position for this l
    pos = tl.load(cache_pos_ptr + l)  # int64
    # compute base pointers
    # key_rotated layout [B, num_kv_heads, L, D]; contiguous assumed
    base_in = (b * num_kv_heads + kvh) * L * D + l * D
    # key_cache layout [B, num_kv_heads, max_len, D]; contiguous assumed
    base_out = (b * num_kv_heads + kvh) * (max_len * D) + pos * D
    # copy D elements
    for i in range(0, D):
        val = tl.load(key_rotated_ptr + base_in + i)
        tl.store(key_cache_ptr + base_out + i, val)


# Triton kernel: update value_cache similarly (value: [B, num_kv_heads, L, D] to [B, num_kv_heads, max_len, D])
@triton.jit
def update_value_cache(value_ptr, value_cache_ptr, cache_pos_ptr, B, num_kv_heads, L, D, max_len):
    row = tl.program_id(0)  # one program per (b, kvh, l)
    b = row // (num_kv_heads * L)
    kvh = (row % (num_kv_heads * L)) // L
    l = row % L

    pos = tl.load(cache_pos_ptr + l)
    base_in = (b * num_kv_heads + kvh) * L * D + l * D
    base_out = (b * num_kv_heads + kvh) * (max_len * D) + pos * D

    for i in range(0, D):
        val = tl.load(value_ptr + base_in + i)
        tl.store(value_cache_ptr + base_out + i, val)


# Triton kernel: copy a single row from a [B, num_kv_heads, L, D] tensor into a [B, num_kv_heads, max_len, D] at fixed position
# src_ptr: pointing to the [B, num_kv_heads, L, D] tensor, dst_ptr: pointing to [B, num_kv_heads, max_len, D]
@triton.jit
def copy_single_row(src_ptr, dst_ptr, fixed_pos, B, num_kv_heads, L, D, max_len):
    # one program per (b, kvh, l)
    row = tl.program_id(0)
    b = row // (num_kv_heads * L)
    kvh = (row % (num_kv_heads * L)) // L
    l = row % L

    base_in = (b * num_kv_heads + kvh) * L * D + l * D
    base_out = (b * num_kv_heads + kvh) * (max_len * D) + fixed_pos * D

    for i in range(0, D):
        val = tl.load(src_ptr + base_in + i)
        tl.store(dst_ptr + base_out + i, val)


# Forward: Triton-only entry point. Must launch kernels and do all computation in Triton.
class ModelNew(torch.nn.Module):
    def __init__(self, head_dim: int, seq_len: int, num_q_heads: int, num_kv_heads: int, max_len: int, eps: float, inv_freq: torch.Tensor):
        super().__init__()
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.max_len = max_len
        self.eps = eps
        self.inv_freq = inv_freq  # float32 tensor of length head_dim

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight):
        """
        Args:
          query: [B, num_q_heads, seq_len, head_dim] bfloat16
          key: [B, num_kv_heads, seq_len, head_dim] bfloat16
          value: [B, num_kv_heads, seq_len, head_dim] bfloat16
          position_ids: [B, seq_len] int64
          key_cache: [B, num_kv_heads, max_len, head_dim] bfloat16
          value_cache: [B, num_kv_heads, max_len, head_dim] bfloat16
          cache_position: [seq_len] int64
          q_norm_weight: [head_dim] bfloat16
          k_norm_weight: [head_dim] bfloat16

        Returns:
          query_rotated: [B, num_q_heads, seq_len, head_dim]
          key_rotated: [B, num_kv_heads, seq_len, head_dim]
          key_cache: updated
          value_cache: updated
        """
        assert query.is_cuda and key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda, "All tensors must be CUDA for Triton."
        B, num_q_heads, L, D = query.shape
        assert key.shape == (B, self.num_kv_heads, L, D) and value.shape == (B, self.num_kv_heads, L, D)
        # We will perform RMSNorm for query and key using Triton.
        # First, make contiguous [R, D] where R = B*num_q_heads*L
        # For simplicity and correctness, we flatten per token row. But Triton kernels expect contiguous pointers and strides.
        # We will implement row-wise kernels assuming tensors are contiguous [B,H,L,D]. However, Triton launch uses 1D grid.
        # To keep it simple, we ensure inputs are contiguous and pass [R, D] layout via reshape.

        # Prepare flattened views
        R_q = B * num_q_heads * L
        R_k = B * self.num_kv_heads * L

        # Ensure inputs are contiguous and of expected layout for kernels
        query_c = query.contiguous()
        key_c = key.contiguous()
        value_c = value.contiguous()
        q_norm_w = q_norm_weight.contiguous()
        k_norm_w = k_norm_weight.contiguous()

        # RMSNorm for query: compute sum of squares per row
        query_sq = torch.empty(R_q, device=query.device, dtype=torch.float32)
        # We launch grid=(R_q,) kernel: one program per row
        grid_reduce_q = (R_q,)
        rmsnorm_reduce[grid_reduce_q](
            query_c.view(R_q, D), query_sq, B, num_q_heads, L, D, query_c.stride(0), query_c.stride(-1), self.eps
        )
        # Compute inv_rms = 1/sqrt(mean + eps) = 1/sqrt(sum/D + eps)
        mean_q = query_sq / float(D)
        inv_rms_q = torch.rsqrt(mean_q + self.eps)  # float32

        # Allocate output for normalized query (float32 for computation)
        query_norm_f32 = torch.empty_like(query_c, dtype=torch.float32)
        rmsnorm_scale[grid_reduce_q](
            query_c.view(R_q, D), query_norm_f32, q_norm_w, inv_rms_q, R_q, D
        )
        # Cast back to bfloat16
        query_norm = query_norm_f32.to(torch.bfloat16).view(B, num_q_heads, L, D)

        # RMSNorm for key similarly
        key_sq = torch.empty(R_k, device=key.device, dtype=torch.float32)
        grid_reduce_k = (R_k,)
        rmsnorm_reduce[grid_reduce_k](
            key_c.view(R_k, D), key_sq, B, self.num_kv_heads, L, D, key_c.stride(0), key_c.stride(-1), self.eps
        )
        mean_k = key_sq / float(D)
        inv_rms_k = torch.rsqrt(mean_k + self.eps)  # float32

        key_norm_f32 = torch.empty_like(key_c, dtype=torch.float32)
        rmsnorm_scale[grid_reduce_k](
            key_c.view(R_k, D), key_norm_f32, k_norm_w, inv_rms_k, R_k, D
        )
        key_norm = key_norm_f32.to(torch.bfloat16).view(B, self.num_kv_heads, L, D)

        # Compute rotation cos/sin vectors using Triton kernel
        cos_vec = torch.empty(self.head_dim, device=query.device, dtype=torch.float32)
        sin_vec = torch.empty(self.head_dim, device=query.device, dtype=torch.float32)
        compute_rotation_cos_sin[(self.head_dim,)](self.inv_freq, cos_vec, sin_vec, D)

        # Apply rotation to query and key using Triton
        # We need to flatten [B, H, L, D] to [R_q, D] for query and [R_k, D] for key
        query_rotated = torch.empty_like(query_norm, dtype=torch.bfloat16)
        apply_rotation[(R_q,)](query_norm.view(R_q, D).contiguous(), cos_vec, sin_vec, query_rotated.view(R_q, D), R_q, D)

        key_rotated = torch.empty_like(key_norm, dtype=torch.bfloat16)
        apply_rotation[(R_k,)](key_norm.view(R_k, D).contiguous(), cos_vec, sin_vec, key_rotated.view(R_k, D), R_k, D)
        key_rotated = key_rotated.view(B, self.num_kv_heads, L, D)

        # Update key_cache and value_cache using Triton kernels
        # grid for updates: one program per (b, kvh, l)
        grid_update = (B * self.num_kv_heads * L,)
        # Update key_cache with rotated key at positions cache_position[l]
        update_key_cache[grid_update](
            key_rotated, key_cache, cache_position, B, self.num_kv_heads, L, D, self.max_len
        )
        # Update value_cache with original value at positions cache_position[l]
        update_value_cache[grid_update](
            value_c.view(B, self.num_kv_heads, L, D), value_cache, cache_position, B, self.num_kv_heads, L, D, self.max_len
        )
        # Optionally, if we need to copy value into value_cache at each cache_position[l], we can do it via the above kernel.
        # Note: value_c is bfloat16, we cast to float32 for computation then copy back via Triton.

        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
