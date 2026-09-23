import triton
import triton.language as tl


@triton.jit
def rmsnorm_row_kernel(
    x_ptr,         # *const T, input tensor pointer
    w_ptr,         # *const T, weight pointer (length D)
    y_ptr,         # *T, output tensor pointer
    B_rows,        # int32, total number of rows
    D,             # int32, head_dim
    eps,           # float32
    BLOCK: tl.constexpr,  # compile-time constant, should equal D
):
    # One program per row
    row = tl.program_id(0)
    if row >= B_rows:
        return

    # Indices along head_dim
    idx = tl.arange(0, BLOCK)
    mask = idx < D  # with BLOCK == D, mask is always True; included for safety

    # Compute flat offsets: each row is contiguous of length D
    offs = row * D + idx

    # Load x row and weight, cast to fp32 for accumulation
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + idx, mask=mask, other=1.0)
    x32 = x.to(tl.float32)
    w32 = w.to(tl.float32)

    # Compute mean of squares
    sq = x32 * x32
    mean = tl.sum(sq, axis=0) / D

    # Compute inv_scale
    inv_scale = 1.0 / tl.sqrt(mean + eps)

    # Normalize and apply weight
    y32 = w32 * x32 * inv_scale
    y = y32.to(x.dtype)

    # Store result
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Implement only RMSNorm in Triton; avoid any torch ops in host code.
        # Shapes:
        # query: [B, num_q_heads, seq_len, D]
        # key:   [B, num_kv_heads, seq_len, D]
        # weight: [D] (1D)
        # Output: (query_norm, key_norm, key_cache, value_cache)

        B, num_q_heads, seq_len, D = query.shape
        Bk, num_kv_heads, _, _ = key.shape

        # Allocate outputs
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # Launch Triton kernel for query RMSNorm: grid over flattened rows (B * num_q_heads * seq_len)
        B_rows_query = B * num_q_heads * seq_len
        grid_query = (B_rows_query,)
        rmsnorm_row_kernel[grid_query](
            query, q_norm_weight, query_norm,
            B_rows_query, D, float(rms_norm_eps),
            BLOCK=D,
            num_warps=4,
        )

        # Launch Triton kernel for key RMSNorm: grid over flattened rows (B * num_kv_heads * seq_len)
        B_rows_key = B * num_kv_heads * seq_len
        grid_key = (B_rows_key,)
        rmsnorm_row_kernel[grid_key](
            key, k_norm_weight, key_norm,
            B_rows_key, D, float(rms_norm_eps),
            BLOCK=D,
            num_warps=4,
        )

        # Return normalized query and key, plus original caches
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
