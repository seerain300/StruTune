import torch
import triton
import triton.language as tl

# Triton kernel for RMSNorm per row
# x_ptr: *x, shape [rows, D] (flattened row-major)
# weight_ptr: *weight, shape [D]
# out_ptr: *out, shape [rows, D]
# rows: number of rows (for query: B * num_q_heads * seq_len; for key: B * num_kv_heads * seq_len)
# D: head_dim
# eps: epsilon for RMSNorm
@triton.jit
def rmsnorm_rows_kernel(x_ptr, weight_ptr, out_ptr, rows, D, eps,
                         BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    if pid >= rows:
        return

    row_offset = pid * D

    # Accumulate sum of squares in fp32
    sumsq = 0.0
    for off in range(0, D, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < D
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sumsq += tl.sum(x32 * x32, axis=0)

    mean = sumsq / D
    inv_scale = tl.rsqrt(mean + eps)

    # Write normalized output
    for off in range(0, D, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < D
        x = tl.load(x_ptr + row_offset + idx, mask=mask, other=0.0)
        w = tl.load(weight_ptr + idx, mask=mask, other=1.0)  # weight is [D]
        y = x * inv_scale * w.to(tl.float32)
        tl.store(out_ptr + row_offset + idx, y.to(x.dtype), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Ensure contiguity for predictable row-major layout
        query = query.contiguous()
        key = key.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()

        # RMSNorm for query: rows = B * num_q_heads * seq_len
        B_q, num_q_heads, seq_len, D = query.shape
        rows_q = B_q * num_q_heads * seq_len
        query_norm = torch.empty_like(query)

        grid_q = (rows_q,)
        rmsnorm_rows_kernel[grid_q](
            query, q_norm_weight, query_norm, rows_q, D, rms_norm_eps, BLOCK=128,
            num_warps=4, num_stages=2
        )

        # RMSNorm for key: rows = B * num_kv_heads * seq_len
        B, num_kv_heads, _, _ = key.shape
        rows_k = B * num_kv_heads * seq_len
        key_norm = torch.empty_like(key)

        grid_k = (rows_k,)
        rmsnorm_rows_kernel[grid_k](
            key, k_norm_weight, key_norm, rows_k, D, rms_norm_eps, BLOCK=128,
            num_warps=4, num_stages=2
        )

        # Return normalized query and key, and original caches (no mutation)
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
