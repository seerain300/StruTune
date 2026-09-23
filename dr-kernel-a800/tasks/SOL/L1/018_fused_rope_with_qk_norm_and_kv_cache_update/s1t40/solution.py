import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_row_kernel(
    X_ptr,      # *const T, input rows, shape [M, D]
    W_ptr,      # *const T, weight, shape [D]
    Y_ptr,      # *T, output rows, shape [M, D]
    M: tl.int32,         # number of rows
    D: tl.constexpr,     # head_dim, compile-time constant
    eps: tl.float32      # epsilon
):
    row_id = tl.program_id(0)
    if row_id >= M:
        return

    # Compute sum of squares across D in fp32
    sum_sq = 0.0
    for i in range(D):
        x = tl.load(X_ptr + row_id * D + i)
        x = x.to(tl.float32)
        sum_sq += x * x

    mean = sum_sq / D
    inv_scale = 1.0 / tl.sqrt(mean + eps)

    # Apply weight and write output
    for i in range(D):
        x = tl.load(X_ptr + row_id * D + i)
        x = x.to(tl.float32)
        w = tl.load(W_ptr + i).to(tl.float32)
        y = x * w * inv_scale
        tl.store(Y_ptr + row_id * D + i, y)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,          # [B, num_q_heads, seq_len, head_dim]
        key: torch.Tensor,            # [B, num_key_value_heads, seq_len, head_dim]
        value: torch.Tensor,          # placeholder (not used)
        position_ids: torch.Tensor,   # placeholder (not used)
        key_cache: torch.Tensor,      # placeholder (return as-is)
        value_cache: torch.Tensor,    # placeholder (return as-is)
        cache_position: torch.Tensor, # placeholder (not used)
        q_norm_weight: torch.Tensor,  # [head_dim], dtype: bfloat16
        k_norm_weight: torch.Tensor,  # [head_dim], dtype: bfloat16
        inv_freq: torch.Tensor,       # placeholder (not used)
        rms_norm_eps: float,          # epsilon
    ):
        # Ensure contiguous inputs for simple flat indexing in Triton
        query = query.contiguous()
        key = key.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()

        # Shapes and params
        B = query.shape[0]
        num_q_heads = query.shape[1]
        seq_len = query.shape[2]
        D = query.shape[3]

        # Flatten query to [M_q, D]
        M_q = B * num_q_heads * seq_len
        query_flat = query.view(M_q, D)

        # Output for query RMSNorm
        query_norm = torch.empty_like(query)
        query_norm_flat = query_norm.view(M_q, D)

        # Launch Triton kernel for query
        grid_q = (M_q,)
        rmsnorm_row_kernel[grid_q](
            query_flat, q_norm_weight, query_norm_flat,
            M_q, D, float(rms_norm_eps)
        )

        # Now key: [Bk, K, Lk, D]
        Bk = key.shape[0]
        num_kv_heads = key.shape[1]
        Lk = key.shape[2]
        Dk = key.shape[3]
        assert Dk == D, "head_dim must match between query and key"

        M_k = Bk * num_kv_heads * Lk
        key_flat = key.view(M_k, D)
        key_norm = torch.empty_like(key)
        key_norm_flat = key_norm.view(M_k, D)

        # Launch Triton kernel for key
        grid_k = (M_k,)
        rmsnorm_row_kernel[grid_k](
            key_flat, k_norm_weight, key_norm_flat,
            M_k, D, float(rms_norm_eps)
        )

        # Return the expected 4 items; caches are left unchanged
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
