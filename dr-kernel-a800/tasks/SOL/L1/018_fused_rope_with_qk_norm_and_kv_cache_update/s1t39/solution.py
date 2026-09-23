import triton
import triton.language as tl


@triton.jit
def rmsnorm_row_kernel(
    X_ptr,          # *const T
    W_ptr,          # *const T
    Y_ptr,          # *mut T
    B,              # int32
    NUM_Q_HEADS,    # int32
    SEQ,            # int32
    HEAD_DIM,       # int32 (constexpr)
    EPS,            # fp32
):
    # Flatten grid over all rows: (b, h, l) -> row_id in [0, B*NUM_Q_HEADS*SEQ)
    row_id = tl.program_id(0)
    total = B * NUM_Q_HEADS * SEQ
    if row_id >= total:
        return

    # Compute (b, h, l) indices
    # Note: integer division and modulo are supported by Triton
    NUM_Q_H_SEQ = NUM_Q_HEADS * SEQ
    b = row_id // NUM_Q_H_SEQ
    rem = row_id % NUM_Q_H_SEQ
    h = rem // SEQ
    l = rem % SEQ

    # Compute base offsets
    # Assume input is contiguous in (B, num_q_heads, seq_len, HEAD_DIM)
    # Strides for contiguous:
    # - stride_b = num_q_heads * seq_len * HEAD_DIM
    # - stride_h = seq_len * HEAD_DIM
    # - stride_l = HEAD_DIM
    stride_b = NUM_Q_HEADS * SEQ * HEAD_DIM
    stride_h = SEQ * HEAD_DIM
    stride_l = HEAD_DIM

    base = b * stride_b + h * stride_h + l * stride_l

    # Vector of indices along the head_dim
    idx = tl.arange(0, HEAD_DIM)
    mask = idx < HEAD_DIM

    # Load x and weight; cast to fp32 for accumulation
    x = tl.load(X_ptr + base + idx, mask=mask, other=0.0)
    w = tl.load(W_ptr + idx, mask=mask, other=1.0)
    x32 = x.to(tl.float32)
    w32 = w.to(tl.float32)

    # Compute sum of squares over the row
    sum_sq = tl.sum(x32 * x32, axis=0)  # scalar

    # Variance and inv_scale
    D = HEAD_DIM
    mean_sq = sum_sq / D
    inv_scale = 1.0 / tl.sqrt(mean_sq + EPS)

    # Normalize and apply weight
    y32 = x32 * (w32 * inv_scale)
    y = y32.to(x.dtype)

    # Store
    tl.store(Y_ptr + base + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
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
        rms_norm_eps: float,
    ):
        # We only implement RMSNorm in Triton. No torch operations here.
        # Compute shapes
        B = query.shape[0]
        num_q_heads = query.shape[1]
        seq_len = query.shape[2]
        head_dim = query.shape[3]

        # Allocate outputs for normalized query and key
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # Launch Triton kernels for query and key
        # Grid size: one program per row (b, h, l)
        grid = (B * num_q_heads * seq_len,)

        # query RMSNorm
        rmsnorm_row_kernel[grid](
            query, q_norm_weight, query_norm,
            B, num_q_heads, seq_len, head_dim, rms_norm_eps,
            num_warps=1,  # small row, 1 warp is sufficient
        )

        # key RMSNorm
        rmsnorm_row_kernel[grid](
            key, k_norm_weight, key_norm,
            B, num_q_heads, seq_len, head_dim, rms_norm_eps,
            num_warps=1,
        )

        # Return normalized query and key, and original caches (no mutation here)
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
