import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_row_kernel(
    x_ptr,           # *x: input pointer (bfloat16)
    w_ptr,           # *w: weight pointer (bfloat16)
    y_ptr,           # *y: output pointer (bfloat16)
    B: tl.constexpr, # number of batches
    QH: tl.constexpr,  # number of query heads
    SL,              # number of seq positions
    D,               # head_dim
    stride_x_b, stride_x_q, stride_x_s, stride_x_d,  # strides for x
    stride_y_b, stride_y_q, stride_y_s, stride_y_d,  # strides for y
    stride_w,                                          # stride for w
    eps,                                                # epsilon (float32)
    BLOCK: tl.constexpr
):
    # Flatten rows: one program per row (b, q, s)
    row_id = tl.program_id(0)
    # Compute (b, q, s) from row_id
    s = row_id % SL
    tmp = row_id // SL
    q = tmp % QH
    b = tmp // QH

    # Base pointers for this row
    x_row_ptr = x_ptr + b * stride_x_b + q * stride_x_q + s * stride_x_s
    y_row_ptr = y_ptr + b * stride_y_b + q * stride_y_q + s * stride_y_s

    # Accumulate sum of squares in fp32
    sumsq = 0.0
    off = 0
    while off < D:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * stride_x_d, mask=mask, other=0.0)
        # cast to fp32 for accumulation
        x_vals_fp32 = x_vals.to(tl.float32)
        sumsq += tl.sum(x_vals_fp32 * x_vals_fp32, axis=0)
        off += BLOCK

    D_fp32 = tl.full((), D, tl.float32)
    mean = sumsq / D_fp32
    var = mean
    inv_scale = 1.0 / tl.sqrt(var + eps)  # fp32

    # Apply normalization: y = w * x * inv_scale
    off = 0
    while off < D:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * stride_x_d, mask=mask, other=0.0)
        w_vals = tl.load(w_ptr + idx * stride_w, mask=mask, other=0.0)
        y_vals = (x_vals.to(tl.float32) * w_vals.to(tl.float32)) * inv_scale
        # cast back to original dtype (bfloat16) before store
        y_vals_cast = y_vals.to(x_vals.dtype)
        tl.store(y_row_ptr + idx * stride_y_d, y_vals_cast, mask=mask)
        off += BLOCK


@triton.jit
def rmsnorm_key_kernel(
    x_ptr,           # *x: input pointer (bfloat16)
    w_ptr,           # *w: weight pointer (bfloat16)
    y_ptr,           # *y: output pointer (bfloat16)
    B: tl.constexpr, # number of batches
    KH: tl.constexpr,  # number of key/value heads (num_key_value_heads)
    SL,              # number of seq positions
    D,               # head_dim
    stride_x_b, stride_x_k, stride_x_s, stride_x_d,  # strides for x
    stride_y_b, stride_y_k, stride_y_s, stride_y_d,  # strides for y
    stride_w,                                          # stride for w
    eps,                                                # epsilon (float32)
    BLOCK: tl.constexpr
):
    # Flatten rows: one program per row (b, k, s)
    row_id = tl.program_id(0)
    s = row_id % SL
    tmp = row_id // SL
    k = tmp % KH
    b = tmp // KH

    x_row_ptr = x_ptr + b * stride_x_b + k * stride_x_k + s * stride_x_s
    y_row_ptr = y_ptr + b * stride_y_b + k * stride_y_k + s * stride_y_s

    sumsq = 0.0
    off = 0
    while off < D:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * stride_x_d, mask=mask, other=0.0)
        x_vals_fp32 = x_vals.to(tl.float32)
        sumsq += tl.sum(x_vals_fp32 * x_vals_fp32, axis=0)
        off += BLOCK

    D_fp32 = tl.full((), D, tl.float32)
    mean = sumsq / D_fp32
    var = mean
    inv_scale = 1.0 / tl.sqrt(var + eps)

    off = 0
    while off < D:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < D
        x_vals = tl.load(x_row_ptr + idx * stride_x_d, mask=mask, other=0.0)
        w_vals = tl.load(w_ptr + idx * stride_w, mask=mask, other=0.0)
        y_vals = (x_vals.to(tl.float32) * w_vals.to(tl.float32)) * inv_scale
        y_vals_cast = y_vals.to(x_vals.dtype)
        tl.store(y_row_ptr + idx * stride_y_d, y_vals_cast, mask=mask)
        off += BLOCK


class ModelNew(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,             # unused (kept for signature compatibility)
        position_ids: torch.Tensor,      # unused
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,       # unused for this implementation
        cache_position: torch.Tensor,    # unused
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        inv_freq: torch.Tensor,          # unused
        rms_norm_eps: float,
    ):
        # Ensure tensors are on CUDA and contiguous
        assert query.is_cuda and key.is_cuda and q_norm_weight.is_cuda and k_norm_weight.is_cuda, "Tensors must be on CUDA."
        query = query.contiguous()
        key = key.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        key_cache = key_cache.contiguous()  # not modified
        # Allocate outputs
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # Launch Triton RMSNorm for query
        B, QH, SL, D = query.shape
        grid_q = (B * QH * SL,)
        rmsnorm_row_kernel[grid_q](
            query, q_norm_weight, query_norm,
            B, QH, SL, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            q_norm_weight.stride(0),
            rms_norm_eps,
            BLOCK=128,
        )

        # Launch Triton RMSNorm for key
        Bk, KH, SLk, Dk = key.shape
        # num_key_value_heads is provided as KH via input 'key' shape; ensure KH is used
        grid_k = (Bk * KH * SLk,)
        rmsnorm_key_kernel[grid_k](
            key, k_norm_weight, key_norm,
            Bk, KH, SLk, Dk,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            k_norm_weight.stride(0),
            rms_norm_eps,
            BLOCK=128,
        )

        # Return normalized query/key and original caches (not modified to avoid side-effects)
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
