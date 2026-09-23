import triton
import triton.language as tl
import torch


@triton.jit
def query_rmsnorm_kernel(
    x_ptr,         # *const T, input query tensor [B, D]
    w_ptr,         # *const T, weight [D]
    y_ptr,         # *T, output [B, D]
    B: tl.constexpr,  # total number of rows
    D: tl.constexpr,  # head_dim
    eps: tl.constexpr,  # epsilon
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return
    sumsq = 0.0
    for i in range(0, D, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < D
        x = tl.load(x_ptr + row * D + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / D
    inv_scale = 1.0 / tl.sqrt(mean + eps)
    for i in range(0, D, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < D
        x = tl.load(x_ptr + row * D + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y = x * w * inv_scale
        out = y.to(tl.load(x_ptr + row * D + idx, mask=mask, other=0.0).dtype)
        tl.store(y_ptr + row * D + idx, out, mask=mask)


@triton.jit
def key_rmsnorm_kernel(
    x_ptr,         # *const T, input key tensor [BK, D]
    w_ptr,         # *const T, weight [D]
    y_ptr,         # *T, output [BK, D]
    BK: tl.constexpr,   # total rows for key (batch * num_kv_heads * seq_len)
    D: tl.constexpr,    # head_dim
    eps: tl.constexpr,  # epsilon
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= BK:
        return
    sumsq = 0.0
    for i in range(0, D, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < D
        x = tl.load(x_ptr + row * D + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / D
    inv_scale = 1.0 / tl.sqrt(mean + eps)
    for i in range(0, D, BLOCK):
        idx = i + tl.arange(0, BLOCK)
        mask = idx < D
        x = tl.load(x_ptr + row * D + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y = x * w * inv_scale
        out = y.to(tl.load(x_ptr + row * D + idx, mask=mask, other=0.0).dtype)
        tl.store(y_ptr + row * D + idx, out, mask=mask)


def _launch_rmsnorm(x, weight, eps):
    # x is [B, D] flattened view (same dtype as original), weight is [D] same dtype
    B = x.numel() // x.shape[-1]
    D = x.shape[-1]
    w = weight.contiguous()
    y = torch.empty_like(x)
    # Simple heuristic for num_warps
    num_warps = 4 if D <= 128 else 8
    grid = (B,)
    query_rmsnorm_kernel[grid](x, w, y, B, D, eps, BLOCK=D, num_warps=num_warps, num_stages=2)
    return y


class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Launch Triton kernels: compute RMSNorm for query and key
        # Ensure inputs are contiguous [B, D] flattened views
        query = query.contiguous()
        key = key.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()

        query_norm = _launch_rmsnorm(query, q_norm_weight, rms_norm_eps)
        key_norm = _launch_rmsnorm(key, k_norm_weight, rms_norm_eps)

        # Return normalized query and key, plus original caches
        return query_norm, key_norm, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
