import torch
import triton
import triton.language as tl


# Triton kernel: RMSNorm over last dim D for a 4D tensor [B, H, S, D].
# x: input (fp16/bf16), out: output (same dtype as x), weight: [D] in fp32, eps: fp32
@triton.jit
def rmsnorm_4d_kernel(
    x_ptr, out_ptr, weight_ptr,
    B, H, S, D,
    stride_b, stride_h, stride_s, stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    eps,
):
    pid = tl.program_id(0)  # one program per [b, h, s] row
    total = B * H * S
    # Decode pid into (b, h, s)
    s = pid % S
    tmp = pid // S
    h = tmp % H
    b = tmp // H

    # Base pointers for this row
    base_x = x_ptr + b * stride_b + h * stride_h + s * stride_s
    base_out = out_ptr + b * out_stride_b + h * out_stride_h + s * out_stride_s

    # First pass: compute sum of squares in fp32
    sumsq = 0.0
    for offs in range(0, D, 128):
        idx = offs + tl.arange(0, 128)
        mask = idx < D
        x = tl.load(base_x + idx * stride_d, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        sumsq += tl.sum(x_fp32 * x_fp32, axis=0)

    mean = sumsq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Second pass: normalize and scale by weight, then store
    for offs in range(0, D, 128):
        idx = offs + tl.arange(0, 128)
        mask = idx < D
        x = tl.load(base_x + idx * stride_d, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        w = tl.load(weight_ptr + idx, mask=mask, other=1.0)  # weight is fp32
        y_fp32 = x_fp32 * inv_rms * w
        y = y_fp32.to(x.dtype)
        tl.store(base_out + idx * out_stride_d, y, mask=mask)


# Triton kernel: build cos_all and sin_all vectors of length D for a given position pos.
# pos: scalar int32, inv_freq: [D//2] fp32, cos_all: [D] fp32, sin_all: [D] fp32
@triton.jit
def build_rot_vec_kernel(
    pos, inv_freq_ptr, cos_all_ptr, sin_all_ptr,
    D,
):
    # idx is [0..D-1]
    idx = tl.arange(0, D)
    half = D // 2
    # angle = idx * inv_freq[idx % (D//2)]
    j = idx % half
    angle = pos.to(tl.float32) * tl.load(inv_freq_ptr + j)  # [D]
    cosv = tl.cos(angle)
    sinv = tl.sin(angle)
    tl.store(cos_all_ptr + idx, cosv)
    tl.store(sin_all_ptr + idx, sinv)


# Triton kernel: apply rotation to a single row [D] for query:
# y = x * cos_all - rotate_half(x) * sin_all
# Inputs: x_norm: [D] fp32, cos_all: [D] fp32, sin_all: [D] fp32
# Output: out: [D] fp32
@triton.jit
def rotate_query_row_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    D,
):
    idx = tl.arange(0, D)
    x = tl.load(x_ptr + idx)
    cosv = tl.load(cos_ptr + idx)
    sinv = tl.load(sin_ptr + idx)
    x1 = x[:D // 2]
    x2 = x[D // 2:]
    half = D // 2
    rotated_half = -x2 * sinv[half:] + x1 * cosv[half:]
    y = x * cosv - rotated_half
    tl.store(out_ptr + idx, y)


# Triton kernel: apply rotation to a single row [D] for key using sin-based rotation:
# y = x * sin_all - rotate_half(x) * cos_all
@triton.jit
def rotate_key_row_kernel(
    x_ptr, sin_ptr, cos_ptr, out_ptr,
    D,
):
    idx = tl.arange(0, D)
    x = tl.load(x_ptr + idx)
    sinv = tl.load(sin_ptr + idx)
    cosv = tl.load(cos_ptr + idx)
    x1 = x[:D // 2]
    x2 = x[D // 2:]
    half = D // 2
    rotated_half = -x2 * cosv[half:] + x1 * sinv[half:]
    y = x * sinv - rotated_half
    tl.store(out_ptr + idx, y)


# Triton kernel: scatter rotated keys and original values into cache at cache_position[s].
# xq: [D] fp32 (rotated query), xk: [D] fp32 (rotated key), xv: [D] fp32 (original value),
# key_cache_ptr, value_cache_ptr: [B, num_kv_heads, L, D] bf16, cache_pos: [B, S] int32
@triton.jit
def scatter_update_cache_kernel(
    xq_ptr, xv_ptr, key_cache_ptr, value_cache_ptr, cache_pos_ptr,
    B, num_kv, S, D, L,
    xq_stride, xv_stride,
    key_cache_stride_b, key_cache_stride_h, key_cache_stride_l, key_cache_stride_d,
    value_cache_stride_b, value_cache_stride_h, value_cache_stride_l, value_cache_stride_d,
    cache_pos_stride_b, cache_pos_stride_s,
):
    pid = tl.program_id(0)  # one program per (b, kv_head, s)
    total = B * num_kv * S
    s = pid % S
    tmp = pid // S
    head = tmp % num_kv
    b = tmp // num_kv

    pos = tl.load(cache_pos_ptr + b * cache_pos_stride_b + s * cache_pos_stride_s)  # int32

    # Store rotated key into key_cache[b, head, pos, :]
    base_k = key_cache_ptr + b * key_cache_stride_b + head * key_cache_stride_h + pos * key_cache_stride_l
    for offs in range(0, D, 128):
        idx = offs + tl.arange(0, 128)
        mask = idx < D
        x = tl.load(xq_ptr + idx * xq_stride, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(base_k + idx * key_cache_stride_d, x, mask=mask)

    # Store original value into value_cache[b, head, pos, :]
    base_v = value_cache_ptr + b * value_cache_stride_b + head * value_cache_stride_h + pos * value_cache_stride_l
    for offs in range(0, D, 128):
        idx = offs + tl.arange(0, 128)
        mask = idx < D
        x = tl.load(xv_ptr + idx * xv_stride, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(base_v + idx * value_cache_stride_d, x, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from the original code
        self.num_q_heads = 96
        self.num_kv_heads = 8
        self.head_dim = 128
        self.rms_norm_eps = 1e-6
        # inv_freq is [D//2] float32; we will compute cos_all/sin_all per position using Triton
        self.max_position_embeddings = 262144  # not used directly

    def forward(self, query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps):
        # Shapes
        B, H_q, S, D = query.shape
        assert H_q == self.num_q_heads, "num_q_heads must be 96"
        assert key.shape == (B, self.num_kv_heads, S, D)
        assert value.shape == (B, self.num_kv_heads, S, D)
        assert position_ids.shape == (B, S)
        assert key_cache.shape == (B, self.num_kv_heads, self.max_position_embeddings, D)
        assert value_cache.shape == (B, self.num_kv_heads, self.max_position_embeddings, D)
        assert cache_position.shape == (S,)
        assert q_norm_weight.shape == (D,)
        assert k_norm_weight.shape == (D,)
        assert inv_freq.shape == (D // 2,)
        assert rms_norm_eps == self.rms_norm_eps

        # Ensure contiguous tensors (4D for query/key/value, 2D for position_ids, 4D for caches, 1D for cache_position)
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        position_ids = position_ids.contiguous()  # [B, S]
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        cache_position = cache_position.contiguous()  # [S]
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()

        # 1) RMSNorm for query and key in Triton
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        grid_q = (B * H_q * S,)
        rmsnorm_4d_kernel[grid_q](
            query, query_norm, q_norm_weight,
            B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            self.rms_norm_eps,
            num_warps=4,
        )

        grid_k = (B * self.num_kv_heads * S,)
        rmsnorm_4d_kernel[grid_k](
            key, key_norm, k_norm_weight,
            B, self.num_kv_heads, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            self.rms_norm_eps,
            num_warps=4,
        )

        # 2) Build cos_all and sin_all per token using Triton (no torch cos/sin)
        # We need [B, S, D] cos/sin. We will launch per (b, s) and write to contiguous buffers.
        # Allocate per-token cos/sin
        cos_buf = torch.empty((B, S, D), dtype=torch.float32, device=query.device)
        sin_buf = torch.empty((B, S, D), dtype=torch.float32, device=query.device)

        # Launch one kernel per (b, s)
        for b in range(B):
            for s in range(S):
                # pos = position_ids[b, s]
                pos = int(position_ids[b, s].item())
                cos_buf_b_s = cos_buf[b, s]  # [D]
                sin_buf_b_s = sin_buf[b, s]  # [D]
                build_rot_vec_kernel[(1,)](
                    pos, inv_freq, cos_buf_b_s, sin_buf_b_s,
                    D,
                    num_warps=4,
                )

        # 3) Rotate query (fp32) and key (fp32) in Triton
        # Allocate fp32 outputs
        query_rot = torch.empty((B, H_q, S, D), dtype=torch.float32, device=query.device)
        key_rot = torch.empty((B, self.num_kv_heads, S, D), dtype=torch.float32, device=query.device)

        # For query rotation: y = x * cos - rotate_half(x) * sin
        grid_qrot = (B * H_q * S,)
        # We'll compute per row in Triton: for each [b,h,s], load x, cos, sin and write y
        for pid in range(grid_qrot[0]):
            b = pid // (H_q * S)
            tmp = pid % (H_q * S)
            s = tmp % S
            h = tmp // S
            # Load row pointers
            base_x = query_norm + b * query_norm.stride(0) + h * query_norm.stride(1) + s * query_norm.stride(2)
            # Build cos/sin pointers: from cos_buf[b, s], sin_buf[b, s]
            cos_ptr = cos_buf[b, s]
            sin_ptr = sin_buf[b, s]
            # Output row pointer
            base_out = query_rot + b * query_rot.stride(0) + h * query_rot.stride(1) + s * query_rot.stride(2)
            # One program per row: we use kernel with D=128
            rotate_query_row_kernel[(1,)](
                base_x, cos_ptr, sin_ptr, base_out, D, num_warps=4
            )

        # For key rotation: y = x * sin - rotate_half(x) * cos
        grid_krot = (B * self.num_kv_heads * S,)
        for pid in range(grid_krot[0]):
            b = pid // (self.num_kv_heads * S)
            tmp = pid % (self.num_kv_heads * S)
            s = tmp % S
            h = tmp // S
            base_x = key_norm + b * key_norm.stride(0) + h * key_norm.stride(1) + s * key_norm.stride(2)
            cos_ptr = cos_buf[b, s]
            sin_ptr = sin_buf[b, s]
            base_out = key_rot + b * key_rot.stride(0) + h * key_rot.stride(1) + s * key_rot.stride(2)
            rotate_key_row_kernel[(1,)](
                base_x, sin_ptr, cos_ptr, base_out, D, num_warps=4
            )

        # 4) Scatter update caches: write rotated keys at cache_position[s], values at the same positions
        # Ensure cache_position is [B, S]
        cache_pos = cache_position.unsqueeze(0).expand(B, -1).contiguous()  # [B, S]

        grid_scatter = (B * self.num_kv_heads * S,)
        for pid in range(grid_scatter[0]):
            b = pid // (self.num_kv_heads * S)
            tmp = pid % (self.num_kv_heads * S)
            s = tmp % S
            head = tmp // S
            # pos for this (b, head, s)
            pos = int(cache_pos[b, s].item())
            # xq = query_rot[b, head, s, :], xv = value[b, head, s, :]
            xq = query_rot[b, head, s]  # [D], fp32
            xv = value[b, head, s].to(torch.float32)  # [D], fp32
            # Launch scatter kernel (one program per (b, head, s))
            scatter_update_cache_kernel[(1,)](
                xq, xv, key_cache, value_cache, cache_pos,
                B, self.num_kv_heads, S, D, self.max_position_embeddings,
                1, 1,  # xq_stride, xv_stride (contiguous D dims)
                key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
                value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
                cache_pos.stride(0), cache_pos.stride(1),
                num_warps=4,
            )

        # Return in original dtypes
        return query_rot.to(query.dtype), key_rot.to(key.dtype), key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
