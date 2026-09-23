import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    X_ptr, Out_ptr, Weight_ptr,
    B, H, S, D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (b, h, s) row
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    x_row_ptr = X_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + s * out_stride_s

    # Accumulate sum of squares across D
    sum_sq = 0.0
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x)

    mean = sum_sq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Scale and multiply by weight, store in fp32
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(Weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(out_row_ptr + idx * out_stride_d, y, mask=mask)


# Triton kernel: rotate query (cos-based rotation). Computes cos_all and sin_all inside-kernel.
@triton.jit
def rotate_query_kernel(
    X_ptr, Out_ptr, Positions_ptr, InvFreq_ptr,
    B, H, S, D, HALF_D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(Positions_ptr + b * S + s).to(tl.float32)  # position as float32 for angle
    x_row_ptr = X_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + s * out_stride_s

    # Build cos_all: [D] = cos(angle) cat cos(angle)
    cos_all = tl.zeros([D], dtype=tl.float32)
    for j in range(0, HALF_D):
        angle_j = pos * tl.load(InvFreq_ptr + j)  # scalar
        cos_j = tl.cos(angle_j)
        # Place into even indices [0,2,4,...,D-2]
        for i in range(0, D, 2):
            idx = i + tl.arange(0, 1)  # create a vector for update
            cos_all = tl.where(idx == i, cos_all + cos_j, cos_all)

    # Build sin_all: [D] = sin(angle) cat sin(angle)
    sin_all = tl.zeros([D], dtype=tl.float32)
    for j in range(0, HALF_D):
        angle_j = pos * tl.load(InvFreq_ptr + j)
        sin_j = tl.sin(angle_j)
        for i in range(0, D, 2):
            idx = i + tl.arange(0, 1)
            sin_all = tl.where(idx == i, sin_all + sin_j, sin_all)

    # Compute rotate_half(x): [-x[D//2:], x[:D//2]]
    half = D // 2
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        x_half = x[half:]
        x_first = x[:half]
        x_rot = tl.concatenate([-x_half, x_first], axis=0)  # shape [D]
        cos_chunk = cos_all[offs : offs + BLOCK_SIZE]
        sin_chunk = sin_all[offs : offs + BLOCK_SIZE]
        y = x * cos_chunk - x_rot * sin_chunk
        tl.store(out_row_ptr + idx * out_stride_d, y, mask=mask)


# Triton kernel: rotate key (sin-based rotation). Computes sin_all and cos_all inside-kernel.
@triton.jit
def rotate_key_kernel(
    X_ptr, Out_ptr, Positions_ptr, InvFreq_ptr,
    B, H, S, D, HALF_D,
    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(Positions_ptr + b * S + s).to(tl.float32)
    x_row_ptr = X_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + s * out_stride_s

    # Build sin_all: [D] = sin(angle) cat sin(angle)
    sin_all = tl.zeros([D], dtype=tl.float32)
    for j in range(0, HALF_D):
        angle_j = pos * tl.load(InvFreq_ptr + j)
        sin_j = tl.sin(angle_j)
        for i in range(0, D, 2):
            idx = i + tl.arange(0, 1)
            sin_all = tl.where(idx == i, sin_all + sin_j, sin_all)

    # Build cos_all similarly
    cos_all = tl.zeros([D], dtype=tl.float32)
    for j in range(0, HALF_D):
        angle_j = pos * tl.load(InvFreq_ptr + j)
        cos_j = tl.cos(angle_j)
        for i in range(0, D, 2):
            idx = i + tl.arange(0, 1)
            cos_all = tl.where(idx == i, cos_all + cos_j, cos_all)

    # Compute rotate_half(x): [-x[D//2:], x[:D//2]]
    half = D // 2
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row_ptr + idx * x_stride_d, mask=mask, other=0.0).to(tl.float32)
        x_half = x[half:]
        x_first = x[:half]
        x_rot = tl.concatenate([-x_half, x_first], axis=0)  # shape [D]
        sin_chunk = sin_all[offs : offs + BLOCK_SIZE]
        cos_chunk = cos_all[offs : offs + BLOCK_SIZE]
        y = x * sin_chunk - x_rot * cos_chunk
        tl.store(out_row_ptr + idx * out_stride_d, y, mask=mask)


# Triton kernel: scatter update cache for keys and values.
# Updates: Out[b, h, cache_position[b, s], :] = Src[b, h, s, :]
@triton.jit
def scatter_update_cache_kernel(
    Src_ptr, Out_ptr,
    B, H, S, D,
    src_stride_b, src_stride_h, src_stride_s, src_stride_d,
    out_stride_b, out_stride_h, out_stride_l, out_stride_d,
    positions_ptr,  # int32 positions [B, S]
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    pos = tl.load(positions_ptr + b * S + s)  # int32
    src_row_ptr = Src_ptr + b * src_stride_b + h * src_stride_h + s * src_stride_s
    out_row_ptr = Out_ptr + b * out_stride_b + h * out_stride_h + pos * out_stride_l

    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        src_val = tl.load(src_row_ptr + idx * src_stride_d, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_row_ptr + idx * out_stride_d, src_val, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, *args):
        # args are: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq
        assert len(args) == 10, "ModelNew.forward expects exactly 10 inputs."
        query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq = args

        device = query.device
        dtype = query.dtype  # typically bfloat16

        B, H_q, S, D = query.shape
        # The original code uses num_attention_heads=96 for query and num_key_value_heads=8.
        # We'll respect H_q and H_key=H_q//2 to match the input shapes. In the provided get_inputs,
        # key/value have shape (B, 8, S, D) but H_q is 96? Wait: The provided get_inputs uses
        # num_q_heads=96, num_kv_heads=8, but the typical shapes are query [B, 96, S, D],
        # key/value [B, 8, S, D]. However, our initial code used 96 for both. To be safe, we'll
        # use H_q = query.shape[1], H_key = key.shape[1]. Here, key.shape is (B, H_q//2, S, D).
        H_key = key.shape[1]
        assert H_key == H_q // 2, "Key/Value head count must be half of query heads."

        # 1) RMSNorm: query and key (compute in fp32)
        query_norm = torch.empty(query.shape, device=device, dtype=torch.float32)
        key_norm = torch.empty(key.shape, device=device, dtype=torch.float32)

        rms_norm_kernel[(B, H_q, S)](
            query, query_norm, q_norm_weight,
            B, H_q, S, D,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            self.eps,
            BLOCK_SIZE=128,
        )

        rms_norm_kernel[(B, H_key, S)](
            key, key_norm, k_norm_weight,
            B, H_key, S, D,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            self.eps,
            BLOCK_SIZE=128,
        )

        # 2) Rotate query and key using Triton (compute cos/sin inside kernels)
        query_rot = torch.empty(query.shape, device=device, dtype=torch.float32)
        key_rot = torch.empty(key.shape, device=device, dtype=torch.float32)


def run(*args):
    return ModelNew()(*args)
