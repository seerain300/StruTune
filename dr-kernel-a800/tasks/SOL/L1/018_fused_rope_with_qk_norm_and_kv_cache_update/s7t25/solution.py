import torch
import triton
import triton.language as tl


# ----------------------------
# Triton kernels: RMSNorm and rotation
# ----------------------------

@triton.jit
def rmsnorm_kernel(x_ptr, y_ptr, weight_ptr, B, H, S, D, eps,
                    x_stride_b, x_stride_h, x_stride_s, x_stride_d,
                    y_stride_b, y_stride_h, y_stride_s, y_stride_d):
    # Each program handles one [b, h, s] row; reduce across D and normalize
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    base_y = y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s

    sum_sq = 0.0
    for d in range(0, D):
        v = tl.load(base_x + d * x_stride_d)
        sum_sq += v * v
    mean = sum_sq / D
    inv_rms = tl.math.rsqrt(mean + eps)  # fp32 scalar

    for d in range(0, D):
        v = tl.load(base_x + d * x_stride_d)
        w = tl.load(weight_ptr + d)  # scalar weight
        out = v * inv_rms * w
        tl.store(base_y + d * y_stride_d, out)


@triton.jit
def rotate_half_2d(x_ptr, y_ptr, cos_ptr, sin_ptr, B, H, S, D,
                   x_stride_b, x_stride_h, x_stride_s, x_stride_d,
                   y_stride_b, y_stride_h, y_stride_s, y_stride_d):
    # Each program handles one [b, h, s] row; apply 2D rotation across D
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = x_ptr + b * x_stride_b + h * x_stride_h + s * x_stride_s
    base_y = y_ptr + b * y_stride_b + h * y_stride_h + s * y_stride_s

    # Load per-(b,s) cos/sin vectors
    cos_vec = tl.load(cos_ptr + s * D + tl.arange(0, D))  # [D]
    sin_vec = tl.load(sin_ptr + s * D + tl.arange(0, D))  # [D]

    for d in range(0, D):
        x = tl.load(base_x + d * x_stride_d)
        # Split into two halves; Triton supports slicing on vectors
        x1 = x[:D // 2]
        x2 = x[D // 2:]
        c = cos_vec[d]
        sn = sin_vec[d]
        rotated = x1 * c - x2 * sn
        tl.store(base_y + d * y_stride_d, rotated)


# ----------------------------
# ModelNew: Triton-optimized forward
# ----------------------------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                position_ids: torch.Tensor, key_cache: torch.Tensor,
                value_cache: torch.Tensor, cache_position: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                inv_freq: torch.Tensor, rms_norm_eps: float):
        # Shapes and asserts
        B, H_q, S, D = query.shape
        # We assume num_q_heads == num_kv_heads == 96 per provided axes; ensure keys and values match
        assert key.shape == (B, H_q, S, D), "key shape must be [B, H_q, S, D]"
        assert value.shape == (B, H_q, S, D), "value shape must be [B, H_q, S, D]"

        # Ensure CUDA tensors
        assert query.is_cuda and key.is_cuda and value.is_cuda and position_ids.is_cuda and \
               key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda, "All tensors must be on CUDA"

        # 1) RMSNorm for query and key using Triton
        q_weight = q_norm_weight.to(device=query.device, dtype=torch.float32).contiguous()
        k_weight = k_norm_weight.to(device=key.device, dtype=torch.float32).contiguous()

        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        grid_rms = (B, H_q, S)
        rmsnorm_kernel[grid_rms](
            query, query_norm, q_weight,
            B, H_q, S, D, rms_norm_eps,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
        )

        rmsnorm_kernel[grid_rms](
            key, key_norm, k_weight,
            B, H_q, S, D, rms_norm_eps,
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
        )

        # 2) Compute rotation vectors using PyTorch (host-side outside forward)
        # angle = pos * inv_freq, inv_freq has length D//2=64
        pos = position_ids.to(torch.float32)  # [B, S]
        half = D // 2
        angle = pos * inv_freq  # [B, S, half]
        # For query: cos_all_q = cos(angle), sin_all_q = sin(angle), both length D
        cos_all_q = torch.cos(angle).expand(B, S, D).contiguous()  # [B, S, D]
        sin_all_q = torch.sin(angle).expand(B, S, D).contiguous()  # [B, S, D]
        # For key: original code uses sin for keys: y = x * sin - rotate_half(x) * cos
        cos_all_k = torch.sin(angle).expand(B, S, D).contiguous()
        sin_all_k = torch.cos(angle).expand(B, S, D).contiguous()

        # Cast to appropriate dtype for Triton kernels
        cos_all_q = cos_all_q.to(query.dtype)
        sin_all_q = sin_all_q.to(query.dtype)
        cos_all_k = cos_all_k.to(key.dtype)
        sin_all_k = sin_all_k.to(key.dtype)

        # 3) Apply rotation using Triton
        query_rot = torch.empty_like(query_norm)
        key_rot = torch.empty_like(key_norm)

        grid_qrot = (B, H_q, S)
        rotate_half_2d[grid_qrot](
            query_norm, query_rot, cos_all_q, sin_all_q,
            B, H_q, S, D,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
        )

        grid_krot = (B, H_q, S)
        rotate_half_2d[grid_krot](
            key_norm, key_rot, cos_all_k, sin_all_k,
            B, H_q, S, D,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
        )

        # 4) Cache update: write rotated keys and original values at cache_position (PyTorch indexing)
        L = key_cache.shape[2]
        # cache_position is [S]; cast to long for indexing
        cp = cache_position.to(torch.int64)
        for b in range(B):
            for s in range(S):
                pos_idx = int(cp[s].item())
                # Update key_cache: [B, H_q, L, D]; assign rotated key slice
                key_rot_slice = key_rot[b, :, s, :]  # shape [H_q, D]
                value_slice = value[b, :, s, :]      # shape [H_q, D]
                # Assign into key/value caches for each head h in H_q
                for h in range(H_q):
                    key_cache[b, h, pos_idx, :] = key_rot_slice[h, :]
                    value_cache[b, h, pos_idx, :] = value_slice[h, :]

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
