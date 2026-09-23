import torch
import triton
import triton.language as tl

# -----------------------
# Triton kernels
# -----------------------

@triton.jit
def rms_sum_kernel(query_ptr, sum_ptr,
                    B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                    H: tl.constexpr,
                    batch_stride_q, h_stride_q, s_stride_q,
                    num_warps=4):
    # One program per (b, h, s)
    pid = tl.program_id(0)  # in [0, B * num_q_heads * S)
    total = B * num_q_heads * S
    # decode indices
    b = pid // (num_q_heads * S)
    rem = pid % (num_q_heads * S)
    h = rem // S
    s = rem % S
    # compute sum of squares across head_dim
    sum_val = tl.zeros((), dtype=tl.float32)
    for d in range(0, H):
        offs = b * batch_stride_q + h * h_stride_q + s * s_stride_q + d
        x = tl.load(query_ptr + offs, mask=True)  # bfloat16
        x = x.to(tl.float32)
        sum_val += x * x
    # store per-token sum for later normalization
    tl.store(sum_ptr + pid, sum_val)


@triton.jit
def rms_norm_kernel(query_ptr, q_norm_weight_ptr, output_ptr, sum_ptr,
                    B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                    H: tl.constexpr,
                    batch_stride_q, h_stride_q, s_stride_q,
                    batch_stride_out, h_stride_out, s_stride_out,
                    eps: tl.constexpr,
                    num_warps=4):
    # One program per (b, h, s)
    pid = tl.program_id(0)  # in [0, B * num_q_heads * S)
    total = B * num_q_heads * S
    b = pid // (num_q_heads * S)
    rem = pid % (num_q_heads * S)
    h = rem // S
    s = rem % S
    sum_val = tl.load(sum_ptr + pid)  # float32
    mean = sum_val / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)  # float32
    # load query, apply normalization and weight
    for d in range(0, H):
        offs_in = b * batch_stride_q + h * h_stride_q + s * s_stride_q + d
        x = tl.load(query_ptr + offs_in, mask=True)  # bfloat16
        x = x.to(tl.float32)
        x = x * inv_rms
        w = tl.load(q_norm_weight_ptr + d, mask=True)  # bfloat16
        w = w.to(tl.float32)
        y = x * w
        offs_out = b * batch_stride_out + h * h_stride_out + s * s_stride_out + d
        tl.store(output_ptr + offs_out, y)


@triton.jit
def rotate_sin_cos_kernel(position_ids_ptr, inv_freq_ptr,
                           cos_ptr, sin_ptr,
                           B: tl.constexpr, S: tl.constexpr,
                           head_dim: tl.constexpr,  # H
                           num_warps=4):
    # One program per token s
    pid = tl.program_id(0)  # in [0, S)
    s = pid
    # load position id
    pos = tl.load(position_ids_ptr + 0, mask=True)  # but we need per-s position
    # Note: position_ids has shape [B, S]; we pass it as 1D but must index per s.
    # To be robust, we can read position_ids[b, s] by passing B as constexpr and computing b via program_id(1),
    # but here we assume forward passes a 1D tensor of [S]. We'll read from global position_ids_ptr at index s.
    # However, position_ids is [B, S]; we need to pick b. We'll fix b by launching grid (B, S).
    # Instead, we relaunch: better approach is to pass a 1D position_ids and assume b=0, or compute b from pid.
    # For simplicity, assume we launch grid (B, S) and read position_ids[b, s] via pointer arithmetic.
    # We'll restructure launch to pass b as program_id(1).
    # Relaunch suggestion: define kernel with grid (S,). This kernel cannot read b directly.
    # Therefore, we revise: we launch grid (B, S) and pass b via program_id(1). Here we redefine the kernel.

    # Below is the corrected rotate_sin_cos_kernel reading per-batch per-token position:
    # We need to define a kernel with two dims: b and s. Triton permits two program_id axes. We'll define it.

    # Since Triton doesn't allow redefining here, we keep an alternative implementation below.
    # For now, this kernel will not be used as it cannot read b. We will provide the correct kernel next.

    pass  # placeholder; actual kernel below.


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr,
                               cos_ptr, sin_ptr,
                               B: tl.constexpr, S: tl.constexpr,
                               head_dim: tl.constexpr,  # H
                               num_warps=4):
    # Two-dimensional grid: (B, S)
    b = tl.program_id(0)
    s = tl.program_id(1)
    # Load position id for (b, s)
    pos = tl.load(position_ids_ptr + b * S + s)  # int64
    # Build emb vector of length H: emb[i] = pos * inv_freq[i//2] (i < H//2), else duplicated
    emb = tl.zeros((head_dim,), dtype=tl.float32)
    half = head_dim // 2
    for i in range(0, half):
        val = pos.to(tl.float32) * tl.load(inv_freq_ptr + i)
        emb[i] = val
        emb[i + half] = val
    # Compute cos/sin
    for i in range(0, head_dim):
        emb_i = emb[i]
        c = tl.cos(emb_i)
        s_i = tl.sin(emb_i)
        tl.store(cos_ptr + b * (S * head_dim) + s * head_dim + i, c)
        tl.store(sin_ptr + b * (S * head_dim) + s * head_dim + i, s_i)


@triton.jit
def apply_rotation_kernel(query_norm_ptr, key_norm_ptr,
                           cos_ptr, sin_ptr,
                           query_rot_ptr, key_rot_ptr,
                           B: tl.constexpr, num_q_heads: tl.constexpr, S: tl.constexpr,
                           H: tl.constexpr,
                           batch_stride_qn, h_stride_qn, s_stride_qn,
                           batch_stride_qr, h_stride_qr, s_stride_qr,
                           batch_stride_kn, h_stride_kn, s_stride_kn,
                           batch_stride_kr, h_stride_kr, s_stride_kr,
                           num_warps=4):
    # One program per (b, h, s)
    pid = tl.program_id(0)  # in [0, B * num_q_heads * S)
    total = B * num_q_heads * S
    b = pid // (num_q_heads * S)
    rem = pid % (num_q_heads * S)
    h = rem // S
    s = rem % S

    # First half vector indices
    for d in range(0, H):
        # Load x, cos, sin for this token (scalar)
        # We need per-token cos/sin vectors, which we precomputed in rotate_sin_cos_kernel and passed as tensors.
        # Here, we assume cos_ptr/sin_ptr are [B, S, H] contiguous, so offset = b*(S*H) + s*H + d
        cos_val = tl.load(cos_ptr + b * (S * H) + s * H + d)
        sin_val = tl.load(sin_ptr + b * (S * H) + s * H + d)

        x = tl.load(query_norm_ptr + b * batch_stride_qn + h * h_stride_qn + s * s_stride_qn + d, mask=True).to(tl.float32)
        y1 = x * cos_val
        y2 = -tl.load(key_norm_ptr + b * batch_stride_kn + h * h_stride_kn + s * s_stride_kn + (d + H // 2), mask=True).to(tl.float32) * sin_val
        y = y1 + y2
        tl.store(query_rot_ptr + b * batch_stride_qr + h * h_stride_qr + s * s_stride_qr + d, y)


@triton.jit
def update_cache_kernel(key_rot_ptr, value_ptr,
                        key_cache_ptr, value_cache_ptr,
                        cache_position_ptr,  # int64 [S]
                        B: tl.constexpr, num_kv_heads: tl.constexpr, S: tl.constexpr,
                        H: tl.constexpr,
                        batch_stride_kr, h_stride_kr, s_stride_kr, d_stride_kr,
                        batch_stride_v, h_stride_v, s_stride_v, d_stride_v,
                        batch_stride_kc, h_stride_kc, dest_stride_kc, d_stride_kc,
                        batch_stride_vc, h_stride_vc, dest_stride_vc, d_stride_vc,
                        num_warps=4):
    # Grid: (B, num_kv_heads * S)
    pid_b = tl.program_id(0)
    pid2 = tl.program_id(1)
    k = pid2 // S
    s = pid2 % S
    dest = tl.load(cache_position_ptr + s).to(tl.int64)

    arange = tl.arange(0, H)
    key_vals = tl.load(key_rot_ptr + pid_b * batch_stride_kr + k * h_stride_kr + s * s_stride_kr + arange * d_stride_kr, mask=True).to(tl.float32)
    tl.store(key_cache_ptr + pid_b * batch_stride_kc + k * h_stride_kc + dest * dest_stride_kc + arange * d_stride_kc, key_vals)

    vals = tl.load(value_ptr + pid_b * batch_stride_v + k * h_stride_v + s * s_stride_v + arange * d_stride_v, mask=True).to(tl.float32)
    tl.store(value_cache_ptr + pid_b * batch_stride_vc + k * h_stride_vc + dest * dest_stride_vc + arange * d_stride_vc, vals)


# -----------------------
# ModelNew.forward
# -----------------------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value,
                position_ids,  # [B, S], int64
                key_cache, value_cache,
                cache_position,  # [S], int64
                q_norm_weight, k_norm_weight,  # [H], bfloat16
                inv_freq,  # [H//2], float32
                rms_norm_eps):
        # Ensure contiguity
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        inv_freq = inv_freq.contiguous()
        # Prepare outputs
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)
        query_rot = torch.empty_like(query_norm)
        key_rot = torch.empty_like(key_norm)

        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # 1) RMSNorm sums
        sum_sums = torch.empty(B * num_q_heads * S, dtype=torch.float32, device=query.device)
        grid_rms_sum = (B * num_q_heads * S,)
        rms_sum_kernel[grid_rms_sum](query, sum_sums,
                                     B=B, num_q_heads=num_q_heads, S=S,
                                     H=H,
                                     batch_stride_q=query.stride(0), h_stride_q=query.stride(1),
                                     s_stride_q=query.stride(2))

        # 2) RMSNorm normalization for query
        grid_norm_q = (B * num_q_heads * S,)
        rms_norm_kernel[grid_norm_q](query, q_norm_weight, query_norm,
                                     sum_sums,
                                     B=B, num_q_heads=num_q_heads, S=S,
                                     H=H,
                                     batch_stride_q=query.stride(0), h_stride_q=query.stride(1),
                                     s_stride_q=query.stride(2),
                                     batch_stride_out=query_norm.stride(0), h_stride_out=query_norm.stride(1),
                                     s_stride_out=query_norm.stride(2),
                                     eps=rms_norm_eps)

        # 3) RMSNorm normalization for key (same as above, but with key and k_norm_weight)
        sum_sums_k = torch.empty(B * num_q_heads * S, dtype=torch.float32, device=query.device)
        rms_sum_kernel[grid_rms_sum](key, sum_sums_k,
                                     B=B, num_q_heads=num_q_heads, S=S,
                                     H=H,
                                     batch_stride_q=key.stride(0), h_stride_q=key.stride(1),
                                     s_stride_q=key.stride(2))
        grid_norm_k = (B * num_q_heads * S,)
        rms_norm_kernel[grid_norm_k](key, k_norm_weight, key_norm,
                                     sum_sums_k,
                                     B=B, num_q_heads=num_q_heads, S=S,
                                     H=H,
                                     batch_stride_q=key.stride(0), h_stride_q=key.stride(1),
                                     s_stride_q=key.stride(2),
                                     batch_stride_out=key_norm.stride(0), h_stride_out=key_norm.stride(1),
                                     s_stride_out=key_norm.stride(2),
                                     eps=rms_norm_eps)

        # 4) Precompute cos/sin for rotation using Triton kernel over (B, S)
        # position_ids is [B, S], make contiguous 1D for per-token indexing by reading b from grid dims.
        # We pass as 1D and read per b, s in kernel. To do this robustly, we launch grid (B, S) and read position_ids[b, s].
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        grid_rc = (B, S)
        rotate_sin_cos_kernel[grid_rc](position_ids, inv_freq, cos, sin,
                                       B=B, S=S, head_dim=H)

        # 5) Apply rotation to query and key
        grid_rot = (B * num_q_heads * S,)
        apply_rotation_kernel[grid_rot](query_norm, key_norm,
                                        cos, sin,
                                        query_rot, key_rot,
                                        B=B, num_q_heads=num_q_heads, S=S,
                                        H=H,
                                        batch_stride_qn=query_norm.stride(0), h_stride_qn=query_norm.stride(1), s_stride_qn=query_norm.stride(2),
                                        batch_stride_qr=query_rot.stride(0), h_stride_qr=query_rot.stride(1), s_stride_qr=query_rot.stride(2),
                                        batch_stride_kn=key_norm.stride(0), h_stride_kn=key_norm.stride(1), s_stride_kn=key_norm.stride(2),
                                        batch_stride_kr=key_rot.stride(0), h_stride_kr=key_rot.stride(1), s_stride_kr=key_rot.stride(2),
                                        num_warps=4)

        # 6) Update caches at positions cache_position[s] for each (b, k, s)
        grid_update = (B, num_kv_heads * S)
        update_cache_kernel[grid_update](key_rot, value,
                                         key_cache, value_cache,
                                         cache_position,
                                         B=B, num_kv_heads=num_kv_heads, S=S,
                                         H=H,
                                         batch_stride_kr=key_rot.stride(0), h_stride_kr=key_rot.stride(1), s_stride_kr=key_rot.stride(2), d_stride_kr=1,
                                         batch_stride_v=value.stride(0), h_stride_v=value.stride(1), s_stride_v=value.stride(2), d_stride_v=1,
                                         batch_stride_kc=key_cache.stride(0), h_stride_kc=key_cache.stride(1),
                                         dest_stride=key_cache.stride(2), d_stride_kc=1,
                                         batch_stride_vc=value_cache.stride(0), h_stride_vc=value_cache.stride(1),
                                         dest_stride_vc=value_cache.stride(2), d_stride_vc=1,
                                         num_warps=4)

        return query_rot, key_rot, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
