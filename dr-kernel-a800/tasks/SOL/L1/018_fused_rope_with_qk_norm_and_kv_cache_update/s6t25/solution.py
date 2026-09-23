import torch
import triton
import triton.language as tl


@triton.jit
def rms_sum_kernel(x_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    batch_stride_x, h_stride_x, s_stride_x,
                    BLOCK_H: tl.constexpr):
    # Each program handles one (b, head, s) and computes sum of squares over H
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_sq = 0.0
    # Reduce across H in chunks of BLOCK_H
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
    tl.store(sum_sums_ptr + pid, sum_sq)


@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, sum_sums_ptr,
                    B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                    eps: tl.float32,
                    batch_stride_x, h_stride_x, s_stride_x,
                    batch_stride_out, h_stride_out, s_stride_out,
                    BLOCK_H: tl.constexpr):
    # Each program applies RMSNorm to one (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    sum_sq = tl.load(sum_sums_ptr + pid)
    Hf = tl.full((), H, tl.float32)
    inv_rms = 1.0 / tl.sqrt(sum_sq / Hf + eps)

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_x = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        y_vals = x_vals * inv_rms * w_vals
        offs_out = b * batch_stride_out + head * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + offs_out, y_vals, mask=mask)


@triton.jit
def rotate_sin_cos_kernel_b_s(position_ids_ptr, inv_freq_ptr, cos_ptr, sin_ptr,
                               B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                               position_stride, cos_stride0, cos_stride1, cos_stride2,
                               sin_stride0, sin_stride1, sin_stride2,
                               BLOCK_H: tl.constexpr):
    # One program per (b, s)
    pid = tl.program_id(0)
    if pid >= B * S:
        return
    b = pid // S
    s = pid % S

    pos = tl.load(position_ids_ptr + b * position_stride + s).to(tl.float32)

    # Prepare emb: [H] = [freqs, freqs], where freqs = pos * inv_freq[:H//2]
    # We'll write emb in blocks and then compute cos/sin
    half = H // 2
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        first_mask = mask & (idx < half)

        # Load inv_freq only for first half
        inv_freq_idx = idx // 2  # only valid for first half
        inv_freq_vals = tl.load(inv_freq_ptr + inv_freq_idx, mask=first_mask, other=0.0).to(tl.float32)
        emb_first = pos * inv_freq_vals
        # Duplicate to second half
        emb = tl.where(first_mask, emb_first, emb_first)  # emb[i] = emb_first[i] for first half; undefined for second half; we will not use second half here because sin/cos are per full H and we duplicate the first half

        c = tl.cos(emb)
        s_ = tl.sin(emb)

        base = b * cos_stride0 + s * cos_stride1
        tl.store(cos_ptr + base + idx * cos_stride2, c, mask=mask)
        tl.store(sin_ptr + base + idx * sin_stride2, s_, mask=mask)


@triton.jit
def apply_rotation_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                           B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                           batch_stride_x, h_stride_x, s_stride_x,
                           batch_stride_out, h_stride_out, s_stride_out,
                           cos_stride0, cos_stride1, cos_stride2,
                           sin_stride0, sin_stride1, sin_stride2,
                           BLOCK_H: tl.constexpr):
    # One program per (b, head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    head = (pid % (num_heads * S)) // S
    s = pid % S

    base = b * cos_stride0 + s * cos_stride1

    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H

        # Load x
        offs_x = b * batch_stride_x + head * h_stride_x + s * s_stride_x + idx
        x_vals = tl.load(x_ptr + offs_x, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for this (b, s)
        cos_vals = tl.load(cos_ptr + base + idx * cos_stride2, mask=mask, other=0.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + base + idx * sin_stride2, mask=mask, other=0.0).to(tl.float32)

        half = H // 2

        # Build rotated for first half and second half
        x_first = tl.load(x_ptr + offs_x, mask=(mask & (idx < half)), other=0.0).to(tl.float32)
        x_second = tl.load(x_ptr + offs_x, mask=(mask & (idx >= half)), other=0.0).to(tl.float32)

        rotated_first = -x_second  # for idx < half, rotated[idx] = -x[half + idx]
        rotated_second = x_first   # for idx >= half, rotated[idx] = x[idx - half]

        rotated_vals = tl.zeros((BLOCK_H,), dtype=tl.float32)
        rotated_vals = tl.where(idx < half, rotated_first, rotated_vals)
        rotated_vals = tl.where(idx >= half, rotated_second, rotated_vals)

        y_vals = x_vals * cos_vals + rotated_vals * sin_vals

        offs_out = b * batch_stride_out + head * h_stride_out + s * s_stride_out + idx
        tl.store(out_ptr + offs_out, y_vals, mask=mask)


@triton.jit
def update_cache_kernel(src_ptr, dest_ptr, pos_ptr,
                         B: tl.constexpr, num_heads: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                         batch_stride_src, h_stride_src, s_stride_src,
                         batch_stride_dest, kv_h_stride_dest, pos_stride,
                         BLOCK_H: tl.constexpr):
    # One program per (b, kv_head, s)
    pid = tl.program_id(0)
    if pid >= B * num_heads * S:
        return
    b = pid // (num_heads * S)
    kv_head = (pid % (num_heads * S)) // S
    s = pid % S

    dest_pos = tl.load(pos_ptr + s * pos_stride)

    # Load src row (b, kv_head, s, :)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        offs_src = b * batch_stride_src + kv_head * h_stride_src + s * s_stride_src + idx
        src_vals = tl.load(src_ptr + offs_src, mask=mask, other=0.0).to(tl.float32)

        offs_dest = b * batch_stride_dest + kv_head * kv_h_stride_dest + dest_pos * H + idx
        tl.store(dest_ptr + offs_dest, src_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
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
                rms_norm_eps: float):
        """
        Triton-optimized forward that mimics the original behavior:
        - RMSNorm for query and key (using q_norm_weight and k_norm_weight respectively)
        - Compute rotation cos/sin per (b, s)
        - Apply rotation to normalized query and key
        - Update caches at positions specified by cache_position
        """
        # Ensure CUDA tensors for Triton
        assert query.is_cuda and key.is_cuda and value.is_cuda and key_cache.is_cuda and value_cache.is_cuda, "All tensors must be on CUDA for Triton."
        assert position_ids.is_cuda and cache_position.is_cuda, "position_ids and cache_position must be CUDA tensors."
        B, num_q_heads, S, H = query.shape
        num_kv_heads = key.shape[1]

        # 1) RMS sum for query
        sum_sums_q = torch.empty((B * num_q_heads * S,), dtype=torch.float32, device=query.device)
        rms_sum_kernel[(B * num_q_heads * S,)](
            query, sum_sums_q,
            B, num_q_heads, S, H,
            query.stride(0), query.stride(1), query.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 2) RMSNorm for query -> query_norm (bfloat16)
        query_norm = torch.empty_like(query)
        rms_norm_kernel[(B * num_q_heads * S,)](
            query, q_norm_weight, query_norm, sum_sums_q,
            B, num_q_heads, S, H,
            float(rms_norm_eps),
            query.stride(0), query.stride(1), query.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 3) Compute rotation cos/sin per (b, s) in float32 -> [B, S, H]
        cos = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        sin = torch.empty((B, S, H), dtype=torch.float32, device=query.device)
        rotate_sin_cos_kernel_b_s[(B, S)](
            position_ids, inv_freq, cos, sin,
            B, S, H,
            position_ids.stride(0),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 4) Apply rotation to query_norm -> query_rot (bfloat16)
        query_rot = torch.empty_like(query_norm)
        apply_rotation_kernel[(B * num_q_heads * S,)](
            query_norm, cos, sin, query_rot,
            B, num_q_heads, S, H,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            BLOCK_H=64,
            num_warps=4
        )

        # 5) RMS


def run(*args):
    return ModelNew()(*args)
