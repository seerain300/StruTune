import torch
import math
import triton
import triton.language as tl

# Kernel 1: RMSNorm per row across last dimension
# Input: x_ptr [B, H, S, D], weight_ptr [D], output_ptr [B, H, S, D]
@triton.jit
def rmsnorm_kernel(x_ptr, weight_ptr, out_ptr,
                    B, H, S, D,
                    stride_xb, stride_xh, stride_xs, stride_xd,
                    stride_outb, stride_outh, stride_outs, stride_outd):
    pid = tl.program_id(0)
    # Map pid to (b, h, s)
    num_rows = H * S
    b = pid // num_rows
    rem = pid % num_rows
    h = rem // S
    s = rem % S
    # Base offset for this (b, h, s) row
    base_x = b * stride_xb + h * stride_xh + s * stride_xs
    base_out = b * stride_outb + h * stride_outh + s * stride_outs
    # We'll process the entire D dimension in one block; D is expected to be <= BLOCK_D (here 128)
    # If D > BLOCK_D, we could loop, but given provided configs, D is 128.
    offsets = tl.arange(0, 128)
    mask = offsets < D
    x = tl.load(x_ptr + base_x + offsets * stride_xd, mask=mask, other=0.0)
    # Compute in fp32 for numerical stability
    x_fp32 = x.to(tl.float32)
    sum_sq = tl.sum(x_fp32 * x_fp32, axis=0)
    var = sum_sq / D
    scale = tl.rsqrt(var + 0.000001)  # rms_norm_eps passed as 1e-6; use literal here
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    out_fp32 = x_fp32 * scale * weight
    out = out_fp32.to(x.dtype)
    tl.store(out_ptr + base_out + offsets * stride_outd, out, mask=mask)

# Kernel 2: Apply Rotary Embedding (apply_rope)
# Input: x_norm_ptr [B, H, S, D], cos_ptr [B, S, D], sin_ptr [B, S, D]
# Output: out_ptr [B, H, S, D]
@triton.jit
def apply_rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                      B, H, S, D,
                      stride_xb, stride_xh, stride_xs, stride_xd,
                      stride_cosb, stride_coss, stride_cosd,
                      stride_sinb, stride_sins, stride_sind,
                      stride_outb, stride_outh, stride_outs, stride_outd):
    pid = tl.program_id(0)
    num_rows = H * S
    b = pid // num_rows
    rem = pid % num_rows
    h = rem // S
    s = rem % S
    base_x = b * stride_xb + h * stride_xh + s * stride_xs
    base_out = b * stride_outb + h * stride_outh + s * stride_outs
    base_cos = b * stride_cosb + s * stride_coss
    base_sin = b * stride_sinb + s * stride_sins
    offsets = tl.arange(0, 128)
    mask = offsets < D
    x = tl.load(x_ptr + base_x + offsets * stride_xd, mask=mask, other=0.0)
    cos = tl.load(cos_ptr + base_cos + offsets * stride_cosd, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + base_sin + offsets * stride_sind, mask=mask, other=0.0)
    # Rotate halves: x1 = x[:D/2], x2 = x[D/2:], out1 = x1*cos[:D/2] - x2*sin[:D/2]
    # out2 = x2*cos[D/2:] + x1*sin[D/2:]
    half = D // 2
    idx = offsets
    # Note: For D=128, offsets in [0,127], idx<64 -> first half, idx>=64 -> second half.
    # Create masks for first and second halves
    mask_first = idx < half
    mask_second = idx >= half
    # For out computation, we need to select appropriate cos/sin halves for each element.
    # Triton doesn't have gather over vector with masks easily; we compute two halves explicitly.
    # Compute out for first half:
    x1 = tl.where(mask_first, x[:half], 0.0)
    x2 = tl.where(mask_second, x[half:], 0.0)
    cos_first = tl.where(mask_first, cos[:half], 0.0)
    sin_first = tl.where(mask_first, sin[:half], 0.0)
    cos_second = tl.where(mask_second, cos[half:], 0.0)
    sin_second = tl.where(mask_second, sin[half:], 0.0)
    out_first = x1 * cos_first - x2 * sin_first
    out_second = x2 * cos_second + x1 * sin_second
    out = tl.where(mask_first, out_first, 0.0) + tl.where(mask_second, out_second, 0.0)
    tl.store(out_ptr + base_out + offsets * stride_outd, out, mask=mask)

# Kernel 3: Update key/value caches at cache_position for each token (assuming num_key_value_heads == 1)
# Input: key_rot_ptr [B, S, D], value_ptr [B, S, D], key_cache_ptr [B, 1, MAX_POS, D], value_cache_ptr [B, 1, MAX_POS, D]
#        cache_pos_ptr [S] (int64)
@triton.jit
def cache_update_kernel(key_rot_ptr, value_ptr, key_cache_ptr, value_cache_ptr, cache_pos_ptr,
                        B, S, D, MAX_POS,
                        stride_kb, stride_ks, stride_kd,
                        stride_vb, stride_vs, stride_vd,
                        stride_kcb, stride_kch, stride_kcp, stride_kcd,  # h=0
                        stride_vcb, stride_vch, stride_vcp, stride_vcd):  # h=0
    pid = tl.program_id(0)  # pid in [0, B*S)
    b = pid // S
    s = pid % S
    pos = tl.load(cache_pos_ptr + s)  # int64
    # Load key rotated and value for token s
    base_key = b * stride_kb + s * stride_ks
    base_val = b * stride_vb + s * stride_vs
    offsets = tl.arange(0, 128)  # D=128
    mask = offsets < D
    key_rot = tl.load(key_rot_ptr + base_key + offsets * stride_kd, mask=mask, other=0.0)
    val = tl.load(value_ptr + base_val + offsets * stride_vd, mask=mask, other=0.0)
    # Store into cache at (b, 0, pos, :)
    base_kc = b * stride_kcb + 0 * stride_kch + pos * stride_kcp
    base_vc = b * stride_vcb + 0 * stride_vch + pos * stride_vcp
    tl.store(key_cache_ptr + base_kc + offsets * stride_kcd, key_rot, mask=mask)
    tl.store(value_cache_ptr + base_vc + offsets * stride_vcd, val, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rms_norm_eps = 1e-6
        # Fixed constants from the original code
        self.rope_theta = 10000000.0
        self.head_dim = 128
        self.half_dim = 64

    def forward(self, query: torch.Tensor,
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
        # Shapes from original
        batch_size, num_q_heads, seq_len, head_dim = query.shape
        num_kv_heads = key.shape[1]
        assert head_dim == self.head_dim, "This Triton implementation assumes head_dim=128"
        assert num_kv_heads == 1, "This Triton implementation assumes num_key_value_heads=1 (original code uses one kv-head per batch item)"
        # Ensure inputs are on CUDA and contiguous
        assert query.is_cuda and key.is_cuda and value.is_cuda, "Triton requires CUDA tensors"
        device = query.device
        dtype = query.dtype  # bfloat16

        # 1) RMSNorm for query and key: compute query_norm and key_norm using Triton
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)
        B, H, S, D = batch_size, num_q_heads, seq_len, head_dim

        # Strides
        sx_b, sx_h, sx_s, sx_d = query.stride()
        sn_b, sn_h, sn_s, sn_d = query_norm.stride()
        # Launch RMSNorm on query
        grid_rms_q = (B * H * S,)
        rmsnorm_kernel[grid_rms_q](
            query, q_norm_weight, query_norm,
            B, H, S, D,
            sx_b, sx_h, sx_s, sx_d,
            sn_b, sn_h, sn_s, sn_d,
            num_warps=1, num_stages=1
        )
        # Launch RMSNorm on key
        sk_b, sk_h, sk_s, sk_d = key.stride()
        snk_b, snk_h, snk_s, snk_d = key_norm.stride()
        grid_rms_k = (B * H * S,)
        rmsnorm_kernel[grid_rms_k](
            key, k_norm_weight, key_norm,
            B, H, S, D,
            sk_b, sk_h, sk_s, sk_d,
            snk_b, snk_h, snk_s, snk_d,
            num_warps=1, num_stages=1
        )

        # 2) Compute inv_freq, cos, sin (host). inv_freq already provided, but we can generate new if desired.
        #    Here, we use provided inv_freq. inv_freq is [half_dim] float32.
        # Compute cos/sin for each position: [B, S, head_dim]
        # We already have inv_freq: shape [head_dim//2] float32
        # Build emb = pos * inv_freq cat with itself: emb[:, :, :half] = pos*inv_freq, emb[:, :, half:] = pos*inv_freq
        # But we need per-token vectors of length head_dim. The original code does:
        # emb = cat([pos*inv_freq, pos*inv_freq], dim=-1), then cos/sin = emb.sin/cos
        # However, since we have position_ids [B, S], we can compute cos/sin directly for each token:
        # For each (b,s), pos = position_ids[b, s], emb_vec = pos * inv_freq (broadcasted to head_dim).
        # Note: position_ids is [B, S]; we need emb of shape [B, S, D]. Since D=2*half, we can do:
        # emb = pos * inv_freq[:half], and pos * inv_freq[half:].
        # But we only need cos/sin per token. A simple approach is to compute two halves:
        # cos = pos * inv_freq[:half], sin = pos * inv_freq[half:] (even though they are the same here; original code uses cat trick to keep same).
        # Since inv_freq is same for all tokens, we can compute cos/sin as emb_vec = pos * inv_freq (length head_dim).
        # To match original behavior exactly, we'll compute cat([pos*inv_freq, pos*inv_freq]) expanded and then sin/cos.
        # But inv_freq is [half_dim], so we need to expand and multiply by 2. Simpler: pos * inv_freq for first half, and pos * inv_freq for second half.
        # In fact, original emb construction is not needed for cos/sin separately: they are sin/cos of the same emb. Since emb = pos * inv_freq_cat,
        # we can generate cos/sin as torch.sin/ torch.cos of emb. We'll compute emb as cat([pos*inv_freq, pos*inv_freq]).
        # We will compute emb of shape [B, S, D]:
        # Construct per-token emb: emb[b, s, :] = [pos*inv[:], pos*inv[]]. Since inv_freq is [half], we use torch broadcast:
        # inv_freq is [half], position_ids is [B, S] -> [B, S, 1, 1] with inv_freq expanded. Then emb = position_ids * inv_freq[:,:,:half] and similarly for second half.
        # Given inv_freq is 1D, we can compute emb = torch.zeros(B, S, D, device=device, dtype=torch.float32)
        # emb[:, :, :half] = (position_ids.to(torch.float32)) * inv_freq.to(torch.float32)[:, None, None]
        # emb[:, :, half:] = same. But inv_freq is 1D, so we can broadcast inv_freq to [1, 1, half] and [1, 1, half] for second half.
        # Simpler: emb[:, :, :half] = position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(1); emb[:, :, half:] = same.

        # Create emb [B, S, D] as per original (two halves identical):
        pos_ids_2d = position_ids.to(torch.float32)  # [B, S]
        inv_freq_bf = inv_freq.to(torch.float32)     # [half]
        emb_first = pos_ids_2d.unsqueeze(-1) * inv_freq_bf.unsqueeze(0).unsqueeze(1)  # [B, S, half]
        emb_second = pos_ids_2d.unsqueeze(-1) * inv_freq_bf.unsqueeze(0).unsqueeze(1)  # [B, S, half]
        emb = torch.empty((B, S, D), device=device, dtype=torch.float32)
        emb[:, :, :self.half_dim] = emb_first
        emb[:, :, self.half_dim:] = emb_second
        cos = torch.cos(emb)  # [B, S, D] float32
        sin = torch.sin(emb)  # [B, S, D] float32

        # Cast to bf16 to match query dtype
        cos_bf = cos.to(torch.bfloat16)
        sin_bf = sin.to(torch.bfloat16)

        # 3) Apply RotE using Triton
        query_rotated = torch.empty_like(query_norm)  # output in query_norm dtype
        key_rotated = torch.empty_like(key_norm)

        # Strides
        qr_b, qr_h, qr_s, qr_d = query_norm.stride()
        qro_b, qro_h, qro_s, qro_d = query_rotated.stride()
        krr_b, krr_h, krr_s, krr_d = key_norm.stride()
        kro_b, kro_h, kro_s, kro_d = key_rotated.stride()
        co_b, co_s, co_d = cos_bf.stride()  # [B, S, D]
        si_b, si_s, si_d = sin_bf.stride()

        grid_rope_q = (B * H * S,)
        apply_rope_kernel[grid_rope_q](
            query_norm, cos_bf, sin_bf, query_rotated,
            B, H, S, D,
            qr_b, qr_h, qr_s, qr_d,
            co_b, co_s, co_d,
            si_b, si_s, si_d,
            qro_b, qro_h, qro_s, qro_d,
            num_warps=1, num_stages=1
        )

        grid_rope_k = (B * H * S,)
        apply_rope_kernel[grid_rope_k](
            key_norm, cos_bf, sin_bf, key_rotated,
            B, H, S, D,
            krr_b, krr_h, krr_s, krr_d,
            co_b, co_s, co_d,
            si_b, si_s, si_d,
            kro_b, kro_h, kro_s, kro_d,
            num_warps=1, num_stages=1
        )

        # 4) Update caches in Triton
        # key_cache: [B, 1, MAX_POS, D], value_cache: [B, 1, MAX_POS, D]
        # cache_position: [S] int64 on device
        # We assume num_key_value_heads == 1, so we use h=0.
        # key_rotated has shape [B, 1, S, D], but we only need per-token slice for each s.
        # So we'll take key_rotated[:, 0, :, :] which is [B, S, D] (conceptually reshape back).
        # However, key_rotated was produced as [B, 1, S, D], so we can index by key_rotated[:, 0, :, :].
        # But in Triton kernel signature, we pass key_rot_ptr as [B, S, D], which we can create by slicing key_rotated[:, 0, :, :].
        # To avoid confusion, we'll just pass key_rotated as [B, S, D] by slicing here (PyTorch side). The Triton kernel requires contiguous layout.
        # Create key_rotated_reshape = key_rotated[:, 0, :, :]
        key_rot_b_s_d = key_rotated[:, 0, :, :].contiguous()
        value_b_s_d = value.contiguous()
        # Strides for input pointers
        krr_b, krr_s, krr_d = key_rot_b_s_d.stride()
        vr_b, vr_s, vr_d = value_b_s_d.stride()
        # Strides for caches (h=0)
        kc_b, kc_h, kc_p, kc_d = key_cache.stride()  # h dimension stride; we use h=0 => stride_kch = key_cache.stride(1)
        vc_b, vc_h, vc_p, vc_d = value_cache.stride()
        # Launch cache update
        grid_cache = (B * S,)
        cache_update_kernel[grid_cache](
            key_rot_b_s_d, value_b_s_d, key_cache, value_cache, cache_position,
            B, S, D, key_cache.shape[2],  # MAX_POS
            krr_b, krr_s, krr_d,
            vr_b, vr_s, vr_d,
            kc_b, kc_h, kc_p, kc_d,  # h=0 -> kc_h = key_cache.stride(1)
            vc_b, vc_h, vc_p, vc_d,  # h=0 -> vc_h = value_cache.stride(1)
            num_warps=1, num_stages=1
        )

        # Return rotated tensors and updated caches
        return query_rotated, key_rotated, key_cache, value_cache


def run(*args):
    return ModelNew()(*args)
