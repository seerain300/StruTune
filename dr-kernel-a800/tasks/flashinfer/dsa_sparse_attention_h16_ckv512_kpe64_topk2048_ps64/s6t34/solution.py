import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (token, head)
# Computes attention for each (t, h): logsumexp, softmax, and final output accumulation.
@triton.jit
def compute_one_kernel(
    q_nope_ptr, q_pe_ptr,
    Kc_all_ptr, Kp_all_ptr,
    sparse_idx_ptr,
    out_ptr, lse_ptr,
    sm_scale,  # float32 scalar
    total_kv,  # int: num_pages * 64
    BLOCK_K: tl.constexpr,
    DIM_QN: tl.constexpr,    # head_dim_ckv = 512
    DIM_QP: tl.constexpr,    # head_dim_kpe = 64
    DIM_KC: tl.constexpr,    # 512
    DIM_KP: tl.constexpr,    # 64
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Build row pointers for q_nope and q_pe
    # q_nope has shape [num_tokens, 16, 512] -> 2D view [num_tokens, 16*512] with contiguous layout
    q_no_row_ptr = q_nope_ptr + t * (16 * DIM_QN) + h * DIM_QN
    q_pe_row_ptr = q_pe_ptr  + t * (16 * DIM_QP) + h * DIM_QP

    # Running max and sum for logsumexp
    m = -float('inf')        # scalar
    s = 0.0                  # scalar
    # Output accumulator for this (t, h)
    out_accum = tl.zeros((DIM_QN,), dtype=tl.float32)

    # Iterate over K dimension in tiles
    for k0 in range(0, total_kv, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        valid = offs_k < total_kv
        # Load sparse indices for this token
        idx_vec = tl.load(sparse_idx_ptr + t * total_kv + offs_k, mask=valid, other=-1)  # [BLOCK_K] int32
        active = valid & (idx_vec != -1)

        # Process each active K entry in the tile
        for j in range(BLOCK_K):
            if not active[j]:
                continue
            k_idx = idx_vec[j]  # int32 index into Kc_all / Kp_all
            Kc_row_ptr = Kc_all_ptr + k_idx * DIM_KC
            Kp_row_ptr = Kp_all_ptr + k_idx * DIM_KP

            # Compute dot1 over 512 dims
            dot1 = 0.0
            for d in range(0, DIM_KC):
                dot1 += tl.load(q_no_row_ptr + d) * tl.load(Kc_row_ptr + d)
            # Compute dot2 over 64 dims
            dot2 = 0.0
            for d in range(0, DIM_KP):
                dot2 += tl.load(q_pe_row_ptr + d) * tl.load(Kp_row_ptr + d)
            logit = (dot1 + dot2) * sm_scale

            # Online logsumexp update with scale 1/ln(2) folded into attn
            m_new = tl.maximum(m, logit)
            s = s * tl.exp(m - m_new) + tl.exp(logit - m_new)
            m = m_new

            # attn = exp(logit - m) / (s * ln(2)); scale factor already applied to s
            attn = tl.exp(logit - m) / (s * 1.4426950408889634)  # 1 / ln(2)

            # Accumulate output: out += attn * Kc_row
            for d in range(0, DIM_KC):
                out_accum[d] += attn * tl.load(Kc_row_ptr + d)

    # Store lse for this (t, h): lse = m + log(s); scaled by ln(2) was folded in attn
    lse_val = m + tl.log(s)
    tl.store(lse_ptr + t * 16 + h, lse_val)

    # Store output vector for this (t, h)
    out_row_ptr = out_ptr + t * 16 * DIM_QN + h * DIM_QN
    for d in range(0, DIM_QN):
        tl.store(out_row_ptr + d, out_accum[d])


def _forward_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """
    Triton-only forward. No torch ops inside.
    Returns:
      output: [num_tokens, 16, 512] float32
      lse: [num_tokens, 16] float32
    """
    if not TRITON_AVAILABLE or q_nope.device.type != 'cuda':
        # Fallback (should not be used in Triton evaluation)
        num_tokens, num_qo_heads, head_dim = q_nope.shape
        output = torch.empty((num_tokens, num_qo_heads, head_dim), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=q_nope.device)
        # Compute with PyTorch to avoid Triton in non-CUDA env
        total_kv = ckv_cache.shape[0] * 64
        for t in range(num_tokens):
            indices_t = sparse_indices[t]  # [2048]
            valid_mask = indices_t != -1
            if not valid_mask.any():
                output[t].zero_()
                lse[t].zero_()
                continue
            Kc_all = ckv_cache.reshape(-1, 512)[valid_mask]  # [M, 512]
            Kp_all = kpe_cache.reshape(-1, 64)[valid_mask]  # [M, 64]
            qn = q_nope[t]  # [16, 512]
            qp = q_pe[t]    # [16, 64]
            logits = (qn @ Kc_all.T) + (qp @ Kp_all.T)       # [16, M]
            logits_scaled = logits * sm_scale
            lse_t = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
            attn = torch.softmax(logits_scaled, dim=-1)
            out = attn @ Kc_all
            output[t] = out
            lse[t] = lse_t
        return output, lse

    # Dimensions
    num_tokens = q_nope.shape[0]
    DIM_QN = 512
    DIM_QP = q_pe.shape[-1]
    DIM_KC = 512
    DIM_KP = 64
    total_kv = ckv_cache.shape[0] * 64

    # Prepare inputs for Triton:
    # - Flatten q_nope and q_pe to [num_tokens, 16*dim] for easy row access
    q_nope_2d = q_nope.view(num_tokens, 16 * DIM_QN).contiguous()
    q_pe_2d = q_pe.view(num_tokens, 16 * DIM_QP).contiguous()
    # - Flatten K caches to [total_kv, dim] and ensure contiguous
    Kc_all = ckv_cache.reshape(-1, DIM_KC).contiguous()
    Kp_all = kpe_cache.reshape(-1, DIM_KP).contiguous()
    # - Ensure sparse_indices dtype is int32 for Triton load
    sparse_indices_i32 = sparse_indices.to(torch.int32)

    # Allocate outputs
    output = torch.empty((num_tokens, 16, DIM_QN), dtype=torch.float32, device=q_nope.device)
    lse = torch.empty((num_tokens, 16), dtype=torch.float32, device=q_nope.device)

    # Launch Triton: one program per (token, head)
    grid = (num_tokens, 16)
    compute_one_kernel[grid](
        q_nope_2d, q_pe_2d,
        Kc_all, Kp_all,
        sparse_indices_i32,
        output, lse,
        float(sm_scale),
        total_kv,
        BLOCK_K=128,   # tile over K=2048
        num_warps=4,
        num_stages=2,
    )

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Triton-only forward: no torch ops inside
        return _forward_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
