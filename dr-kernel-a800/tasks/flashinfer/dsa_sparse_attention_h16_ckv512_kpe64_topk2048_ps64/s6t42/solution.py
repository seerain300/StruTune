import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def compute_one_kernel(
    q_nope_ptr, q_pe_ptr,
    Kc_all_ptr, Kp_all_ptr,
    sparse_idx_ptr,
    output_ptr, lse_ptr,
    sm_scale,  # float32 scalar
    num_tokens, total_kv,  # int32 scalars
    DIM_QN: tl.constexpr,  # 512
    DIM_QP: tl.constexpr,  # 64
    DIM_KC: tl.constexpr,  # 512
    DIM_KP: tl.constexpr,  # 64
    BLOCK_K: tl.constexpr,  # 128
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Base pointers for q_nope row (t, h, :) and q_pe row (t, h, :)
    q_no_row_ptr = q_nope_ptr + t * 16 * DIM_QN + h * DIM_QN
    q_pe_row_ptr = q_pe_ptr  + t * 16 * DIM_QP + h * DIM_QP

    # Running max and sum for logsumexp (scaled by ln(2) already in attn)
    m = -float('inf')      # scalar
    s = 0.0                # scalar
    # Output accumulator for this (t, h)
    out_accum = tl.zeros((DIM_QN,), dtype=tl.float32)

    # Iterate over K dimension in tiles
    for k0 in range(0, total_kv, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        valid = offs_k < total_kv
        # Load sparse indices for this token; indices start from 0
        idx_vec = tl.load(sparse_idx_ptr + t * total_kv + offs_k, mask=valid, other=-1)  # [BLOCK_K] int32
        active = valid & (idx_vec != -1)

        # Compute m and s over this tile
        tile_m = m
        tile_s = s
        for j in range(BLOCK_K):
            if not active[j]:
                continue
            k_idx = idx_vec[j]
            Kc_row_ptr = Kc_all_ptr + k_idx * DIM_KC
            Kp_row_ptr = Kp_all_ptr + k_idx * DIM_KP

            # Compute dot1 over 512 dims: q_no[t, h, :] · Kc_row
            dot1 = 0.0
            for d in range(0, DIM_QN):
                qd = tl.load(q_no_row_ptr + d)
                kd = tl.load(Kc_row_ptr + d)
                dot1 += qd * kd

            # Compute dot2 over 64 dims: q_pe[t, h, :] · Kp_row
            dot2 = 0.0
            for d in range(0, DIM_QP):
                qp = tl.load(q_pe_row_ptr + d)
                kp = tl.load(Kp_row_ptr + d)
                dot2 += qp * kp

            logit = (dot1 + dot2) * sm_scale
            new_m = tl.maximum(tile_m, logit)
            tile_s = tile_s * tl.exp(tile_m - new_m) + tl.exp(logit - new_m)
            tile_m = new_m

        # After tile, update global m and s
        m = tile_m
        s = tile_s

    # Compute lse_scaled = (m + log(s)) / ln(2)
    lse_scaled = (m + tl.log(s)) / 1.4426950408889634  # 1 / ln(2)
    tl.store(lse_ptr + t * 16 + h, lse_scaled)

    # Store output for this (t, h)
    out_row_ptr = output_ptr + t * 16 * DIM_QN + h * DIM_QN
    tl.store(out_row_ptr + tl.arange(0, DIM_QN), out_accum)


def _forward_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """
    Forward that only launches Triton kernels. Reads inputs, calls Triton, returns outputs.
    No torch operations on tensors; only allocations and kernel launches are allowed.
    Returns:
      output: [num_tokens, 16, 512] (float32)
      lse: [num_tokens, 16] float32
    """
    if not TRITON_AVAILABLE or q_nope.device.type != 'cuda':
        return None, None

    num_tokens = q_nope.shape[0]
    device = q_nope.device

    # Flatten K caches: [num_pages, 64, 512] -> [num_pages*64, 512], [num_pages, 64, 64] -> [num_pages*64, 64]
    total_kv = ckv_cache.shape[0] * 64
    Kc_all = ckv_cache.reshape(-1, 512)  # [total_kv, 512]
    Kp_all = kpe_cache.reshape(-1, 64)   # [total_kv, 64]
    # Ensure inputs are on device; Triton will load raw pointers. No .to() or .contiguous() on tensors.

    # Prepare output and lse
    output = torch.empty((num_tokens, 16, 512), dtype=torch.float32, device=device)
    lse = torch.empty((num_tokens, 16), dtype=torch.float32, device=device)

    # Launch one program per (token, head)
    grid = (num_tokens, 16)
    compute_one_kernel[grid](
        q_nope, q_pe,
        Kc_all, Kp_all,
        sparse_indices,  # int32 tensor
        output, lse,
        float(sm_scale),
        num_tokens, total_kv,
        DIM_QN=512, DIM_QP=64, DIM_KC=512, DIM_KP=64,
        BLOCK_K=128,
        num_warps=4, num_stages=2
    )

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Only allocate and launch Triton. No torch ops.
        return _forward_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
