import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (token, head)
@triton.jit
def compute_one_kernel(
    q_nope_ptr, q_pe_ptr,
    Kc_all_ptr, Kp_all_ptr,
    sparse_idx_ptr,
    out_ptr, lse_ptr,
    sm_scale, ln2_inv,  # sm_scale (float) and 1/ln(2) (float)
    total_kv: tl.constexpr,  # num_pages * 64
    DIM_QN: tl.constexpr,    # 512
    DIM_QP: tl.constexpr,    # 64
    DIM_KC: tl.constexpr,    # 512
    DIM_KP: tl.constexpr,    # 64
    BLOCK_K: tl.constexpr,   # tile size, e.g., 128
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Base strides for q_nope row (t, h, :) and q_pe row (t, h, :)
    q_no_row_ptr = q_nope_ptr + t * DIM_QN + h * 0  # q_nope is [T, 16, 512], row size = 512
    q_pe_row_ptr = q_pe_ptr  + t * DIM_QP + h * 0  # q_pe is [T, 16, 64], row size = 64

    # Running max and sum for logsumexp (scaled)
    m = -float('inf')
    s = 0.0

    # Output accumulator (will be stored as bfloat16)
    out_accum = tl.zeros((DIM_QN,), dtype=tl.float32)

    # Iterate over K dimension in tiles
    for k0 in range(0, total_kv, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        valid = offs_k < total_kv
        # Load sparse indices for this token
        idx_vec = tl.load(sparse_idx_ptr + t * total_kv + offs_k, mask=valid, other=-1)  # [BLOCK_K] int32
        active = valid & (idx_vec != -1)  # [BLOCK_K] boolean mask

        # Process each active K entry in the tile
        for j in range(BLOCK_K):
            if not active[j]:
                continue
            k_idx = idx_vec[j]  # int32 index into Kc_all / Kp_all rows
            # Pointer to the k-th row in Kc_all and Kp_all
            Kc_row_ptr = Kc_all_ptr + k_idx * DIM_KC
            Kp_row_ptr = Kp_all_ptr + k_idx * DIM_KP

            # Compute dot1 over 512 dims (q_nope row · Kc_row)
            dot1 = 0.0
            for d in range(0, DIM_QN):
                qd = tl.load(q_no_row_ptr + d)  # q_nope[t, h, d]
                kd = tl.load(Kc_row_ptr + d)    # Kc_all[k_idx, d]
                dot1 += qd * kd
            # Compute dot2 over 64 dims (q_pe row · Kp_row)
            dot2 = 0.0
            for d in range(0, DIM_QP):
                qp = tl.load(q_pe_row_ptr + d)  # q_pe[t, h, d]
                kp = tl.load(Kp_row_ptr + d)    # Kp_all[k_idx, d]
                dot2 += qp * kp
            # Scaled logit
            logit = (dot1 + dot2) * sm_scale

            # Update running max and sum for numerical stability
            # new_m = max(m, logit), new_s = s*exp(m-new_m) + exp(logit - new_m) if new_m == logit else s + exp(logit - m)
            # Implement via branch
            if logit > m:
                s = s * tl.exp(m - logit)
                m = logit
            # else keep m, add exp(logit - m)
            s += tl.exp(logit - m)

    # Compute lse = (m + log(s)) * ln2_inv (ln2_inv = 1/ln(2))
    lse_val = (m + tl.log(s)) * ln2_inv
    tl.store(lse_ptr + t * 16 + h, lse_val)

    # Recompute attn for each K entry and accumulate output
    # Iterate again over tiles; we recompute logit per entry to get attn
    for k0 in range(0, total_kv, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        valid = offs_k < total_kv
        idx_vec = tl.load(sparse_idx_ptr + t * total_kv + offs_k, mask=valid, other=-1)
        active = valid & (idx_vec != -1)

        for j in range(BLOCK_K):
            if not active[j]:
                continue
            k_idx = idx_vec[j]
            Kc_row_ptr = Kc_all_ptr + k_idx * DIM_KC
            Kp_row_ptr = Kp_all_ptr + k_idx * DIM_KP

            dot1 = 0.0
            for d in range(0, DIM_QN):
                qd = tl.load(q_no_row_ptr + d)
                kd = tl.load(Kc_row_ptr + d)
                dot1 += qd * kd
            dot2 = 0.0
            for d in range(0, DIM_QP):
                qp = tl.load(q_pe_row_ptr + d)
                kp = tl.load(Kp_row_ptr + d)
                dot2 += qp * kp
            logit = (dot1 + dot2) * sm_scale

            # attn = exp(logit - m) / (s * ln2_inv)
            attn = tl.exp(logit - m) * (1.0 / (s * ln2_inv))
            # Accumulate output
            for d in range(0, DIM_QN):
                kd_out = tl.load(Kc_row_ptr + d)
                out_accum[d] += attn * kd_out

    # Store output for this (t, h) as bfloat16
    out_row_ptr = out_ptr + t * 16 * 512 + h * 512
    # Cast to bfloat16 for storage
    # Triton stores float32 pointer with bfloat16 values by casting if target tensor is bfloat16
    # Ensure we write bfloat16; Triton will handle dtype if out_ptr is bfloat16 tensor
    tl.store(out_row_ptr + tl.arange(0, 512), out_accum)


def _forward_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """
    Forward that only launches Triton kernels. No torch operations on tensors.
    Returns:
      output: [num_tokens, 16, 512], dtype bfloat16
      lse: [num_tokens, 16], dtype float32
    """
    assert TRITON_AVAILABLE and q_nope.device.type == 'cuda', "Triton/CUDA required"
    num_tokens = q_nope.shape[0]
    device = q_nope.device

    # Flatten caches for row-wise access
    Kc_all = ckv_cache.reshape(-1, 512)  # [num_pages*64, 512]
    Kp_all = kpe_cache.reshape(-1, 64)   # [num_pages*64, 64]

    # Allocate outputs
    output = torch.empty((num_tokens, 16, 512), dtype=torch.bfloat16, device=device)
    lse = torch.empty((num_tokens, 16), dtype=torch.float32, device=device)

    # Launch: one program per (token, head)
    grid = (num_tokens, 16)
    # total_kv is num_pages * 64 (from shape); here 8462 * 64 = 541568
    total_kv = ckv_cache.shape[0] * 64

    # sm_scale and ln(2) inverse
    sm_scale = float(sm_scale)
    ln2_inv = 1.4426950408889634  # 1 / ln(2)

    compute_one_kernel[grid](
        q_nope, q_pe,
        Kc_all, Kp_all,
        sparse_indices,  # int32
        output, lse,
        sm_scale, ln2_inv,
        total_kv,
        DIM_QN=512, DIM_QP=64, DIM_KC=512, DIM_KP=64,
        BLOCK_K=128,
        num_warps=4, num_stages=2
    )

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Triton-only forward: no torch ops
        return _forward_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
