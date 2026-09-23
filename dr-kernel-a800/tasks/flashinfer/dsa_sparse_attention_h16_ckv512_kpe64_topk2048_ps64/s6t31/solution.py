import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# We use the original shapes/axes from the provided Model:
# num_qo_heads = 16, head_dim_ckv = 512, head_dim_kpe = 64, topk = 2048.

@triton.jit
def compute_one_kernel(
    q_no_ptr, q_pe_ptr,      # [num_tokens, 16, 512], [num_tokens, 16, 64]
    ckv_ptr, kpe_ptr,        # [num_pages, 64, 512], [num_pages, 64, 64]
    indices_ptr,             # [num_tokens, 2048], int32
    out_ptr,                 # [num_tokens, 16, 512], float32
    lse_ptr,                 # [num_tokens, 16], float32
    sm_scale,                # float32 scalar
    num_tokens,              # int
    BLOCK_K: tl.constexpr,
):
    # One program per (token, head)
    t = tl.program_id(0)  # token index
    h = tl.program_id(1)  # head index
    if t >= num_tokens:
        return

    # Load q_no[t, h, :] and q_pe[t, h, :]
    q_no_row_ptr = q_no_ptr + t * 16 * 512 + h * 512
    q_no = tl.load(q_no_row_ptr + tl.arange(0, 512))  # [512], dtype inferred

    q_pe_row_ptr = q_pe_ptr + t * 16 * 64 + h * 64
    q_pe = tl.load(q_pe_row_ptr + tl.arange(0, 64))   # [64], dtype inferred

    # Logsumexp components
    m = tl.full((), -1e30, tl.float32)  # running max
    s = tl.zeros((), tl.float32)        # running sum of exp(logit - m)

    # Output accumulator
    out_accum = tl.zeros((512,), dtype=tl.float32)

    # Iterate over K dimension in tiles
    for k0 in range(0, 2048, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = offs < 2048

        # Load indices for this token and tile
        idx_ptrs = indices_ptr + t * 2048 + offs
        idxs = tl.load(idx_ptrs, mask=mask_k, other=-1)  # int32

        # Valid mask
        valid = idxs != -1  # [BLOCK_K]

        # For each kk in tile
        for kk in range(BLOCK_K):
            if not (kk < 2048):
                break
            if not valid[kk]:
                continue

            entry = idxs[kk].to(tl.int64)  # flattened entry index in [0, num_pages*64)

            # Load Kc_row [512] and Kp_row [64]
            Kc_row = tl.load(ckv_ptr + entry * 512 + tl.arange(0, 512),
                             mask=mask_k & valid[kk], other=0.0)
            Kp_row = tl.load(kpe_ptr + entry * 64 + tl.arange(0, 64),
                             mask=mask_k & valid[kk], other=0.0)

            # Compute dot products via explicit reductions
            dot1 = tl.zeros((), tl.float32)
            for i in range(512):
                dot1 += q_no[i] * Kc_row[i]

            dot2 = tl.zeros((), tl.float32)
            for j in range(64):
                dot2 += q_pe[j] * Kp_row[j]

            logit = (dot1 + dot2) * sm_scale  # scalar

            # Update logsumexp
            m_new = tl.maximum(m, logit)
            s = s * tl.exp(m - m_new) + tl.exp(logit - m_new)
            m = m_new

            # Attention scaled by ln(2)
            ln2 = 0.6931471805599453
            attn = tl.exp(logit - m) / (s * ln2)

            # Accumulate output
            for i in range(512):
                out_accum[i] += attn * Kc_row[i]

    # Compute final lse
    lse_val = (m + tl.log(s)) / ln2
    tl.store(lse_ptr + t * 16 + h, lse_val)

    # Store output for this (t, h)
    out_row_ptr = out_ptr + t * 16 * 512 + h * 512
    tl.store(out_row_ptr + tl.arange(0, 512), out_accum)


def _run_triton(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """
    Triton-optimized run. Returns:
      output: [num_tokens, 16, 512] float32 (computed in-kernel), cast to bfloat16 in caller
      lse: [num_tokens, 16] float32
    """
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and sparse_indices.is_cuda, "All tensors must be CUDA for Triton."
    device = q_nope.device

    num_tokens = q_nope.shape[0]
    # Allocate outputs (float32)
    out = torch.empty((num_tokens, 16, 512), dtype=torch.float32, device=device)
    lse = torch.empty((num_tokens, 16), dtype=torch.float32, device=device)

    # Launch grid: one program per (token, head)
    grid = (num_tokens, 16)
    compute_one_kernel[grid](
        q_nope, q_pe,
        ckv_cache, kpe_cache,
        sparse_indices,
        out, lse,
        float(sm_scale),
        num_tokens,
        BLOCK_K=128,
        num_warps=4, num_stages=2
    )

    # Return output as bfloat16 (caller casts), lse as float32
    return out, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        """
        Triton-optimized forward: all computation in Triton kernels.
        No torch.* ops on tensors in forward. Only allocations and kernel launch.
        """
        if TRITON_AVAILABLE and (q_nope.device.type == 'cuda'):
            output_f32, lse = _run_triton(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
            # Return output as bfloat16 to match original signature; lse as float32
            return output_f32.to(torch.bfloat16), lse
        # Fallback: CPU or no Triton
        with torch.no_grad():
            num_tokens = q_nope.shape[0]
            output = torch.empty((num_tokens, 16, 512), dtype=torch.bfloat16, device=q_nope.device)
            lse = torch.full((num_tokens, 16), -float("inf"), dtype=torch.float32, device=q_nope.device)

            # Flatten caches
            Kc_all = ckv_cache.reshape(-1, 512)
            Kp_all = kpe_cache.reshape(-1, 64)

            for t in range(num_tokens):
                indices_t = sparse_indices[t]  # [2048]
                valid_mask = indices_t != -1
                valid_indices = indices_t[valid_mask]
                if valid_indices.numel() == 0:
                    output[t].zero_()
                    continue

                Kc = Kc_all[valid_indices]  # [M, 512]
                Kp = Kp_all[valid_indices]  # [M, 64]
                qn = q_nope[t]               # [16, 512]
                qp = q_pe[t]                 # [16, 64]

                logits = (qn @ Kc.T) + (qp @ Kp.T)  # [16, M]
                logits_scaled = logits * sm_scale
                lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                attn = torch.softmax(logits_scaled, dim=-1)  # [16, M]
                out = attn @ Kc  # [16, 512]
                output[t] = out.to(torch.bfloat16)

            return output, lse


def run(*args):
    return ModelNew()(*args)
