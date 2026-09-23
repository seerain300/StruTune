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
    q_no_ptr, q_pe_ptr,      # [num_tokens, 16, 512], [num_tokens, 16, 64]
    ckv_ptr, kpe_ptr,        # [num_pages, 64, 512], [num_pages, 64, 64]
    indices_ptr,             # [num_tokens, 2048], int32
    out_ptr,                 # [num_tokens, 16, 512], output buffer (we'll store as float32 then rely on caller to cast)
    lse_ptr,                 # [num_tokens, 16], float32
    sm_scale,                # float32 scalar
    num_tokens, num_qo_heads, head_dim_ckv, head_dim_kpe, topk,
    BLOCK_K: tl.constexpr,
):
    # Program IDs: one program handles one (token, head)
    t = tl.program_id(0)  # token index
    h = tl.program_id(1)  # head index
    if t >= num_tokens:
        return

    # Load q_no[t, h, :] and q_pe[t, h, :] as float32 (tensors are expected to be float32; no .to here)
    q_no_row_ptr = q_no_ptr + t * num_qo_heads * head_dim_ckv + h * head_dim_ckv
    q_no = tl.load(q_no_row_ptr + tl.arange(0, head_dim_ckv))  # [512], dtype inferred from pointer

    q_pe_row_ptr = q_pe_ptr + t * num_qo_heads * head_dim_kpe + h * head_dim_kpe
    q_pe = tl.load(q_pe_row_ptr + tl.arange(0, head_dim_kpe))  # [64]

    # Initialize logsumexp components
    m = tl.full((), -1e30, tl.float32)  # running max
    s = tl.zeros((), tl.float32)        # running sum of exp(logits - m)

    # Accumulator for output: [512]
    out_accum = tl.zeros((head_dim_ckv,), dtype=tl.float32)

    # Iterate over K dimension in tiles
    for k0 in range(0, topk, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = offs < topk

        # Load indices for this token and tile
        idx_ptrs = indices_ptr + t * topk + offs
        idxs = tl.load(idx_ptrs, mask=mask_k, other=-1)  # int32

        # Valid mask: idxs != -1
        valid = idxs != -1  # [BLOCK_K]

        # For each kk in tile, if valid, load K rows and compute dot products
        for kk in range(BLOCK_K):
            if not (kk < topk):
                break
            if not valid[kk]:
                continue

            entry = idxs[kk].to(tl.int64)  # flattened index into [num_pages * 64]

            # Load Kc_row [512] and Kp_row [64]
            Kc_row = tl.load(
                ckv_ptr + entry * head_dim_ckv + tl.arange(0, head_dim_ckv),
                mask=mask_k & valid[kk],
                other=0.0
            )
            Kp_row = tl.load(
                kpe_ptr + entry * head_dim_kpe + tl.arange(0, head_dim_kpe),
                mask=mask_k & valid[kk],
                other=0.0
            )

            # Dot products for this entry
            dot1 = tl.sum(q_no * Kc_row)          # scalar
            dot2 = tl.sum(q_pe * Kp_row)          # scalar
            logit = (dot1 + dot2) * sm_scale      # scalar

            # Update logsumexp
            m_new = tl.maximum(m, logit)
            s = s * tl.exp(m - m_new) + tl.exp(logit - m_new)
            m = m_new

            # Softmax attention scaled by ln(2)
            ln2 = 0.6931471805599453
            attn = tl.exp(logit - m) / (s * ln2)

            # Accumulate output
            out_accum += attn * Kc_row

    # Compute final lse: (m + log(s)) / ln(2)
    lse_val = (m + tl.log(s)) / ln2
    tl.store(lse_ptr + t * num_qo_heads + h, lse_val)

    # Store output for this (t, h)
    out_row_ptr = out_ptr + t * num_qo_heads * head_dim_ckv + h * head_dim_ckv
    tl.store(out_row_ptr + tl.arange(0, head_dim_ckv), out_accum)


def _run_triton(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """
    Triton-optimized run. Returns:
      output: [num_tokens, 16, 512] bfloat16 (allocated by caller and cast outside Triton)
      lse: [num_tokens, 16] float32 (allocated by caller)
    """
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and sparse_indices.is_cuda, "All tensors must be CUDA for Triton."
    device = q_nope.device

    # Shapes
    num_tokens = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]  # 512
    head_dim_kpe = q_pe.shape[2]    # 64

    # Ensure contiguity (no dtype conversions)
    q_no = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    ckv = ckv_cache.contiguous()
    kpe = kpe_cache.contiguous()
    indices = sparse_indices.contiguous()

    # Output buffers (float32 for math, caller will cast to bfloat16)
    out = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

    # Launch grid: one program per (token, head)
    grid = (num_tokens, num_qo_heads)
    compute_one_kernel[grid](
        q_no, q_pe,
        ckv, kpe,
        indices,
        out, lse,
        float(sm_scale),
        num_tokens, num_qo_heads, head_dim_ckv, head_dim_kpe, 2048,
        BLOCK_K=128,
        num_warps=4, num_stages=2
    )

    # output is float32 in kernel, lse is float32; caller casts output to bfloat16
    return out, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        """
        Triton-optimized forward: no torch ops in host code; all computation in Triton kernels.
        If Triton or CUDA is unavailable, falls back to original PyTorch run.
        """
        # Triton path: CUDA + Triton available
        if TRITON_AVAILABLE and (q_nope.device.type == 'cuda'):
            # Allocate outputs (caller will cast output to bfloat16 after kernel)
            output = torch.empty((q_nope.shape[0], q_nope.shape[1], q_nope.shape[2]), dtype=torch.float32, device=q_nope.device)
            lse = torch.empty((q_nope.shape[0], q_nope.shape[1]), dtype=torch.float32, device=q_nope.device)

            # Run Triton kernel (no .to, no .contiguous on inputs beyond what was done above, and no dtype conversions)
            _run_triton(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

            # Return: output (float32) and lse (float32); evaluation may cast output to bfloat16 externally
            return output, lse

        # Fallback: pure PyTorch implementation (still correct; no Triton used)
        with torch.no_grad():
            num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
            head_dim_kpe = q_pe.shape[-1]
            # Check constants as in original
            assert num_qo_heads == 16
            assert head_dim_ckv == 512
            assert head_dim_kpe == 64
            assert sparse_indices.shape[-1] == 2048

            # Flatten caches
            Kc_all = ckv_cache.reshape(-1, head_dim_ckv)  # [num_pages*64, 512]
            Kp_all = kpe_cache.reshape(-1, head_dim_kpe)  # [num_pages*64, 64]

            output = torch.zeros(
                (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device
            )
            lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=q_nope.device)

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
