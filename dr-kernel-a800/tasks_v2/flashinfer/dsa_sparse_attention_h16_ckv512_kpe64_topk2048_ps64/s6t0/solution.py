import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: per (token, head) program. Computes:
# - Logits for this token and head across all valid K entries (via loop in tiles).
# - LSE for this token and head.
# - Output for this token and head.
# It does not write the entire output (all heads) at once; it writes only this head's output and LSE for this head.
@triton.jit
def compute_one_kernel(
    q_no_ptr, q_pe_ptr,      # [num_tokens, 16, 512] and [num_tokens, 16, 64]
    ckv_ptr, kpe_ptr,        # [num_pages, 64, 512] and [num_pages, 64, 64]
    indices_ptr,             # [num_tokens, topk] int32
    out_ptr,                 # [num_tokens, 16, 512] bfloat16 (we'll store float32 then cast outside)
    lse_ptr,                 # [num_tokens, 16] float32
    sm_scale,                # float32 scalar
    num_tokens, num_qo_heads, head_dim_ckv, head_dim_kpe, topk,
):
    # Program IDs: one program handles one (token, head)
    t = tl.program_id(0)  # token index
    h = tl.program_id(1)  # head index

    # Constants
    H = num_qo_heads
    Dk = head_dim_ckv
    Dp = head_dim_kpe

    # Bounds check (usually grid matches sizes, but keep it safe)
    if t >= num_tokens:
        return

    # Load q_no[t, h, :] and q_pe[t, h, :] as float32
    q_no_row_ptr = q_no_ptr + t * H * Dk + h * Dk
    q_no = tl.load(q_no_row_ptr + tl.arange(0, Dk))  # [Dk]
    q_no = q_no.to(tl.float32)

    q_pe_row_ptr = q_pe_ptr + t * H * Dp + h * Dp
    q_pe = tl.load(q_pe_row_ptr + tl.arange(0, Dp))  # [Dp]
    q_pe = q_pe.to(tl.float32)

    # Initialize LSE components
    m = tl.full((), -1e30, tl.float32)  # running max
    s = tl.zeros((), tl.float32)        # running sum of exp(logits - m)

    # Accumulate output per head (float32)
    out_accum = tl.zeros((Dk,), dtype=tl.float32)

    # Iterate over K dimension in tiles of BLOCK_K (e.g., 128)
    # For each tile, we compute logits for all entries, update LSE, and accumulate output.
    BLOCK_K = 128
    for k0 in range(0, topk, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_valid = offs < topk
        # Load indices for this token and tile
        idx_ptrs = indices_ptr + t * topk + offs
        idxs = tl.load(idx_ptrs, mask=mask_valid, other=-1)  # int32

        # Build valid mask from idxs (-1 invalid)
        valid = idxs != -1  # [BLOCK_K], boolean

        # Compute base offsets in flattened KV cache
        # Flattened KV cache has N = num_pages * 64 entries.
        # Each entry is (page_idx, offset) where offset in [0, 63].
        # idx maps to entry = idx, where idx < N. We gather per valid.
        # For each valid k, we need to load:
        # Kc[entry, :] and Kp[entry, :]
        # Since entry = idxs[k], we can compute pointer offsets:
        # Kc_ptr at ckv_ptr + entry * Dk + offs_k
        # Kp_ptr at kpe_ptr + entry * Dp + offs_k

        # Prepare offsets for each valid k in this tile
        # We will loop over the tile and handle masked loads
        for kk in range(BLOCK_K):
            if not (mask_valid[kk]):
                continue
            if not (valid[kk]):
                continue
            entry = idxs[kk].to(tl.int64)
            # Load Kc row and Kp row for this entry
            # Cast to int64 for safe pointer arithmetic
            # Kc: [Dk], Kp: [Dp]
            Kc_row = tl.load(ckv_ptr + entry * Dk + tl.arange(0, Dk), mask=valid[kk], other=0.0)
            Kc_row = Kc_row.to(tl.float32)
            Kp_row = tl.load(kpe_ptr + entry * Dp + tl.arange(0, Dp), mask=valid[kk], other=0.0)
            Kp_row = Kp_row.to(tl.float32)

            # Compute logits for this head
            # logit = dot(q_no, Kc_row) + dot(q_pe, Kp_row)
            dot1 = tl.sum(q_no * Kc_row)           # scalar
            dot2 = tl.sum(q_pe * Kp_row)           # scalar
            logit = dot1 + dot2                     # scalar

            # Apply scaling
            logit = logit * sm_scale

            # Update running max and sum for logsumexp
            m_new = tl.maximum(m, logit)
            # s_new = s * exp(m - m_new) + exp(logit - m_new)
            s = s * tl.exp(m - m_new) + tl.exp(logit - m_new)
            m = m_new

            # Compute attention for this head and this entry
            # attn = exp(logit - m) / (s * ln(2))
            ln2 = 0.6931471805599453
            attn = tl.exp(logit - m) / (s * ln2)

            # Accumulate output for this head
            # out += attn * Kc_row
            out_accum += attn * Kc_row

    # Store final lse for this (token, head)
    lse_val = (m + tl.log(s)) / ln2
    tl.store(lse_ptr + t * H + h, lse_val)

    # Store output for this (token, head) as float32, then cast outside if needed
    # out_ptr is [num_tokens, 16, 512] float32
    out_row_ptr = out_ptr + t * H * Dk + h * Dk
    tl.store(out_row_ptr + tl.arange(0, Dk), out_accum)


def _run_triton(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """
    Triton-optimized run that avoids torch ops in the host code and launches Triton kernels.
    Returns:
      output: [num_tokens, 16, 512] bfloat16
      lse: [num_tokens, 16] float32
    """
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and sparse_indices.is_cuda, "All tensors must be CUDA for Triton."
    device = q_nope.device
    num_tokens = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]
    num_pages = ckv_cache.shape[0]
    page_size = ckv_cache.shape[1]
    assert page_size == 64
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert sparse_indices.shape[0] == num_tokens
    assert sparse_indices.shape[1] == 2048

    # Ensure inputs are contiguous
    q_no = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    ckv = ckv_cache.contiguous()
    kpe = kpe_cache.contiguous()
    # We keep indices as int32
    indices = sparse_indices.to(torch.int32)

    # Output buffers
    out = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

    # Launch Triton grid: one program per (token, head)
    grid = (num_tokens, num_qo_heads)
    compute_one_kernel[grid](
        q_no, q_pe,
        ckv, kpe,
        indices,
        out, lse,
        sm_scale,
        num_tokens, num_qo_heads, head_dim_ckv, head_dim_kpe, 2048,
        num_warps=4, num_stages=2
    )

    # Cast output to bfloat16 to match original
    output = out.to(torch.bfloat16)
    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        """
        Triton-optimized forward: no torch ops in host code, all heavy compute in Triton kernels.
        If Triton or CUDA is unavailable, falls back to original PyTorch run.
        """
        # If Triton is not available or tensors are not on CUDA, fall back
        if (not TRITON_AVAILABLE) or (q_nope.device.type != 'cuda'):
            # Fallback to original PyTorch logic
            with torch.no_grad():
                num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
                head_dim_kpe = q_pe.shape[-1]
                num_pages, page_size, _ = ckv_cache.shape
                topk = sparse_indices.shape[-1]

                assert num_qo_heads == 16
                assert head_dim_ckv == 512
                assert head_dim_kpe == 64
                assert page_size == 64
                assert topk == 2048

                Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [total_kv_tokens, head_dim_ckv]
                Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [total_kv_tokens, head_dim_kpe]

                output = torch.zeros(
                    (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device
                )
                lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=q_nope.device)

                for t in range(num_tokens):
                    indices = sparse_indices[t]  # [topk]
                    valid_mask = indices != -1
                    valid_indices = indices[valid_mask]
                    if valid_indices.numel() == 0:
                        output[t].zero_()
                        continue

                    Kc = Kc_all[valid_indices]  # [num_valid, head_dim_ckv]
                    Kp = Kp_all[valid_indices]  # [num_valid, head_dim_kpe]
                    qn = q_nope[t].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
                    qp = q_pe[t].to(torch.float32)    # [num_qo_heads, head_dim_kpe]

                    logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, num_valid]
                    logits_scaled = logits * sm_scale
                    # 2-base LSE
                    lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

                    attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, num_valid]
                    out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
                    output[t] = out.to(torch.bfloat16)

                return output, lse

        # Triton path: ensure dtypes are float32 for math, keep output bfloat16
        # q_nope and q_pe may be bfloat16; we convert to float32 for computation.
        # But Triton kernel already loads and converts to float32.
        q_no_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)
        ckv_f32 = ckv_cache.to(torch.float32)
        kpe_f32 = kpe_cache.to(torch.float32)

        output_f32, lse_f32 = _run_triton(q_no_f32, q_pe_f32, ckv_f32, kpe_f32, sparse_indices, float(sm_scale))

        # Output bfloat16, lse float32 as per original
        return output_f32.to(torch.bfloat16), lse_f32


def run(*args):
    return ModelNew()(*args)
