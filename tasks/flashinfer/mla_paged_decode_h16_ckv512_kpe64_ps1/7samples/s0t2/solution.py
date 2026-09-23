import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_logits_and_lse_kernel(
    qn_ptr,       # *f32, [D] per head
    qp_ptr,       # *f32, [DP] per head
    Kc_ptr,       # *f32, [L_TOKENS, D], row-major
    Kp_ptr,       # *f32, [L_TOKENS, DP], row-major
    tok_idx_ptr,  # *i32, [L_TOKENS]
    lse_ptr,      # *f32, [1] (single scalar for this head)
    sm_scale,     # f32 scalar
    D: tl.constexpr,       # 512
    DP: tl.constexpr,      # 64
    L_TOKENS: tl.constexpr # number of tokens
):
    # Each program handles one head
    h = tl.program_id(0)

    # Load q vectors for this head
    qn = tl.load(qn_ptr + h * D)       # [D]
    qp = tl.load(qp_ptr + h * DP)      # [DP]

    # Compute logits vector for all tokens and LSE
    # Use tl.arange to form indices [0..L_TOKENS)
    t = tl.arange(0, L_TOKENS)                  # [L_TOKENS]
    idx = tl.load(tok_idx_ptr + t)              # [L_TOKENS], int32
    Kc_rows = tl.load(Kc_ptr + idx * D + t)     # [L_TOKENS]
    Kp_rows = tl.load(Kp_ptr + idx * DP + t)    # [L_TOKENS]

    logits = tl.dot(qn, Kc_rows) + tl.dot(qp, Kp_rows)     # scalar
    # logits is a scalar? We want a vector: one dot per token.
    # Correction: we need per-token dot products; better to build a vectorized computation.
    # The above simplification is incorrect: qn and Kc_rows are vectors of length D/DP.
    # We should compute dot(qn, Kc_rows[t]) per t. Triton can do this via broadcasting and tl.sum.
    # Fix: compute dot per token by gathering Kc_rows[t], Kp_rows[t] and then summing.

    # Proper computation: for each token t, dot(qn, Kc_rows[t]) + dot(qp, Kp_rows[t])
    # We'll do it elementwise and then reduce.
    # First, define Kc_rows_vec[t] and Kp_rows_vec[t] as vectors of length D and DP respectively.
    # However, Triton doesn't allow indexing into loaded vectors by t; better approach: compute in chunks.
    # To avoid loops (which caused issues), we can compute the dot products by iterating t explicitly.
    # Triton supports Python loops with constexpr bounds, so we loop over t in [0, L_TOKENS).
    # Initialize logits_vec as a vector of zeros.
    logits_vec = tl.zeros((L_TOKENS,), dtype=tl.float32)

    # Compute per-token logits: logits_vec[t] = dot(qn, Kc_rows[t]) + dot(qp, Kp_rows[t])
    # For t in 0..L_TOKENS-1:
    for ti in range(0, L_TOKENS):
        Kc_row_t = tl.load(Kc_ptr + idx[ti] * D + tl.arange(0, D))  # [D]
        Kp_row_t = tl.load(Kp_ptr + idx[ti] * DP + tl.arange(0, DP))  # [DP]
        # Reduce dot products
        dot_qn = 0.0
        for i in range(0, D):
            dot_qn += qn[i] * Kc_row_t[i]
        dot_qp = 0.0
        for j in range(0, DP):
            dot_qp += qp[j] * Kp_row_t[j]
        logits_vec[ti] = dot_qn + dot_qp

    # Scale and compute logsumexp base-2
    logits_scaled = logits_vec * sm_scale
    m = tl.max(logits_scaled, axis=0)
    s = tl.sum(tl.exp(logits_scaled - m), axis=0)
    lse_scalar = m + tl.log(s) / math.log(2.0)
    # Write out LSE for this head
    tl.store(lse_ptr + h, lse_scalar)


@triton.jit
def _compute_attention_output_kernel(
    qn_ptr,       # *f32, [D]
    qp_ptr,       # *f32, [DP]
    Kc_ptr,       # *f32, [L_TOKENS, D]
    Kp_ptr,       # *f32, [L_TOKENS, DP]
    tok_idx_ptr,  # *i32, [L_TOKENS]
    out_ptr,      # *f32, [D] (output vector for this head)
    sm_scale,     # f32 scalar
    D: tl.constexpr,       # 512
    DP: tl.constexpr,      # 64
    L_TOKENS: tl.constexpr # number of tokens
):
    h = tl.program_id(0)
    qn = tl.load(qn_ptr + h * D)
    qp = tl.load(qp_ptr + h * DP)

    # Recompute logits vector (we need it to form attn)
    t = tl.arange(0, L_TOKENS)
    idx = tl.load(tok_idx_ptr + t)
    logits_vec = tl.zeros((L_TOKENS,), dtype=tl.float32)
    for ti in range(0, L_TOKENS):
        Kc_row_t = tl.load(Kc_ptr + idx[ti] * D + tl.arange(0, D))  # [D]
        Kp_row_t = tl.load(Kp_ptr + idx[ti] * DP + tl.arange(0, DP))  # [DP]
        dot_qn = 0.0
        for i in range(0, D):
            dot_qn += qn[i] * Kc_row_t[i]
        dot_qp = 0.0
        for j in range(0, DP):
            dot_qp += qp[j] * Kp_row_t[j]
        logits_vec[ti] = dot_qn + dot_qp

    logits_scaled = logits_vec * sm_scale
    m = tl.max(logits_scaled, axis=0)
    s = tl.sum(tl.exp(logits_scaled - m), axis=0)
    attn = tl.exp(logits_scaled - m) / s  # [L_TOKENS]

    # Compute output = sum_t attn[t] * Kc_rows[t]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for ti in range(0, L_TOKENS):
        Kc_row_t = tl.load(Kc_ptr + idx[ti] * D + tl.arange(0, D))  # [D]
        out_vec += attn[ti] * Kc_row_t

    tl.store(out_ptr + h * D + tl.arange(0, D), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Input checks: keep behavior similar to original
        B, num_qo_heads, D = q_nope.shape
        _, _, DP = q_pe.shape
        assert num_qo_heads == 16
        assert D == 512
        assert DP == 64

        device = q_nope.device
        # Ensure tensors on the same device
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        output = torch.empty((B, num_qo_heads, D), dtype=torch.float32, device=device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(B):
            # Get number of tokens for this batch
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                # No tokens for this batch element: output zeros, lse -inf
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Gather selected rows
            tok_idx = kv_indices[start:end].to(torch.int32)  # [L_tokens]
            Kc_selected = Kc_all[tok_idx]  # [L_tokens, 512]
            Kp_selected = Kp_all[tok_idx]  # [L_tokens, 64]

            # Prepare pointers to this batch's data (using base pointers and strides)
            # Triton expects contiguous memory; ensure row-major contiguous
            Kc_selected = Kc_selected.contiguous()
            Kp_selected = Kp_selected.contiguous()
            tok_idx = tok_idx.contiguous()

            # Launch Triton per batch (grid = 1), loop over heads inside kernel
            # Note: Triton supports loops with constexpr bounds; num_qo_heads is 16.
            _compute_logits_and_lse_kernel[(1,)](
                q_nope_f32[b], q_pe_f32[b], Kc_selected, Kp_selected, tok_idx,
                lse[b], sm_scale,
                D=D, DP=DP, L_TOKENS=L_tokens,
                num_warps=4
            )

            # Compute attention output vector per head (PyTorch reduction)
            for h in range(num_qo_heads):
                # Launch Triton kernel to compute attention-weighted output for head h
                # out_ptr is a 1D vector of length D
                out_vec = torch.empty((D,), dtype=torch.float32, device=device)
                _compute_attention_output_kernel[(1,)](
                    q_nope_f32[b], q_pe_f32[b], Kc_selected, Kp_selected, tok_idx,
                    out_vec, sm_scale,
                    D=D, DP=DP, L_TOKENS=L_tokens,
                    num_warps=4
                )
                output[b, h] = out_vec

        # Cast output to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
