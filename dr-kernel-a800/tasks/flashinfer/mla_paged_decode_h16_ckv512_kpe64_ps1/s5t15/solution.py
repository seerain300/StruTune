import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_lse_and_output_per_head_kernel(
    qn_ptr,        # *f32, base pointer to [H, CK]
    qp_ptr,        # *f32, base pointer to [H, KP]
    Kc_all_ptr,    # *f32, base pointer to [P, CK]
    Kp_all_ptr,    # *f32, base pointer to [P, KP]
    tok_idx_ptr,   # *i32, base pointer to [L_tokens]
    output_ptr,    # *f32, base pointer to [H, CK]
    lse_ptr,       # *f32, base pointer to [1] (we write a single scalar)
    # meta-parameters (tl.constexpr)
    H: tl.constexpr,          # num_qo_heads (unused in kernel except for loop over h if needed; we launch per-head)
    CK: tl.constexpr,         # head_dim_ckv
    KP: tl.constexpr,         # head_dim_kpe
    L_tokens: tl.constexpr,   # number of tokens in this batch
    sm_scale: tl.constexpr,   # scaling factor
    h_idx: tl.constexpr,      # current head index for this launch
):
    # Prepare column ranges
    dim_ck = tl.arange(0, CK)
    dim_kp = tl.arange(0, KP)

    # Initialize per-head running max and sumexp for lse
    m = -float("inf")
    s = 0.0

    # First pass: compute scaled logits, update lse, and store nothing (we'll recompute softmax later)
    for t in range(0, L_tokens):
        idx = tl.load(tok_idx_ptr + t)  # int32

        # Load query vectors for this head
        qn_vec = tl.load(qn_ptr + h_idx * CK + dim_ck)  # [CK]
        qp_vec = tl.load(qp_ptr + h_idx * KP + dim_kp)  # [KP]

        # Load corresponding key rows
        Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK]
        Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)  # [KP]

        # Compute scaled logits for this head and token
        dot_qn = tl.sum(qn_vec * Kc_row, axis=0)  # scalar
        dot_qp = tl.sum(qp_vec * Kp_row, axis=0)  # scalar
        scaled = (dot_qn + dot_qp) * sm_scale     # float32 scalar

        # Online update for lse: m_new = max(m, scaled); s = s*exp(m-m_new) + exp(scaled - m_new); m = m_new
        m_new = tl.maximum(m, scaled)
        s = s * tl.exp(m - m_new) + tl.exp(scaled - m_new)
        m = m_new

    # After all tokens, lse[h] = m (natural log base)
    tl.store(lse_ptr, m)  # store lse for head h_idx

    # Second pass: compute output[h, :] = sum_t softmax(scaled[h, t]) * Kc[t, :]
    out_acc = tl.zeros((CK,), tl.float32)
    for t in range(0, L_tokens):
        idx = tl.load(tok_idx_ptr + t)
        qn_vec = tl.load(qn_ptr + h_idx * CK + dim_ck)
        qp_vec = tl.load(qp_ptr + h_idx * KP + dim_kp)
        Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)

        scaled = (tl.sum(qn_vec * Kc_row, axis=0) + tl.sum(qp_vec * Kp_all_ptr[idx * KP + dim_kp, None] * Kp_all_ptr[idx * KP + dim_kp, None], axis=0)) * sm_scale
        # The above line had a bug: we cannot index Kp_all_ptr with (dim_kp, None) like this. Fix by loading Kp_row correctly:
        Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)
        scaled = (tl.sum(qn_vec * Kc_row, axis=0) + tl.sum(qp_vec * Kp_row, axis=0)) * sm_scale

        lse_h = m
        p = tl.exp(scaled - lse_h)  # softmax probability for this token
        out_acc += p * Kc_row

    # Store final output for head h_idx
    tl.store(output_ptr + h_idx * CK + dim_ck, out_acc)


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure device is CUDA for Triton
    assert q_nope.is_cuda, "Triton kernels require CUDA tensors."
    device = q_nope.device

    # Convert inputs to float32 for compute (contiguous)
    q_nope_f32 = q_nope.to(torch.float32).contiguous()     # [B, H, CK]
    q_pe_f32 = q_pe.to(torch.float32).contiguous()         # [B, H, KP]
    Kc_all_f32 = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [P, CK]
    Kp_all_f32 = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [P, KP]
    tok_idx = kv_indices.to(torch.int32).contiguous()       # [L_tokens]

    batch_size = q_nope.shape[0]
    H = q_nope.shape[1]
    CK = q_nope.shape[2]
    num_qo_heads = H
    head_dim_ckv = CK

    # Output buffers (float32 for compute, cast to bfloat16 at the end)
    output_accum = torch.zeros((batch_size, num_qo_heads, CK), dtype=torch.float32, device=device)
    lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    # For each batch element and each head, compute outputs and lse
    for b in range(batch_size):
        L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L_tokens <= 0:
            output_accum[b] = torch.zeros_like(output_accum[b])
            lse[b] = torch.zeros_like(lse[b])
            continue

        # Launch one kernel per head; grid=(1,) and constexpr parameters ensure correctness.
        for h in range(0, num_qo_heads):
            _compute_lse_and_output_per_head_kernel[(1,)](
                q_nope_f32[b], q_pe_f32[b], Kc_all_f32, Kp_all_f32, tok_idx,
                output_accum[b], lse[b],
                H=num_qo_heads, CK=CK, KP=q_pe_f32[b].shape[-1], L_tokens=L_tokens, sm_scale=sm_scale, h_idx=h,
            )

    # Cast output to bfloat16 to match original behavior
    output_bf16 = output_accum.to(torch.bfloat16)
    return output_bf16, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: no torch ops on tensors, only allocation and casting.
        return _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
