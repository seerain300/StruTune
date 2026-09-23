import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_lse_and_output_kernel(
    qn_ptr,       # *f32, [H, CK]
    qp_ptr,       # *f32, [H, KP]
    Kc_ptr,       # *f32, [N_tokens, CK] (we will index with tok_idx)
    Kp_ptr,       # *f32, [N_tokens, KP]
    tok_idx_ptr,  # *i32, [N_tokens]
    out_ptr,      # *f32, [H, CK] (output per head)
    lse_ptr,      # *f32, [H]    (lse per head, natural log)
    H: tl.constexpr,
    CK: tl.constexpr,
    KP: tl.constexpr,
    N_tokens: tl.constexpr,  # L_tokens
    sm_scale: tl.float32,
    BLOCK_T: tl.constexpr = 128,
):
    h = tl.program_id(0)

    # Initialize lse (logsumexp) and output accumulator
    # lse uses natural log
    # We'll accumulate sumexp after computing max
    m = -float("inf")
    sumexp = 0.0
    out = tl.zeros((CK,), dtype=tl.float32)

    # Loop over tokens in tiles
    for t0 in range(0, N_tokens, BLOCK_T):
        offs = t0 + tl.arange(0, BLOCK_T)
        mask = offs < N_tokens
        # Load token indices for this tile
        tok_idx = tl.load(tok_idx_ptr + offs, mask=mask, other=0)  # int32
        # Load corresponding Kc and Kp rows for each token in the tile
        # We'll do an unrolled loop across the tile since BLOCK_T is constexpr
        for dt in range(BLOCK_T):
            valid = mask[dt]
            # Only proceed if within bounds
            if valid:
                t = t0 + dt
                # Compute dot products for this token t:
                # qn[h] · Kc[t, :]
                qn_row = tl.load(qn_ptr + h * CK + tl.arange(0, CK), mask=True, other=0.0)  # [CK]
                Kc_row = tl.load(Kc_ptr + tok_idx[dt] * CK + tl.arange(0, CK), mask=True, other=0.0)  # [CK]
                dot_qn = 0.0
                for i in range(CK):
                    dot_qn += qn_row[i] * Kc_row[i]
                # qp[h] · Kp[t, :]
                qp_row = tl.load(qp_ptr + h * KP + tl.arange(0, KP), mask=True, other=0.0)  # [KP]
                Kp_row = tl.load(Kp_ptr + tok_idx[dt] * KP + tl.arange(0, KP), mask=True, other=0.0)  # [KP]
                dot_qp = 0.0
                for i in range(KP):
                    dot_qp += qp_row[i] * Kp_row[i]
                scaled = sm_scale * (dot_qn + dot_qp)
                # Update max and sumexp for logsumexp
                if m is None:  # initialize once
                    m = scaled
                elif scaled > m:
                    m_new = scaled
                    sumexp = sumexp * tl.exp(m - m_new) + tl.exp(m - m_new)  # redundant first time; harmless
                    m = m_new
                else:
                    sumexp = sumexp + tl.exp(scaled - m)
                # Accumulate output: out += softmax * Kc_row
                prob = tl.exp(scaled - m)
                for i in range(CK):
                    out[i] += prob * Kc_row[i]

    # Finalize lse
    lse = tl.log(sumexp) + m
    # Store outputs
    tl.store(out_ptr + h * CK + tl.arange(0, CK), out)
    tl.store(lse_ptr + h, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on the same device and contiguous
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Tensors must be on CUDA for Triton"
        # Cast to float32 for compute
        q_nope_f = q_nope.to(torch.float32).contiguous()
        q_pe_f = q_pe.to(torch.float32).contiguous()
        # Flatten ckv_cache to [num_tokens, head_dim_ckv] ignoring batch in dim 0
        # Note: ckv_cache shape is [num_pages, 1, CK] -> flatten to [num_tokens, CK]
        num_tokens = ckv_cache.shape[0]
        CK = q_nope_f.shape[-1]
        KP = q_pe_f.shape[-1]
        Kc_all = ckv_cache.view(num_tokens, CK).contiguous()
        Kp_all = kpe_cache.view(num_tokens, KP).contiguous()
        # Prepare output tensor [batch, num_qo_heads, CK]
        batch_size = q_nope_f.shape[0]
        H = q_nope_f.shape[1]
        out = torch.empty((batch_size, H, CK), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Determine L_tokens from kv_indptr[b:b+1]
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            # Slice token indices and Kc/Kp rows for this batch
            tok_idx = kv_indices[b : b + L_tokens].to(torch.int32).contiguous()
            # Launch kernel for this batch
            _compute_lse_and_output_kernel[(H,)](
                q_nope_f[b], q_pe_f[b], Kc_all, Kp_all, tok_idx, out[b], lse[b],
                H=H, CK=CK, KP=KP, N_tokens=L_tokens, sm_scale=float(sm_scale),
                BLOCK_T=128,
            )

        # Return output cast to bfloat16, and lse as float32
        return out.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
