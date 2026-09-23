import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_logits_kernel(
    qn_ptr,       # *f32, shape [H, CK] flattened as 1D
    qp_ptr,       # *f32, shape [H, KP] flattened as 1D
    Kc_ptr,       # *f32, shape [L_tokens, CK]
    Kp_ptr,       # *f32, shape [L_tokens, KP]
    logits_ptr,   # *f32, shape [H*L_tokens]
    max_ptr,      # *f32, shape [H]
    H: tl.constexpr, CK: tl.constexpr, KP: tl.constexpr,
    L_tokens: tl.constexpr, sm_scale: tl.constexpr
):
    h = tl.program_id(0)
    t = tl.program_id(1)

    if (h >= H) or (t >= L_tokens):
        return

    qn_base = h * CK
    qp_base = h * KP

    dot_qn = 0.0
    dot_qp = 0.0

    # Dot products via scalar accumulation
    for i in range(CK):
        qn_i = tl.load(qn_ptr + qn_base + i)
        Kc_it = tl.load(Kc_ptr + t * CK + i)
        dot_qn += qn_i * Kc_it

    for i in range(KP):
        qp_i = tl.load(qp_ptr + qp_base + i)
        Kp_it = tl.load(Kp_ptr + t * KP + i)
        dot_qp += qp_i * Kp_it

    val = (dot_qn + dot_qp) * sm_scale
    idx = h * L_tokens + t
    tl.store(logits_ptr + idx, val)

    # Maintain max per head for later lse stability
    tl.store(max_ptr + h, tl.maximum(tl.load(max_ptr + h), val))


@triton.jit
def _compute_lse_kernel(
    logits_ptr,   # *f32, shape [H*L_tokens]
    max_ptr,      # *f32, shape [H]
    lse_ptr,      # *f32, shape [H]
    H: tl.constexpr, L_tokens: tl.constexpr
):
    h = tl.program_id(0)
    m = tl.load(max_ptr + h)
    sumexp = 0.0
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        sumexp += tl.exp(val - m)
    lse = tl.log(sumexp)
    tl.store(lse_ptr + h, lse)


@triton.jit
def _compute_output_kernel(
    qn_ptr,       # *f32, shape [H, CK] flattened as 1D
    qp_ptr,       # *f32, shape [H, KP] flattened as 1D
    Kc_ptr,       # *f32, shape [L_tokens, CK]
    logits_ptr,   # *f32, shape [H*L_tokens]
    out_ptr,      # *f32, shape [H, CK], flattened
    H: tl.constexpr, CK: tl.constexpr, KP: tl.constexpr,
    L_tokens: tl.constexpr, sm_scale: tl.constexpr
):
    h = tl.program_id(0)
    # Initialize output to zero
    for i in range(CK):
        tl.store(out_ptr + h * CK + i, 0.0)

    # Accumulate: out[h, :] += softmax(logits[h, t] * sm_scale) * Kc[t, :]
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        prob = tl.exp(val * sm_scale)  # softmax over logits_scaled == logits * sm_scale
        # Multiply by Kc[t, :] in tiles along CK
        for i in range(0, CK, 128):
            offs = i + tl.arange(0, 128)
            mask = offs < CK
            Kc_vec = tl.load(Kc_ptr + t * CK + offs, mask=mask, other=0.0)
            out_vec = prob * Kc_vec
            # Store into out[h, offs]
            tl.store(out_ptr + h * CK + offs, out_ptr[h * CK + offs] + out_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-optimized forward. Returns (output, lse) with:
          - output: [batch_size, num_qo_heads, head_dim_ckv] in bfloat16
          - lse: [batch_size, num_qo_heads] in float32
        """
        device = q_nope.device

        # Cast and make contiguous
        qn = q_nope.to(torch.float32).contiguous()     # [B, H, CK]
        qp = q_pe.to(torch.float32).contiguous()       # [B, H, KP]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, CK]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, KP]

        B = qn.shape[0]
        H = qn.shape[1]
        CK = qn.shape[2]
        KP = qp.shape[2]

        # We expect batch_size=1 in provided inputs; handle general B by looping
        output = torch.empty((B, H, CK), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(B):
            # Determine L_tokens for this batch element using kv_indptr
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # Slice token indices
            tok_idx = kv_indices[:L_tokens].to(torch.long)

            # Select Kc and Kp slices
            Kc = Kc_all[tok_idx]  # [L_tokens, CK]
            Kp = Kp_all[tok_idx]  # [L_tokens, KP]

            # Flatten qn[h, :] and qp[h, :] for kernel
            qn_flat = qn[b].reshape(H * CK).contiguous()  # [H*CK]
            qp_flat = qp[b].reshape(H * KP).contiguous()  # [H*KP]

            # Allocate buffers
            logits = torch.empty((H * L_tokens,), dtype=torch.float32, device=device)
            max_vals = torch.empty((H,), dtype=torch.float32, device=device)  # init to -inf
            # Triton will overwrite with max of computed logits per head

            # Launch Triton kernel to compute logits and per-head max
            grid = (H, L_tokens)
            _compute_logits_kernel[grid](
                qn_flat, qp_flat, Kc, Kp, logits, max_vals,
                H=H, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=float(sm_scale)
            )

            # Launch Triton kernel to compute lse per head
            _compute_lse_kernel[(H,)](
                logits, max_vals, lse[b],
                H=H, L_tokens=L_tokens
            )

            # Launch Triton kernel to compute output per head
            out_flat = output[b].reshape(H * CK).contiguous()
            _compute_output_kernel[(H,)](
                qn_flat, qp_flat, Kc, logits, out_flat,
                H=H, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=float(sm_scale)
            )

        # Return output in bfloat16, lse in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
