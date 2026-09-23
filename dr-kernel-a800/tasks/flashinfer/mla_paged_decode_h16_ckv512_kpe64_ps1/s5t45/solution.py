import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_logits_scaled_kernel(
    qn_ptr,        # *f32, shape [H, CK]
    qp_ptr,        # *f32, shape [H, KP]
    Kc_ptr,        # *f32, shape [L_tokens, CK]
    Kp_ptr,        # *f32, shape [L_tokens, KP]
    tok_idx_ptr,   # *i32, shape [L_tokens]
    logits_ptr,    # *f32, shape [H, L_tokens]
    sm_scale: tl.constexpr,   # scalar f32
    H: tl.constexpr,          # int
    CK: tl.constexpr,         # int (head_dim_ckv)
    KP: tl.constexpr,         # int (head_dim_kpe)
    L_tokens: tl.constexpr,   # int
):
    h = tl.program_id(0)
    t = tl.program_id(1)
    if h >= H or t >= L_tokens:
        return

    # Load qn[h, :] and qp[h, :]
    qn = tl.load(qn_ptr + h * CK + tl.arange(0, CK))
    qp = tl.load(qp_ptr + h * KP + tl.arange(0, KP))

    # Load Kc[t, :] and Kp[t, :]
    idx = tl.load(tok_idx_ptr + t)  # int32
    Kc = tl.load(Kc_ptr + idx * CK + tl.arange(0, CK))
    Kp = tl.load(Kp_ptr + idx * KP + tl.arange(0, KP))

    # Dot products: qn · Kc and qp · Kp
    dot_qn = 0.0
    for i in range(CK):
        dot_qn += qn[i] * Kc[i]
    dot_qp = 0.0
    for i in range(KP):
        dot_qp += qp[i] * Kp[i]

    scaled = sm_scale * (dot_qn + dot_qp)
    tl.store(logits_ptr + h * L_tokens + t, scaled)


@triton.jit
def _lse_kernel(
    logits_ptr,    # *f32, shape [H, L_tokens]
    lse_ptr,       # *f32, shape [H]
    inv_ln2: tl.constexpr,     # scalar f32 = 1 / ln(2)
    H: tl.constexpr,           # int
    L_tokens: tl.constexpr,    # int
):
    h = tl.program_id(0)
    if h >= H:
        return
    max_val = -1e30
    sum_exp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        max_val = tl.maximum(max_val, val)
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        sum_exp += tl.exp(val - max_val)
    lse = tl.log(sum_exp) * inv_ln2
    tl.store(lse_ptr + h, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Cast inputs to float32 for compute
        q_nope = q_nope.to(torch.float32)
        q_pe = q_pe.to(torch.float32)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, CK]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, KP]
        # Shapes
        H = q_nope.shape[1]
        CK = Kc_all.shape[-1]  # head_dim_ckv == 512
        KP = Kp_all.shape[-1]  # head_dim_kpe == 64

        device = q_nope.device
        # Initialize outputs
        output = torch.empty((q_nope.shape[0], H, CK), dtype=torch.float32, device=device)
        lse = torch.empty((q_nope.shape[0], H), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(q_nope.shape[0]):
            # If no tokens in this batch slice
            if int(kv_indptr[b].item()) == int(kv_indptr[b + 1].item()):
                # No valid kv entries for this batch, output zeros and lse = -inf
                output[b].zero_()
                lse[b].fill_(-float('inf'))
                continue

            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].to(torch.int32)  # [L_tokens]
            # Prepare qn and qp
            qn = q_nope[b].to(torch.float32)  # [H, CK]
            qp = q_pe[b].to(torch.float32)    # [H, KP]
            # Prepare K slices
            Kc = Kc_all[tok_idx]              # [L_tokens, CK]
            Kp = Kp_all[tok_idx]              # [L_tokens, KP]
            # Prepare logits buffer [H, L_tokens]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            # Launch compute_logits_scaled_kernel
            _compute_logits_scaled_kernel[(H, L_tokens)](
                qn, qp, Kc, Kp, tok_idx, logits,
                sm_scale=float(sm_scale),
                H=H, CK=CK, KP=KP, L_tokens=L_tokens
            )
            # Compute lse per head in Triton: base-2 logsumexp
            inv_ln2 = 1.0 / math.log(2.0)
            _lse_kernel[(H,)](
                logits, lse[b], inv_ln2,
                H=H, L_tokens=L_tokens
            )
            # Compute output per head: out[h, :] = sum_t softmax(logits_scaled[h, t]) * Kc[t, :]
            # Use PyTorch for exact softmax and accumulation to guarantee correctness.
            # logits is [H, L_tokens] float32
            probs = torch.softmax(logits, dim=-1)  # [H, L_tokens]
            # Broadcast probs to [H, L_tokens, CK] and multiply by Kc, then sum over tokens
            # output[b] shape: [H, CK]
            output[b] = (probs.unsqueeze(-1) * Kc.to(torch.float32)).sum(dim=1)

        # Cast output back to bfloat16 to match original dtype
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
