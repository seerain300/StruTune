import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_scaled_logits_kernel(
    qn_ptr,        # *f32, base pointer to [H, CK]
    qp_ptr,        # *f32, base pointer to [H, KP]
    Kc_all_ptr,    # *f32, base pointer to [P, CK]
    Kp_all_ptr,    # *f32, base pointer to [P, KP]
    tok_idx_ptr,   # *i32, base pointer to [L_tokens]
    scaled_ptr,    # *f32, base pointer to [H, L_tokens]
    H: tl.constexpr,
    CK: tl.constexpr,
    KP: tl.constexpr,
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
):
    # Grid is (H, L_tokens)
    h = tl.program_id(0)
    t = tl.program_id(1)
    if (h >= H) or (t >= L_tokens):
        return

    # Vectors for this head
    dim_ck = tl.arange(0, CK)
    dim_kp = tl.arange(0, KP)
    qn_vec = tl.load(qn_ptr + h * CK + dim_ck)   # [CK]
    qp_vec = tl.load(qp_ptr + h * KP + dim_kp)   # [KP]

    # Token index and corresponding rows in Kc/Kp
    idx = tl.load(tok_idx_ptr + t)               # int32
    Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK]
    Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)  # [KP]

    # Dot-products
    dot_qn = tl.sum(qn_vec * Kc_row)            # scalar
    dot_qp = tl.sum(qp_vec * Kp_row)            # scalar

    # Compute scaled logit for this head and token
    scaled = (dot_qn + dot_qp) * sm_scale
    tl.store(scaled_ptr + h * L_tokens + t, scaled)


@triton.jit
def _output_kernel(
    scaled_ptr,    # *f32, base pointer to [H, L_tokens]
    Kc_all_ptr,    # *f32, base pointer to [P, CK]
    tok_idx_ptr,   # *i32, base pointer to [L_tokens]
    out_ptr,       # *f32, base pointer to [H, CK]
    lse_ptr,       # *f32, base pointer to [H]
    H: tl.constexpr,
    CK: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)
    if h >= H:
        return

    # Load per-head lse (this is logsumexp of scaled_logits in natural log, then divided by ln(2) in host)
    lse = tl.load(lse_ptr + h)  # scalar float32

    # Accumulate output[h, :] = sum_t exp(scaled[h, t] - lse) * Kc[t, :]
    out_acc = tl.zeros((CK,), tl.float32)
    for t in range(0, L_tokens):
        s = tl.load(scaled_ptr + h * L_tokens + t)  # scaled logit for this head and token
        soft = tl.exp(s - lse)                      # softmax value for this token
        idx = tl.load(tok_idx_ptr + t)             # int32 token index
        Kc_row = tl.load(Kc_all_ptr + idx * CK + tl.arange(0, CK))  # [CK]
        out_acc += soft * Kc_row                   # vector accumulate

    tl.store(out_ptr + h * CK + tl.arange(0, CK), out_acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes:
        # q_nope: [B, H, CK], q_pe: [B, H, KP], ckv_cache: [num_pages, 1, CK], kpe_cache: [num_pages, 1, KP]
        device = q_nope.device
        dtype_f32 = torch.float32

        B, H, CK = q_nope.shape
        _, _, KP = q_pe.shape
        num_pages = ckv_cache.shape[0]

        # Convert inputs to float32 for compute
        qn = q_nope.to(dtype_f32)   # [B, H, CK]
        qp = q_pe.to(dtype_f32)     # [B, H, KP]
        Kc_all = ckv_cache.to(dtype_f32).squeeze(1)  # [num_pages, CK]
        Kp_all = kpe_cache.to(dtype_f32).squeeze(1)  # [num_pages, KP]

        # Allocate outputs
        output = torch.empty((B, H, CK), dtype=torch.float32, device=device)
        lse_host = torch.empty((B, H), dtype=torch.float32, device=device)  # per-batch, per-head lse

        # Process each batch element
        for b in range(B):
            # Determine L_tokens for this batch slice: number of indices between indptr[b] and indptr[b+1]
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = max(0, page_end - page_beg)

            if L_tokens == 0:
                # If no tokens, output zeros and lse -inf; but output should be zero, lse not used in forward
                output[b].zero_()
                lse_host[b].fill_(0.0)
                continue

            # Token indices for this batch
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # [L_tokens]

            # Allocate scaled logits buffer [H, L_tokens]
            scaled_logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch kernel to compute scaled logits for all heads and tokens
            _compute_scaled_logits_kernel[(H, L_tokens)](
                qn[b],                 # qn[b] is [H, CK]
                qp[b],                 # qp[b] is [H, KP]
                Kc_all,                # [P, CK]
                Kp_all,                # [P, KP]
                tok_idx,               # [L_tokens]
                scaled_logits,         # [H, L_tokens]
                H=H, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=sm_scale,
                num_warps=4, num_stages=2,
            )

            # Compute lse per head in host: logsumexp of scaled_logits (natural log), then divide by ln(2) to match reference
            lse_host[b] = torch.logsumexp(scaled_logits, dim=-1) / math.log(2.0)

            # Launch kernel to compute final output per head (one program per head)
            _output_kernel[(H,)](
                scaled_logits,         # [H, L_tokens]
                Kc_all,                # [P, CK]
                tok_idx,               # [L_tokens]
                output[b],             # [H, CK]
                lse_host[b],           # [H]
                H=H, CK=CK, L_tokens=L_tokens,
                num_warps=4, num_stages=2,
            )

        # Return output as bfloat16 to match original signature, and lse in float32
        return output.to(torch.bfloat16), lse_host


def run(*args):
    return ModelNew()(*args)
