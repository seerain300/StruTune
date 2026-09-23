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
    # 2D grid: (H, L_tokens)
    h = tl.program_id(0)
    t = tl.program_id(1)
    if (h >= H) or (t >= L_tokens):
        return

    # Load qn[h, :] and qp[h, :]
    dim_ck = tl.arange(0, CK)
    qn_vec = tl.load(qn_ptr + h * CK + dim_ck)   # [CK]
    dim_kp = tl.arange(0, KP)
    qp_vec = tl.load(qp_ptr + h * KP + dim_kp)   # [KP]

    # Load token index for this position
    idx = tl.load(tok_idx_ptr + t)               # int32

    # Load corresponding rows from Kc_all and Kp_all
    Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)   # [CK]
    Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)   # [KP]

    # Compute dot-products
    dot_qn = tl.sum(qn_vec * Kc_row)            # scalar
    dot_qp = tl.sum(qp_vec * Kp_row)            # scalar
    scaled = (dot_qn + dot_qp) * sm_scale       # base-2 logsumexp will divide by ln(2) elsewhere

    # Store scaled logit for this head and token
    tl.store(scaled_ptr + h * L_tokens + t, scaled)


@triton.jit
def _lse_kernel(
    scaled_ptr,     # *f32, base pointer to [H, L_tokens]
    lse_ptr,        # *f32, base pointer to [H] (base-2 logsumexp)
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # Per-head reduction: compute lse[h] = max_t scaled[h, t] + log(sum_t exp(scaled[h, t] - max)) / ln(2)
    h = tl.program_id(0)
    if h >= H:
        return

    # 1) Find max over t
    max_s = tl.full((), -float("inf"), tl.float32)
    for t in range(0, L_tokens):
        s = tl.load(scaled_ptr + h * L_tokens + t)
        max_s = tl.maximum(max_s, s)

    # 2) Sum exp(s - max)
    sum_exp = tl.full((), 0.0, tl.float32)
    for t in range(0, L_tokens):
        s = tl.load(scaled_ptr + h * L_tokens + t)
        sum_exp += tl.exp(s - max_s)

    # lse in base-2: log2(sum_exp) = ln(sum_exp) / ln(2)
    ln2 = 0.6931471805599453
    lse_base2 = max_s + tl.log(sum_exp) / ln2
    tl.store(lse_ptr + h, lse_base2)


@triton.jit
def _output_kernel(
    scaled_ptr,     # *f32, base pointer to [H, L_tokens] (base-2 scaled logits)
    Kc_all_ptr,     # *f32, base pointer to [P, CK]
    tok_idx_ptr,    # *i32, base pointer to [L_tokens]
    lse_ptr,        # *f32, base pointer to [H] (base-2 lse)
    out_ptr,        # *f32, base pointer to [H, CK]
    H: tl.constexpr,
    CK: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # Per-head output accumulation: out[h, :] = sum_t exp(scaled[h, t] - lse[h]) * Kc[t, :]
    h = tl.program_id(0)
    if h >= H:
        return

    lse = tl.load(lse_ptr + h)  # base-2 lse
    ln2 = 0.6931471805599453
    lse_ln = lse * ln2          # convert base-2 lse to natural log scale

    out_acc = tl.zeros((CK,), tl.float32)
    for t in range(0, L_tokens):
        s = tl.load(scaled_ptr + h * L_tokens + t)
        soft = tl.exp(s - lse_ln)  # softmax in natural log scale
        idx = tl.load(tok_idx_ptr + t)  # int32
        Kc_row = tl.load(Kc_all_ptr + idx * CK + tl.arange(0, CK))  # [CK]
        out_acc += soft * Kc_row

    tl.store(out_ptr + h * CK + tl.arange(0, CK), out_acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Inputs:
        #   q_nope: [B, H, CK], q_pe: [B, H, KP]
        #   ckv_cache: [P, 1, CK], kpe_cache: [P, 1, KP]
        #   kv_indptr: [B+1], kv_indices: [M], sm_scale: float
        device = q_nope.device
        dtype_f32 = torch.float32

        B, H, CK = q_nope.shape
        _, _, KP = q_pe.shape
        P = ckv_cache.shape[0]  # number of cached pages

        # Convert inputs to float32 for compute
        qn = q_nope.to(dtype_f32)       # [B, H, CK]
        qp = q_pe.to(dtype_f32)         # [B, H, KP]
        Kc_all = ckv_cache.to(dtype_f32).squeeze(1)  # [P, CK]
        Kp_all = kpe_cache.to(dtype_f32).squeeze(1)  # [P, KP]

        # Outputs
        output = torch.empty((B, H, CK), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Process per batch
        for b in range(B):
            # Determine token slice for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = max(0, page_end - page_beg)

            # Slice of token indices used in the original code
            tok_idx = kv_indices[page_beg:page_end]  # [L_tokens], int32
            tok_idx = tok_idx.to(torch.int32).to(device)

            # Allocate intermediate scaled logits [H, L_tokens]
            scaled = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch compute_scaled_logits_kernel
            grid = (H, L_tokens)
            _compute_scaled_logits_kernel[grid](
                qn[b], qp[b], Kc_all, Kp_all, tok_idx, scaled,
                H=H, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=float(sm_scale),
            )

            # Launch lse_kernel: compute per-head lse (base-2) and store in lse[b, :]
            _lse_kernel[(H,)](
                scaled, lse[b], H=H, L_tokens=L_tokens
            )

            # Launch output_kernel: compute final output for this batch element
            _output_kernel[(H,)](
                scaled, Kc_all, tok_idx, lse[b], output[b], H=H, CK=CK, L_tokens=L_tokens
            )

        # Return output as bfloat16 and lse as float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
