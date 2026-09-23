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
    logits_ptr,    # *f32, base pointer to [H, L_tokens]
    H: tl.constexpr,
    CK: tl.constexpr,
    KP: tl.constexpr,
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,  # scalar, e.g., 1.0
):
    # Grid: (H, L_tokens)
    h = tl.program_id(0)
    t = tl.program_id(1)
    if (h >= H) or (t >= L_tokens):
        return

    # Load query vectors
    dim_ck = tl.arange(0, CK)
    dim_kp = tl.arange(0, KP)
    qn_vec = tl.load(qn_ptr + h * CK + dim_ck)   # [CK]
    qp_vec = tl.load(qp_ptr + h * KP + dim_kp)   # [KP]

    # Load K rows for this token
    idx = tl.load(tok_idx_ptr + t)               # int32
    Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK]
    Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)  # [KP]

    # Dot-products
    dot_qn = tl.sum(qn_vec * Kc_row)            # scalar
    dot_qp = tl.sum(qp_vec * Kp_row)            # scalar

    scaled = (dot_qn + dot_qp) * sm_scale       # scalar
    tl.store(logits_ptr + h * L_tokens + t, scaled)


@triton.jit
def _lse_base2_kernel(
    scaled_ptr,    # *f32, base pointer to [H, L_tokens]
    lse_ptr,       # *f32, base pointer to [H]
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # Compute per-head lse in base-2: lse = max(scaled) + log(sum(exp(scaled - max))) / ln(2)
    h = tl.program_id(0)
    if h >= H:
        return

    # Pass 1: find max
    max_val = -float("inf")
    for t in range(0, L_tokens):
        s = tl.load(scaled_ptr + h * L_tokens + t)
        if s > max_val:
            max_val = s

    # Pass 2: sum exp(s - max)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        s = tl.load(scaled_ptr + h * L_tokens + t)
        sum_exp += tl.exp(s - max_val)

    # lse = max + log(sum) / ln(2)
    lse_base2 = max_val + (tl.log(sum_exp) * 1.4426950408889634)  # 1/ln(2) = ~1.442695...
    tl.store(lse_ptr + h, lse_base2)


@triton.jit
def _compute_output_kernel(
    scaled_ptr,    # *f32, base pointer to [H, L_tokens] (already scaled by sm_scale, base-2 factor included implicitly because we apply base-2 lse)
    Kc_all_ptr,    # *f32, base pointer to [P, CK]
    tok_idx_ptr,   # *i32, base pointer to [L_tokens]
    out_ptr,       # *f32, base pointer to [H, CK]
    lse_ptr,       # *f32, base pointer to [H]
    H: tl.constexpr,
    CK: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return

    lse_h = tl.load(lse_ptr + h)  # base-2 logsumexp

    # Accumulate output[h, :]
    out_acc = tl.zeros((CK,), tl.float32)
    for t in range(0, L_tokens):
        s = tl.load(scaled_ptr + h * L_tokens + t)  # base-2 scaled logit
        soft = tl.exp(s - lse_h)                    # softmax value in base-2 sense
        idx = tl.load(tok_idx_ptr + t)
        Kc_row = tl.load(Kc_all_ptr + idx * CK + tl.arange(0, CK))  # [CK]
        out_acc += soft * Kc_row

    tl.store(out_ptr + h * CK + tl.arange(0, CK), out_acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Inputs: q_nope [B, H, CK], q_pe [B, H, KP], ckv_cache [num_pages, 1, CK], kpe_cache [num_pages, 1, KP]
        device = q_nope.device
        dtype_f32 = torch.float32

        B, H, CK = q_nope.shape
        _, _, KP = q_pe.shape
        num_pages = ckv_cache.shape[0]

        # Convert inputs to float32 for compute
        qn = q_nope.to(dtype_f32)               # [B, H, CK]
        qp = q_pe.to(dtype_f32)                 # [B, H, KP]
        Kc_all = ckv_cache.to(dtype_f32).squeeze(1)  # [P, CK]
        Kp_all = kpe_cache.to(dtype_f32).squeeze(1)  # [P, KP]

        # Allocate intermediates
        scaled_logits = torch.empty((B * H, 0), dtype=torch.float32, device=device)  # placeholder, we'll compute per b
        output = torch.empty((B, H, CK), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Process per batch
        for b in range(B):
            # Determine L_tokens for this batch slice: number of indices between indptr[b] and indptr[b+1]
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = max(0, page_end - page_beg)

            # Allocate per-batch work tensors
            H_b = H  # number of heads
            scaled_ptr = torch.empty((H_b * L_tokens,), dtype=torch.float32, device=device)
            tok_idx_ptr = kv_indices[page_beg:page_end].to(torch.int32).contiguous()

            # Launch kernel to compute scaled_logits[h, t] for all heads and tokens
            grid = (H_b, L_tokens)
            _compute_scaled_logits_kernel[grid](
                qn[b], qp[b], Kc_all, Kp_all, tok_idx_ptr, scaled_ptr,
                H=H_b, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=sm_scale,
                num_warps=1, num_stages=1
            )

            # Compute lse per head (base-2) in Triton
            grid_lse = (H_b,)
            _lse_base2_kernel[grid_lse](
                scaled_ptr, lse[b], H=H_b, L_tokens=L_tokens,
                num_warps=1, num_stages=1
            )

            # Compute final output per head in Triton
            _compute_output_kernel[grid](
                scaled_ptr, Kc_all, tok_idx_ptr, output[b], lse[b],
                H=H_b, CK=CK, L_tokens=L_tokens,
                num_warps=1, num_stages=1
            )

        # Cast output back to bfloat16 to match original behavior
        output_bf16 = output.to(torch.bfloat16)

        # Return output and lse (lse kept as float32; original code divides by ln(2) at host, but here we mimic base-2 lse computed in-kernel)
        # Note: The original returns (output, lse). We keep dtype: output bfloat16, lse float32.
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
