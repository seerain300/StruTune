import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_logits_scaled_kernel(
    qn_ptr,        # *f32, base pointer to [H, CK]
    qp_ptr,        # *f32, base pointer to [H, KP]
    Kc_all_ptr,    # *f32, base pointer to [P, CK]
    Kp_all_ptr,    # *f32, base pointer to [P, KP]
    tok_idx_ptr,   # *i32, base pointer to [L_tokens]
    logits_ptr,    # *f32, base pointer to [H, L_tokens]
    H: tl.constexpr,           # number of heads
    CK: tl.constexpr,          # head_dim_ckv
    KP: tl.constexpr,          # head_dim_kpe
    L_tokens: tl.constexpr,    # number of tokens in slice
    sm_scale: tl.constexpr,    # scaling factor (float)
):
    # 2D grid: (head, token)
    h = tl.program_id(0)
    t = tl.program_id(1)
    if (h >= H) or (t >= L_tokens):
        return

    # Load qn[h, :] and qp[h, :]
    dim_ck = tl.arange(0, CK)
    qn_vec = tl.load(qn_ptr + h * CK + dim_ck)  # [CK], float32

    dim_kp = tl.arange(0, KP)
    qp_vec = tl.load(qp_ptr + h * KP + dim_kp)  # [KP], float32

    # Load Kc_row and Kp_row for this token
    idx = tl.load(tok_idx_ptr + t)              # int32 token index
    Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK]
    Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)  # [KP]

    # Compute dot-products
    dot_qn = tl.sum(qn_vec * Kc_row, axis=0)    # scalar float32
    dot_qp = tl.sum(qp_vec * Kp_row, axis=0)    # scalar float32
    scaled = (dot_qn + dot_qp) * sm_scale       # float32

    # Store scaled logits[h, t]
    tl.store(logits_ptr + h * L_tokens + t, scaled)


@triton.jit
def _lse_kernel(
    logits_ptr,   # *f32, base pointer to [H, L_tokens]
    lse_ptr,      # *f32, base pointer to [H]
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return
    # Compute max for numerical stability
    max_val = tl.full((), -float('inf'), tl.float32)
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        max_val = tl.maximum(max_val, val)

    # Sum exp(s - max)
    sum_exp = tl.zeros((), tl.float32)
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        sum_exp += tl.exp(val - max_val)

    # lse = log(sum_exp) + max_val  (natural log)
    lse = tl.log(sum_exp) + max_val
    tl.store(lse_ptr + h, lse)


@triton.jit
def _output_kernel(
    logits_ptr,   # *f32, base pointer to [H, L_tokens]
    Kc_all_ptr,   # *f32, base pointer to [P, CK]
    tok_idx_ptr,  # *i32, base pointer to [L_tokens]
    out_ptr,      # *f32, base pointer to [H, CK]
    lse_ptr,      # *f32, base pointer to [H]
    H: tl.constexpr,
    CK: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return
    lse = tl.load(lse_ptr + h)  # scalar float32

    # Accumulate output[h, :] = sum_t exp(logits[h, t] - lse) * Kc[t, :]
    out_acc = tl.zeros((CK,), tl.float32)
    for t in range(0, L_tokens):
        s = tl.load(logits_ptr + h * L_tokens + t)  # scaled logit for this head and token
        soft = tl.exp(s - lse)                      # softmax value for this token
        idx = tl.load(tok_idx_ptr + t)             # int32 token index
        Kc_row = tl.load(Kc_all_ptr + idx * CK + tl.arange(0, CK))  # [CK]
        out_acc += soft * Kc_row                   # vector accumulate

    tl.store(out_ptr + h * CK + tl.arange(0, CK), out_acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Inputs: q_nope [B, H, CK], q_pe [B, H, KP], ckv_cache [num_pages, 1, CK], kpe_cache [num_pages, 1, KP]
        device = q_nope.device
        dtype_f32 = torch.float32
        B, H, CK = q_nope.shape
        _, _, KP = q_pe.shape
        num_pages = ckv_cache.shape[0]

        # Convert inputs to float32 for compute (original tensors are bfloat16)
        qn = q_nope.to(dtype_f32)
        qp = q_pe.to(dtype_f32)
        Kc_all = ckv_cache.to(dtype_f32).squeeze(1)  # [num_pages, CK]
        Kp_all = kpe_cache.to(dtype_f32).squeeze(1)  # [num_pages, KP]

        # Outputs
        output = torch.empty((B, H, CK), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Process per batch
        for b in range(B):
            # Determine L_tokens for this batch slice: number of indices between indptr[b] and indptr[b+1]
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = max(0, page_end - page_beg)

            if L_tokens == 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)

            # Intermediate logits: [H, L_tokens]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Kernel 1: compute scaled logits
            grid_logits = (H, L_tokens)
            _compute_logits_scaled_kernel[grid_logits](
                qn[b],               # [H, CK]
                qp[b],               # [H, KP]
                Kc_all,              # [P, CK]
                Kp_all,              # [P, KP]
                tok_idx,             # [L_tokens]
                logits,              # [H, L_tokens]
                H=H, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=float(sm_scale),
                num_warps=1,
            )

            # Kernel 2: compute per-head lse (natural log)
            grid_lse = (H,)
            _lse_kernel[grid_lse](
                logits,              # [H, L_tokens]
                lse[b],              # [H]
                H=H, L_tokens=L_tokens,
                num_warps=1,
            )

            # Kernel 3: compute final output using softmax(exp(s - lse)) * Kc
            grid_out = (H,)
            _output_kernel[grid_out](
                logits,              # [H, L_tokens]
                Kc_all,              # [P, CK]
                tok_idx,             # [L_tokens]
                output[b],           # [H, CK]
                lse[b],              # [H]
                H=H, CK=CK, L_tokens=L_tokens,
                num_warps=1,
            )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
