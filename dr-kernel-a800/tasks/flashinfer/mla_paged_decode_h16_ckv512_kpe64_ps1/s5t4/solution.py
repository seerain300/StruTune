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
    H: tl.constexpr,            # num_qo_heads
    CK: tl.constexpr,           # head_dim_ckv
    KP: tl.constexpr,           # head_dim_kpe
    L_tokens: tl.constexpr,     # number of tokens in this batch
    sm_scale: tl.constexpr,     # scaling factor
):
    # Grid: one program per (head, token)
    h = tl.program_id(0)
    t = tl.program_id(1)
    if h >= H or t >= L_tokens:
        return

    # Vectors for head and cache dimensions
    dim_ck = tl.arange(0, CK)
    dim_kp = tl.arange(0, KP)

    # Load qn[h, :] and qp[h, :]
    qn_vec = tl.load(qn_ptr + h * CK + dim_ck)  # [CK]
    qp_vec = tl.load(qp_ptr + h * KP + dim_kp)  # [KP]

    # Gather Kc_row and Kp_row for this token index
    idx = tl.load(tok_idx_ptr + t)               # int32
    Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK]
    Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)  # [KP]

    # Compute dot products
    dot_qn = tl.sum(qn_vec * Kc_row)  # scalar
    dot_qp = tl.sum(qp_vec * Kp_row)  # scalar
    scaled = (dot_qn + dot_qp) * sm_scale

    # Store scaled logits at [h, t]
    tl.store(logits_ptr + h * L_tokens + t, scaled)


@triton.jit
def _lse_kernel(
    scaled_ptr,   # *f32, base pointer to [H, L_tokens]
    lse_ptr,      # *f32, base pointer to [H]
    H: tl.constexpr,          # num_qo_heads
    L_tokens: tl.constexpr,   # number of tokens
):
    # One program per head
    h = tl.program_id(0)
    if h >= H:
        return

    # Pass 1: compute max over tokens
    max_val = -float("inf")
    for t in range(0, L_tokens):
        val = tl.load(scaled_ptr + h * L_tokens + t)
        if val > max_val:
            max_val = val

    # Pass 2: sum exp(s - max)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(scaled_ptr + h * L_tokens + t)
        sum_exp += tl.exp(val - max_val)

    lse = tl.log(sum_exp)
    tl.store(lse_ptr + h, lse)


@triton.jit
def _output_kernel(
    scaled_ptr,   # *f32, base pointer to [H, L_tokens]
    Kc_all_ptr,   # *f32, base pointer to [P, CK]
    tok_idx_ptr,  # *i32, base pointer to [L_tokens]
    out_ptr,      # *f32, base pointer to [H, CK]
    H: tl.constexpr,          # num_qo_heads
    CK: tl.constexpr,         # head_dim_ckv
    L_tokens: tl.constexpr,   # number of tokens
    lse_ptr,      # *f32, base pointer to [H]
):
    # One program per head
    h = tl.program_id(0)
    if h >= H:
        return

    # Load lse for this head
    lse_h = tl.load(lse_ptr + h)

    # Accumulate output over tokens: out[h, :] += sum_t exp(scaled[h, t] - lse_h) * Kc[t, :]
    dim_ck = tl.arange(0, CK)
    out_acc = tl.zeros((CK,), tl.float32)

    for t in range(0, L_tokens):
        # softmax contribution for token t
        scaled = tl.load(scaled_ptr + h * L_tokens + t)
        p = tl.exp(scaled - lse_h)

        # Gather Kc_row and accumulate
        idx = tl.load(tok_idx_ptr + t)
        Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK]
        out_acc += p * Kc_row

    # Store output for this head
    tl.store(out_ptr + h * CK + dim_ck, out_acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation of the original run function.
        Computes:
          - output: [batch_size, num_qo_heads, head_dim_ckv], bfloat16
          - lse: [batch_size, num_qo_heads], float32 (raw logsumexp; original divides by log(2), optional to apply on host)
        """
        device = q_nope.device
        H = q_nope.shape[1]
        CK = q_nope.shape[2]
        KP = q_pe.shape[2]

        # Ensure inputs are on CUDA and float32 for compute
        qn = q_nope.contiguous().to(torch.float32)
        qp = q_pe.contiguous().to(torch.float32)
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [P, CK]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [P, KP]

        # Batch size from indptr
        batch_size = kv_indptr.shape[0] - 1

        # Allocate outputs and intermediates
        output = torch.empty((batch_size, H, CK), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(batch_size):
            # Compute L_tokens for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg

            # Early continue if no tokens
            if L_tokens <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # Token indices for this batch element
            tok_idx = kv_indices[page_beg:page_end].contiguous().to(torch.int32)

            # 1) Compute scaled_logits [H, L_tokens]
            scaled_logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            grid = (H, L_tokens)
            _compute_scaled_logits_kernel[grid](
                qn_ptr=qn,
                qp_ptr=qp,
                Kc_all_ptr=Kc_all,
                Kp_all_ptr=Kp_all,
                tok_idx_ptr=tok_idx,
                logits_ptr=scaled_logits,
                H=H, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=float(sm_scale),
                num_warps=1, num_stages=1
            )

            # 2) Compute lse per head
            grid_lse = (H,)
            _lse_kernel[grid_lse](
                scaled_ptr=scaled_logits,
                lse_ptr=lse[b],
                H=H, L_tokens=L_tokens,
                num_warps=1, num_stages=1
            )

            # 3) Compute final output per head
            grid_out = (H,)
            _output_kernel[grid_out](
                scaled_ptr=scaled_logits,
                Kc_all_ptr=Kc_all,
                tok_idx_ptr=tok_idx,
                out_ptr=output[b],
                H=H, CK=CK, L_tokens=L_tokens, lse_ptr=lse[b],
                num_warps=1, num_stages=1
            )

        # Cast output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
