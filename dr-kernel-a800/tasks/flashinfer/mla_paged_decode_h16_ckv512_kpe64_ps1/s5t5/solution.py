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
    H: tl.constexpr,          # num_qo_heads
    CK: tl.constexpr,         # head_dim_ckv
    KP: tl.constexpr,         # head_dim_kpe
    L_tokens: tl.constexpr,   # number of tokens for this batch
    sm_scale: tl.constexpr,   # scaling factor
):
    # 2D grid: (H, L_tokens)
    h = tl.program_id(0)
    t = tl.program_id(1)
    if (h >= H) or (t >= L_tokens):
        return

    dim_ck = tl.arange(0, CK)
    dim_kp = tl.arange(0, KP)

    # load qn[h, :] and qp[h, :]
    qn_vec = tl.load(qn_ptr + h * CK + dim_ck)   # [CK], f32
    qp_vec = tl.load(qp_ptr + h * KP + dim_kp)   # [KP], f32

    # token index and corresponding rows from cache
    idx = tl.load(tok_idx_ptr + t)               # i32
    Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK], f32
    Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)  # [KP], f32

    # dot products
    dot_qn = tl.sum(qn_vec * Kc_row)             # scalar f32
    dot_qp = tl.sum(qp_vec * Kp_row)             # scalar f32
    scaled = (dot_qn + dot_qp) * sm_scale        # f32

    # store to logits buffer
    tl.store(logits_ptr + h * L_tokens + t, scaled)


@triton.jit
def _lse_kernel(
    scaled_ptr,   # *f32, base pointer to [H, L_tokens]
    lse_ptr,      # *f32, base pointer to [H]
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # one program per head; grid can be (H, 1)
    h = tl.program_id(0)
    if h >= H:
        return

    # compute max across tokens
    max_val = -float("inf")
    for t in range(L_tokens):
        v = tl.load(scaled_ptr + h * L_tokens + t)  # f32
        max_val = tl.maximum(max_val, v)

    # compute sum exp(scaled - max)
    sum_exp = 0.0
    for t in range(L_tokens):
        v = tl.load(scaled_ptr + h * L_tokens + t)  # f32
        sum_exp += tl.exp(v - max_val)

    lse = tl.log(sum_exp) + max_val  # natural logsumexp
    tl.store(lse_ptr + h, lse)


@triton.jit
def _output_kernel(
    qn_ptr,        # *f32, [H, CK]
    Kc_all_ptr,    # *f32, [P, CK]
    tok_idx_ptr,   # *i32, [L_tokens]
    lse_ptr,       # *f32, [H]
    out_ptr,       # *f32, [H, CK]
    H: tl.constexpr,
    CK: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # one program per head; grid can be (H, 1)
    h = tl.program_id(0)
    if h >= H:
        return

    lse_val = tl.load(lse_ptr + h)  # f32

    # accumulator for output[h, :]
    out_acc = tl.zeros((CK,), tl.float32)

    # iterate tokens to accumulate output
    for t in range(L_tokens):
        # softmax component for this token
        scaled = tl.load(scaled_ptr + h * L_tokens + t)  # f32
        soft = tl.exp(scaled - lse_val)  # softmax at this token

        # corresponding Kc row
        idx = tl.load(tok_idx_ptr + t)            # i32
        Kc_row = tl.load(Kc_all_ptr + idx * CK + tl.arange(0, CK))  # [CK], f32

        out_acc += soft * Kc_row

    # store output for head h
    for i in range(CK):
        tl.store(out_ptr + h * CK + i, out_acc[i])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on same device and cast to float32 for compute
        device = q_nope.device
        qn = q_nope.to(torch.float32).contiguous()  # [B,


def run(*args):
    return ModelNew()(*args)
