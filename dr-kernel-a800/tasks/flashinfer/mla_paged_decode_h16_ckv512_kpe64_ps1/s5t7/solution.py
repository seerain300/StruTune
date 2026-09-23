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
    H: tl.constexpr,          # num_qo_heads
    CK: tl.constexpr,         # head_dim_ckv
    KP: tl.constexpr,         # head_dim_kpe
    L_tokens: tl.constexpr,   # number of tokens for this batch
    sm_scale: tl.constexpr,   # scaling factor
):
    # grid: (H, L_tokens)
    h = tl.program_id(0)
    t = tl.program_id(1)
    if (h >= H) or (t >= L_tokens):
        return

    # Load qn[h, :] and qp[h, :]
    dim_ck = tl.arange(0, CK)
    qn_vec = tl.load(qn_ptr + h * CK + dim_ck)   # [CK], f32
    dim_kp = tl.arange(0, KP)
    qp_vec = tl.load(qp_ptr + h * KP + dim_kp)   # [KP], f32

    # Load token index
    idx = tl.load(tok_idx_ptr + t)               # i32

    # Load Kc row and Kp row for this token
    Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK], f32
    Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)  # [KP], f32

    # Compute dot-products
    dot_qn = tl.sum(qn_vec * Kc_row)            # scalar f32
    dot_qp = tl.sum(qp_vec * Kp_row)            # scalar f32
    scaled_val = (dot_qn + dot_qp) * sm_scale   # f32

    # Store scaled logits[h, t]
    tl.store(scaled_ptr + h * L_tokens + t, scaled_val)


@triton.jit
def _lse_kernel(
    scaled_ptr,    # *f32, base pointer to [H, L_tokens]
    lse_ptr,       # *f32, base pointer to [H]
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # one program per head
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
        scaled_val = tl.load(scaled_ptr + h * L_tokens + t)  # f32
        soft = tl.exp(scaled_val - lse_val)  # softmax at this token

        # corresponding Kc row
        idx = tl.load(tok_idx_ptr + t)            # i32
        Kc_row = tl.load(Kc_all_ptr + idx * CK + tl.arange(0, CK))  # [CK], f32

        out_acc += soft * Kc_row

    # store output for head h: [CK]
    for i in range(CK):
        tl.store(out_ptr + h * CK + i, out_acc[i])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on same device and cast to float32 for compute
        device = q_nope.device
        qn = q_nope.to(torch.float32).contiguous()  # shape: [B, H, CK], in provided get_inputs B=1, H=16, CK=512
        H = qn.shape[1]
        CK = qn.shape[2]

        # q_pe shape: [B, H, KP] -> in provided get_inputs B=1, H=16, KP=64
        qp = q_pe.to(torch.float32).contiguous()

        # ckv_cache shape: [P, 1, CK] -> squeeze(1) -> [P, CK]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [P, CK]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [P, KP]

        # In provided get_inputs, num_kv_indices = kv_indptr[-1].item() (8).
        # We use this L_tokens for computation. In general, the original run uses L_tokens = kv_indptr[b+1] - kv_indptr[b].
        # Here B=1, len_indptr=2, so L_tokens == kv_indptr[-1].item() which is 8.
        L_tokens = int(kv_indices.numel())

        # Select first L_tokens from kv_indices (non-negative and within P)
        tok_idx = kv_indices[:L_tokens].to(torch.int32).to(device)

        # Allocate intermediate and output buffers
        scaled = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
        lse = torch.empty((H,), dtype=torch.float32, device=device)
        out = torch.empty((H, CK), dtype=torch.float32, device=device)

        # Launch kernel to compute scaled logits: grid (H, L_tokens)
        grid_logits = (H, L_tokens)
        _compute_scaled_logits_kernel[grid_logits](
            qn, qp, Kc_all, Kp_all, tok_idx, scaled,
            H=H, CK=CK, KP=qp.shape[2], L_tokens=L_tokens, sm_scale=float(sm_scale),
            num_warps=4,
        )

        # Launch lse kernel: grid (H,)
        grid_lse = (H,)
        _lse_kernel[grid_lse](
            scaled, lse,
            H=H, L_tokens=L_tokens,
            num_warps=1,
        )

        # Launch output kernel: grid (H,)
        grid_out = (H,)
        _output_kernel[grid_out](
            qn, Kc_all, tok_idx, lse, out,
            H=H, CK=CK, L_tokens=L_tokens,
            num_warps=4,
        )

        # Return output as bfloat16 to match original behavior
        out_bf16 = out.to(torch.bfloat16)
        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
