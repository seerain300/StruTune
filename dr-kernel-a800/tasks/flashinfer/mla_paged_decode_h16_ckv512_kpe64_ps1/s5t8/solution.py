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
    qn_vec = tl.load(qn_ptr + h * CK + dim_ck)  # [CK], f32
    dim_kp = tl.arange(0, KP)
    qp_vec = tl.load(qp_ptr + h * KP + dim_kp)  # [KP], f32

    # Load token index and corresponding Kc row and Kp row
    idx = tl.load(tok_idx_ptr + t)             # i32
    Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK], f32
    Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)  # [KP], f32

    # Compute dot-products
    dot_qn = tl.sum(qn_vec * Kc_row)          # scalar f32
    dot_qp = tl.sum(qp_vec * Kp_row)          # scalar f32
    scaled_val = (dot_qn + dot_qp) * sm_scale  # f32

    # Store scaled logits
    tl.store(scaled_ptr + h * L_tokens + t, scaled_val)


@triton.jit
def _lse_kernel(
    scaled_ptr,   # *f32, base pointer to [H, L_tokens]
    lse_ptr,      # *f32, base pointer to [H]
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # one program per head
    h = tl.program_id(0)
    if h >= H:
        return

    # compute max over tokens
    max_val = -float("inf")
    for t in range(L_tokens):
        v = tl.load(scaled_ptr + h * L_tokens + t)  # f32
        max_val = tl.maximum(max_val, v)

    # compute sum exp(v - max)
    sum_exp = 0.0
    for t in range(L_tokens):
        v = tl.load(scaled_ptr + h * L_tokens + t)  # f32
        sum_exp += tl.exp(v - max_val)

    lse_val = tl.log(sum_exp) + max_val  # natural logsumexp
    tl.store(lse_ptr + h, lse_val)


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
    # one program per head
    h = tl.program_id(0)
    if h >= H:
        return

    # load lse for this head
    lse_val = tl.load(lse_ptr + h)  # f32

    # accumulator for output[h, :]
    out_acc = tl.zeros((CK,), tl.float32)

    # iterate tokens and accumulate
    for t in range(L_tokens):
        # softmax component for this token
        scaled_val = tl.load(scaled_ptr + h * L_tokens + t)  # f32
        soft = tl.exp(scaled_val - lse_val)  # f32

        # corresponding Kc row
        idx = tl.load(tok_idx_ptr + t)            # i32
        Kc_row = tl.load(Kc_all_ptr + idx * CK + tl.arange(0, CK))  # [CK], f32

        out_acc += soft * Kc_row

    # store final output[h, :]
    for i in range(CK):
        tl.store(out_ptr + h * CK + i, out_acc[i])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are float32 and contiguous
        device = q_nope.device
        # q_nope: [B, H, CK] -> take first batch: [H, CK]
        qn = q_nope.to(torch.float32).contiguous()
        # q_pe: [B, H, KP] -> take first batch: [H, KP]
        qp = q_pe.to(torch.float32).contiguous()
        H = qn.shape[0]  # number of heads
        CK = qn.shape[1]  # head_dim_ckv
        KP = qp.shape[1]  # head_dim_kpe

        # ckv_cache: [P, 1, CK] -> [P, CK]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [P, CK]
        # kpe_cache: [P, 1, KP] -> [P, KP]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [P, KP]

        # Batch size B and token range from kv_indptr for b=0
        B = q_nope.shape[0]
        b = 0
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        L_tokens = end - start
        tok_idx = kv_indices[start:end].to(torch.int32).to(device)

        # Allocate buffers
        scaled = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
        lse = torch.empty((H,), dtype=torch.float32, device=device)
        out = torch.empty((H, CK), dtype=torch.float32, device=device)

        # Launch kernels (ensure all are launched)
        grid_logits = (H, L_tokens)
        _compute_scaled_logits_kernel[grid_logits](
            qn, qp, Kc_all, Kp_all, tok_idx, scaled,
            H=H, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=float(sm_scale),
            num_warps=4,
        )

        grid_lse = (H,)
        _lse_kernel[grid_lse](
            scaled, lse,
            H=H, L_tokens=L_tokens,
            num_warps=1,
        )

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
