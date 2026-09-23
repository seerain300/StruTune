import math
import torch
import triton
import triton.language as tl


# Triton kernels: matmul for qn @ Kc.T and qp @ Kp.T
@triton.jit
def matmul_qn_KcT_fp32_tiled(
    qn_ptr,     # *fp32, [H, D], contiguous
    Kc_ptr,     # *fp32, [L_tokens, D], contiguous
    acc_ptr,    # *fp32, [H, L_tokens], contiguous
    H: tl.constexpr,         # number of heads
    D: tl.constexpr,         # head_dim_ckv (e.g., 512)
    L_tokens: tl.constexpr,  # tokens per batch
    BLOCK_H: tl.constexpr,   # tile over heads
    BLOCK_T: tl.constexpr,   # tile over tokens
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    acc = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)

    for d0 in range(0, D, BLOCK_T):
        d_offsets = d0 + tl.arange(0, BLOCK_T)
        # q_sub: [BLOCK_H, BLOCK_T]
        q_sub = tl.load(qn_ptr + h_offsets[:, None] * D + d_offsets[None, :],
                        mask=(h_offsets[:, None] < H) & (d_offsets[None, :] < D),
                        other=0.0)
        # k_sub: [BLOCK_T, BLOCK_T]
        k_sub = tl.load(Kc_ptr + t_offsets[:, None] * D + d_offsets[None, :],
                        mask=(t_offsets[:, None] < L_tokens) & (d_offsets[None, :] < D),
                        other=0.0)
        # acc += q_sub @ k_sub.T
        acc += tl.dot(q_sub, tl.trans(k_sub))

    tl.store(acc_ptr + h_offsets[:, None] * L_tokens + t_offsets[None, :],
             acc,
             mask=(h_offsets[:, None] < H) & (t_offsets[None, :] < L_tokens))


@triton.jit
def matmul_qp_KpT_fp32_tiled(
    qp_ptr,     # *fp32, [H, Dp], contiguous
    Kp_ptr,     # *fp32, [L_tokens, Dp], contiguous
    acc_ptr,    # *fp32, [H, L_tokens], contiguous
    H: tl.constexpr,
    Dp: tl.constexpr,        # head_dim_kpe (e.g., 64)
    L_tokens: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    acc = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)

    for dp0 in range(0, Dp, BLOCK_T):
        dp_offsets = dp0 + tl.arange(0, BLOCK_T)
        q_sub = tl.load(qp_ptr + h_offsets[:, None] * Dp + dp_offsets[None, :],
                        mask=(h_offsets[:, None] < H) & (dp_offsets[None, :] < Dp),
                        other=0.0)
        k_sub = tl.load(Kp_ptr + t_offsets[:, None] * Dp + dp_offsets[None, :],
                        mask=(t_offsets[:, None] < L_tokens) & (dp_offsets[None, :] < Dp),
                        other=0.0)
        acc += tl.dot(q_sub, tl.trans(k_sub))

    tl.store(acc_ptr + h_offsets[:, None] * L_tokens + t_offsets[None, :],
             acc,
             mask=(h_offsets[:, None] < H) & (t_offsets[None, :] < L_tokens))


# Helper kernel: add vec to each row of mat and store to out
@triton.jit
def add_vec_to_mat_fp32(
    vec_ptr,     # *fp32, [H], contiguous
    mat_ptr,     # *fp32, [H, L_tokens], contiguous
    out_ptr,     # *fp32, [H, L_tokens], contiguous
    H: tl.constexpr,
    L_tokens: tl.constexpr,
    scale: tl.float32,
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)

    h = pid_h
    t = pid_t

    vec = tl.load(vec_ptr + h)
    mat = tl.load(mat_ptr + h * L_tokens + t)
    out = mat + vec * scale
    tl.store(out_ptr + h * L_tokens + t, out)


# Per-row logsumexp (base-2), three-pass for correctness:
# 1) compute max per row; 2) compute sum(exp(logits - max)); 3) write normalized softmax and lse
@triton.jit
def softmax_lse_rows_fp32(
    logits_ptr,   # *fp32, [H, L_tokens], contiguous
    probs_ptr,    # *fp32, [H, L_tokens], contiguous (scratch)
    lse_ptr,      # *fp32, [H], contiguous
    H: tl.constexpr,
    L_tokens: tl.constexpr,
    scale: tl.float32,   # 1 / ln(2) for base-2 logsumexp
):
    pid_h = tl.program_id(0)
    h = pid_h

    # Pass 1: max
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + h * L_tokens + t)
        m = tl.maximum(m, logit)

    # Pass 2: sum exp(logits - m)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + h * L_tokens + t)
        sum_exp += tl.exp(logit - m)

    # Pass 3: write softmax and lse
    lse_val = tl.log(sum_exp) + m
    lse_val = lse_val * scale
    tl.store(lse_ptr + h, lse_val)
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + h * L_tokens + t)
        soft = tl.exp(logit - m) / sum_exp
        tl.store(probs_ptr + h * L_tokens + t, soft)


# Matmul for softmax @ Kc (per row across tokens, accumulate into output)
@triton.jit
def matmul_vec_rows_fp32(
    mat_ptr,      # *fp32, [H, L_tokens] contiguous (softmax rows)
    vec_ptr,      # *fp32, [L_tokens, D] contiguous (Kc tiles)
    out_ptr,      # *fp32, [H, D] contiguous
    H: tl.constexpr,
    D: tl.constexpr,
    L_tokens: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_d = tl.program_id(1)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    out = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    for t0 in range(0, L_tokens, BLOCK_D):
        t_offsets = t0 + tl.arange(0, BLOCK_D)
        mat_sub = tl.load(mat_ptr + h_offsets[:, None] * L_tokens + t_offsets[None, :],
                          mask=(h_offsets[:, None] < H) & (t_offsets[None, :] < L_tokens),
                          other=0.0)
        vec_sub = tl.load(vec_ptr + t_offsets[:, None] * D + d_offsets[None, :],
                          mask=(t_offsets[:, None] < L_tokens) & (d_offsets[None, :] < D),
                          other=0.0)
        out += tl.dot(mat_sub, vec_sub)  # [BLOCK_H, BLOCK_D]

    tl.store(out_ptr + h_offsets[:, None] * D + d_offsets[None, :],
             out,
             mask=(h_offsets[:, None] < H) & (d_offsets[None, :] < D))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device

        # Ensure dtypes and contiguity
        q_nope = q_nope.to(torch.float32).contiguous()  # [B, H, D]
        q_pe = q_pe.to(torch.float32).contiguous()     # [B, H, Dp]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D] -> we'll slice by tokens
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Dp]

        # Integers
        kv_indptr = kv_indptr.to(torch.int32)
        kv_indices = kv_indices.to(torch.int64)

        batch_size = q_nope.shape[0]
        H = q_nope.shape[1]
        D = q_nope.shape[2]
        Dp = q_pe.shape[2]
        assert H == 16 and D == 512 and Dp == 64, "Shape assertions for this implementation"

        output = torch.empty((batch_size, H, D), dtype=torch.float32, device=device)  # store fp32, convert later
        lse = torch.full((batch_size, H), -float("inf"), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Per-batch token count
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Gather tokens
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(device)  # int64
            Kc = Kc_all[tok_idx].to(torch.float32).contiguous()  # [L_tokens, D]
            Kp = Kp_all[tok_idx].to(torch.float32).contiguous()  # [L_tokens, Dp]

            # Allocate intermediates
            acc1 = torch.empty((H, L_tokens), dtype=torch.float32, device=device)  # qn @ Kc.T
            acc2 = torch.empty((H, L_tokens), dtype=torch.float32, device=device)  # qp @ Kp.T
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            probs = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Matmuls with tiling
            BLOCK_H = 8
            BLOCK_T = 64
            grid = (triton.cdiv(H, BLOCK_H), triton.cdiv(L_tokens, BLOCK_T))
            matmul_qn_KcT_fp32_tiled[grid](q_nope[b], Kc, acc1, H=H, D=D, L_tokens=L_tokens, BLOCK_H=BLOCK_H, BLOCK_T=BLOCK_T)
            matmul_qp_KpT_fp32_tiled[grid](q_pe[b], Kp, acc2, H=H, Dp=Dp, L_tokens=L_tokens, BLOCK_H=BLOCK_H, BLOCK_T=BLOCK_T)

            # Add and scale: logits = (acc1 + acc2) * sm_scale
            scale = sm_scale
            add_vec_to_mat_fp32[grid](acc1, acc2, logits, H=H, L_tokens=L_tokens, scale=scale)

            # lse and softmax per row, base-2 logsumexp
            ln2 = 1.4426950408889634  # 1 / ln(2)
            softmax_lse_rows_fp32[(H,)](logits, probs, lse[b], H=H, L_tokens=L_tokens, scale=1.0 / ln2)

            # Output: softmax @ Kc
            out_row = torch.empty((H, D), dtype=torch.float32, device=device)
            BLOCK_Hm = 8
            BLOCK_Dm = 128
            grid_out = (triton.cdiv(H, BLOCK_Hm), triton.cdiv(D, BLOCK_Dm))
            matmul_vec_rows_fp32[grid_out](probs, Kc, out_row, H=H, D=D, L_tokens=L_tokens, BLOCK_H=BLOCK_Hm, BLOCK_D=BLOCK_Dm)

            output[b] = out_row

        # Return output in bfloat16, lse in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
