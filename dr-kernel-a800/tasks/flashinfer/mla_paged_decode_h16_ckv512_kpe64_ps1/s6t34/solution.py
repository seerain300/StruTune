import math
import torch
import triton
import triton.language as tl


@triton.jit
def per_batch_logits_lse_out_kernel(
    qn_ptr,         # *fp32, contiguous tensor of shape [H, D]
    qp_ptr,         # *fp32, contiguous tensor of shape [H, Dp]
    Kc_ptr,         # *fp32, contiguous tensor of shape [L_tokens, D]
    Kp_ptr,         # *fp32, contiguous tensor of shape [L_tokens, Dp]
    out_ptr,        # *fp32, contiguous tensor of shape [H, D], flattened
    lse_ptr,        # *fp32, contiguous tensor of shape [H]
    H: tl.constexpr,         # number of heads (compile-time constant for Triton loops)
    D: tl.constexpr,         # head_dim_ckv = 512 (compile-time constant)
    Dp: tl.constexpr,        # head_dim_kpe = 64 (compile-time constant)
    L_tokens,       # runtime int: number of tokens for this batch element
    sm_scale,       # runtime float32 scaling factor
):
    # Compute per-head lse and output by looping over h (we process one head at a time)
    for h in range(0, H):
        # 1) Compute logits_scaled[h, t] for t in 0..L_tokens-1
        logits_row = [0.0] * L_tokens  # list of floats
        for t in range(0, L_tokens):
            acc1 = 0.0
            # Dot qn[h, :] with Kc[t, :]
            for kk in range(0, D):
                val = tl.load(qn_ptr + h * D + kk)  # qn_ptr[h, kk]
                kv = tl.load(Kc_ptr + t * D + kk)   # Kc[t, kk]
                acc1 += val * kv
            acc2 = 0.0
            # Dot qp[h, :] with Kp[t, :]
            for kk in range(0, Dp):
                val = tl.load(qp_ptr + h * Dp + kk) # qp_ptr[h, kk]
                kv = tl.load(Kp_ptr + t * Dp + kk)  # Kp[t, kk]
                acc2 += val * kv
            logits_row[t] = (acc1 + acc2) * sm_scale

        # 2) Compute lse[h] = logsumexp(logits_row) / ln(2)
        max_logit = -float("inf")
        for t in range(0, L_tokens):
            max_logit = tl.maximum(max_logit, logits_row[t])
        sum_exp = 0.0
        for t in range(0, L_tokens):
            sum_exp += tl.exp(logits_row[t] - max_logit)
        lse_val = tl.log(sum_exp) + max_logit  # logsumexp
        lse_val = lse_val / math.log(2.0)
        tl.store(lse_ptr + h, lse_val)

        # 3) Compute output[h, :] = softmax(logits_row) @ Kc[:, :]
        out_vec = [0.0] * D
        sum_exp = 0.0
        for t in range(0, L_tokens):
            sum_exp += tl.exp(logits_row[t] - max_logit)
        for kk in range(0, D):
            acc = 0.0
            for t in range(0, L_tokens):
                p = tl.exp(logits_row[t] - max_logit) / sum_exp
                kv = tl.load(Kc_ptr + t * D + kk)
                acc += p * kv
            tl.store(out_ptr + h * D + kk, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Cast inputs to float32 for Triton computation
        device = q_nope.device
        assert q_nope.dtype in (torch.float32, torch.bfloat16)
        assert q_pe.dtype in (torch.float32, torch.bfloat16)
        q_nope = q_nope.to(torch.float32)
        q_pe = q_pe.to(torch.float32)

        # Kc_all and Kp_all are cached per token; cast to float32
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        batch_size = q_nope.shape[0]
        H = q_nope.shape[1]  # num_qo_heads
        D = q_nope.shape[2]  # head_dim_ckv = 512
        Dp = q_pe.shape[2]   # head_dim_kpe = 64

        output = torch.empty((batch_size, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        # Process one batch element at a time to ensure correct slicing
        for b in range(batch_size):
            # Determine number of tokens for this batch element
            b0 = b
            b1 = b + 1
            assert b1 < kv_indptr.numel(), "kv_indptr length must cover batch"
            L_tokens = int(kv_indptr[b1].item() - kv_indptr[b0].item())
            if L_tokens <= 0:
                # No KV entries for this batch element: output zeros, lse = -inf
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Gather token indices for this batch element
            tok_idx = kv_indices[kv_indptr[b0]:kv_indptr[b1]].to(torch.int32)

            # Slice Kc and Kp for these tokens
            Kc = Kc_all[tok_idx]  # [L_tokens, D]
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

            # Launch Triton per-batch kernel
            grid = (1,)  # single program handles all heads and tokens
            per_batch_logits_lse_out_kernel[grid](
                q_nope[b], q_pe[b], Kc, Kp,
                output[b], lse[b],
                H=H, D=D, Dp=Dp, L_tokens=L_tokens, sm_scale=float(sm_scale),
                num_warps=1, num_stages=1
            )

        # Return output as bfloat16 and lse as float32 to match original behavior
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
