import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_scaled_per_batch_kernel(
    qn_ptr,         # *fp32, [H, D], contiguous
    qp_ptr,         # *fp32, [H, Dp], contiguous
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, [L_tokens, Dp], contiguous
    logits_ptr,     # *fp32, [H, L_tokens], contiguous
    H,              # int32
    L_tokens,       # int32
    sm_scale,       # fp32
    D: tl.constexpr,        # 512
    Dp: tl.constexpr,       # 64
):
    # Grid (H, L_tokens): each program computes one logits[h, t]
    h = tl.program_id(0)
    t = tl.program_id(1)

    acc1 = 0.0
    acc2 = 0.0

    # Dot qn[h, :] with Kc[t, :]
    for kk in range(0, D):
        q = tl.load(qn_ptr + h * D + kk)  # qn[h, kk]
        k = tl.load(Kc_ptr + t * D + kk)  # Kc[t, kk]
        acc1 += q * k

    # Dot qp[h, :] with Kp[t, :]
    for kk in range(0, Dp):
        q = tl.load(qp_ptr + h * Dp + kk)  # qp[h, kk]
        k = tl.load(Kp_ptr + t * Dp + kk)  # Kp[t, kk]
        acc2 += q * k

    logit = (acc1 + acc2) * sm_scale
    tl.store(logits_ptr + h * L_tokens + t, logit)


@triton.jit
def compute_lse_per_batch_kernel(
    logits_ptr,     # *fp32, [H, L_tokens]
    lse_ptr,        # *fp32, [H]
    H,              # int32
    L_tokens,       # int32
):
    h = tl.program_id(0)
    m = -float("inf")
    # Compute max for numerical stability
    for t in range(0, L_tokens):
        v = tl.load(logits_ptr + h * L_tokens + t)
        if v > m:
            m = v

    # Compute sum(exp(logits - m))
    sum_exp = 0.0
    for t in range(0, L_tokens):
        v = tl.load(logits_ptr + h * L_tokens + t)
        sum_exp += tl.exp(v - m)

    lse = tl.log(sum_exp) + m
    lse = lse / math.log(2.0)  # base-2 logsumexp
    tl.store(lse_ptr + h, lse)


@triton.jit
def compute_output_per_batch_kernel(
    logits_ptr,     # *fp32, [H, L_tokens]
    Kc_ptr,         # *fp32, [L_tokens, D]
    out_ptr,        # *fp32, [H, D]
    H,              # int32
    L_tokens,       # int32
    D: tl.constexpr,        # 512
):
    h = tl.program_id(0)
    # Compute output[h, kk] = sum_t softmax(logits[h, t]) * Kc[t, kk]
    for kk in range(0, D):
        acc = 0.0
        den = 0.0
        # Compute denominator sum exp(logits[h, t])
        for t in range(0, L_tokens):
            v = tl.load(logits_ptr + h * L_tokens + t)
            den += tl.exp(v)
        # Compute numerator and accumulate
        for t in range(0, L_tokens):
            v = tl.load(logits_ptr + h * L_tokens + t)
            p = tl.exp(v) / den
            k = tl.load(Kc_ptr + t * D + kk)
            acc += p * k
        tl.store(out_ptr + h * D + kk, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA."

        batch_size = q_nope.shape[0]
        H = q_nope.shape[1]
        D = q_nope.shape[2]
        Dp = q_pe.shape[2]
        device = q_nope.device

        # Prepare Kc_all and Kp_all as fp32; shape [num_pages, D] and [num_pages, Dp]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, Dp]

        # Output and lse buffers
        output = torch.empty((batch_size, H, D), dtype=torch.bfloat16, device=device)  # final output dtype
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Compute valid tokens for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = max(page_end - page_beg, 0)
            if L_tokens == 0:
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Gather token indices and corresponding Kc/Kp rows
            tok_idx = kv_indices[page_beg:page_end].to(torch.long)  # [L_tokens]
            Kc = Kc_all[tok_idx].contiguous().to(torch.float32)    # [L_tokens, D]
            Kp = Kp_all[tok_idx].contiguous().to(torch.float32)    # [L_tokens, Dp]

            # Prepare q for this batch element (convert to fp32 for Triton)
            qn = q_nope[b].to(torch.float32).contiguous()          # [H, D]
            qp = q_pe[b].to(torch.float32).contiguous()            # [H, Dp]

            # Allocate intermediate Triton buffers
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
            out = torch.empty((H, D), dtype=torch.float32, device=device)

            # Launch kernel to compute logits_scaled[h, t] for this batch
            grid = (H, L_tokens)
            compute_logits_scaled_per_batch_kernel[grid](
                qn, qp, Kc, Kp, logits, H, L_tokens, float(sm_scale),
                D=512, Dp=64,
                num_warps=4,
            )

            # Launch kernel to compute lse[h] for this batch
            grid_lse = (H,)
            compute_lse_per_batch_kernel[grid_lse](
                logits, lse[b],
                H, L_tokens,
                num_warps=1,
            )

            # Launch kernel to compute output[h, :] for this batch
            grid_out = (H,)
            compute_output_per_batch_kernel[grid_out](
                logits, Kc, out, H, L_tokens,
                D=512,
                num_warps=4,
            )

            # Store output as bfloat16
            output[b] = out.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
