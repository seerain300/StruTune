import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_scaled_kernel(
    qn_ptr,         # *fp32, shape [H, D], contiguous
    qp_ptr,         # *fp32, shape [H, Dp], contiguous
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, shape [L_tokens, Dp], contiguous
    logits_ptr,     # *fp32, flattened buffer [H*L_tokens], contiguous
    H: tl.constexpr,          # number of heads (runtime int, but we'll pass as normal; not used in kernel math here)
    D: tl.constexpr,          # 512 (compile-time const)
    Dp: tl.constexpr,         # 64 (compile-time const)
    L_tokens,       # int32 (runtime)
):
    # Grid: 2D (h, t) where h in [0, H), t in [0, L_tokens)
    h = tl.program_id(0)
    t = tl.program_id(1)

    # Compute dot with qn[h, :] and Kc[t, :]
    acc1 = 0.0
    for k in range(0, D):
        q = tl.load(qn_ptr + h * D + k)
        kc = tl.load(Kc_ptr + t * D + k)
        acc1 += q * kc

    # Compute dot with qp[h, :] and Kp[t, :]
    acc2 = 0.0
    for k in range(0, Dp):
        q = tl.load(qp_ptr + h * Dp + k)
        kp = tl.load(Kp_ptr + t * Dp + k)
        acc2 += q * kp

    logit = (acc1 + acc2)
    tl.store(logits_ptr + h * L_tokens + t, logit)


@triton.jit
def compute_lse_kernel(
    logits_ptr,     # *fp32, shape [H, L_tokens], contiguous
    lse_ptr,        # *fp32, shape [H], contiguous
    L_tokens,       # int32
    ln2,            # float32, natural log(2) ~ 0.6931472
):
    h = tl.program_id(0)
    # First pass: find max for numerical stability
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + h * L_tokens + t)
        if logit > m:
            m = logit
    # Second pass: sum exp(logits - m)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + h * L_tokens + t)
        exp_t = tl.exp(logit - m)
        sum_exp += exp_t
    lse = tl.log(sum_exp) / ln2
    tl.store(lse_ptr + h, lse)


@triton.jit
def compute_output_kernel(
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    logits_ptr,     # *fp32, shape [H, L_tokens], contiguous
    output_ptr,     # *fp32, shape [H, D], contiguous
    L_tokens,       # int32
):
    h = tl.program_id(0)

    # Pass 1: compute lse for softmax normalization
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + h * L_tokens + t)
        if logit > m:
            m = logit
    sum_exp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + h * L_tokens + t)
        exp_t = tl.exp(logit - m)
        sum_exp += exp_t
    lse = tl.log(sum_exp)

    # Pass 2: compute output[h, :] = softmax(logits[h, :]) @ Kc[:, :]
    # Initialize output to zeros
    for k in range(0, 512):
        tl.store(output_ptr + h * 512 + k, 0.0)
    # Accumulate contributions
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + h * L_tokens + t)
        attn = tl.exp(logit - lse)  # softmax probability for token t
        # out[k] += attn * Kc[t, k]
        for k in range(0, 512):
            kc = tl.load(Kc_ptr + t * 512 + k)
            out_val = tl.load(output_ptr + h * 512 + k)
            out_val += attn * kc
            tl.store(output_ptr + h * 512 + k, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        # We assume D=512, Dp=64 as per the original constraints.
        D = 512
        Dp = 64

        # Ensure caches are float32 for computation
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, Dp]

        output = torch.empty((B, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No tokens, set zeros and -inf lse
                output[b] = 0.0
                lse[b] = -float("inf")
                continue

            # Gather token indices and corresponding Kc, Kp rows
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]]  # [L_tokens]
            # Gather rows from caches
            Kc = Kc_all[tok_idx]  # [L_tokens, D]
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

            # Cast queries to float32
            qn = q_nope[b].to(torch.float32)  # [H, D]
            qp = q_pe[b].to(torch.float32)    # [H, Dp]

            # Allocate logits buffer for this batch
            logits = torch.empty((H * L_tokens,), dtype=torch.float32, device=device)

            # Launch Triton kernel to compute logits_scaled[h, t]
            grid = (H, L_tokens)
            compute_logits_scaled_kernel[grid](
                qn, qp, Kc, Kp, logits,
                H=H, D=512, Dp=64, L_tokens=L_tokens
            )

            # Compute lse per head (divide by ln(2) to match base-2 logsumexp)
            ln2 = 0.6931472
            grid_lse = (H,)
            compute_lse_kernel[grid_lse](
                logits, lse[b], L_tokens, ln2
            )

            # Compute output[h, :] = softmax(logits[h, :]) @ Kc
            grid_out = (H,)
            compute_output_kernel[grid_out](
                Kc, logits, output[b], L_tokens
            )

        # Return output in bfloat16 as per original, lse in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
