import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_scaled_kernel(
    qn_ptr,         # *fp32, [H, D], contiguous
    qp_ptr,         # *fp32, [H, Dp], contiguous
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, [L_tokens, Dp], contiguous
    logits_ptr,     # *fp32, [H * L_tokens], contiguous flattened
    H,              # int32
    D: tl.constexpr,         # 512
    Dp: tl.constexpr,        # 64
    L_tokens,       # int32
):
    h = tl.program_id(0)  # head id in 0..H-1
    # For each token t, compute dot product with Kc and Kp and store to logits[h * L_tokens + t]
    for t in range(0, L_tokens):
        acc1 = 0.0
        # Dot qn[h, :] with Kc[t, :]
        for k in range(0, D):
            q = tl.load(qn_ptr + h * D + k)
            kc = tl.load(Kc_ptr + t * D + k)
            acc1 += q * kc
        acc2 = 0.0
        # Dot qp[h, :] with Kp[t, :]
        for k in range(0, Dp):
            q = tl.load(qp_ptr + h * Dp + k)
            kp = tl.load(Kp_ptr + t * Dp + k)
            acc2 += q * kp
        logit = acc1 + acc2
        tl.store(logits_ptr + h * L_tokens + t, logit)


@triton.jit
def compute_lse_kernel(
    logits_ptr,     # *fp32, [H * L_tokens], contiguous
    lse_ptr,        # *fp32, [H], contiguous
    H,              # int32
    L_tokens,       # int32
    ln2,            # fp32, 1.4426950408889634
):
    h = tl.program_id(0)  # head id
    # First pass: find max for numerical stability
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + h * L_tokens + t)
        if logit > m:
            m = logit
    # Second pass: compute sum exp(logits - m)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + h * L_tokens + t)
        sum_exp += tl.exp(logit - m)
    lse = tl.log(sum_exp) / ln2
    tl.store(lse_ptr + h, lse)


@triton.jit
def compute_output_kernel(
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    logits_ptr,     # *fp32, [H * L_tokens], contiguous
    output_ptr,     # *fp32, [H * D], contiguous
    H,              # int32
    D: tl.constexpr,         # 512
    L_tokens,       # int32
):
    h = tl.program_id(0)  # head id
    # Compute output[h, :] = softmax(logits_scaled[h, :]) @ Kc[:, :]
    # We need logits_scaled[h, :] = logits_ptr[h * L_tokens + t]
    for k in range(0, D):
        tl.store(output_ptr + h * D + k, 0.0)  # initialize output row to zero
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + h * L_tokens + t)
        # softmax probability for token t
        # sum_exp was computed in lse kernel as exp(logit - m). Here we need to recompute or rely on lse; but we can compute probabilities directly.
        # Since we don't have lse here, compute probability with max m found in lse kernel by reading it? Instead, compute exp(logit - max) and sum in this kernel.
        # However, we need the sum of exp(logit - m) across all t. We'll compute that here to avoid extra kernel.
        # Define m as max of logits for this head. We can find m by scanning logits_ptr in this kernel.
        m = -float("inf")
        for it in range(0, L_tokens):
            logit_it = tl.load(logits_ptr + h * L_tokens + it)
            if logit_it > m:
                m = logit_it
        sum_exp = 0.0
        for it in range(0, L_tokens):
            logit_it = tl.load(logits_ptr + h * L_tokens + it)
            sum_exp += tl.exp(logit_it - m)
        attn = tl.exp(logit - m) / sum_exp
        # Add contribution from Kc[t, :] scaled by attn to output[h, :]
        for kk in range(0, D):
            kc = tl.load(Kc_ptr + t * D + kk)
            out = tl.load(output_ptr + h * D + kk)
            out += attn * kc
            tl.store(output_ptr + h * D + kk, out)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes and assertions
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        num_pages, _, _ = ckv_cache.shape
        _, _, _ = kpe_cache.shape

        # Cast caches to fp32 for computation
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

        device = q_nope.device

        # Output buffers (we will compute in fp32 and cast to bfloat16 at the end)
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        H = num_qo_heads
        D = head_dim_ckv  # 512
        Dp = head_dim_kpe  # 64
        ln2 = 1.4426950408889634  # math.log(2)

        # Iterate per batch element
        for b in range(batch_size):
            # Number of valid tokens for this batch element
            # len_indptr = kv_indptr.shape[0] is the number of entries (batch_size + 1 usually)
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV tokens for this batch element
                lse[b].fill_(-float("inf"))
                output[b].zero_()
                continue

            # Gather tokens indices for this batch
            tok_idx = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())]  # [L_tokens]
            # Gather Kc and Kp rows for those tokens
            Kc = Kc_all[tok_idx].contiguous()  # [L_tokens, D] fp32
            Kp = Kp_all[tok_idx].contiguous()  # [L_tokens, Dp] fp32

            # Cast q_nope and q_pe to fp32 for kernel computation
            qn = q_nope[b].to(torch.float32).contiguous()  # [H, D]
            qp = q_pe[b].to(torch.float32).contiguous()   # [H, Dp]

            # Allocate logits buffer [H * L_tokens]
            logits = torch.empty(H * L_tokens, dtype=torch.float32, device=device)

            # Launch Triton kernel to compute logits_scaled
            # Grid over heads (1D): H programs
            grid = (H,)
            compute_logits_scaled_kernel[grid](
                qn, qp, Kc, Kp, logits,
                H, D, Dp, L_tokens,
                num_warps=1, num_stages=1
            )

            # Launch Triton kernel to compute lse per head
            grid = (H,)
            compute_lse_kernel[grid](
                logits, lse[b],
                H, L_tokens, ln2,
                num_warps=1, num_stages=1
            )

            # Launch Triton kernel to compute output per head
            grid = (H,)
            output[b].zero_()  # ensure output row is zero
            compute_output_kernel[grid](
                Kc, logits, output[b],
                H, D, L_tokens,
                num_warps=1, num_stages=1
            )

        # Return output in bfloat16, lse in float32, matching original signature
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
