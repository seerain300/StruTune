import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_scaled_per_batch_kernel(
    qn_ptr,         # *fp32, shape [H, D], contiguous
    qp_ptr,         # *fp32, shape [H, Dp], contiguous
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, shape [L_tokens, Dp], contiguous
    logits_ptr,     # *fp32, shape [H, L_tokens], contiguous
    H,              # int32
    D: tl.constexpr,         # compile-time constant: 512
    Dp: tl.constexpr,        # compile-time constant: 64
    L_tokens,       # int32 runtime
):
    # One program per head h
    h = tl.program_id(0)
    # For each token t, compute logits_scaled[h, t]
    for t in range(0, L_tokens):
        # Dot qn[h, :] with Kc[t, :]
        acc1 = 0.0
        # Vectorize over D using tl.arange; D is constexpr
        for k in range(0, D):
            q = tl.load(qn_ptr + h * D + k)     # qn[h, k]
            kc = tl.load(Kc_ptr + t * D + k)    # Kc[t, k]
            acc1 += q * kc
        # Dot qp[h, :] with Kp[t, :]
        acc2 = 0.0
        # Vectorize over Dp using tl.arange; Dp is constexpr
        for k in range(0, Dp):
            q = tl.load(qp_ptr + h * Dp + k)    # qp[h, k]
            kp = tl.load(Kp_ptr + t * Dp + k)   # Kp[t, k]
            acc2 += q * kp
        logit = (acc1 + acc2)
        # Store logits[h, t] to contiguous buffer
        tl.store(logits_ptr + h * L_tokens + t, logit)


@triton.jit
def compute_lse_and_output_per_batch_kernel(
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    logits_ptr,     # *fp32, shape [H, L_tokens], contiguous
    output_ptr,     # *fp32, shape [H, D], contiguous
    lse_ptr,        # *fp32, shape [H], contiguous
    H,              # int32
    D: tl.constexpr,         # 512
    L_tokens,       # int32
):
    h = tl.program_id(0)

    # Compute lse for this head: logsumexp(logits_scaled[h, :]) / ln(2)
    ln2 = 1.4426950408889634  # math.log(2)
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
        exp_t = tl.exp(logit - m)
        sum_exp += exp_t
    lse = tl.log(sum_exp) / ln2
    tl.store(lse_ptr + h, lse)

    # Third pass: compute output[h, :] = softmax(logits) @ Kc
    # Initialize output[h, :] to zero
    for k in range(0, D):
        tl.store(output_ptr + h * D + k, 0.0)
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + h * L_tokens + t)
        attn = tl.exp(logit - lse)  # softmax probability for token t
        # Add contribution from Kc[t, :] scaled by attn
        for k in range(0, D):
            kc = tl.load(Kc_ptr + t * D + k)
            out = tl.load(output_ptr + h * D + k, eviction_policy='evict_last')  # safe read
            out += attn * kc
            tl.store(output_ptr + h * D + k, out)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure all inputs are on CUDA
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be CUDA"

        # Cast K caches to float32 and make contiguous
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Dp]

        batch_size = q_nope.shape[0]
        H = q_nope.shape[1]
        D = q_nope.shape[2]  # 512
        Dp = q_pe.shape[2]   # 64

        # Output tensors (compute in fp32, cast to bf16 later)
        output = torch.empty((batch_size, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        # Process each batch element on the host to ensure exact semantics
        for b in range(batch_size):
            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No valid tokens: output zeros and lse -inf
                output[b] = torch.zeros((H, D), dtype=torch.float32, device=device)
                lse[b] = torch.full((H,), -float("inf"), dtype=torch.float32, device=device)
                continue

            # Gather token indices for this batch and corresponding K vectors
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(torch.long)  # [L_tokens]
            Kc = Kc_all[tok_idx].contiguous()  # [L_tokens, D]
            Kp = Kp_all[tok_idx].contiguous()  # [L_tokens, Dp]

            # q_nope[b] and q_pe[b] as float32
            qn = q_nope[b].to(torch.float32).contiguous()  # [H, D]
            qp = q_pe[b].to(torch.float32).contiguous()    # [H, Dp]

            # Allocate per-batch buffers
            logits_scaled = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch kernel to compute logits_scaled[h, t]
            grid = (H,)
            compute_logits_scaled_per_batch_kernel[grid](
                qn, qp, Kc, Kp, logits_scaled, H, D, Dp, L_tokens,
                num_warps=1, num_stages=1,
            )

            # Launch kernel to compute lse and output for each head
            output[b] = torch.zeros((H, D), dtype=torch.float32, device=device)
            lse[b] = torch.full((H,), -float("inf"), dtype=torch.float32, device=device)
            compute_lse_and_output_per_batch_kernel[grid](
                Kc, logits_scaled, output[b], lse[b], H, D, L_tokens,
                num_warps=1, num_stages=1,
            )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
