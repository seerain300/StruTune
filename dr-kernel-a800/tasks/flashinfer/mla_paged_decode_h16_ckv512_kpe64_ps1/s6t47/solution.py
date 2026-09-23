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
    H,              # int32
    D: tl.constexpr,          # 512
    Dp: tl.constexpr,         # 64
    L_tokens,       # int32
):
    # Grid: (h in 0..H-1, t in 0..L_tokens-1)
    h = tl.program_id(0)
    t = tl.program_id(1)

    # Compute dot products
    acc1 = 0.0
    for k in range(0, D):
        q = tl.load(qn_ptr + h * D + k)   # qn[h, k]
        kc = tl.load(Kc_ptr + t * D + k)  # Kc[t, k]
        acc1 += q * kc

    acc2 = 0.0
    for k in range(0, Dp):
        q = tl.load(qp_ptr + h * Dp + k)  # qp[h, k]
        kp = tl.load(Kp_ptr + t * Dp + k) # Kp[t, k]
        acc2 += q * kp

    logit = (acc1 + acc2)
    tl.store(logits_ptr + h * L_tokens + t, logit)


@triton.jit
def compute_lse_kernel(
    logits_ptr,     # *fp32, [H, L_tokens] flattened, but indexed as row_start + t
    lse_ptr,        # *fp32, [H]
    H,              # int32
    L_tokens,       # int32
    ln2_inv,        # fp32 = 1 / ln(2)
):
    # One program per head
    h = tl.program_id(0)
    row_start = h * L_tokens

    # Pass 1: max for numerical stability
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        m = tl.maximum(m, logit)

    # Pass 2: sum exp(logits - m)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        sum_exp += tl.exp(logit - m)

    lse = tl.log(sum_exp) * ln2_inv
    tl.store(lse_ptr + h, lse)


@triton.jit
def compute_output_kernel(
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    logits_ptr,     # *fp32, [H, L_tokens] flattened
    output_ptr,     # *fp32, [H, D], contiguous
    H,              # int32
    D: tl.constexpr,          # 512
    L_tokens,       # int32
    ln2_inv,        # fp32
):
    # One program per head
    h = tl.program_id(0)
    row_start = h * L_tokens

    # First, compute lse for this head
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        m = tl.maximum(m, logit)

    sum_exp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        sum_exp += tl.exp(logit - m)

    lse = tl.log(sum_exp) * ln2_inv

    # Compute output[h, :] = softmax(logits) @ Kc
    # Initialize output to zero
    for k in range(0, D):
        tl.store(output_ptr + h * D + k, 0.0)

    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        attn = tl.exp(logit - lse)  # softmax probability for token t
        # Add contribution from Kc[t, :] scaled by attn
        for k in range(0, D):
            kc = tl.load(Kc_ptr + t * D + k)
            tl.store(output_ptr + h * D + k, tl.load(output_ptr + h * D + k) + kc * attn)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Cast caches to fp32 and remove singleton dim
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, Dp]

        batch_size = q_nope.shape[0]
        H = q_nope.shape[1]
        D = q_nope.shape[2]
        Dp = q_pe.shape[2]

        device = q_nope.device
        # Constants for this problem
        assert D == 512 and Dp == 64, "This implementation assumes D=512 and Dp=64."
        ln2_inv = 1.4426950408889634  # 1 / math.log(2)

        # Output and lse buffers
        output = torch.empty((batch_size, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV tokens for this batch element: output zeros, lse = -inf
                lse[b] = -float("inf")
                continue

            # Slice indices for this batch
            tok_idx = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())].to(torch.long)
            if tok_idx.numel() == 0:
                lse[b] = -float("inf")
                continue

            # Gather Kc and Kp for selected tokens
            Kc = Kc_all[tok_idx].contiguous()  # [L_tokens, D]
            Kp = Kp_all[tok_idx].contiguous()  # [L_tokens, Dp]

            # Current batch q_nope and q_pe (H x D, H x Dp)
            qn = q_nope[b].to(torch.float32).contiguous()  # [H, D]
            qp = q_pe[b].to(torch.float32).contiguous()    # [H, Dp]

            # Allocate logits buffer for this batch: [H, L_tokens] flattened as [H*L_tokens]
            logits = torch.empty((H * L_tokens,), dtype=torch.float32, device=device)

            # Launch kernel to compute logits_scaled[h, t]
            grid = (H, L_tokens)
            compute_logits_scaled_kernel[grid](
                qn, qp, Kc, Kp, logits,
                H, D, Dp, L_tokens,
                num_warps=4
            )

            # Launch kernel to compute lse for each head
            grid_lse = (H,)
            compute_lse_kernel[grid_lse](
                logits, lse[b],
                H, L_tokens, ln2_inv,
                num_warps=1
            )

            # Launch kernel to compute output[h, :] = softmax(logits) @ Kc
            grid_out = (H,)
            compute_output_kernel[grid_out](
                Kc, logits, output[b],
                H, D, L_tokens, ln2_inv,
                num_warps=4
            )

        # Return output in bfloat16, lse in float32, matching original
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
