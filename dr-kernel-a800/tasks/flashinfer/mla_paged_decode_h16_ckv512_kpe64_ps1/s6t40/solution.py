import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits_scaled[h, t] for a single batch b
# Grid: (H, L_tokens). We use scalar loops because D and Dp are compile-time constants in this problem.
@triton.jit
def compute_logits_scaled_per_batch_kernel(
    qn_ptr,         # *fp32, shape [H, D], contiguous
    qp_ptr,         # *fp32, shape [H, Dp], contiguous
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, shape [L_tokens, Dp], contiguous
    logits_ptr,     # *fp32, buffer [H*L_tokens], indexed via h*L_tokens + t
    H,              # int32 (runtime)
    D: tl.constexpr,          # compile-time constant: 512
    Dp: tl.constexpr,         # compile-time constant: 64
    L_tokens,       # int32 (runtime)
    b,              # int32 batch id (unused in kernel, but kept for future use)
    sm_scale,       # float32
):
    h = tl.program_id(0)
    t = tl.program_id(1)

    # Accumulate dot products in float32
    acc1 = 0.0
    for kk in range(0, D):
        val = tl.load(qn_ptr + h * D + kk)  # qn[h, kk]
        kv = tl.load(Kc_ptr + t * D + kk)   # Kc[t, kk]
        acc1 += val * kv

    acc2 = 0.0
    for kk in range(0, Dp):
        val = tl.load(qp_ptr + h * Dp + kk)
        kv = tl.load(Kp_ptr + t * Dp + kk)
        acc2 += val * kv

    logit = (acc1 + acc2) * sm_scale
    idx = h * L_tokens + t
    tl.store(logits_ptr + idx, logit)


# Triton kernel: compute lse[h] for a single batch b using its logits buffer
@triton.jit
def compute_lse_per_batch_kernel(
    logits_ptr,     # *fp32, buffer [H*L_tokens], indexed via h*L_tokens + t
    lse_ptr,        # *fp32, buffer [H], indexed via h
    H,              # int32
    L_tokens,       # int32
    b,              # int32
):
    h = tl.program_id(0)
    base = b * H + h
    row_start = base * L_tokens

    # Compute max for numerical stability across tokens t
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        m = tl.maximum(m, logit)

    sumexp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        sumexp += tl.exp(logit - m)

    lse = m + tl.log(sumexp)  # natural log
    lse = lse / math.log(2.0)  # convert to base-2
    tl.store(lse_ptr + base, lse)


# Triton kernel: compute output[h, :] for a single batch b using its logits buffer and Kc
@triton.jit
def compute_output_per_batch_kernel(
    logits_ptr,     # *fp32, buffer [H*L_tokens], indexed via h*L_tokens + t
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    output_ptr,     # *fp32, buffer [H*D], indexed via h*D + d
    H,              # int32
    D: tl.constexpr,          # compile-time constant: 512
    L_tokens,       # int32
    b,              # int32
):
    h = tl.program_id(0)
    base = b * H + h

    # output[h, d] = sum_t softmax(logits_scaled[h, t]) * Kc[t, d]
    for d in range(0, D):
        numerator = 0.0
        denom = 0.0
        row_start = base * L_tokens
        for t in range(0, L_tokens):
            logit = tl.load(logits_ptr + row_start + t)
            expv = tl.exp(logit)
            denom += expv
            kv = tl.load(Kc_ptr + t * D + d)  # Kc[t, d]
            numerator += expv * kv
        out_d = numerator / denom
        tl.store(output_ptr + base * D + d, out_d)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes from original model
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        device = q_nope.device
        # Prepare Kc_all, Kp_all as float32 (contiguous)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 64]

        # Allocate outputs
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Iterate batch elements in host to ensure correct per-batch semantics
        for b in range(batch_size):
            # If no tokens for this batch element, set lse to -inf and output zeros; skip
            if kv_indptr[b] >= kv_indptr[b + 1]:
                lse[b].fill_(-float("inf"))
                output[b].zero_()
                continue

            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1] - kv_indptr[b])
            # Indices for this batch element
            tok_idx = kv_indices[kv_indptr[b] : kv_indptr[b + 1]].to(torch.long)  # [L_tokens]
            # Gather Kc and Kp for this batch
            Kc = Kc_all[tok_idx]  # [L_tokens, 512]
            Kp = Kp_all[tok_idx]  # [L_tokens, 64]

            # Prepare inputs as float32 contiguous for kernels; keep original shapes
            qn = q_nope[b].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[b].to(torch.float32).contiguous()    # [16, 64]

            # Buffer for logits_scaled[h, t] for this batch
            logits_flat = torch.empty((num_qo_heads * L_tokens,), dtype=torch.float32, device=device)

            # Launch Triton kernel to compute logits_scaled for all (h, t)
            grid = (num_qo_heads, L_tokens)
            compute_logits_scaled_per_batch_kernel[grid](
                qn, qp, Kc, Kp, logits_flat, num_qo_heads, 512, 64, L_tokens, b, sm_scale
            )

            # Compute lse per head for this batch
            grid_lse = (num_qo_heads,)
            compute_lse_per_batch_kernel[grid_lse](logits_flat, lse[b], num_qo_heads, L_tokens, b)

            # Compute output per head for this batch
            output_flat = torch.empty((num_qo_heads * head_dim_ckv,), dtype=torch.float32, device=device)
            grid_out = (num_qo_heads,)
            compute_output_per_batch_kernel[grid_out](logits_flat, Kc, output_flat, num_qo_heads, 512, L_tokens, b)

            # Reshape and cast output to bfloat16
            output[b] = output_flat.view(num_qo_heads, head_dim_ckv).to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
