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
    logits_ptr,     # *fp32, buffer [H*L_tokens]
    H,              # int32 (runtime)
    D: tl.constexpr,          # compile-time constant: 512
    Dp: tl.constexpr,         # compile-time constant: 64
    L_tokens,       # int32 (runtime per-batch tokens)
    b,              # int32 (runtime scalar, used for indexing only)
    sm_scale,       # float32 scaling
):
    # 2D grid: (h in 0..H-1, t in 0..L_tokens-1)
    h = tl.program_id(0)
    t = tl.program_id(1)

    # idx is row-major for [H, L_tokens]
    idx = h * L_tokens + t

    # Load qn row for head h: [D]
    acc1 = 0.0
    for kk in range(0, D):
        val = tl.load(qn_ptr + h * D + kk)
        kv = tl.load(Kc_ptr + t * D + kk)
        acc1 += val * kv

    # Load qp row for head h: [Dp]
    acc2 = 0.0
    for kk in range(0, Dp):
        val = tl.load(qp_ptr + h * Dp + kk)
        kv = tl.load(Kp_ptr + t * Dp + kk)
        acc2 += val * kv

    logit = (acc1 + acc2) * sm_scale
    tl.store(logits_ptr + idx, logit)


@triton.jit
def compute_lse_per_head_kernel(
    logits_ptr,     # *fp32, buffer [H*L_tokens] for this batch
    lse_ptr,        # *fp32, buffer [H] for this batch
    H,              # int32
    L_tokens,       # int32
):
    # grid: (h,)
    h = tl.program_id(0)
    row_start = h * L_tokens

    # Compute max for numerical stability
    m = -float("inf")
    for t in range(0, L_tokens):
        x = tl.load(logits_ptr + row_start + t)
        if x > m:
            m = x

    # Compute sum(exp(x - m))
    s = 0.0
    for t in range(0, L_tokens):
        x = tl.load(logits_ptr + row_start + t)
        s += tl.exp(x - m)

    # lse = log(sum_exp) / log(2)
    ln2 = 0.6931471805599453
    lse_val = tl.log(s) / ln2
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def compute_output_per_head_kernel(
    logits_ptr,     # *fp32, buffer [H*L_tokens] for this batch
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    output_ptr,     # *bf16, buffer [H*D]
    H,              # int32
    D: tl.constexpr,          # compile-time constant: 512
    L_tokens,       # int32
):
    # grid: (h in 0..H-1, d in 0..D-1)
    h = tl.program_id(0)
    d = tl.program_id(1)

    row_start = h * L_tokens

    # First pass: compute max
    m = -float("inf")
    for t in range(0, L_tokens):
        x = tl.load(logits_ptr + row_start + t)
        if x > m:
            m = x

    # Second pass: compute sum of exp(x - m)
    s = 0.0
    for t in range(0, L_tokens):
        x = tl.load(logits_ptr + row_start + t)
        s += tl.exp(x - m)

    # Third pass: accumulate output[h, d] = sum_t softmax_t * Kc[t, d]
    out_val = 0.0
    for t in range(0, L_tokens):
        x = tl.load(logits_ptr + row_start + t)
        p = tl.exp(x - m) / s
        kv = tl.load(Kc_ptr + t * D + d)  # Kc[:, d] at column d
        out_val += p * kv

    # Store output[h, d] as bfloat16
    tl.store(output_ptr + h * D + d, out_val.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA and contiguous
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA."
        device = q_nope.device

        # Constants
        D = 512  # head_dim_ckv
        Dp = 64  # head_dim_kpe
        H = q_nope.shape[1]  # num_qo_heads, should be 16

        # Create output and lse
        batch_size = q_nope.shape[0]
        output = torch.empty((batch_size, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        # Prepare Kc_all and Kp_all as fp32 contiguous
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Dp]

        # Process each batch element
        for b in range(batch_size):
            # Compute per-batch L_tokens
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg
            if L_tokens <= 0:
                # No tokens for this batch element -> output zeros and lse = -inf
                lse[b].fill_(-float("inf"))
                # Fill output with zeros in bfloat16
                output[b].zero_()
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()

            # Slice Kc_all and Kp_all according to tok_idx
            Kc = Kc_all[tok_idx]  # [L_tokens, D]
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

            # qn and qp rows per head, shape [H, D] and [H, Dp]
            qn = q_nope[b].to(torch.float32).contiguous()  # [H, D]
            qp = q_pe[b].to(torch.float32).contiguous()    # [H, Dp]

            # Allocate logits buffer for this batch [H*L_tokens]
            logits = torch.empty((H * L_tokens,), dtype=torch.float32, device=device)

            # Launch kernel to compute logits_scaled[b, :, :]
            grid = (H, L_tokens)
            compute_logits_scaled_per_batch_kernel[grid](
                qn, qp, Kc, Kp, logits,
                H, D, Dp, L_tokens, b, sm_scale,
                num_warps=1, num_stages=1,
            )

            # Compute lse per head for this batch
            grid_lse = (H,)
            lse[b] = compute_lse_per_head_kernel[grid_lse](
                logits, lse[b],
                H, L_tokens,
                num_warps=1, num_stages=1,
            )

            # Compute output[b, :, :] per head
            for h in range(H):
                out_row = torch.empty((D,), dtype=torch.bfloat16, device=device)
                grid_out = (1, D)
                compute_output_per_head_kernel[grid_out](
                    logits, Kc, out_row,
                    H, D, L_tokens,
                    num_warps=1, num_stages=1,
                )
                output[b, h, :] = out_row

        return output, lse


def run(*args):
    return ModelNew()(*args)
