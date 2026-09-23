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
    logits_ptr,     # *fp32, flattened [H*L_tokens]
    H,              # int32
    D: tl.constexpr,          # compile-time constant: 512
    Dp: tl.constexpr,         # compile-time constant: 64
    L_tokens,       # int32
    sm_scale,       # float32
):
    # 2D grid: (h in 0..H-1, t in 0..L_tokens-1)
    h = tl.program_id(0)
    t = tl.program_id(1)

    # Accumulate dot products in fp32
    acc1 = 0.0
    for k in range(0, D):
        val = tl.load(qn_ptr + h * D + k)  # qn[h, k]
        kv = tl.load(Kc_ptr + t * D + k)   # Kc[t, k]
        acc1 += val * kv

    acc2 = 0.0
    for k in range(0, Dp):
        val = tl.load(qp_ptr + h * Dp + k) # qp[h, k]
        kv = tl.load(Kp_ptr + t * Dp + k)  # Kp[t, k]
        acc2 += val * kv

    logit = (acc1 + acc2) * sm_scale
    idx = h * L_tokens + t
    tl.store(logits_ptr + idx, logit)


@triton.jit
def compute_lse_row_kernel(
    logits_ptr,     # *fp32, flattened [H*L_tokens]
    lse_ptr,        # *fp32, shape [H]
    H,              # int32
    L_tokens,       # int32
    h,              # int32
):
    # Single program per head
    base = h * L_tokens
    # Pass 1: compute max for numerical stability
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + base + t)
        m = tl.maximum(m, logit)

    # Pass 2: compute sum(exp(logits - m))
    sumexp = 0.0
    ln2 = 1.4426950408889634  # 1 / ln(2)
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + base + t)
        sumexp += tl.exp(logit - m) / ln2

    lse = tl.log(sumexp) + m
    tl.store(lse_ptr + h, lse)


@triton.jit
def compute_output_row_kernel(
    logits_ptr,     # *fp32, flattened [H*L_tokens]
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    output_ptr,     # *fp32, shape [H, D] flattened as [H*D]
    H,              # int32
    D: tl.constexpr,          # 512
    L_tokens,       # int32
    h,              # int32
):
    # Single program per head
    base = h * L_tokens

    # Compute softmax over tokens: s[t] = exp(logits_scaled[t] - m) / Z
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + base + t)
        m = tl.maximum(m, logit)

    Z = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + base + t)
        Z += tl.exp(logit - m)

    # Now compute output = s @ Kc
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + base + t)
        s = tl.exp(logit - m) / Z  # softmax value for token t
        for k in range(0, D):
            kv = tl.load(Kc_ptr + t * D + k)  # Kc[t, k]
            out_vec[k] += s * kv

    out_base = h * D
    for k in range(0, D):
        tl.store(output_ptr + out_base + k, out_vec[k])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; everything is computed in Triton.

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure all tensors are on the same CUDA device
        device = q_nope.device
        assert q_pe.device == device and ckv_cache.device == device and kpe_cache.device == device and kv_indptr.device == device and kv_indices.device == device, \
            "All tensors must be on the same CUDA device."

        # Cast q to float32 and ensure contiguous
        qn = q_nope.to(torch.float32).contiguous()  # [1, H, D] but we index H by stride, so H=q_nope.shape[1]
        qp = q_pe.to(torch.float32).contiguous()    # [1, H, Dp] -> same H
        # Select Kc and Kp for batch b=0
        b = 0
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if page_beg >= page_end:
            # No KV cache for this batch element; match original behavior
            H = qn.shape[1]
            D = qn.shape[2]
            output = torch.zeros((1, H, D), dtype=torch.bfloat16, device=device)
            lse = torch.full((1, H), -float("inf"), dtype=torch.float32, device=device)
            return output, lse

        L_tokens = page_end - page_beg
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [L_tokens]

        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, Dp]

        Kc = Kc_all[tok_idx]  # [L_tokens, D]
        Kp = Kp_all[tok_idx]  # [L_tokens, Dp]
        Kc = Kc.contiguous()
        Kp = Kp.contiguous()

        H = qn.shape[1]
        D = qn.shape[2]
        Dp = qp.shape[2]
        assert D == 512 and Dp == 64, "Expected head_dim_ckv=512 and head_dim_kpe=64."

        # Allocate intermediate and output buffers
        logits = torch.empty((H * L_tokens,), dtype=torch.float32, device=device)
        lse = torch.empty((H,), dtype=torch.float32, device=device)
        output = torch.empty((H * D,), dtype=torch.float32, device=device)  # we'll reshape to [H, D] later

        # Launch Triton kernels
        grid_logit = (H, L_tokens)
        compute_logits_scaled_kernel[grid_logit](
            qn.view(H, D), qp.view(H, Dp), Kc, Kp, logits, H, D, Dp, L_tokens, sm_scale
        )

        grid_lse = (H,)
        compute_lse_row_kernel[grid_lse](logits, lse, H, L_tokens)

        grid_out = (H,)
        compute_output_row_kernel[grid_out](logits, Kc, output, H, D, L_tokens)

        # Reshape output to [H, D] and cast to bfloat16
        output = output.view(H, D).to(torch.bfloat16)

        # Return shape [1, H, D] for output and [1, H] for lse to match original
        output = output.unsqueeze(0)
        lse = lse.unsqueeze(0)

        return output, lse


def run(*args):
    return ModelNew()(*args)
