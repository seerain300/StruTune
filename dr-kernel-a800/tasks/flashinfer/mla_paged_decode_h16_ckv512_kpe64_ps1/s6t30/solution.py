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
    D: tl.constexpr,          # compile-time constant: 512
    Dp: tl.constexpr,         # compile-time constant: 64
    L_tokens,       # int32
    sm_scale,       # fp32
):
    # 2D grid: (h, t)
    h = tl.program_id(0)
    t = tl.program_id(1)

    # Row pointers for qn/h and Kc/t
    qn_row = qn_ptr + h * D
    Kc_row = Kc_ptr + t * D
    qp_row = qp_ptr + h * Dp
    Kp_row = Kp_ptr + t * Dp

    acc1 = 0.0
    for kk in range(0, D):
        val = tl.load(qn_row + kk)
        kv = tl.load(Kc_row + kk)
        acc1 += val * kv

    acc2 = 0.0
    for kk in range(0, Dp):
        val = tl.load(qp_row + kk)
        kv = tl.load(Kp_row + kk)
        acc2 += val * kv

    logit_scaled = (acc1 + acc2) * sm_scale
    # Store to logits[H, L_tokens] contiguous: offset = h * L_tokens + t
    tl.store(logits_ptr + h * L_tokens + t, logit_scaled)


@triton.jit
def compute_lse_per_batch_kernel(
    logits_ptr,     # *fp32, shape [H, L_tokens], contiguous
    lse_ptr,        # *fp32, shape [H], contiguous
    H,              # int32
    L_tokens,       # int32
):
    # Grid: (h,)
    h = tl.program_id(0)
    row_ptr = logits_ptr + h * L_tokens

    # Pass 1: max for numerical stability
    m = -float("inf")
    for t in range(0, L_tokens):
        x = tl.load(row_ptr + t)
        if x > m:
            m = x

    # Pass 2: sum exp(x - m) / ln(2)
    ln2 = 1.4426950408889634  # 1 / ln(2)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        x = tl.load(row_ptr + t)
        sum_exp += tl.exp(x - m)

    lse_val = tl.log(sum_exp) + m  # logsumexp
    lse_val = lse_val / ln2        # divide by ln(2) as per original code
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def compute_output_per_batch_kernel(
    logits_ptr,     # *fp32, shape [H, L_tokens], contiguous
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    out_ptr,        # *fp32, shape [H, D], contiguous
    H,              # int32
    D: tl.constexpr,          # compile-time constant: 512
    L_tokens,       # int32
):
    # 2D grid: (h, kk)
    h = tl.program_id(0)
    kk = tl.program_id(1)

    row_ptr = logits_ptr + h * L_tokens
    out_row = out_ptr + h * D

    # Accumulate over t: out[h, kk] = sum_t exp(logits_scaled[h, t]) * Kc[t, kk]
    acc = 0.0
    for t in range(0, L_tokens):
        x = tl.load(row_ptr + t)
        e = tl.exp(x)
        kv = tl.load(Kc_ptr + t * D + kk)
        acc += e * kv

    tl.store(out_row + kk, acc)


def run_triton_per_batch(q_nope, q_pe, Kc_all, Kp_all, kv_indptr, kv_indices, sm_scale):
    """
    Compute output and lse per batch using Triton kernels.
    q_nope: [B, H, D] bfloat16 or float32
    q_pe: [B, H, Dp] bfloat16 or float32
    Kc_all: [num_pages, D] bfloat16 -> cast to fp32 for kernel
    Kp_all: [num_pages, Dp] bfloat16 -> cast to fp32 for kernel
    kv_indptr: [B+1] int32
    kv_indices: [num_kv_indices] int32
    sm_scale: float32
    Returns:
      output: [B, H, D] bfloat16
      lse: [B, H] float32
    """
    assert q_nope.is_cuda and q_pe.is_cuda and Kc_all.is_cuda and Kp_all.is_cuda, "Tensors must be on CUDA device."
    device = q_nope.device
    B, H, D = q_nope.shape
    _, _, Dp = q_pe.shape
    assert D == 512 and Dp == 64, "Expected head dims: D=512, Dp=64."

    # Cast inputs to fp32 for computation
    q_nope_fp32 = q_nope.to(torch.float32)
    q_pe_fp32 = q_pe.to(torch.float32)
    Kc_all_fp32 = Kc_all.to(torch.float32)
    Kp_all_fp32 = Kp_all.to(torch.float32)

    output = torch.empty((B, H, D), dtype=torch.float32, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    for b in range(B):
        # Compute number of valid tokens for this batch element
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L_tokens = max(page_end - page_beg, 0)

        if L_tokens <= 0:
            # No valid tokens: output zeros, lse = -inf
            output[b].zero_()
            lse[b].fill_(-float("inf"))
            continue

        # Slice indices for this batch
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()  # [L_tokens]
        # Gather Kc, Kp for these tokens
        Kc = Kc_all_fp32[tok_idx]  # [L_tokens, D]
        Kp = Kp_all_fp32[tok_idx]  # [L_tokens, Dp]

        # Prepare qn (for each head) and qp (for each head) contiguous
        qn = q_nope_fp32[b]  # [H, D]
        qp = q_pe_fp32[b]    # [H, Dp]

        # Launch kernel to compute logits_scaled [H, L_tokens]
        logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
        grid = (H, L_tokens)
        compute_logits_scaled_per_batch_kernel[grid](
            qn, qp, Kc, Kp, logits, H, D, Dp, L_tokens, sm_scale,
            num_warps=1, num_stages=1
        )

        # Launch kernel to compute lse per head
        grid_lse = (H,)
        compute_lse_per_batch_kernel[grid_lse](logits, lse[b], H, L_tokens, num_warps=1, num_stages=1)

        # Launch kernel to compute output [H, D]
        output[b] = torch.empty((H, D), dtype=torch.float32, device=device)
        grid_out = (H, D)
        compute_output_per_batch_kernel[grid_out](logits, Kc, output[b], H, D, L_tokens, num_warps=1, num_stages=1)

    # Cast output to bfloat16 to match original return dtype
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA device
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA device."

        # Use Triton kernels to compute output and lse per batch
        output, lse = run_triton_per_batch(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)
        return output, lse


def run(*args):
    return ModelNew()(*args)
