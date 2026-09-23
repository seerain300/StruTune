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
    logits_ptr,     # *fp32, flattened buffer [B*H*L_tokens], we fill only for the current batch
    H,              # int32
    D: tl.constexpr,          # compile-time constants for dims
    Dp: tl.constexpr,
    L_tokens,       # int32
    b,              # int32 batch id (runtime scalar)
    sm_scale,       # float32 scaling factor
):
    # 2D grid: (h in 0..H-1, t in 0..L_tokens-1)
    h = tl.program_id(0)
    t = tl.program_id(1)

    # Compute flat index for logits[b, h, t]
    idx = (b * H + h) * L_tokens + t

    # Compute acc1 = qn[h, :] @ Kc[t, :]
    acc1 = 0.0
    for kk in range(0, D):
        val = tl.load(qn_ptr + h * D + kk)  # qn is [H, D], row-major
        kv = tl.load(Kc_ptr + t * D + kk)   # Kc is [L_tokens, D], row-major
        acc1 += val * kv

    # Compute acc2 = qp[h, :] @ Kp[t, :]
    acc2 = 0.0
    for kk in range(0, Dp):
        val = tl.load(qp_ptr + h * Dp + kk)  # qp is [H, Dp]
        kv = tl.load(Kp_ptr + t * Dp + kk)   # Kp is [L_tokens, Dp]
        acc2 += val * kv

    logit = (acc1 + acc2) * sm_scale
    tl.store(logits_ptr + idx, logit)


@triton.jit
def compute_lse_per_batch_kernel(
    logits_ptr,     # *fp32, flattened buffer [B*H*L_tokens]
    lse_ptr,        # *fp32, buffer [B*H]
    H,              # int32
    L_tokens,       # int32
    b,              # int32
):
    # Grid: (h in 0..H-1)
    h = tl.program_id(0)
    base = b * H + h
    row_start = base * L_tokens

    # Compute max for numerical stability
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        if logit > m:
            m = logit

    # Compute sum of exp(logits - m)
    sumexp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        sumexp += tl.exp(logit - m)

    lse_val = (m + tl.log(sumexp)) / math.log(2.0)  # convert to base-2
    tl.store(lse_ptr + base, lse_val)


@triton.jit
def compute_output_per_batch_matvec_kernel(
    logits_ptr,     # *fp32, flattened buffer [B*H*L_tokens]
    Kc_ptr,         # *fp32, [L_tokens, D]
    output_ptr,     # *fp32, flattened buffer [B*H*D], we write per head
    H,              # int32
    D: tl.constexpr,
    L_tokens,       # int32
    b,              # int32
):
    # Grid: (h in 0..H-1)
    h = tl.program_id(0)
    base = b * H + h

    # Compute softmax over L_tokens per row, then matrix multiply by Kc to get output
    # output[h, kk] = sum_t softmax(logits_scaled[h, t]) * Kc[t, kk]
    for kk in range(0, D):
        m = -float("inf")
        for t in range(0, L_tokens):
            logit = tl.load(logits_ptr + base * L_tokens + t)
            if logit > m:
                m = logit
        sumexp = 0.0
        for t in range(0, L_tokens):
            logit = tl.load(logits_ptr + base * L_tokens + t)
            sumexp += tl.exp(logit - m)
        inv_sumexp = 1.0 / sumexp
        out_kk = 0.0
        for t in range(0, L_tokens):
            logit = tl.load(logits_ptr + base * L_tokens + t)
            p = tl.exp(logit - m) * inv_sumexp
            kc = tl.load(Kc_ptr + t * D + kk)
            out_kk += p * kc
        tl.store(output_ptr + base * D + kk, out_kk)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, D] (B=1 in provided get_inputs, but we handle generic)
        q_pe: [B, H, Dp]
        ckv_cache: [num_pages, 1, D]
        kpe_cache: [num_pages, 1, Dp]
        kv_indptr: [B+1] int32
        kv_indices: [N] int32, tokens per batch
        sm_scale: float32
        Returns:
        output: [B, H, D] in bfloat16
        lse: [B, H] in float32
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA"
        B, H, D = q_nope.shape
        Bp, _, Dp = q_pe.shape
        assert B == Bp, "Batch size mismatch between q_nope and q_pe"
        assert D == 512 and Dp == 64, "Expected head dims 512 and 64 respectively"
        num_pages = ckv_cache.shape[0]
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "Expected single page dimension"
        assert kv_indptr.shape[0] == B + 1, "kv_indptr length must be B+1"

        # Cast input queries to fp32 for computation
        q_nope_f = q_nope.to(torch.float32).contiguous()   # [B, H, D]
        q_pe_f = q_pe.to(torch.float32).contiguous()       # [B, H, Dp]
        # Gather all Kc/Kp for entire cache
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Dp]

        output = torch.empty((B, H, D), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        for b in range(B):
            # Compute number of tokens for this batch element
            L


def run(*args):
    return ModelNew()(*args)
