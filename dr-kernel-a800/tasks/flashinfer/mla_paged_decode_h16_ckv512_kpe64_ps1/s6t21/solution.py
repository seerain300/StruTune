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
    sm_scale,       # float32
):
    # 2D grid: (h in 0..H-1, t in 0..L_tokens-1)
    h = tl.program_id(0)
    t = tl.program_id(1)

    base = h * L_tokens

    # Accumulate dot products over D and Dp (both compile-time constants)
    acc1 = 0.0
    D_const = 512
    Dp_const = 64
    for kk in range(0, D_const):
        acc1 += tl.load(qn_ptr + h * D_const + kk) * tl.load(Kc_ptr + t * D_const + kk)

    acc2 = 0.0
    for kk in range(0, Dp_const):
        acc2 += tl.load(qp_ptr + h * Dp_const + kk) * tl.load(Kp_ptr + t * Dp_const + kk)

    logit = (acc1 + acc2) * sm_scale
    tl.store(logits_ptr + base + t, logit)


@triton.jit
def compute_lse_per_batch_kernel(
    logits_ptr,     # *fp32, [H, L_tokens], contiguous
    lse_ptr,        # *fp32, [H], contiguous
    H,              # int32
    L_tokens,       # int32
):
    # 1D grid: (h in 0..H-1)
    h = tl.program_id(0)
    base = h * L_tokens

    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + base + t)
        m = tl.maximum(m, logit)

    sumexp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + base + t)
        sumexp += tl.exp(logit - m)

    lse_val = (m + tl.log(sumexp)) / math.log(2.0)
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def compute_output_per_batch_matvec_kernel(
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    logits_ptr,     # *fp32, [H, L_tokens], contiguous
    output_ptr,     # *fp32, [H, D], contiguous
    H,              # int32
    D: tl.constexpr,         # compile-time D=512
    L_tokens,       # int32
    sm_scale,       # float32 (not used here, but kept for signature consistency)
):
    # 1D grid: (h in 0..H-1)
    h = tl.program_id(0)
    base_log = h * L_tokens

    out = tl.zeros((D,), dtype=tl.float32)
    for t in range(0, L_tokens):
        p = tl.load(logits_ptr + base_log + t)
        p = tl.exp(p)  # softmax over logits_scaled
        for kk in range(0, D):
            out[kk] += p * tl.load(Kc_ptr + t * D + kk)

    tl.store(output_ptr + h * D + tl.arange(0, D), out)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, D] (B expected 16, H=16, D=512)
        q_pe: [B, H, Dp] (Dp=64)
        ckv_cache: [num_pages, 1, D]
        kpe_cache: [num_pages, 1, Dp]
        kv_indptr: [B+1] int32
        kv_indices: [N] int32 (tokens used per batch; we use full range [0..N-1])
        sm_scale: float32 scalar
        Returns:
        output: [B, H, D] in bfloat16
        lse: [B, H] in float32
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA"
        B, H, D = q_nope.shape
        Bp, Hp, Dp = q_pe.shape
        assert B == Bp and H == Hp, "Shape mismatch between q_nope and q_pe"
        assert D == 512 and Dp == 64, "Expected D=512 and Dp=64"
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "Expected single cache dim"
        assert kv_indptr.shape[0] == B + 1, "kv_indptr length must be B+1"

        # Cast queries to fp32 for compute
        q_nope_f = q_nope.to(torch.float32).contiguous()   # [B, H, D]
        q_pe_f = q_pe.to(torch.float32).contiguous()       # [B, H, Dp]
        # Gather all cached Kc/Kp
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Dp]

        output = torch.empty((B, H, D), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        for b in range(B):
            # Compute number of tokens for this batch element
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                # No tokens for this batch element: output zeros, lse -inf
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Prepare per-batch queries
            qn_b = q_nope_f[b]                       # [H, D]
            qp_b = q_pe_f[b]                        # [H, Dp]

            # Prepare cached Kc/Kp for this batch's tokens
            Kc_batch = Kc_all[start:end].contiguous()  # [L_tokens, D]
            Kp_batch = Kp_all[start:end].contiguous()  # [L_tokens, Dp]

            # Allocate logits buffer [H, L_tokens]
            logits_buf = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)

            # Launch kernel to compute logits_scaled
            grid_logits = (H, L_tokens)
            compute_logits_scaled_per_batch_kernel[grid_logits](
                qn_b, qp_b, Kc_batch, Kp_batch, logits_buf, H, L_tokens, sm_scale
            )

            # Launch kernel to compute lse per head
            grid_lse = (H,)
            compute_lse_per_batch_kernel[grid_lse](logits_buf, lse[b], H, L_tokens)

            # Launch kernel to compute output per head (matvec)
            grid_out = (H,)
            compute_output_per_batch_matvec_kernel[grid_out](
                Kc_batch, logits_buf, output[b], H, D, L_tokens, sm_scale
            )

        # Return output in bfloat16, lse in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
