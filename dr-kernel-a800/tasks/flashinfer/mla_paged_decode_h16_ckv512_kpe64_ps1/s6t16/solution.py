import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits_scaled[b, h, t] for a given batch b, head h, all tokens t
@triton.jit
def compute_logits_scaled_per_batch_kernel(
    qn_ptr,         # *fp32, [H, D], contiguous
    qp_ptr,         # *fp32, [H, Dp], contiguous
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, [L_tokens, Dp], contiguous
    logits_ptr,     # *fp32, flattened [B*H*L_tokens]
    H,              # int32 runtime
    D: tl.constexpr,          # 512 (compile-time constant)
    Dp: tl.constexpr,         # 64 (compile-time constant)
    L_tokens,       # int32 runtime
    b,              # int32 batch id
    sm_scale,       # float32
):
    # Grid: (h, t) where t in [0, L_tokens)
    h = tl.program_id(0)
    t = tl.program_id(1)

    # Compute address for logits[b, h, t]
    idx = (b * H + h) * L_tokens + t

    # Load qn row [D] and Kc row [D], accumulate dot
    acc = 0.0
    for kk in range(0, D):
        qk = tl.load(qn_ptr + h * D + kk)  # qn_ptr is contiguous: row offset is h*D
        kc = tl.load(Kc_ptr + t * D + kk)  # Kc_ptr row offset is t*D
        acc += qk * kc

    # Load qp row [Dp] and Kp row [Dp], accumulate dot
    for kk in range(0, Dp):
        qk = tl.load(qp_ptr + h * Dp + kk)
        kp = tl.load(Kp_ptr + t * Dp + kk)
        acc += qk * kp

    logit_scaled = acc * sm_scale
    tl.store(logits_ptr + idx, logit_scaled)


# Triton kernel: compute lse[b, h] = logsumexp(logits_scaled[b, h, :]) / ln(2)
@triton.jit
def compute_lse_per_batch_kernel(
    logits_ptr,     # *fp32, flattened [B*H*L_tokens]
    lse_ptr,        # *fp32, [B*H]
    H,              # int32
    L_tokens,       # int32
    b,              # int32
):
    # Grid: (h,)
    h = tl.program_id(0)
    base = b * H + h
    row_start = base * L_tokens

    # Compute max for numerical stability
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        if logit > m:
            m = logit

    # Compute sum exp(logits - m)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        sum_exp += tl.exp(logit - m)

    lse_val = tl.log(sum_exp) / math.log(2.0)
    tl.store(lse_ptr + base, lse_val)


# Triton kernel: compute output[b, h, :] = softmax(logits_scaled[b, h, :]) @ Kc[:, :]
@triton.jit
def compute_output_per_batch_kernel(
    logits_ptr,     # *fp32, flattened [B*H*L_tokens]
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    output_ptr,     # *fp32, flattened [B*H*D]
    H,              # int32
    D: tl.constexpr,          # 512
    L_tokens,       # int32
    b,              # int32
):
    # Grid: (h, d) where d in [0, D)
    h = tl.program_id(0)
    d = tl.program_id(1)

    base = b * H + h
    row_start = base * L_tokens
    out_row = output_ptr + (b * H + h) * D + d  # output_ptr is contiguous across H, then D

    # Compute softmax over logits_scaled[b, h, :]
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        if logit > m:
            m = logit

    sum_exp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        sum_exp += tl.exp(logit - m)

    # Accumulate output[b, h, d] = sum_t softmax_t * Kc[t, d]
    acc = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        p = tl.exp(logit - m) / sum_exp
        kc = tl.load(Kc_ptr + t * D + d)
        acc += p * kc

    tl.store(output_ptr + (b * H + h) * D + d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation mirroring the original PyTorch logic.
        Returns:
          - output: [batch_size, num_qo_heads, head_dim_ckv] in bfloat16
          - lse: [batch_size, num_qo_heads] in float32
        """
        # Shapes/consts from original assumptions
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape
        num_pages = ckv_cache.shape[0]
        D = head_dim_ckv  # 512
        Dp = head_dim_kpe # 64
        assert num_qo_heads == 16
        assert D == 512
        assert Dp == 64
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1
        # Ensure CUDA tensors
        if not q_nope.is_cuda or not q_pe.is_cuda or not ckv_cache.is_cuda or not kpe_cache.is_cuda:
            raise RuntimeError("Inputs must be CUDA tensors for Triton kernels.")
        # Cast caches to fp32 for computation
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, Dp]

        # Output and lse buffers
        output = torch.empty((batch_size, num_qo_heads, D), dtype=torch.float32, device=q_nope.device)  # compute in fp32, cast later
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q_nope.device)

        # Precompute strides for q_nope and q_pe (contiguous per batch)
        q_nope_fp32 = q_nope.to(torch.float32).contiguous()   # [B, H, D]
        q_pe_fp32 = q_pe.to(torch.float32).contiguous()       # [B, H, Dp]

        for b in range(batch_size):
            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item() - kv_indptr[b].item())  # runtime integer
            # If no tokens, zero output for this batch, lse = -inf
            if L_tokens <= 0:
                lse[b, :] = -float("inf")
                output[b] = 0.0
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[kv_indptr[b].item():kv_indptr[b + 1].item()].to(torch.int32).contiguous()  # [L_tokens]

            # Slice caches for these tokens
            Kc = Kc_all[tok_idx]  # [L_tokens, D]
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

            # Allocate per-(H, L_tokens) logits buffer for this batch
            logits_flat = torch.empty((num_qo_heads * L_tokens,), dtype=torch.float32, device=q_nope.device)

            # Launch kernel: compute logits_scaled[b, h, t] for all h, t
            # Grid: (H, L_tokens)
            grid = (num_qo_heads, L_tokens)
            compute_logits_scaled_per_batch_kernel[grid](
                q_nope_fp32[b], q_pe_fp32[b], Kc, Kp, logits_flat, num_qo_heads, D, Dp, L_tokens, b, sm_scale
            )

            # Compute lse per head for this batch
            grid2 = (num_qo_heads,)
            compute_lse_per_batch_kernel[grid2](logits_flat, lse[b], num_qo_heads, L_tokens, b)

            # Compute output per head for this batch
            grid3 = (num_qo_heads, D)
            compute_output_per_batch_kernel[grid3](logits_flat, Kc, output[b], num_qo_heads, D, L_tokens, b)

        # Cast output to bfloat16 to match original return dtype
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
