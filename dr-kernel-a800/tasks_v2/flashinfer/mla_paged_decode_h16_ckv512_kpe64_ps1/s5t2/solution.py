import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_logits_kernel(
    qn_ptr,        # [H, CK], base pointer
    qp_ptr,        # [H, KP], base pointer
    Kc_all_ptr,    # [P, CK], base pointer
    Kp_all_ptr,    # [P, KP], base pointer
    tok_idx_ptr,   # [L_tokens], int32
    logits_ptr,    # [H, L_tokens], float32, base pointer
    H: tl.constexpr,          # num_qo_heads
    CK: tl.constexpr,         # head_dim_ckv
    KP: tl.constexpr,         # head_dim_kpe
    L_tokens: tl.constexpr,   # number of tokens for this batch element
):
    # Grid: (H, L_tokens)
    h = tl.program_id(0)
    t = tl.program_id(1)
    if (h >= H) or (t >= L_tokens):
        return

    # Load qn[h, :] and qp[h, :]
    dim_ck = tl.arange(0, CK)
    dim_kp = tl.arange(0, KP)
    qn_vec = tl.load(qn_ptr + h * CK + dim_ck)   # [CK]
    qp_vec = tl.load(qp_ptr + h * KP + dim_kp)   # [KP]

    # Load Kc_row and Kp_row for this token
    idx = tl.load(tok_idx_ptr + t)               # int32
    Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK]
    Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)  # [KP]

    # Compute dot-products
    dot_qn = tl.sum(qn_vec * Kc_row)            # scalar
    dot_qp = tl.sum(qp_vec * Kp_row)            # scalar
    logits_h_t = dot_qn + dot_qp                 # unscaled logits for this (h, t)

    # Store to logits[H, L_tokens]
    tl.store(logits_ptr + h * L_tokens + t, logits_h_t)


@triton.jit
def _lse_kernel(
    logits_ptr,     # [H, L_tokens], float32
    lse_ptr,        # [H], float32
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # Grid: (H,)
    h = tl.program_id(0)
    if h >= H:
        return

    # Compute max across tokens
    max_val = -1.0e30
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        max_val = tl.maximum(max_val, val)

    # Compute sum of exp(logits - max) across tokens
    sumexp = 0.0
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        sumexp += tl.exp(val - max_val)

    # lse is log(sumexp); we'll divide by ln(2) on host to match original
    lse_h = tl.log(sumexp)
    tl.store(lse_ptr + h, lse_h)


@triton.jit
def _compute_output_kernel(
    scaled_logits_ptr,  # [H, L_tokens], float32 (already scaled by sm_scale)
    lse_ptr,            # [H], float32
    Kc_all_ptr,         # [P, CK], float32
    tok_idx_ptr,        # [L_tokens], int32
    out_ptr,            # [H, CK], float32
    H: tl.constexpr,
    CK: tl.constexpr,
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,  # should match scaling used in scaled_logits
):
    # Grid: (H,)
    h = tl.program_id(0)
    if h >= H:
        return

    # Load per-head lse
    lse_h = tl.load(lse_ptr + h)

    # Compute output[h, :] = sum_t softmax(scaled_logits[h, t]) * Kc[t, :]
    # Initialize accumulator
    out_acc = tl.zeros((CK,), tl.float32)

    for t in range(L_tokens):
        # Load scaled logits for this token
        val = tl.load(scaled_logits_ptr + h * L_tokens + t)
        p = tl.exp(val - lse_h)  # softmax probability for this token
        # Gather Kc_row corresponding to tok_idx[t]
        idx = tl.load(tok_idx_ptr + t)  # int32
        Kc_row = tl.load(Kc_all_ptr + idx * CK + tl.arange(0, CK))  # [CK]
        out_acc += p * Kc_row

    # Store output
    tl.store(out_ptr + h * CK + tl.arange(0, CK), out_acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton."

        # Shapes
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages_ckv = ckv_cache.shape[0]
        num_pages_kpe = kpe_cache.shape[0]
        assert num_pages_ckv == num_pages_kpe, "ckv_cache and kpe_cache must have same first dimension."
        assert head_dim_ckv == 512 and head_dim_kpe == 64, "Expected head dims: CK=512, KP=64."

        device = q_nope.device

        # Outputs
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse_host = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Convert caches to float32 for compute; Triton will read float32
        Kc_all = ckv_cache.to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.to(torch.float32).contiguous()  # [num_pages, 64]

        # Flatten q_nope and q_pe to [B*H, D] for base pointers
        qn_flat = q_nope.to(torch.float32).contiguous().view(batch_size * num_qo_heads, head_dim_ckv)  # [B*H, CK]
        qp_flat = q_pe.to(torch.float32).contiguous().view(batch_size * num_qo_heads, head_dim_kpe)  # [B*H, KP]

        for b in range(batch_size):
            # Compute L_tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                output[b].zero_()
                lse_host[b] = -float("inf")
                continue

            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32).contiguous()  # [L_tokens]

            # Base pointers for qn and qp for this batch
            qn_batch_ptr = qn_flat[b * num_qo_heads: (b + 1) * num_qo_heads]  # [H, CK]
            qp_batch_ptr = qp_flat[b * num_qo_heads: (b + 1) * num_qo_heads]  # [H, KP]

            # Allocate logits buffer [H, L_tokens], float32
            logits = torch.empty((num_qo_heads, L_tokens), dtype=torch.float32, device=device)

            # Launch Triton kernel to compute logits (unscaled)
            grid_log = (num_qo_heads, L_tokens)
            _compute_logits_kernel[grid_log](
                qn_batch_ptr, qp_batch_ptr,
                Kc_all, Kp_all,
                tok_idx,
                logits,
                H=num_qo_heads,
                CK=head_dim_ckv,
                KP=head_dim_kpe,
                L_tokens=L_tokens,
                num_warps=4,
            )

            # Compute scaled_logits = logits * sm_scale
            scaled_logits = logits * sm_scale

            # Compute lse per head using Triton reduction
            grid_lse = (num_qo_heads,)
            lse_per_head = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
            _lse_kernel[grid_lse](
                scaled_logits,
                lse_per_head,
                H=num_qo_heads,
                L_tokens=L_tokens,
                num_warps=1,
            )

            # Compute output per head using Triton
            out_accum = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
            _compute_output_kernel[grid_lse](
                scaled_logits,
                lse_per_head,
                Kc_all,
                tok_idx,
                out_accum,
                H=num_qo_heads,
                CK=head_dim_ckv,
                L_tokens=L_tokens,
                sm_scale=sm_scale,
                num_warps=4,
            )

            # Store outputs: cast to bfloat16 and place in output[b]
            output[b] = out_accum.to(torch.bfloat16)
            lse_host[b] = lse_per_head

        # Divide lse_host by ln(2) to match original behavior
        if lse_host.numel() > 0:
            lse_host = lse_host / math.log(2.0)

        return output, lse_host


def run(*args):
    return ModelNew()(*args)
